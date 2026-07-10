"""
Tests for the GTSAM alignment axis (sim3 / sl4 pose2point variant).

The aligned pose-to-point factor warps the CURRENT frame's measured camera
points (per-frame extras variable keyed symbol('a', frame)); previous-frame
observations stay un-warped and anchor the landmark scale. Only SE(3) poses
are written back ("estimate + report", mirroring the GEDF backend).
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import pypose as pp

gtsam = pytest.importorskip("gtsam")

from Module.Map import MatchObs, PointNode                                  # noqa: E402
from Module.Optimization.GTSAM.Graphs import (                              # noqa: E402
    GTSAM_GraphInput, GTSAM_Pose2Point,
)
from Module.Optimization.GTSAM.Optimizer import GTSAM_Graph                 # noqa: E402
from Module.Optimization.TwoFramePGO.Graphs import GraphInput               # noqa: E402
from Utility.GTSAM_Utils import (                                           # noqa: E402
    make_aligned_pose_to_point_factor, make_alignment_warp, pypose_to_pose3,
)
from Utility.Point import point2pixel_NED                                   # noqa: E402

from test_gedf_registration import (                                        # noqa: E402
    K_INTRINSIC, pose_error, scene_registration_points,
)


# --------------------------------------------------------------------------- #
# Factor-level Jacobian verification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a_type,dim", [("sim3", 1), ("sl4", 9)])
def test_aligned_factor_jacobians(a_type: str, dim: int):
    rng = np.random.default_rng(0)
    pose = gtsam.Pose3.Expmap(rng.normal(size=6) * 0.1)
    x = rng.normal(size=dim) * 0.05
    l_w = rng.normal(size=3) + np.array([2.0, 0.0, 0.0])
    obs = rng.normal(size=3) + np.array([2.0, 0.0, 0.0])

    pose_key, extras_key, lmk_key = (gtsam.symbol(c, 0) for c in "pal")
    factor = make_aligned_pose_to_point_factor(
        pose_key, extras_key, lmk_key, obs,
        gtsam.noiseModel.Unit.Create(3), make_alignment_warp(a_type))

    values = gtsam.Values()
    values.insert(pose_key, pose)
    values.insert(extras_key, x)
    values.insert(lmk_key, l_w)

    jf = factor.linearize(values)
    A = jf.getA()                     # (3, 6 + dim + 3), unit noise -> raw Jacobian
    assert A.shape == (3, 6 + dim + 3)

    def err_at(pose_v, x_v, l_v) -> np.ndarray:
        v = gtsam.Values()
        v.insert(pose_key, pose_v)
        v.insert(extras_key, x_v)
        v.insert(lmk_key, l_v)
        return factor.unwhitenedError(v)

    eps = 1e-6
    num = np.zeros((3, 6 + dim + 3))
    for i in range(6):                                    # pose (retract)
        d = np.zeros(6); d[i] = eps
        num[:, i] = (err_at(pose.retract(d), x, l_w) - err_at(pose.retract(-d), x, l_w)) / (2 * eps)
    for i in range(dim):                                  # extras (vector)
        d = np.zeros(dim); d[i] = eps
        num[:, 6 + i] = (err_at(pose, x + d, l_w) - err_at(pose, x - d, l_w)) / (2 * eps)
    for i in range(3):                                    # landmark
        d = np.zeros(3); d[i] = eps
        num[:, 6 + dim + i] = (err_at(pose, x, l_w + d) - err_at(pose, x, l_w - d)) / (2 * eps)

    np.testing.assert_allclose(A, num, atol=1e-5)


# --------------------------------------------------------------------------- #
# End-to-end pose2point recovery
# --------------------------------------------------------------------------- #
P0_TRUE = pp.SE3(torch.tensor([[0.1, 1.5, 1.2, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64))
P1_TRUE = pp.SE3(torch.tensor([[0.3, 1.5, 1.2, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64))
P2_TRUE = pp.SE3(torch.tensor([[0.5, 1.55, 1.15, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64))


def obs_from_pose(pose: pp.LieTensor, points_w: torch.Tensor,
                  warp_cam=None) -> tuple[torch.Tensor, torch.Tensor]:
    p_c = pp.SE3(pose).Inv().Act(points_w)
    assert bool((p_c[:, 0] > 0.1).all())
    if warp_cam is not None:
        p_c = warp_cam(p_c)
    return point2pixel_NED(p_c, K_INTRINSIC), p_c[:, 0:1].clone()


def make_pair(from_idx: int, frame_idx: int, pose_a: pp.LieTensor, pose_b: pp.LieTensor,
              points_w: torch.Tensor, warp_b=None, sigma: float = 1e-2) -> GraphInput:
    """A GraphInput whose pixel1_* observe from pose_a and pixel2_* from pose_b."""
    N = points_w.shape[0]
    uv1, d1 = obs_from_pose(pose_a, points_w)
    uv2, d2 = obs_from_pose(pose_b, points_w, warp_cam=warp_b)
    eye_cov = torch.eye(3, dtype=torch.float64).expand(N, 3, 3).clone() * sigma ** 2
    obs = MatchObs.init({
        "pixel1_uv": uv1, "pixel1_d": d1, "obs1_covTc": eye_cov.clone(),
        "pixel2_uv": uv2, "pixel2_d": d2, "obs2_covTc": eye_cov.clone(),
    })
    pts = PointNode.init({
        "pos_Tw": points_w.clone(),
        "cov_Tw": eye_cov.clone(),
    })
    return GraphInput(
        frame_idx=torch.tensor([frame_idx]), from_idx=torch.tensor([from_idx]),
        from_pose=pp.SE3(pose_a), init_motion=pp.SE3(P1_TRUE),
        baseline=torch.tensor([0.1]), observations=obs, points=pts,
        images_intrinsic=K_INTRINSIC, edges_index=torch.zeros(N, dtype=torch.long),
        device="cpu",
    )


def make_gtsam_input(warp_current=None) -> GTSAM_GraphInput:
    pts_w = scene_registration_points()
    prev = make_pair(0, 1, P0_TRUE, P1_TRUE, pts_w)
    curr = make_pair(1, 2, P1_TRUE, P2_TRUE, pts_w, warp_b=warp_current)
    N = pts_w.shape[0]
    return GTSAM_GraphInput(previous_graph_data=prev, current_graph_data=curr,
                            indexes_prev_curr=[(i, i) for i in range(N)])


def run_pose2point(alignment_type: str, warp_current=None):
    graph = GTSAM_Pose2Point(alignment_type=alignment_type)
    graph.parse_graph_data(make_gtsam_input(warp_current=warp_current))
    graph.run_gtsam_optimization()
    return graph.write_back()


def test_gtsam_sim3_recovery_scaled_depth():
    scaled = lambda p: 1.2 * p

    out = run_pose2point("sim3", warp_current=scaled)
    rot, trans = pose_error(out.pose_estimates[1].double(), P2_TRUE)
    assert rot < 1.0 and trans < 0.03, f"sim3: {rot:.3f} deg {trans:.4f} m"
    assert out.alignment_type == "sim3"
    assert out.alignment_state is not None
    assert float(out.alignment_state.item()) == pytest.approx(-math.log(1.2), abs=0.03)
    assert out.scale == pytest.approx(1 / 1.2, rel=0.03)

    out_se3 = run_pose2point("se3", warp_current=scaled)
    _, trans_se3 = pose_error(out_se3.pose_estimates[1].double(), P2_TRUE)
    assert trans_se3 > 0.05, f"se3 unexpectedly absorbed the scale bias ({trans_se3:.4f} m)"
    assert out_se3.alignment_state is None and out_se3.scale is None


def test_gtsam_sl4_recovery_sheared_depth():
    S = torch.eye(3, dtype=torch.float64)
    S[0, 1] = S[1, 0] = 0.05
    shear = lambda p: p @ S.T

    out_sl4 = run_pose2point("sl4", warp_current=shear)
    out_se3 = run_pose2point("se3", warp_current=shear)
    _, trans_sl4 = pose_error(out_sl4.pose_estimates[1].double(), P2_TRUE)
    _, trans_se3 = pose_error(out_se3.pose_estimates[1].double(), P2_TRUE)

    assert trans_sl4 < trans_se3, f"sl4 {trans_sl4:.4f} !< se3 {trans_se3:.4f}"
    assert trans_sl4 < 0.05
    assert out_sl4.alignment_state is not None
    assert out_sl4.alignment_state.shape == (9,)
    assert bool(torch.isfinite(out_sl4.alignment_state).all())


def test_gtsam_alignment_prior_holds_on_clean_data():
    out = run_pose2point("sim3")
    rot, trans = pose_error(out.pose_estimates[1].double(), P2_TRUE)
    assert rot < 1.0 and trans < 0.02
    assert out.alignment_state is not None
    assert abs(float(out.alignment_state.item())) < 0.02


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def gtsam_cfg(graph_type: str = "pose2point",
              alignment: SimpleNamespace | None = None) -> SimpleNamespace:
    ns = SimpleNamespace(graph_type=graph_type, device="cpu", vectorize=True,
                         parallel=False, autodiff=True)
    if alignment is not None:
        ns.alignment = alignment
    return ns


def test_gtsam_alignment_config_validation():
    GTSAM_Graph.is_valid_config(gtsam_cfg())                        # back-compat
    GTSAM_Graph.is_valid_config(gtsam_cfg(alignment=SimpleNamespace(type="sim3")))
    GTSAM_Graph.is_valid_config(
        gtsam_cfg(alignment=SimpleNamespace(type="sl4", prior_weight=50.0)))

    with pytest.raises(ValueError, match="pose2point"):
        GTSAM_Graph.is_valid_config(
            gtsam_cfg("isam", alignment=SimpleNamespace(type="sim3")))
    with pytest.raises((ValueError, KeyError)):
        GTSAM_Graph.is_valid_config(
            gtsam_cfg(alignment=SimpleNamespace(type="sim4")))
    with pytest.raises((ValueError, KeyError)):
        GTSAM_Graph.is_valid_config(
            gtsam_cfg(alignment=SimpleNamespace(type="sim3", prior_weight=-1.0)))
