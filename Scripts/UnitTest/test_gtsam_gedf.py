"""
Tests for the GTSAM+G-EDF hybrid (graph_type "pose2point+gedf"): GTSAM's
pose2point solve ("GTSAM's ICP" — custom pose-to-point factors with landmarks
re-estimated jointly) fused with ONE batched G-EDF field factor on the current
pose, registering the frame against the whole accumulated map inside the same
joint solve.
"""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

gtsam = pytest.importorskip("gtsam")

from Module.Optimization.GTSAM.Graphs import (                              # noqa: E402
    GTSAM_Pose2Point, make_field_eval,
)
from Module.Optimization.GTSAM.Optimizer import GTSAM_Graph                 # noqa: E402
from Utility.GTSAM_Utils import make_gedf_field_factor                     # noqa: E402

from test_gedf_registration import (                                        # noqa: E402,F401
    FIELD_NS, UNCLAMPED, pose_error, scene_map_bin, scene_mapper,
    scene_registration_points,
)
from test_gtsam_alignment import P2_TRUE, make_gtsam_input                  # noqa: E402


# --------------------------------------------------------------------------- #
# Factor-level tests
# --------------------------------------------------------------------------- #
def test_field_factor_jacobian(scene_mapper):
    """The batched analytic Jacobian of the field factor must match numerical
    differentiation over the GTSAM Pose3 retract."""
    pts_w = torch.from_numpy(
        scene_registration_points(n_per=10).numpy())            # (30, 3) on surfaces
    pose = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([0.02, -0.03, 0.05])),
                       np.array([0.35, 1.45, 1.15]))
    points_Tc = np.stack([pose.transformTo(p) for p in pts_w.double().numpy()])

    N = points_Tc.shape[0]
    pose_key = gtsam.symbol('p', 0)
    # UNCLAMPED: the gradient-norm clamp is a deliberate solver-side deviation
    # from the true derivative and must be off when verifying the Jacobian
    # (same convention as the pypose GEDF Jacobian tests).
    factor = make_gedf_field_factor(
        pose_key, points_Tc, make_field_eval(scene_mapper, UNCLAMPED),
        gtsam.noiseModel.Unit.Create(N))

    values = gtsam.Values()
    values.insert(pose_key, pose)
    A = factor.linearize(values).getA()                          # (N, 6), unit noise
    assert A.shape == (N, 6)

    def err_at(pose_v) -> np.ndarray:
        v = gtsam.Values()
        v.insert(pose_key, pose_v)
        return factor.unwhitenedError(v)

    eps = 1e-6
    num = np.zeros((N, 6))
    for i in range(6):
        d = np.zeros(6); d[i] = eps
        num[:, i] = (err_at(pose.retract(d)) - err_at(pose.retract(-d))) / (2 * eps)
    np.testing.assert_allclose(A, num, atol=1e-4)


def test_field_factor_oob(scene_mapper):
    """Points far outside the map: constant oob_residual, zero Jacobian."""
    far = np.full((5, 3), 40.0) + np.arange(5).reshape(-1, 1)
    pose_key = gtsam.symbol('p', 0)
    factor = make_gedf_field_factor(
        pose_key, far, make_field_eval(scene_mapper, FIELD_NS),
        gtsam.noiseModel.Unit.Create(5))
    values = gtsam.Values()
    values.insert(pose_key, gtsam.Pose3())
    r = factor.unwhitenedError(values)
    np.testing.assert_allclose(r, np.full(5, FIELD_NS.oob_residual))
    assert (factor.linearize(values).getA() == 0).all()


# --------------------------------------------------------------------------- #
# Joint-solve recovery
# --------------------------------------------------------------------------- #
def test_pose2point_gedf_recovery(scene_mapper):
    """Adding the field factor to the pose2point solve must keep (not break)
    pose recovery on consistent data; the reported pose stays near truth."""
    graph = GTSAM_Pose2Point(field=scene_mapper, field_cfg=FIELD_NS)
    graph.parse_graph_data(make_gtsam_input())
    graph.run_gtsam_optimization()
    out = graph.write_back()
    rot, trans = pose_error(out.pose_estimates[1].double(), P2_TRUE)
    assert rot < 1.0 and trans < 0.05, f"pose2point+gedf: {rot:.3f} deg {trans:.4f} m"


def test_field_inert_when_map_not_ready():
    """With a not-ready map the hybrid must behave exactly like pose2point."""
    from Module.Optimization.GEDF import GEDFConfig, GEDFMapper
    empty = GEDFMapper(GEDFConfig(device="cpu"))
    assert not empty.is_ready

    out_hybrid = _run(GTSAM_Pose2Point(field=empty, field_cfg=FIELD_NS))
    out_plain = _run(GTSAM_Pose2Point())
    torch.testing.assert_close(out_hybrid.pose_estimates[1], out_plain.pose_estimates[1])


def _run(graph: GTSAM_Pose2Point):
    graph.parse_graph_data(make_gtsam_input())
    graph.run_gtsam_optimization()
    return graph.write_back()


# --------------------------------------------------------------------------- #
# Config / context / optimizer plumbing
# --------------------------------------------------------------------------- #
def hybrid_cfg(map_path: str | None, source: str = "prebuilt",
               viz: SimpleNamespace | None = None,
               graph_type: str = "pose2point+gedf") -> SimpleNamespace:
    gedf = SimpleNamespace(
        map=SimpleNamespace(source=source, path=map_path, insert_keypoints=True,
                            min_gaussians=50, online=None),
        field=FIELD_NS,
    )
    if viz is not None:
        gedf.viz = viz
    return SimpleNamespace(graph_type=graph_type, device="cpu", vectorize=True,
                           parallel=False, autodiff=False, gedf=gedf)


def test_hybrid_config_validation(scene_map_bin):
    GTSAM_Graph.is_valid_config(hybrid_cfg(scene_map_bin))
    # gedf block required for the hybrid graph type
    with pytest.raises(ValueError, match="gedf"):
        GTSAM_Graph.is_valid_config(SimpleNamespace(
            graph_type="pose2point+gedf", device="cpu", vectorize=True,
            parallel=False, autodiff=False))
    # the field factor is unwarped: alignment stays se3-only
    bad = hybrid_cfg(scene_map_bin)
    bad.alignment = SimpleNamespace(type="sim3")
    with pytest.raises(ValueError, match="SE3"):
        GTSAM_Graph.is_valid_config(bad)
    # plain pose2point config untouched by the new spec (back-compat)
    GTSAM_Graph.is_valid_config(SimpleNamespace(
        graph_type="pose2point", device="cpu", vectorize=True,
        parallel=False, autodiff=False))


def test_hybrid_context_and_optimize(scene_map_bin):
    cfg = hybrid_cfg(scene_map_bin)
    ctx = GTSAM_Graph.init_context(cfg)
    assert ctx["gedf_map"] is not None and ctx["gedf_map"].is_ready
    assert ctx["graph"].field is ctx["gedf_map"]

    _, out = GTSAM_Graph._optimize(ctx, make_gtsam_input())
    rot, trans = pose_error(out.pose_estimates[1].double(), P2_TRUE)
    assert rot < 1.0 and trans < 0.05


def test_hybrid_online_map_grows():
    cfg = hybrid_cfg(None, source="online")
    ctx = GTSAM_Graph.init_context(cfg)
    assert ctx["gedf_map"].num_cubes == 0
    _, _ = GTSAM_Graph._optimize(ctx, make_gtsam_input())
    assert ctx["gedf_map"].num_cubes > 0        # landmarks were inserted pre-solve


def test_hybrid_snapshot_plumbing(scene_map_bin):
    viz = SimpleNamespace(every=1, iso=0.10, resolution=0.10, max_points=100000,
                          gaussians=True, n_sigma=1.0, max_gaussians=500, cubes=True)
    cfg = hybrid_cfg(scene_map_bin, viz=viz)
    GTSAM_Graph.is_valid_config(cfg)
    ctx = GTSAM_Graph.init_context(cfg)

    gi = make_gtsam_input()
    gi.want_map_snapshot = True
    _, out = GTSAM_Graph._optimize(ctx, gi)
    assert out.gedf_points is not None and out.gedf_dist is not None
    assert out.gedf_gauss_means is not None and out.gedf_gauss_weights is not None
    assert out.gedf_cube_centers is not None and out.gedf_cube_size is not None
    for t in (out.gedf_points, out.gedf_gauss_means, out.gedf_cube_centers):
        assert not t.is_cuda
    out2 = pickle.loads(pickle.dumps(out))
    assert torch.equal(out2.gedf_points, out.gedf_points)

    gi.want_map_snapshot = False
    _, out3 = GTSAM_Graph._optimize(ctx, gi)
    assert out3.gedf_points is None and out3.gedf_cube_centers is None
