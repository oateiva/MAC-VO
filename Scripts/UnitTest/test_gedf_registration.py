"""
Tests for the GEDF factor graphs and the GEDF_PGO optimizer backend.

The pose-recovery fixture uses THREE ORTHOGONAL plane patches: a single plane
leaves 3 degrees of freedom unconstrained under pure point-to-distance-field
registration (translations inside the plane and yaw about its normal), so the
scene must have geometry in every direction.
"""
import math
import pickle
import typing as typ
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import pytest
import torch

from Module.Map import MatchObs, PointNode
from Module.Optimization import IOptimizer
from Module.Optimization.GEDF import (
    Analytic_GEDF_ICP, Analytic_GEDF_Registration,
    GEDF_GraphInput, GEDF_ICP, GEDF_PGO, GEDF_Registration,
    GEDFConfig, GEDFMapper,
)
from Utility.Point import point2pixel_NED

K_INTRINSIC = torch.tensor([[320.0, 0.0, 320.0],
                            [0.0, 320.0, 320.0],
                            [0.0, 0.0, 1.0]], dtype=torch.float64)

FIELD_NS = SimpleNamespace(weighting="mahalanobis", sigma=0.10,
                           oob_value_threshold=19.0, oob_residual=5.0,
                           max_grad_norm=10.0)


# --------------------------------------------------------------------------- #
# Scene / fixture helpers
# --------------------------------------------------------------------------- #
def three_plane_cloud(n_per: int = 3000, seed: int = 0) -> torch.Tensor:
    """Floor z=0.5 plus walls y=2.7 and x=2.7: constrains all 6 DoF."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(0, 3, size=(n_per, 2)), np.full(n_per, 0.5)])
    wall_y = np.column_stack([rng.uniform(0, 3, size=n_per),
                              np.full(n_per, 2.7), rng.uniform(0.5, 2.0, size=n_per)])
    wall_x = np.column_stack([np.full(n_per, 2.7), rng.uniform(0, 3, size=n_per),
                              rng.uniform(0.5, 2.0, size=n_per)])
    return torch.from_numpy(np.concatenate([floor, wall_y, wall_x]))


def scene_registration_points(seed: int = 1, n_per: int = 60) -> torch.Tensor:
    """World points on all three patches, all in front of the test camera."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(0.8, 2.4, size=(n_per, 2)), np.full(n_per, 0.5)])
    wall_y = np.column_stack([rng.uniform(0.8, 2.4, size=n_per),
                              np.full(n_per, 2.7), rng.uniform(0.6, 1.8, size=n_per)])
    wall_x = np.column_stack([np.full(n_per, 2.7), rng.uniform(0.8, 2.4, size=n_per),
                              rng.uniform(0.6, 1.8, size=n_per)])
    return torch.from_numpy(np.concatenate([floor, wall_y, wall_x]))


@pytest.fixture(scope="module")
def scene_mapper() -> GEDFMapper:
    # Wall-intersection cubes contain two orthogonal surfaces; K=16 keeps the
    # map MAE ~0.03 m there (the K=8 default is for locally planar VO scenes).
    mapper = GEDFMapper(GEDFConfig(device="cpu", num_gaussians=16,
                                   lm_iters_cold=60, sample_points=1500))
    mapper.insert(three_plane_cloud(n_per=10000).float())
    mapper.flush()
    assert mapper.is_ready
    return mapper


T_GT = pp.SE3(torch.tensor([[0.3, 1.5, 1.2, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64))


def perturbed_pose(pose: pp.LieTensor, rot_deg: float, trans_m: float,
                   seed: int = 5) -> pp.LieTensor:
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    rot = axis * math.radians(rot_deg)
    tdir = rng.normal(size=3); tdir /= np.linalg.norm(tdir)
    xi = torch.tensor([[*(tdir * trans_m), *rot]], dtype=torch.float64)
    perturbed = typ.cast(pp.LieTensor, pose @ pp.se3(xi).Exp())
    return pp.SE3(perturbed.tensor())


def make_graph_input(pose_gt: pp.LieTensor, init_motion: pp.LieTensor,
                     points_w: torch.Tensor, obs_sigma: float = 1e-2,
                     pts_sigma: float = 1e-2) -> GEDF_GraphInput:
    """Fabricate observations consistent with `pose_gt` viewing `points_w`."""
    N = points_w.shape[0]
    p_c = pp.SE3(pose_gt).Inv().Act(points_w)
    assert bool((p_c[:, 0] > 0.1).all()), "all points must be in front of the camera"

    pixel2_uv = point2pixel_NED(p_c, K_INTRINSIC)
    pixel2_d = p_c[:, 0:1].clone()

    obs = MatchObs.init({
        "pixel2_uv": pixel2_uv,
        "pixel2_d": pixel2_d,
        "obs2_covTc": torch.eye(3, dtype=torch.float64).expand(N, 3, 3).clone() * obs_sigma ** 2,
    })
    pts = PointNode.init({
        "pos_Tw": points_w.clone(),
        "cov_Tw": torch.eye(3, dtype=torch.float64).expand(N, 3, 3).clone() * pts_sigma ** 2,
    })
    return GEDF_GraphInput(
        frame_idx=torch.tensor([1]), from_idx=torch.tensor([0]),
        from_pose=pp.SE3(pose_gt), init_motion=pp.SE3(init_motion),
        baseline=torch.tensor([0.1]), observations=obs, points=pts,
        images_intrinsic=K_INTRINSIC, edges_index=torch.zeros(N, dtype=torch.long),
        device="cpu",
    )


def pose_error(est: torch.Tensor, gt: pp.LieTensor) -> tuple[float, float]:
    """(rotation error deg, translation error m)."""
    delta = typ.cast(pp.LieTensor,
                     pp.SE3(gt).Inv() @ pp.SE3(est.reshape(1, 7).double())).tensor()
    trans = float(delta[0, :3].norm())
    qw = float(delta[0, 6].abs().clamp(max=1.0))
    return math.degrees(2.0 * math.acos(qw)), trans


def make_optimizer_config(graph_type: str, autodiff: bool, map_path: str | None,
                          source: str = "prebuilt") -> SimpleNamespace:
    return SimpleNamespace(
        graph_type=graph_type, device="cpu", vectorize=True, parallel=False,
        autodiff=autodiff,
        map=SimpleNamespace(source=source, path=map_path, insert_keypoints=False,
                            insert_dense=False, min_gaussians=50, online=None),
        field=FIELD_NS,
        solver=SimpleNamespace(coarse_kernel_delta=3.0, coarse_steps=10,
                               fine_kernel_delta=0.5, fine_steps=10),
        viz=SimpleNamespace(every=10, iso=0.10, resolution=0.10, max_points=100000),
    )


# --------------------------------------------------------------------------- #
# Graph-level tests
# --------------------------------------------------------------------------- #
def _make_graph(cls, scene_mapper: GEDFMapper, rot_deg=3.0, trans_m=0.1,
                field_cfg=FIELD_NS):
    init = perturbed_pose(T_GT, rot_deg, trans_m)
    gd = make_graph_input(T_GT, init, scene_registration_points())
    return cls(gd, field=scene_mapper, field_cfg=field_cfg).to(dtype=torch.double)


# The gradient-norm clamp is a deliberate solver-side robustification that
# deviates from autograd; it must be inactive when verifying the Jacobian.
UNCLAMPED = SimpleNamespace(**{**vars(FIELD_NS), "max_grad_norm": 1e9})


def test_jacobian_registration(scene_mapper):
    graph = _make_graph(Analytic_GEDF_Registration, scene_mapper, field_cfg=UNCLAMPED)
    graph()   # module __call__ so the forward-pre-hook records call args
    assert graph.verify_jacobian(graph.build_jacobian())


def test_jacobian_hybrid(scene_mapper):
    graph = _make_graph(Analytic_GEDF_ICP, scene_mapper, field_cfg=UNCLAMPED)
    graph()
    assert graph.verify_jacobian(graph.build_jacobian())


def test_forward_residual_semantics(scene_mapper):
    """At the ground-truth pose, field residuals must be small (points lie on
    the mapped surfaces) and ICP residuals must vanish."""
    gd = make_graph_input(T_GT, T_GT, scene_registration_points())
    graph = GEDF_ICP(gd, field=scene_mapper, field_cfg=FIELD_NS).to(dtype=torch.double)
    r = graph()
    assert float(r[:, :3].abs().max()) < 1e-9          # exact ICP consistency
    assert float(r[:, 3].abs().mean()) < 0.08          # field ~ distance ~ 0 on surface


def test_oob_handling(scene_mapper):
    """Points far outside the map: constant residual, zero Jacobian."""
    far = torch.tensor([[40.0, 40.0, -20.0]], dtype=torch.float64).expand(5, 3) \
        + torch.arange(5, dtype=torch.float64).unsqueeze(-1)
    cam = pp.SE3(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64))
    gd = make_graph_input(cam, cam, far)
    graph = Analytic_GEDF_Registration(gd, field=scene_mapper,
                                       field_cfg=FIELD_NS).to(dtype=torch.double)
    r = graph()
    assert torch.allclose(r, torch.full_like(r, FIELD_NS.oob_residual))
    J = graph.build_jacobian()
    assert (J == 0).all()
    # the autodiff path agrees: constant residual => zero pose gradient
    assert graph.verify_jacobian(J)


def test_cold_start_hybrid_equals_icp(scene_mapper):
    """With a not-ready map the hybrid graph must behave as pure ICP."""
    empty = GEDFMapper(GEDFConfig(device="cpu"))
    assert not empty.is_ready

    init = perturbed_pose(T_GT, 3.0, 0.1)
    gd = make_graph_input(T_GT, init, scene_registration_points())
    graph = Analytic_GEDF_ICP(gd, field=empty, field_cfg=FIELD_NS).to(dtype=torch.double)
    r = graph()
    assert (r[:, 3] == 0).all()
    J = graph.build_jacobian().reshape(-1, 4, 7)
    assert (J[:, 3] == 0).all()
    cov = graph.covariance_array()
    assert torch.allclose(cov[:, 3, 3], torch.ones_like(cov[:, 3, 3]))


def test_graph_input_picklable(scene_mapper):
    gd = make_graph_input(T_GT, T_GT, scene_registration_points())
    gd2 = pickle.loads(pickle.dumps(gd))
    assert torch.equal(gd2.points.data["pos_Tw"], gd.points.data["pos_Tw"])
    assert torch.equal(gd2.observations.data["pixel2_uv"], gd.observations.data["pixel2_uv"])
    assert gd2.map_insert_pos_Tw is None


# --------------------------------------------------------------------------- #
# Optimizer-level tests
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def scene_map_bin(scene_mapper, tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("gedf") / "scene.bin"
    scene_mapper.export_gdf1(path)
    return str(path)


@pytest.mark.parametrize("graph_type", ["gedf", "gedf+icp"])
@pytest.mark.parametrize("autodiff", [False, True])
def test_pose_recovery(scene_map_bin, graph_type, autodiff):
    cfg = make_optimizer_config(graph_type, autodiff, scene_map_bin)
    GEDF_PGO.is_valid_config(cfg)
    context = GEDF_PGO.init_context(cfg)

    init = perturbed_pose(T_GT, 5.0, 0.2)
    rot0, trans0 = pose_error(init.tensor(), T_GT)
    assert rot0 > 3.0 and trans0 > 0.15

    gd = make_graph_input(T_GT, init, scene_registration_points())
    _, out = GEDF_PGO._optimize(context, gd)
    rot, trans = pose_error(out.motion.detach(), T_GT)

    if graph_type == "gedf+icp":
        assert rot < 0.5 and trans < 0.02, f"{graph_type} autodiff={autodiff}: {rot:.3f} deg {trans:.4f} m"
    else:
        # pure field registration is limited by the field MAE (~2-4 cm)
        assert rot < 1.5 and trans < 0.06, f"{graph_type} autodiff={autodiff}: {rot:.3f} deg {trans:.4f} m"


def test_cold_start_returns_init_motion(tmp_path):
    cfg = make_optimizer_config("gedf", False, None, source="online")
    context = GEDF_PGO.init_context(cfg)

    init = perturbed_pose(T_GT, 5.0, 0.2)
    gd = make_graph_input(T_GT, init, scene_registration_points())
    _, out = GEDF_PGO._optimize(context, gd)
    assert torch.allclose(out.motion.detach().double().reshape(-1),
                          init.tensor().reshape(-1))


def test_map_snapshot_plumbing(scene_map_bin):
    """want_map_snapshot must produce a picklable GEDF_GraphOutput snapshot
    (cpu float32); without the request the fields stay None."""
    from Module.Optimization.GEDF import GEDF_GraphOutput

    cfg = make_optimizer_config("gedf+icp", False, scene_map_bin)
    context = GEDF_PGO.init_context(cfg)

    gd = make_graph_input(T_GT, perturbed_pose(T_GT, 2.0, 0.05), scene_registration_points())
    gd.want_map_snapshot = True
    _, out = GEDF_PGO._optimize(context, gd)
    assert isinstance(out, GEDF_GraphOutput)
    assert out.map_points is not None and out.map_dist is not None
    assert out.map_points.shape[0] > 0
    assert out.map_points.dtype == torch.float32 and not out.map_points.is_cuda
    assert bool((out.map_dist < cfg.viz.iso).all())

    out2 = pickle.loads(pickle.dumps(out))
    assert torch.equal(out2.map_points, out.map_points)

    gd.want_map_snapshot = False
    _, out3 = GEDF_PGO._optimize(context, gd)
    assert out3.map_points is None and out3.map_dist is None


def test_registry_and_sequential_optimize(scene_map_bin):
    """GEDF_PGO must be instantiable via the same registry MACVO.from_config uses."""
    cfg = make_optimizer_config("gedf+icp", False, scene_map_bin)
    opt = IOptimizer.instantiate("GEDF_PGO", cfg)
    assert isinstance(opt, GEDF_PGO)

    init = perturbed_pose(T_GT, 4.0, 0.15)
    gd = make_graph_input(T_GT, init, scene_registration_points())
    out = opt.sequential_optimize(gd)
    rot, trans = pose_error(out.motion.detach(), T_GT)
    assert rot < 0.5 and trans < 0.02
