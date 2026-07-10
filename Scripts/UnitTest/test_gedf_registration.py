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
                     pts_sigma: float = 1e-2,
                     warp_cam: typ.Callable[[torch.Tensor], torch.Tensor] | None = None
                     ) -> GEDF_GraphInput:
    """Fabricate observations consistent with `pose_gt` viewing `points_w`.

    `warp_cam` distorts the MEASURED camera points (simulating biased monocular
    depth) while `pos_Tw` keeps the true world points."""
    N = points_w.shape[0]
    p_c = pp.SE3(pose_gt).Inv().Act(points_w)
    assert bool((p_c[:, 0] > 0.1).all()), "all points must be in front of the camera"
    if warp_cam is not None:
        p_c = warp_cam(p_c)

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
                          source: str = "prebuilt",
                          alignment: SimpleNamespace | None = None) -> SimpleNamespace:
    ns = SimpleNamespace(
        graph_type=graph_type, device="cpu", vectorize=True, parallel=False,
        autodiff=autodiff,
        map=SimpleNamespace(source=source, path=map_path, insert_keypoints=False,
                            insert_dense=False, min_gaussians=50, online=None),
        field=FIELD_NS,
        solver=SimpleNamespace(coarse_kernel_delta=3.0, coarse_steps=10,
                               fine_kernel_delta=0.5, fine_steps=10),
        viz=SimpleNamespace(every=10, iso=0.10, resolution=0.10, max_points=100000),
    )
    if alignment is not None:   # attached only when given: default path tests true key absence
        ns.alignment = alignment
    return ns


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


def test_gaussian_snapshot_plumbing(scene_map_bin):
    """map_gauss_* fields must be populated iff viz.gaussians is enabled AND a
    snapshot was requested; defaults (no key) keep the current points-only
    behavior; the validator rejects bad values for the optional keys."""
    cfg = make_optimizer_config("gedf+icp", False, scene_map_bin)
    cfg.viz.gaussians = True
    cfg.viz.n_sigma = 2.0
    cfg.viz.max_gaussians = 500
    GEDF_PGO.is_valid_config(cfg)
    context = GEDF_PGO.init_context(cfg)

    gd = make_graph_input(T_GT, perturbed_pose(T_GT, 2.0, 0.05), scene_registration_points())
    gd.want_map_snapshot = True
    _, out = GEDF_PGO._optimize(context, gd)
    assert out.map_gauss_means is not None and out.map_gauss_sigmas is not None
    assert out.map_gauss_weights is not None and out.map_gauss_mae is not None
    n = out.map_gauss_means.shape[0]
    assert 0 < n <= 500
    assert out.map_gauss_sigmas.shape == (n, 3)
    assert out.map_gauss_weights.shape == (n,) and out.map_gauss_mae.shape == (n,)
    for t in (out.map_gauss_means, out.map_gauss_sigmas,
              out.map_gauss_weights, out.map_gauss_mae):
        assert t.dtype == torch.float32 and not t.is_cuda
    out2 = pickle.loads(pickle.dumps(out))
    assert torch.equal(out2.map_gauss_means, out.map_gauss_means)

    # no snapshot request -> gaussian fields stay None
    gd.want_map_snapshot = False
    _, out3 = GEDF_PGO._optimize(context, gd)
    assert out3.map_gauss_means is None and out3.map_gauss_weights is None

    # back-compat: config without the optional keys -> points only
    cfg_plain = make_optimizer_config("gedf+icp", False, scene_map_bin)
    GEDF_PGO.is_valid_config(cfg_plain)
    ctx_plain = GEDF_PGO.init_context(cfg_plain)
    gd.want_map_snapshot = True
    _, out4 = GEDF_PGO._optimize(ctx_plain, gd)
    assert out4.map_points is not None and out4.map_gauss_means is None

    # validator rejects bad optional values
    cfg_bad = make_optimizer_config("gedf+icp", False, scene_map_bin)
    cfg_bad.viz.n_sigma = -1.0
    with pytest.raises((ValueError, KeyError)):
        GEDF_PGO.is_valid_config(cfg_bad)
    cfg_bad2 = make_optimizer_config("gedf+icp", False, scene_map_bin)
    cfg_bad2.viz.gaussians = "yes"
    with pytest.raises((ValueError, KeyError)):
        GEDF_PGO.is_valid_config(cfg_bad2)


def test_cube_snapshot_plumbing(scene_map_bin):
    """map_cube_* fields must be populated iff viz.cubes is enabled AND a
    snapshot was requested; the payload must be picklable CPU tensors."""
    cfg = make_optimizer_config("gedf+icp", False, scene_map_bin)
    cfg.viz.cubes = True
    GEDF_PGO.is_valid_config(cfg)
    context = GEDF_PGO.init_context(cfg)

    gd = make_graph_input(T_GT, perturbed_pose(T_GT, 2.0, 0.05), scene_registration_points())
    gd.want_map_snapshot = True
    _, out = GEDF_PGO._optimize(context, gd)
    assert out.map_cube_centers is not None and out.map_cube_valid is not None
    assert out.map_cube_mae is not None and out.map_cube_size is not None
    C = out.map_cube_centers.shape[0]
    assert C > 0
    assert out.map_cube_valid.shape == (C,) and out.map_cube_mae.shape == (C,)
    assert out.map_cube_centers.dtype == torch.float32 and not out.map_cube_centers.is_cuda
    assert out.map_cube_valid.dtype == torch.bool
    assert out.map_cube_size > 0
    out2 = pickle.loads(pickle.dumps(out))
    assert torch.equal(out2.map_cube_centers, out.map_cube_centers)

    # no snapshot request -> cube fields stay None
    gd.want_map_snapshot = False
    _, out3 = GEDF_PGO._optimize(context, gd)
    assert out3.map_cube_centers is None and out3.map_cube_size is None

    # off by default (absent key)
    ctx_plain = GEDF_PGO.init_context(make_optimizer_config("gedf+icp", False, scene_map_bin))
    gd.want_map_snapshot = True
    _, out4 = GEDF_PGO._optimize(ctx_plain, gd)
    assert out4.map_cube_centers is None

    # validator rejects a non-bool value
    cfg_bad = make_optimizer_config("gedf+icp", False, scene_map_bin)
    cfg_bad.viz.cubes = "yes"
    with pytest.raises((ValueError, KeyError)):
        GEDF_PGO.is_valid_config(cfg_bad)


# --------------------------------------------------------------------------- #
# Alignment axis (se3 | sim3 | sl4)
# --------------------------------------------------------------------------- #
from Module.Optimization.GEDF import (             # noqa: E402
    SE3Alignment, SL4Alignment, Sim3Alignment,
)
from Module.Optimization.GEDF.Alignment import _sl4_complement_basis  # noqa: E402


def align_ns(a_type: str, prior_weight: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(type=a_type, prior_weight=prior_weight)


def test_alignment_basis_sanity():
    E = _sl4_complement_basis()
    assert E.shape == (9, 4, 4)
    assert torch.allclose(E.diagonal(dim1=-2, dim2=-1).sum(-1),
                          torch.zeros(9, dtype=E.dtype))

    # se(3) generators as 4x4 (3 skew + 3 translation columns)
    se3_gen = torch.zeros((6, 4, 4), dtype=torch.float64)
    se3_gen[0, 1, 2], se3_gen[0, 2, 1] = -1.0, 1.0
    se3_gen[1, 0, 2], se3_gen[1, 2, 0] = 1.0, -1.0
    se3_gen[2, 0, 1], se3_gen[2, 1, 0] = -1.0, 1.0
    for k in range(3):
        se3_gen[3 + k, k, 3] = 1.0
    stacked = torch.cat([se3_gen, E]).reshape(15, 16)
    assert torch.linalg.matrix_rank(stacked) == 15   # disjoint + independent

    # det(exp(sum x_i E_i)) == 1 for random coefficients
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(9, dtype=torch.float64, generator=gen) * 0.1
    W = torch.matrix_exp((x.view(9, 1, 1) * E).sum(0))
    assert float(torch.det(W)) == pytest.approx(1.0, abs=1e-10)

    # zero extras reproduce the identity warp exactly
    pts = torch.randn((50, 3), dtype=torch.float64, generator=gen)
    edges = torch.zeros(50, dtype=torch.long)
    ref = SE3Alignment(pp.SE3(T_GT)).act(pts, edges)
    for cls in (Sim3Alignment, SL4Alignment):
        a = cls(pp.SE3(T_GT)).double()
        torch.testing.assert_close(a.act(pts, edges), ref, atol=1e-12, rtol=0)


def test_sim3_recovery_scaled_depth(scene_map_bin):
    """Monocular depth over-estimated by 1.2x: sim3 recovers the pose AND the
    scale correction; se3 on the same input stays biased."""
    init = perturbed_pose(T_GT, 3.0, 0.1)
    scaled = lambda p: 1.2 * p

    cfg = make_optimizer_config("gedf+icp", True, scene_map_bin, alignment=align_ns("sim3"))
    GEDF_PGO.is_valid_config(cfg)
    ctx = GEDF_PGO.init_context(cfg)
    gd = make_graph_input(T_GT, init, scene_registration_points(), warp_cam=scaled)
    _, out = GEDF_PGO._optimize(ctx, gd)
    rot, trans = pose_error(out.motion.detach(), T_GT)
    assert rot < 0.5 and trans < 0.03, f"sim3: {rot:.3f} deg {trans:.4f} m"
    assert out.alignment_type == "sim3"
    assert out.alignment_state is not None
    assert float(out.alignment_state.item()) == pytest.approx(-math.log(1.2), abs=0.03)
    assert out.scale == pytest.approx(1 / 1.2, rel=0.03)

    # the motivating contrast: rigid-only alignment cannot absorb the scale bias
    cfg_se3 = make_optimizer_config("gedf+icp", True, scene_map_bin)
    ctx_se3 = GEDF_PGO.init_context(cfg_se3)
    gd2 = make_graph_input(T_GT, init, scene_registration_points(), warp_cam=scaled)
    _, out_se3 = GEDF_PGO._optimize(ctx_se3, gd2)
    _, trans_se3 = pose_error(out_se3.motion.detach(), T_GT)
    assert trans_se3 > 0.05, f"se3 unexpectedly absorbed the scale bias ({trans_se3:.4f} m)"


def test_sl4_recovery_sheared_depth(scene_map_bin):
    """Mild shear distortion of the measured camera points: sl4 beats se3."""
    S = torch.eye(3, dtype=torch.float64)
    S[0, 1] = S[1, 0] = 0.05
    shear = lambda p: p @ S.T
    init = perturbed_pose(T_GT, 3.0, 0.1)

    results = {}
    for a_type in ("se3", "sl4"):
        cfg = make_optimizer_config("gedf+icp", True, scene_map_bin,
                                    alignment=align_ns(a_type) if a_type != "se3" else None)
        ctx = GEDF_PGO.init_context(cfg)
        gd = make_graph_input(T_GT, init, scene_registration_points(), warp_cam=shear)
        _, out = GEDF_PGO._optimize(ctx, gd)
        results[a_type] = (pose_error(out.motion.detach(), T_GT), out)

    (_, trans_se3), _ = results["se3"]
    (_, trans_sl4), out_sl4 = results["sl4"]
    assert trans_sl4 < trans_se3, f"sl4 {trans_sl4:.4f} !< se3 {trans_se3:.4f}"
    assert trans_sl4 < 0.05
    assert out_sl4.alignment_state is not None
    assert out_sl4.alignment_state.shape == (9,)
    assert bool(torch.isfinite(out_sl4.alignment_state).all())


def test_sim3_scale_feedforward_at_insertion(scene_map_bin):
    """The damped scale state from one solve must be applied to the NEXT
    call's landmarks (scaled about the previous camera center, covariance
    x s^2) before they reach the map / ICP rows; se3 must leave them
    untouched."""
    from Module.Optimization.GEDF.Optimizer import _ALIGN_FF_ALPHA

    cfg = make_optimizer_config("gedf+icp", True, scene_map_bin,
                                alignment=align_ns("sim3"))
    ctx = GEDF_PGO.init_context(cfg)
    assert ctx["align_scale_prev"] is None

    # solve 1: depth over-estimated by 1.2x -> estimate ~= 1/1.2 enters the
    # state through one log-space EMA step from 1.0
    gd = make_graph_input(T_GT, perturbed_pose(T_GT, 3.0, 0.1),
                          scene_registration_points(), warp_cam=lambda p: 1.2 * p)
    ctx, out = GEDF_PGO._optimize(ctx, gd)
    assert out.scale == pytest.approx(1 / 1.2, rel=0.05)
    s = ctx["align_scale_prev"]
    assert s == pytest.approx((1 / 1.2) ** _ALIGN_FF_ALPHA, rel=0.05)

    # solve 2: landmarks must arrive scaled about the previous camera center.
    # Snapshot the anchor BEFORE the solve: pose2opt shares storage with
    # init_motion (= T_GT here), so the LM update mutates T_GT in place.
    pts_raw = scene_registration_points(seed=2)
    gd2 = make_graph_input(T_GT, T_GT, pts_raw)
    cov_raw = gd2.points.data["cov_Tw"].clone()
    anchor = T_GT.translation().to(pts_raw.dtype).clone()
    ctx, _ = GEDF_PGO._optimize(ctx, gd2)
    torch.testing.assert_close(gd2.points.data["pos_Tw"],
                               anchor + s * (pts_raw - anchor))
    torch.testing.assert_close(gd2.points.data["cov_Tw"], s * s * cov_raw)

    # se3 config: no scale channel, landmarks stay bit-identical
    ctx_se3 = GEDF_PGO.init_context(
        make_optimizer_config("gedf+icp", True, scene_map_bin))
    gd3 = make_graph_input(T_GT, T_GT, pts_raw)
    ctx_se3, _ = GEDF_PGO._optimize(ctx_se3, gd3)
    assert ctx_se3["align_scale_prev"] is None
    assert torch.equal(gd3.points.data["pos_Tw"], pts_raw)


def test_sim3_feedforward_state_gating():
    """The feed-forward state must survive diverged solves and track sustained
    drift without collapsing (regression: plane_nose scale death spiral, where
    raw feed-forward of a transient drove the map and warp to scale ~1e-3)."""
    from Module.Optimization.GEDF.Optimizer import (
        _ALIGN_FF_ALPHA, _update_align_scale_state,
    )

    # first accepted estimate: one EMA step away from identity
    s = _update_align_scale_state(None, 0.8)
    assert s == pytest.approx(0.8 ** _ALIGN_FF_ALPHA)

    # out-of-range / non-finite estimates leave the state untouched
    for bad in (0.01, 50.0, float("nan"), float("inf"), None):
        assert _update_align_scale_state(s, bad) == s
    assert _update_align_scale_state(None, 0.01) is None

    # sustained plausible drift: converges to the estimate, never below it
    st = None
    for _ in range(200):
        st = _update_align_scale_state(st, 0.6)
    assert st == pytest.approx(0.6, rel=0.01)

    # a short transient of the lowest accepted value cannot collapse the state:
    # 10 frames at 0.5 from identity stays above 0.5 (EMA lag bounds the dip)
    st = 1.0
    for _ in range(10):
        st = _update_align_scale_state(st, 0.5)
    assert st is not None and st > 0.5


def test_alignment_prior_holds_scale_on_clean_data(scene_map_bin):
    """On unwarped data the prior must keep sim3 at s ~= 1 without hurting the pose."""
    cfg = make_optimizer_config("gedf+icp", True, scene_map_bin, alignment=align_ns("sim3"))
    ctx = GEDF_PGO.init_context(cfg)
    gd = make_graph_input(T_GT, perturbed_pose(T_GT, 3.0, 0.1), scene_registration_points())
    _, out = GEDF_PGO._optimize(ctx, gd)
    rot, trans = pose_error(out.motion.detach(), T_GT)
    assert rot < 0.5 and trans < 0.02
    assert out.alignment_state is not None
    assert abs(float(out.alignment_state.item())) < 0.02


def test_alignment_config_validation(scene_map_bin):
    # absent key: valid + defaults to se3 (back-compat)
    cfg = make_optimizer_config("gedf+icp", False, scene_map_bin)
    GEDF_PGO.is_valid_config(cfg)
    assert GEDF_PGO.init_context(cfg)["alignment_cfg"].type == "se3"

    # non-se3 alignment requires autodiff
    with pytest.raises(ValueError, match="autodiff"):
        GEDF_PGO.is_valid_config(
            make_optimizer_config("gedf+icp", False, scene_map_bin, alignment=align_ns("sim3")))
    # se3 alignment + analytic is fine
    GEDF_PGO.is_valid_config(
        make_optimizer_config("gedf+icp", False, scene_map_bin, alignment=align_ns("se3")))
    # unknown type / bad prior weight rejected
    with pytest.raises((ValueError, KeyError)):
        GEDF_PGO.is_valid_config(
            make_optimizer_config("gedf+icp", True, scene_map_bin, alignment=align_ns("sim4")))
    with pytest.raises((ValueError, KeyError)):
        GEDF_PGO.is_valid_config(
            make_optimizer_config("gedf+icp", True, scene_map_bin,
                                  alignment=align_ns("sim3", prior_weight=-1.0)))


def test_alignment_output_plumbing(scene_map_bin):
    cfg = make_optimizer_config("gedf+icp", True, scene_map_bin, alignment=align_ns("sim3"))
    ctx = GEDF_PGO.init_context(cfg)
    gd = make_graph_input(T_GT, perturbed_pose(T_GT, 2.0, 0.05), scene_registration_points())
    _, out = GEDF_PGO._optimize(ctx, gd)

    assert out.alignment_state is not None
    assert out.alignment_state.dtype == torch.float32 and not out.alignment_state.is_cuda
    out2 = pickle.loads(pickle.dumps(out))
    torch.testing.assert_close(out2.alignment_state, out.alignment_state)
    pp.SE3(out.motion[0].detach().double())        # write-back stays a valid 7-float SE3

    cfg_se3 = make_optimizer_config("gedf+icp", True, scene_map_bin)
    _, out_se3 = GEDF_PGO._optimize(GEDF_PGO.init_context(cfg_se3), gd)
    assert out_se3.alignment_state is None and out_se3.scale is None
    assert out_se3.alignment_type == "se3"


def test_alignment_residual_shapes(scene_mapper):
    """Locks the (N+P) residual/covariance bookkeeping for both graph types."""
    gd = make_graph_input(T_GT, T_GT, scene_registration_points())
    N = gd.observations.data["pixel2_uv"].shape[0]
    w = 100.0

    reg = GEDF_Registration(gd, field=scene_mapper, field_cfg=FIELD_NS,
                            alignment_cfg=align_ns("sim3", w)).to(dtype=torch.double)
    assert reg().shape == (N + 1, 1)
    cov = reg.covariance_array()
    assert cov.shape == (N + 1, 1, 1)
    assert float(cov[-1, 0, 0]) == pytest.approx(1.0 / w)

    hyb = GEDF_ICP(gd, field=scene_mapper, field_cfg=FIELD_NS,
                   alignment_cfg=align_ns("sim3", w)).to(dtype=torch.double)
    assert hyb().shape == (N + 1, 4)
    cov4 = hyb.covariance_array()
    assert cov4.shape == (N + 1, 4, 4)
    expected = torch.diag(torch.tensor([1.0 / w, 1.0, 1.0, 1.0], dtype=cov4.dtype))
    torch.testing.assert_close(cov4[-1], expected)


# --------------------------------------------------------------------------- #
# Sequential ICP-init -> whole-map field registration ("icp->gedf")
# --------------------------------------------------------------------------- #
def test_field_disabled_graph_is_pure_icp(scene_mapper):
    """field_enabled=False must keep the field rows inert even on a READY map
    (the pure-ICP init stage of "icp->gedf")."""
    assert scene_mapper.is_ready
    init = perturbed_pose(T_GT, 3.0, 0.1)
    gd = make_graph_input(T_GT, init, scene_registration_points())
    graph = Analytic_GEDF_ICP(gd, field=scene_mapper, field_cfg=FIELD_NS,
                              field_enabled=False).to(dtype=torch.double)
    r = graph()
    assert (r[:, 3] == 0).all()
    J = graph.build_jacobian().reshape(-1, 4, 7)
    assert (J[:, 3] == 0).all()
    cov = graph.covariance_array()
    assert torch.allclose(cov[:, 3, 3], torch.ones_like(cov[:, 3, 3]))


@pytest.mark.parametrize("autodiff", [False, True])
def test_icp_gedf_pose_recovery(scene_map_bin, autodiff):
    cfg = make_optimizer_config("icp->gedf", autodiff, scene_map_bin)
    GEDF_PGO.is_valid_config(cfg)
    context = GEDF_PGO.init_context(cfg)

    init = perturbed_pose(T_GT, 5.0, 0.2)
    gd = make_graph_input(T_GT, init, scene_registration_points())
    _, out = GEDF_PGO._optimize(context, gd)
    rot, trans = pose_error(out.motion.detach(), T_GT)
    # the final answer is pure field registration: limited by the field MAE
    assert rot < 1.5 and trans < 0.06, f"icp->gedf autodiff={autodiff}: {rot:.3f} deg {trans:.4f} m"


def test_icp_gedf_recovers_beyond_pure_gedf_basin(scene_map_bin):
    """The motivating case: an init far outside pure field registration's
    convergence basin is recovered because the ICP stage supplies the initial
    estimate; pure "gedf" from the same init must do strictly worse."""
    results = {}
    for graph_type in ("gedf", "icp->gedf"):
        # fresh (seeded, identical) init per run: pose2opt may share storage
        # with init_motion and the first solve would mutate a shared one
        init = perturbed_pose(T_GT, 20.0, 1.0)
        ctx = GEDF_PGO.init_context(make_optimizer_config(graph_type, False, scene_map_bin))
        gd = make_graph_input(T_GT, init, scene_registration_points())
        _, out = GEDF_PGO._optimize(ctx, gd)
        results[graph_type] = pose_error(out.motion.detach(), T_GT)

    rot_seq, trans_seq = results["icp->gedf"]
    _, trans_gedf = results["gedf"]
    assert rot_seq < 1.5 and trans_seq < 0.08, f"icp->gedf: {rot_seq:.3f} deg {trans_seq:.4f} m"
    assert trans_seq < trans_gedf, \
        f"pure gedf unexpectedly matched the ICP-seeded solve ({trans_gedf:.4f} m)"


def test_icp_gedf_cold_start_runs_pure_icp(scene_map_bin):
    """With a not-ready map the sequential mode must degrade to pure ICP (which
    recovers the pose exactly in this fixture), not return init unchanged."""
    cfg = make_optimizer_config("icp->gedf", False, None, source="online")
    context = GEDF_PGO.init_context(cfg)   # insert_keypoints=False: stays not-ready

    init = perturbed_pose(T_GT, 5.0, 0.2)
    gd = make_graph_input(T_GT, init, scene_registration_points())
    _, out = GEDF_PGO._optimize(context, gd)
    rot, trans = pose_error(out.motion.detach(), T_GT)
    assert rot < 0.5 and trans < 0.02, f"cold icp->gedf: {rot:.3f} deg {trans:.4f} m"


def test_icp_gedf_sim3_alignment(scene_map_bin):
    """The alignment axis must survive the two-solve flow: the reported scale
    comes from the final (field) solve and corrects a 1.2x depth bias."""
    cfg = make_optimizer_config("icp->gedf", True, scene_map_bin,
                                alignment=align_ns("sim3"))
    GEDF_PGO.is_valid_config(cfg)
    ctx = GEDF_PGO.init_context(cfg)
    gd = make_graph_input(T_GT, perturbed_pose(T_GT, 3.0, 0.1),
                          scene_registration_points(), warp_cam=lambda p: 1.2 * p)
    ctx, out = GEDF_PGO._optimize(ctx, gd)
    rot, trans = pose_error(out.motion.detach(), T_GT)
    assert rot < 1.5 and trans < 0.06, f"icp->gedf sim3: {rot:.3f} deg {trans:.4f} m"
    assert out.alignment_type == "sim3" and out.scale is not None
    assert out.scale == pytest.approx(1 / 1.2, rel=0.05)
    assert ctx["align_scale_prev"] is not None   # feed-forward uses the field solve


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
