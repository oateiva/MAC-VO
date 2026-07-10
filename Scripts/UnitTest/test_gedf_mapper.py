"""
Tests for the G-EDF fitting pipeline and (further below, added with the
incremental lifecycle) the online GEDFMapper.
"""
import numpy as np
import pytest
import torch

from Module.Optimization.GEDF import GEDFConfig
from Module.Optimization.GEDF.Fitting import (
    edt_targets, fit_batch, gmm_predict, init_nms, sample_training_data,
)

from asset_gedf import (
    plane_edf, sample_plane, sample_sphere, sample_two_bumps, sphere_edf,
)

CFG = GEDFConfig(device="cpu")


def build_cube_batches(points: torch.Tensor, origins: list[tuple[float, float, float]],
                       cube_size: float = 1.0, halo: float = 0.5
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a cloud into per-cube local clouds (cube box + halo), padded."""
    clouds = []
    for o in origins:
        lo = torch.tensor(o, dtype=points.dtype) - halo
        hi = lo + cube_size + 2 * halo
        m = ((points >= lo) & (points <= hi)).all(-1)
        clouds.append(points[m])
    P = max(int(c.shape[0]) for c in clouds)
    B = len(clouds)
    pts = torch.zeros((B, P, 3), dtype=points.dtype)
    mask = torch.zeros((B, P), dtype=torch.bool)
    for i, c in enumerate(clouds):
        pts[i, :c.shape[0]] = c
        mask[i, :c.shape[0]] = True
    return pts, mask, torch.tensor(origins, dtype=points.dtype)


# --------------------------------------------------------------------------- #
# Fitting pipeline
# --------------------------------------------------------------------------- #
def test_edt_targets_exact():
    rng = np.random.default_rng(11)
    cloud = torch.from_numpy(rng.uniform(0, 2, size=(2, 300, 3)))
    mask = torch.rand(2, 300) > 0.3
    samples = torch.from_numpy(rng.uniform(-0.5, 2.5, size=(2, 100, 3)))

    got = edt_targets(samples, cloud, mask)

    for b in range(2):
        pts = cloud[b][mask[b]].numpy()
        expect = np.min(np.linalg.norm(
            samples[b].numpy()[:, None, :] - pts[None], axis=-1), axis=-1)
        np.testing.assert_allclose(got[b].numpy(), expect, atol=1e-12)


def test_sample_training_data_in_box():
    pts, mask, origins = build_cube_batches(sample_plane(z=0.5), [(0, 0, 0), (1, 1, 0)])
    gen = torch.Generator().manual_seed(0)
    X, d = sample_training_data(pts, mask, origins, CFG, gen)

    assert X.shape == (2, CFG.sample_points, 3)
    lo = origins - CFG.margin
    hi = origins + CFG.cube_size + CFG.margin
    assert bool(((X >= lo.unsqueeze(1) - 1e-9) & (X <= hi.unsqueeze(1) + 1e-9)).all())
    assert bool(torch.isfinite(d).all()) and bool((d >= 0).all())
    # roughly half the samples must be near the surface
    near = (d < 0.15).float().mean(-1)
    assert bool((near > 0.3).all())


def test_nms_init_two_bumps():
    """Positive gaussians must land near the interior EDT maximum (z ~ 0.5),
    negative ones near the surfaces (z ~ 0.15 / 0.85)."""
    pts, mask, origins = build_cube_batches(sample_two_bumps(), [(0, 0, 0)])
    gen = torch.Generator().manual_seed(0)
    theta0 = init_nms(pts, mask, origins, CFG, gen)

    w = theta0[0, :, 6]
    z = theta0[0, :, 2]
    assert int((w > 0).sum()) == CFG.num_gaussians // 2
    # NMS emits picks best-first; later positive slots may come from the random
    # fill (any voxel with d > 0), exactly like the C++ fallback. The strongest
    # pick must sit on the interior EDT maximum plane z = 0.5 (0.35 from both
    # patches).
    top_pos_z = z[w > 0][0]
    assert bool((top_pos_z - 0.5).abs() < 0.15)
    # negative gaussians seed at minima, i.e. near the surfaces
    neg_z = z[w < 0]
    near_patch = torch.minimum((neg_z - 0.15).abs(), (neg_z - 0.85).abs())
    assert bool(near_patch.max() < 0.25)
    # init sigma / weight conventions
    assert torch.allclose(theta0[..., 3:6],
                          torch.full_like(theta0[..., 3:6], CFG.init_sigma_param))
    assert bool((theta0[0, :, 6].abs() == pytest.approx(CFG.init_weight)) or
                torch.allclose(w.abs(), torch.full_like(w, CFG.init_weight)))


def _fit_and_probe(cloud: torch.Tensor, origins: list, edf_fn, seed: int = 1,
                   cfg: GEDFConfig = CFG
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pts, mask, org = build_cube_batches(cloud, origins)
    theta, mae, _, usable = fit_batch(pts, mask, org, None, cfg, seed=seed)
    assert bool(usable.all()), f"unusable fits, mae={mae.tolist()}"

    # probe each cube interior near the surface (|d_true| < 0.3)
    rng = np.random.default_rng(seed)
    B = len(origins)
    probes = torch.from_numpy(rng.uniform(0.05, 0.95, size=(B, 400, 3)))
    probes = probes + org.unsqueeze(1)
    d_pred = gmm_predict(theta, probes)
    d_true = edf_fn(probes.reshape(-1, 3)).reshape(B, -1)
    sel = d_true < 0.3
    return d_pred, d_true, sel


def test_fit_plane():
    origins = [(x, y, 0.0) for x in (0.0, 1.0, 2.0) for y in (0.0, 1.0, 2.0)]
    d_pred, d_true, sel = _fit_and_probe(sample_plane(z=0.5), origins, lambda p: plane_edf(p, z=0.5))
    err = (d_pred - d_true).abs()[sel]
    assert float(err.mean()) < 0.05, f"plane field MAE {float(err.mean()):.4f}"


def test_fit_sphere():
    """High-curvature case: these cubes also contain the deep interior of the
    sphere (distances up to ~1 m), which needs more capacity than the K=8
    default - the C++ adaptive loop escalates to K in {16, 32} for such cubes."""
    cfg = GEDFConfig(device="cpu", num_gaussians=24, lm_iters_cold=60)
    origins = [(x, y, z) for x in (1.0, 2.0) for y in (1.0, 2.0) for z in (1.0, 2.0)]
    d_pred, d_true, sel = _fit_and_probe(
        sample_sphere(center=(1.5, 1.5, 1.5), radius=1.0), origins,
        lambda p: sphere_edf(p, center=(1.5, 1.5, 1.5), radius=1.0), cfg=cfg)
    err = (d_pred - d_true).abs()[sel]
    assert float(err.mean()) < 0.05, f"sphere field MAE {float(err.mean()):.4f}"


def test_warm_start_refit():
    """A warm refit on slightly perturbed points must reach cold-fit quality."""
    origins = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    pts, mask, org = build_cube_batches(sample_plane(extent=2.0, z=0.5), origins)
    theta_cold, mae_cold, _, usable = fit_batch(pts, mask, org, None, CFG, seed=2)
    assert bool(usable.all())

    gen = torch.Generator().manual_seed(42)
    pts2 = pts + torch.randn(pts.shape, dtype=pts.dtype, generator=gen) * 0.005
    theta_warm, mae_warm, _, usable2 = fit_batch(pts2, mask, org, theta_cold, CFG, seed=3)
    assert bool(usable2.all())
    assert bool((mae_warm <= mae_cold + 0.01).all()), \
        f"warm {mae_warm.tolist()} vs cold {mae_cold.tolist()}"


# --------------------------------------------------------------------------- #
# Incremental mapper lifecycle
# --------------------------------------------------------------------------- #
from Module.Optimization.GEDF import GEDFMapper  # noqa: E402


def make_mapper(**overrides) -> GEDFMapper:
    cfg = GEDFConfig(device="cpu", budget_cubes_per_frame=8)
    for key, value in overrides.items():
        assert hasattr(cfg, key)
        setattr(cfg, key, value)
    return GEDFMapper(cfg)


def test_insert_dedup_and_cap():
    mapper = make_mapper(max_points_per_cube=64)
    pts = torch.from_numpy(
        np.random.default_rng(0).uniform(0.1, 0.9, size=(50, 3))).float()

    mapper.insert(pts)
    n0 = mapper._n_points[0]
    mapper.insert(pts)                     # exact duplicates: nothing new
    assert mapper._n_points[0] == n0

    dense = torch.from_numpy(
        np.random.default_rng(1).uniform(0.1, 0.9, size=(500, 3))).float()
    mapper.insert(dense)
    assert mapper._n_points[0] <= 64       # cap enforced


def test_cov_gate():
    mapper = make_mapper(cov_trace_gate=0.01)
    pts = torch.rand(20, 3) * 0.8 + 0.1
    cov = torch.eye(3).expand(20, 3, 3).clone() * 0.001    # trace 0.003 -> keep
    cov[10:] = torch.eye(3) * 0.02                          # trace 0.06 -> drop
    mapper.insert(pts, cov)
    assert mapper._n_points[0] <= 10


def test_incremental_fit_serves_field():
    """Insert a plane cloud in chunks with budgeted refits; the final field must
    match the analytic distance near the surface, and stale (dirty but not yet
    refitted) cubes must keep serving their previous fit."""
    mapper = make_mapper()
    cloud = sample_plane(n=6000, extent=3.0, z=0.5).float()
    chunks = torch.chunk(cloud[torch.randperm(cloud.shape[0])], 10)

    for chunk in chunks:
        mapper.insert(chunk)
        mapper.refit(camera_pos=torch.tensor([1.5, 1.5, 0.5]))
    mapper.flush()

    assert mapper.num_valid_cubes >= 9
    assert mapper.is_ready

    rng = np.random.default_rng(4)
    probes = torch.from_numpy(np.column_stack([
        rng.uniform(0.3, 2.7, size=(500, 2)), rng.uniform(0.25, 0.75, size=500)]))
    d, g = mapper.query_with_grad(probes)
    d_true = plane_edf(probes, z=0.5)
    in_map = d < 19.0
    assert float(in_map.float().mean()) > 0.95
    err = (d[in_map] - d_true[in_map]).abs()
    assert float(err.mean()) < 0.06, f"online field MAE {float(err.mean()):.4f}"
    # gradient should point along +-z near a z-plane
    gz = g[in_map & (d_true > 0.1)]
    assert float(gz[:, 2].abs().mean()) > 2.0 * float(gz[:, :2].abs().mean())


def test_dirty_budget_scheduling():
    """20 cold cubes with budget 8 -> 8, 8, 4 refits."""
    mapper = make_mapper(min_points_fit=5)
    rng = np.random.default_rng(7)
    for i in range(20):
        base = np.array([float(i), 0.0, 0.0])
        pts = torch.from_numpy(base + rng.uniform(0.1, 0.9, size=(30, 3))).float()
        mapper.insert(pts)

    fittable = [s for s in mapper._dirty if mapper._n_points.get(s, 0) >= 5]
    assert len(fittable) >= 20
    counts = [mapper.refit()["refitted"] for _ in range(3)]
    assert counts[0] == 8 and counts[1] == 8
    assert sum(counts) >= 20   # a couple of halo-dirtied neighbors may add fits


def test_incremental_equals_batch():
    """Chunked insert+refit must converge to (approximately) the same field as a
    single batch insert + flush. Not bitwise: warm-start paths differ by design."""
    cloud = sample_plane(n=5000, extent=2.0, z=0.5).float()

    m_batch = make_mapper()
    m_batch.insert(cloud)
    m_batch.flush()

    m_inc = make_mapper()
    for chunk in torch.chunk(cloud, 8):
        m_inc.insert(chunk)
        m_inc.refit()
    m_inc.flush()

    rng = np.random.default_rng(9)
    probes = torch.from_numpy(np.column_stack([
        rng.uniform(0.2, 1.8, size=(400, 2)), rng.uniform(0.3, 0.7, size=400)]))
    d_batch = m_batch.query(probes)
    d_inc = m_inc.query(probes)
    both = (d_batch < 19.0) & (d_inc < 19.0)
    assert float(both.float().mean()) > 0.9
    diff = (d_batch[both] - d_inc[both]).abs()
    assert float(diff.median()) < 0.02, f"median diff {float(diff.median()):.4f}"


def test_export_roundtrip_online(tmp_path):
    mapper = make_mapper()
    mapper.insert(sample_plane(n=4000, extent=2.0, z=0.5).float())
    mapper.flush()
    assert mapper.num_valid_cubes > 0

    path = tmp_path / "online.bin"
    mapper.export_gdf1(path)
    frozen = GEDFMapper.from_gdf1(path, dtype=torch.float64)
    assert frozen.is_ready and frozen.frozen

    rng = np.random.default_rng(2)
    probes = torch.from_numpy(np.column_stack([
        rng.uniform(0.2, 1.8, size=(300, 2)), rng.uniform(0.3, 0.7, size=300)]))
    d_live = mapper.query(probes.float()).double()
    d_frozen = frozen.query(probes)
    # f32 storage quantization only
    torch.testing.assert_close(d_frozen, d_live, atol=1e-4, rtol=1e-4)


def test_sample_surface():
    mapper = make_mapper()
    mapper.insert(sample_plane(n=5000, extent=2.0, z=0.5).float())
    mapper.flush()

    points, dist = mapper.sample_surface(resolution=0.1, iso=0.1, max_points=5000)
    assert points.shape[0] > 0 and points.shape[0] <= 5000
    assert points.dtype == torch.float32 and dist.dtype == torch.float32
    assert not points.is_cuda and not dist.is_cuda
    assert bool((dist < 0.1).all())
    # sampled points must hug the true surface (iso + fit error tolerance)
    assert float((points[:, 2] - 0.5).abs().max()) < 0.1 + 0.06

    # cap is enforced
    pts_capped, _ = mapper.sample_surface(resolution=0.05, iso=0.1, max_points=100)
    assert pts_capped.shape[0] == 100

    # empty map returns empty tensors
    empty_pts, empty_dist = make_mapper().sample_surface()
    assert empty_pts.shape == (0, 3) and empty_dist.shape == (0,)


def test_gaussians_accessor():
    """gaussians() must expose exactly the valid, non-padding components as CPU
    float32 (means, sigmas=p**2, signed weights, broadcast cube MAE)."""
    # Fitted map: counts, dtypes, MAE broadcast.
    mapper = make_mapper()
    mapper.insert(sample_plane(n=5000, extent=2.0, z=0.5).float())
    mapper.flush()

    # max_sigma=0 disables the broad-component display filter: ALL components
    means, sigmas, weights, mae = mapper.gaussians(max_sigma=0)
    n = mapper.num_valid_gaussians
    assert means.shape == (n, 3) and sigmas.shape == (n, 3)
    assert weights.shape == (n,) and mae.shape == (n,)
    for t in (means, sigmas, weights, mae):
        assert t.dtype == torch.float32 and not t.is_cuda
    assert bool((sigmas > 0).all())
    # every component's MAE is one of the valid cubes' MAE values
    cube_maes = mapper._mae[mapper._valid].float()
    assert bool(torch.isin(mae, cube_maes).all())

    # cap keeps the top-|weight| components
    _, _, w_cap, _ = mapper.gaussians(max_gaussians=2, max_sigma=0)
    assert w_cap.shape == (2,)
    assert bool((w_cap.abs() >= weights.abs().sort(descending=True).values[1] - 1e-6).all())

    # default max_sigma (2 * cube_size) drops the near-constant broad
    # components the fit uses as DC offsets (sigma can reach ~1e10 m) but
    # keeps the tight surface structure
    m_def, s_def, w_def, _ = mapper.gaussians()
    limit = 2.0 * mapper.cube_size
    assert 0 < m_def.shape[0] <= n
    assert bool((s_def.max(dim=-1).values <= limit).all())
    over = sigmas.max(dim=-1).values > limit
    assert m_def.shape[0] == n - int(over.sum())
    # explicit override behaves the same way
    m_tight, *_ = mapper.gaussians(max_sigma=0.5 * limit)
    assert m_tight.shape[0] <= m_def.shape[0]

    # padding + invalid-cube filtering on a hand-built map
    hand = make_mapper()
    K = hand.cfg.num_gaussians
    h_means = torch.zeros((2, K, 3))
    h_means[0, :3] = torch.tensor([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
    h_p = torch.zeros((2, K, 3))
    h_p[0, :3] = 0.3
    h_w = torch.zeros((2, K))
    h_w[0, :3] = torch.tensor([1.0, -0.5, 0.25])
    hand._append_cubes(
        origin_i=torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int64),
        means=h_means, p_sigma=h_p, weights=h_w,
        n_gauss=torch.tensor([3, K], dtype=torch.int64),
        mae=torch.tensor([0.01, 0.02]), std=torch.tensor([0.005, 0.005]),
        valid=torch.tensor([True, False]),
    )
    m2, s2, w2, mae2 = hand.gaussians()
    assert m2.shape == (3, 3)                                    # invalid cube + padding excluded
    torch.testing.assert_close(m2, h_means[0, :3])
    torch.testing.assert_close(s2, torch.full((3, 3), 0.3 ** 2))  # sigma = p**2
    torch.testing.assert_close(w2, h_w[0, :3])                   # sign preserved
    assert bool((mae2 == 0.01).all())

    # empty map returns empty tensors
    e_m, e_s, e_w, e_mae = make_mapper().gaussians()
    assert e_m.shape == (0, 3) and e_s.shape == (0, 3)
    assert e_w.shape == (0,) and e_mae.shape == (0,)


def test_cubes_accessor():
    """cubes() must expose one entry per allocated cube: grid-aligned centers,
    the valid mask, and per-cube MAE, as CPU tensors."""
    mapper = make_mapper()
    mapper.insert(sample_plane(n=5000, extent=2.0, z=0.5).float())
    mapper.flush()

    centers, valid, mae = mapper.cubes()
    C = mapper.num_cubes
    assert centers.shape == (C, 3) and valid.shape == (C,) and mae.shape == (C,)
    assert centers.dtype == torch.float32 and not centers.is_cuda
    assert valid.dtype == torch.bool and mae.dtype == torch.float32
    assert int(valid.sum()) == mapper.num_valid_cubes > 0
    # centers sit at cube midpoints of the world-anchored grid
    frac = centers / mapper.cube_size - torch.floor(centers / mapper.cube_size)
    torch.testing.assert_close(frac, torch.full_like(frac, 0.5))
    assert bool(torch.isfinite(mae[valid]).all())

    # empty map returns empty tensors
    e_c, e_v, e_mae = make_mapper().cubes()
    assert e_c.shape == (0, 3) and e_v.shape == (0,) and e_mae.shape == (0,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_cuda_parity():
    """Same seed on both devices must give equivalent map QUALITY. Bitwise
    equality is not achievable: LM accept/reject branches can flip on
    float reduction-order differences and the trajectories diverge."""
    cloud = sample_plane(n=3000, extent=2.0, z=0.5).float()
    rng = np.random.default_rng(6)
    probes = torch.from_numpy(np.column_stack([
        rng.uniform(0.2, 1.8, size=(400, 2)), rng.uniform(0.3, 0.7, size=400)]))
    d_true = plane_edf(probes, z=0.5)

    field_mae = {}
    for device in ("cpu", "cuda"):
        m = make_mapper(device=device)
        m.insert(cloud)
        m.flush()
        d = m.query(probes.to(device).double()).cpu()
        in_map = d < 19.0
        assert float(in_map.float().mean()) > 0.9
        field_mae[device] = float((d[in_map] - d_true[in_map]).abs().mean())

    assert abs(field_mae["cpu"] - field_mae["cuda"]) < 0.01, field_mae
    assert max(field_mae.values()) < 0.06, field_mae
