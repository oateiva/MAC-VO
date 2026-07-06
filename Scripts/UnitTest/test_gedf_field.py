"""
Tests for the G-EDF field storage / query path (Module/Optimization/GEDF).

The NumPy reference evaluator vendored below is a direct port of
`G-EDF/scripts/gdf1_field.py` (`_eval_cube` + the blending loop of
`eval_volume`), evaluated at arbitrary points instead of a dense grid. It is the
parity oracle for the torch implementation.
"""
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from Module.Optimization.GEDF import GEDFConfig, GEDFMapper, read_gdf1, write_gdf1

PLANE_NOSE_BIN = Path(r"C:\Users\oat\Documents\Github\G-EDF-Loc\plane_nose_model.bin")

CUBE_SIZE = 1.0
MARGIN = 0.25
OOB = 20.0


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


# --------------------------------------------------------------------------- #
# Independent GDF1 writer (second implementation of the byte layout, used to
# validate Export.read_gdf1 against something other than Export.write_gdf1)
# --------------------------------------------------------------------------- #
def write_gdf1_independent(path: Path, cubes: list[dict], cube_size: float, margin: float) -> None:
    origins = np.stack([c["origin"] for c in cubes])
    bmin, bmax = origins.min(0), origins.max(0) + cube_size
    maes = np.array([c["mae"] for c in cubes])
    buf = bytearray()
    buf += struct.pack("<4s", b"GDF1")
    buf += struct.pack("<I", 1)
    buf += struct.pack("<I", len(cubes))
    buf += struct.pack("<f", float(maes.mean()))
    buf += struct.pack("<f", float(maes.std()))
    buf += struct.pack("<3f", *[float(v) for v in bmin])
    buf += struct.pack("<3f", *[float(v) for v in bmax])
    buf += struct.pack("<f", cube_size)
    buf += struct.pack("<f", 0.25)
    buf += struct.pack("<f", margin)
    buf += bytes(64)
    for c in cubes:
        buf += struct.pack("<3f", *[float(v) for v in c["origin"]])
        buf += struct.pack("<f", float(c["mae"]))
        buf += struct.pack("<f", float(c["std_dev"]))
        buf += struct.pack("<I", c["means"].shape[0])
        for k in range(c["means"].shape[0]):
            buf += struct.pack("<I", k)
            buf += struct.pack("<3f", *[float(v) for v in c["means"][k]])
            buf += struct.pack("<3f", *[float(v) for v in c["sigmas"][k]])
            buf += struct.pack("<f", float(c["weights"][k]))
    path.write_bytes(bytes(buf))


# --------------------------------------------------------------------------- #
# Vendored NumPy reference (port of gdf1_field.py, see module docstring)
# --------------------------------------------------------------------------- #
def numpy_reference_eval(points: np.ndarray, cubes: list[dict],
                         cube_size: float, margin: float, oob: float = OOB) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    wsum = np.zeros(pts.shape[0])
    wtot = np.zeros(pts.shape[0])
    for c in cubes:
        o = c["origin"]
        d0 = np.minimum(pts[:, 0] - o[0], o[0] + cube_size - pts[:, 0])
        d1 = np.minimum(pts[:, 1] - o[1], o[1] + cube_size - pts[:, 1])
        d2 = np.minimum(pts[:, 2] - o[2], o[2] + cube_size - pts[:, 2])
        min_dist = np.minimum(np.minimum(d0, d1), d2)
        t = np.clip(1.0 + min_dist / margin, 0.0, 1.0)
        bw = t * t * (3.0 - 2.0 * t)
        active = bw > 1e-6
        if not np.any(active):
            continue
        lam = c["sigmas"].astype(np.float64) ** 4                  # (G, 3)
        d = pts[:, None, :] - c["means"][None]                     # (N, G, 3)
        with np.errstate(divide="ignore"):                         # lam == 0 -> dsq inf -> exp 0
            dsq = np.sum(d * d / lam[None], axis=-1)               # (N, G)
        val = np.sum(c["weights"][None] * np.exp(-0.5 * dsq), axis=-1)
        wsum += np.where(active, bw * val, 0.0)
        wtot += np.where(active, bw, 0.0)
    out = np.full(pts.shape[0], oob)
    good = wtot > 1e-6
    out[good] = wsum[good] / wtot[good]
    return out


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_synthetic_cubes(seed: int = 7) -> list[dict]:
    """2x2x2 grid-aligned cubes, 3-10 gaussians each, negative weights included."""
    rng = np.random.default_rng(seed)
    cubes = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                origin = np.array([ix, iy, iz], dtype=np.float64) * CUBE_SIZE
                g = int(rng.integers(3, 11))
                means = origin + rng.uniform(-MARGIN, CUBE_SIZE + MARGIN, size=(g, 3))
                sigmas = rng.uniform(0.4, 0.8, size=(g, 3))        # lambda = p^4
                weights = rng.uniform(0.05, 0.6, size=(g,))
                weights[rng.random(g) < 0.3] *= -1.0               # some negative
                cubes.append(dict(origin=origin, mae=rng.uniform(0.01, 0.05),
                                  std_dev=rng.uniform(0.005, 0.02),
                                  means=means, sigmas=sigmas, weights=weights))
    return cubes


def sample_probe_points(seed: int = 13, n: int = 2000) -> np.ndarray:
    """Interior points, blend-zone points, near-outside points and far-outside points."""
    rng = np.random.default_rng(seed)
    interior = rng.uniform(0.3, 1.7, size=(n // 2, 3))             # deep inside cubes
    blend = np.concatenate([
        rng.uniform(-0.2, 2.2, size=(n // 4, 1)),                  # x spans faces/outside
        rng.uniform(0.0, 2.0, size=(n // 4, 2)),
    ], axis=1)
    edges = 1.0 + rng.uniform(-MARGIN, MARGIN, size=(n // 8, 3))   # around the center corner
    far = rng.uniform(10.0, 20.0, size=(n - n // 2 - n // 4 - n // 8, 3))
    pts = np.concatenate([interior, blend, edges, far])
    # avoid exact face-distance ties (measure-zero non-differentiable set)
    return pts + 1.2345e-7


def cubes_to_mapper(cubes: list[dict], tmp_path: Path, device: str = "cpu") -> GEDFMapper:
    path = tmp_path / "synthetic.bin"
    write_gdf1_independent(path, cubes, CUBE_SIZE, MARGIN)
    cfg = GEDFConfig(device=device)
    return GEDFMapper.from_gdf1(path, cfg, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_gdf1_roundtrip(tmp_path: Path):
    cubes = make_synthetic_cubes()
    path = tmp_path / "map.bin"
    write_gdf1_independent(path, cubes, CUBE_SIZE, MARGIN)

    header, loaded = read_gdf1(path)
    assert header["num_cubes"] == len(cubes)
    assert header["cube_size"] == pytest.approx(CUBE_SIZE)
    assert header["margin"] == pytest.approx(MARGIN)
    for orig, got in zip(cubes, loaded):
        np.testing.assert_allclose(got["origin"], orig["origin"], atol=1e-6)
        np.testing.assert_allclose(got["means"], orig["means"], rtol=1e-6)
        np.testing.assert_allclose(got["sigmas"], orig["sigmas"], rtol=1e-6)
        np.testing.assert_allclose(got["weights"], orig["weights"], rtol=1e-6)
        assert got["mae"] == pytest.approx(orig["mae"], abs=1e-6)

    # Our own writer must round-trip through our reader identically
    path2 = tmp_path / "map2.bin"
    C = len(loaded)
    K = max(c["means"].shape[0] for c in loaded)
    means = np.zeros((C, K, 3)); sigmas = np.zeros((C, K, 3)); weights = np.zeros((C, K))
    ng = np.array([c["means"].shape[0] for c in loaded])
    for i, c in enumerate(loaded):
        g = c["means"].shape[0]
        means[i, :g], sigmas[i, :g], weights[i, :g] = c["means"], c["sigmas"], c["weights"]
    write_gdf1(path2, np.stack([c["origin"] for c in loaded]), means, sigmas, weights, ng,
               np.array([c["mae"] for c in loaded]), np.array([c["std_dev"] for c in loaded]),
               CUBE_SIZE, MARGIN)
    header2, loaded2 = read_gdf1(path2)
    assert header2["num_cubes"] == C
    for a, b in zip(loaded, loaded2):
        np.testing.assert_allclose(a["means"], b["means"], rtol=1e-6)
        np.testing.assert_allclose(a["sigmas"], b["sigmas"], rtol=1e-6)
        np.testing.assert_allclose(a["weights"], b["weights"], rtol=1e-6)

    # Mapper must store lambda = sigma**4 (as inverse)
    mapper = GEDFMapper.from_gdf1(path, GEDFConfig(device="cpu"), dtype=torch.float64)
    assert mapper.num_cubes == len(cubes)
    slot0_ng = int(mapper._n_gauss[0].item())
    expect_inv_lam = 1.0 / (loaded[0]["sigmas"] ** 4)
    np.testing.assert_allclose(mapper._inv_lam[0, :slot0_ng].cpu().numpy(),
                               expect_inv_lam, rtol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_field_parity_vs_numpy_reference(tmp_path: Path, device: str):
    cubes = make_synthetic_cubes()
    mapper = cubes_to_mapper(cubes, tmp_path, device=device)
    pts = sample_probe_points()

    # Reference evaluated with the same (grid-snapped == original) origins and the
    # float32-quantized parameters actually stored in the file.
    _, loaded = read_gdf1(tmp_path / "synthetic.bin")
    expected = numpy_reference_eval(pts, loaded, CUBE_SIZE, MARGIN)

    got = mapper.query(torch.from_numpy(pts).to(device)).cpu().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-9)

    oob_rows = expected == OOB
    assert oob_rows.any(), "probe set should include out-of-map points"
    assert (got[oob_rows] == OOB).all()


@pytest.mark.local
@pytest.mark.skipif(not PLANE_NOSE_BIN.exists(), reason="plane_nose_model.bin not present")
def test_field_parity_real_map():
    header, loaded = read_gdf1(PLANE_NOSE_BIN)
    cs = header["cube_size"]
    origins = np.stack([c["origin"] for c in loaded])
    idx = np.floor((origins + 0.01 * cs) / cs)
    assert np.abs(idx * cs - origins).max() < 1e-3, \
        "real map cube origins are not grid aligned; snapped-grid queries would deviate"

    mapper = GEDFMapper.from_gdf1(PLANE_NOSE_BIN, dtype=torch.float64)
    rng = np.random.default_rng(3)
    bmin = np.array(header["bounds_min"]); bmax = np.array(header["bounds_max"])
    pts = rng.uniform(bmin - 0.5, bmax + 0.5, size=(4000, 3))

    snapped = [dict(c, origin=i * cs) for c, i in zip(loaded, idx)]
    expected = numpy_reference_eval(pts, snapped, cs, header["margin"])
    got = mapper.query(torch.from_numpy(pts)).cpu().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("device", _devices())
def test_analytic_grad_vs_autograd(tmp_path: Path, device: str):
    cubes = make_synthetic_cubes()
    mapper = cubes_to_mapper(cubes, tmp_path, device=device)
    pts = torch.from_numpy(sample_probe_points()).to(device)

    dist_a, grad_a = mapper.query_with_grad(pts)

    pts_req = pts.clone().requires_grad_(True)
    dist = mapper.query(pts_req)
    grad_auto = torch.autograd.grad(dist.sum(), pts_req)[0]

    torch.testing.assert_close(dist_a, dist.detach(), atol=1e-12, rtol=0)
    torch.testing.assert_close(grad_a, grad_auto, atol=1e-7, rtol=0)
    # the probe set must actually exercise blend zones for this test to mean anything
    in_blend = (torch.frac(pts / CUBE_SIZE) * CUBE_SIZE)
    in_blend = ((in_blend < MARGIN) | (in_blend > CUBE_SIZE - MARGIN)).any(-1)
    assert bool(in_blend.any())


@pytest.mark.parametrize("device", _devices())
def test_oob_sentinel(tmp_path: Path, device: str):
    mapper = cubes_to_mapper(make_synthetic_cubes(), tmp_path, device=device)
    far = torch.tensor([[50.0, 50.0, 50.0], [-30.0, 0.5, 0.5]],
                       dtype=torch.float64, device=device)

    dist, grad = mapper.query_with_grad(far)
    assert (dist == OOB).all()
    assert (grad == 0).all()

    far_req = far.clone().requires_grad_(True)
    d = mapper.query(far_req)
    assert (d == OOB).all()
    g = torch.autograd.grad(d.sum(), far_req)[0]
    assert (g == 0).all()


def test_blend_continuity(tmp_path: Path):
    """The blended field must be continuous across cube faces (C^1 seams)."""
    mapper = cubes_to_mapper(make_synthetic_cubes(), tmp_path)
    eps = 1e-4
    rng = np.random.default_rng(5)
    yz = rng.uniform(0.2, 1.8, size=(200, 2))
    for face_x in (1.0,):     # interior face between the two cube layers
        lo = np.column_stack([np.full(len(yz), face_x - eps), yz])
        hi = np.column_stack([np.full(len(yz), face_x + eps), yz])
        f_lo = mapper.query(torch.from_numpy(lo)).numpy()
        f_hi = mapper.query(torch.from_numpy(hi)).numpy()
        assert np.abs(f_hi - f_lo).max() < 10.0 * 2 * eps


def test_query_differentiable_in_pypose_graph(tmp_path: Path):
    import pypose as pp

    mapper = cubes_to_mapper(make_synthetic_cubes(), tmp_path)
    pose = pp.Parameter(pp.SE3(torch.tensor([[0.1, 0.05, -0.02, 0.0, 0.0, 0.0, 1.0]],
                                            dtype=torch.float64)))
    pts_c = torch.tensor([[0.5, 0.5, 0.5], [1.2, 0.9, 0.4]], dtype=torch.float64)

    dist = mapper.query(pose.Act(pts_c))
    loss = dist.square().sum()
    loss.backward()
    assert pose.grad is not None
    assert torch.isfinite(pose.grad).all()
    assert pose.grad.abs().sum() > 0
