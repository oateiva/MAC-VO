"""
Tests for the DepthCov adapter (`Module/Network/Depth/DepthCov/network.py`).

DepthCov's GP posterior lives in natural-log depth; the adapter converts it to the metric
depth + metric depth variance that `Covariance_2to3_full` consumes (delta method:
Var(depth) = exp(2*mu) * sigma^2). These tests pin that conversion, the coordinate-mapping
assumption (anchors normalized at frame resolution are valid at inference resolution), and
the constant-depth fallback contract for cold-start frames without a usable depth prior.

The `local` test additionally smoke-tests the real ScanNet checkpoint end-to-end.
"""
from pathlib import Path

import pypose as pp
import pytest
import torch

from DataLoader import CameraData
from Module.Network.Depth.DepthCov import DepthCov
from Module.Network.Depth.DepthCov.network import log_depth_to_metric
from Module.Network.Depth.DepthCov.depth_cov.utils.utils import normalize_coordinates

H, W = 48, 64


def make_camera(height: int = H, width: int = W, depth_prior: torch.Tensor | None = None) -> CameraData:
    K = torch.eye(3).unsqueeze(0)
    K[0, 0, 0], K[0, 1, 1] = 100.0, 100.0
    K[0, 0, 2], K[0, 1, 2] = width / 2, height / 2
    return CameraData.from_mono(
        T_BS=pp.identity_SE3(1), K=K, time_ns=[0], height=height, width=width,
        images=torch.rand(1, 3, height, width), depth_prior=depth_prior,
    )


def make_prior(num_anchors: int, height: int = H, width: int = W) -> torch.Tensor:
    """Sparse metric prior with `num_anchors` nonzero pixels on an even grid."""
    prior = torch.zeros(1, 1, height, width)
    flat = torch.linspace(0, height * width - 1, num_anchors).long()
    v, u = flat // width, flat % width
    prior[0, 0, v, u] = torch.rand(num_anchors) * 4.0 + 1.0
    return prior


def test_delta_method_units():
    """cov / depth^2 == log_var * cov_scale — the dimensionless invariant the backend
    relies on: sigma_zz must be a metric variance matching depth's units squared."""
    log_depth = torch.randn(1, 1, H, W, dtype=torch.float32)
    log_var = torch.rand(1, 1, H, W, dtype=torch.float32) + 1e-3

    depth, cov = log_depth_to_metric(log_depth, log_var)
    assert depth.dtype == torch.float32 and cov.dtype == torch.float32
    assert torch.equal(depth, log_depth.exp())
    torch.testing.assert_close(cov / depth.square(), log_var)
    assert bool((cov > 0).all())

    _, cov_scaled = log_depth_to_metric(log_depth, log_var, cov_scale=4.0)
    torch.testing.assert_close(cov_scaled, cov * 4.0)


def test_normalized_coords_resolution_invariant():
    """The adapter normalizes anchor coords at frame resolution and uses them at the
    (different) inference resolution. `normalize_coordinates` maps pixel centers to
    [-1, 1], so the two agree up to the half-pixel center offset."""
    frame_dims, inf_dims = (480, 640), (192, 256)
    px_frame = torch.tensor([[[120.0, 320.0], [0.0, 0.0], [479.0, 639.0]]])
    scale = torch.tensor(inf_dims, dtype=torch.float32) / torch.tensor(frame_dims, dtype=torch.float32)
    px_inf = px_frame * scale

    n_frame = normalize_coordinates(px_frame, frame_dims)
    n_inf = normalize_coordinates(px_inf, inf_dims)
    # Half-pixel center offset: |1/dim_inf - 1/dim_frame| <= 1/192
    torch.testing.assert_close(n_frame, n_inf, atol=2.0 / 192.0, rtol=0.0)


def test_fallback_no_prior():
    """No depth_prior -> constant init_depth / init_cov at frame resolution."""
    m = DepthCov(weight=None, init_depth=3.0, init_cov=10.0)
    out = m.deepodo_inference(make_camera(depth_prior=None))
    assert out.depth.shape == (1, 1, H, W) and out.cov is not None
    assert torch.equal(out.depth, torch.full((1, 1, H, W), 3.0))
    assert torch.equal(out.cov, torch.full((1, 1, H, W), 10.0))


def test_fallback_too_few_anchors():
    """Fewer than min_points anchors -> same constant fallback."""
    m = DepthCov(weight=None, min_points=4, init_depth=2.0, init_cov=5.0)
    out = m.deepodo_inference(make_camera(depth_prior=make_prior(2)))
    assert torch.equal(out.depth, torch.full((1, 1, H, W), 2.0))
    assert out.cov is not None and torch.equal(out.cov, torch.full((1, 1, H, W), 5.0))


def test_anchor_extraction_subsamples_to_budget():
    m = DepthCov(weight=None, max_points=64, min_points=4)
    anchors = m._extract_anchors(make_prior(300), H, W)
    assert anchors is not None
    coords_norm, sparse_log_depth, mean_log_depth = anchors
    assert coords_norm.shape == (1, 64, 2) and sparse_log_depth.shape == (1, 64, 1)
    assert bool(coords_norm.abs().le(1.0).all()), "normalized coords must lie in [-1, 1]"
    assert bool(torch.isfinite(sparse_log_depth).all()) and bool(torch.isfinite(mean_log_depth))


def test_conditioned_inference_shapes_random_weights():
    """Full GP path with random weights (small inference grid): finite metric depth and
    strictly positive variance at frame resolution, exercising IDepth.Output's
    jaxtyping contract (B 1 H W float32). The UNet halves the grid 5 times, so the
    inference dims must divide by 32."""
    m = DepthCov(weight=None, inference_size=(64, 96), max_points=32)
    m.eval()
    out = m.deepodo_inference(make_camera(depth_prior=make_prior(50)))
    assert out.depth.shape == (1, 1, H, W) and out.depth.dtype == torch.float32
    assert out.cov is not None and out.cov.shape == (1, 1, H, W)
    assert bool(torch.isfinite(out.depth).all()) and bool((out.depth > 0).all())
    assert bool((out.cov > 0).all())


WEIGHT = Path("./Model/depthcov_scannet.ckpt")


@pytest.mark.local
@pytest.mark.skipif(not WEIGHT.exists(), reason="DepthCov ScanNet checkpoint not downloaded")
def test_depthcov_smoke_real_weights():
    """Strict checkpoint load + conditioning sanity: the posterior reproduces the anchor
    depths at the anchor pixels and is less certain far from them."""
    import cv2
    import numpy as np

    m = DepthCov(weight=str(WEIGHT))  # strict=True load must succeed
    m.deepodo_initialize(type("C", (), {"device": "cuda" if torch.cuda.is_available() else "cpu"})())

    img_paths = sorted(Path("./Scripts/UnitTest/assets/test_sequence/TartanAir2_abs_P000").rglob("*.png"))
    assert img_paths, "test sequence assets missing"
    bgr = cv2.imread(str(img_paths[0]), cv2.IMREAD_COLOR)
    rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1)[None].float() / 255.0
    h, w = rgb.shape[-2:]

    # Smooth synthetic scene: depth ramp 2m..6m, 64 anchors on a grid.
    v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    gt_depth = 2.0 + 4.0 * v.float() / h
    prior = torch.zeros(1, 1, h, w)
    vs = torch.linspace(h * 0.1, h * 0.9, 8).long()
    us = torch.linspace(w * 0.1, w * 0.9, 8).long()
    gv, gu = torch.meshgrid(vs, us, indexing="ij")
    prior[0, 0, gv.flatten(), gu.flatten()] = gt_depth[gv.flatten(), gu.flatten()]

    frame = CameraData.from_mono(
        T_BS=pp.identity_SE3(1), K=torch.eye(3).unsqueeze(0), time_ns=[0],
        height=h, width=w, images=rgb, depth_prior=prior,
    )
    out = m.deepodo_inference(frame)

    assert out.cov is not None
    assert bool(torch.isfinite(out.depth).all()) and bool((out.cov > 0).all())
    depth_cpu, cov_cpu = out.depth.cpu(), out.cov.cpu()
    at_anchor = depth_cpu[0, 0, gv.flatten(), gu.flatten()]
    gt_anchor = gt_depth[gv.flatten(), gu.flatten()]
    assert bool(((at_anchor - gt_anchor).abs() / gt_anchor < 0.25).float().mean() > 0.9), \
        "posterior should reproduce anchor depths at anchor pixels"
    assert float(cov_cpu[0, 0, gv.flatten(), gu.flatten()].mean()) < float(cov_cpu.mean()), \
        "variance should be lower at anchors than on average"
