"""
Tests for the median-relative depth-covariance filter (`depth_cov_rel`) in
MappingPointSelector and CovAwareSelector_NoDepth: reliable pixels must be
selected by their RELATIVE covariance ordering even when all absolute
covariances are large (the monocular-depth case).
"""
from types import SimpleNamespace

import pytest
import torch

from Module.Frontend.StereoDepth import IDepth
from Module.Frontend.Matching import IMatcher
from Module.KeypointSelector import CovAwareSelector_NoDepth, MappingPointSelector

H, W = 64, 96
MASK_W = 8


def make_depth() -> IDepth.Output:
    """Constant depth; covariance large everywhere (far above any sane absolute
    threshold) but ~10x larger on the right half of the image, with realistic
    per-pixel variation so the median cuts strictly between the two halves."""
    gen = torch.Generator().manual_seed(1)
    depth = torch.full((1, 1, H, W), 3.0)
    cov = 5.0 + torch.rand((1, 1, H, W), generator=gen)          # right: ~5-6
    cov[..., : W // 2] = 0.5 + 0.1 * torch.rand((1, 1, H, W // 2), generator=gen)
    return IDepth.Output(depth=depth, cov=cov)


def make_match(batch: int = 1) -> IMatcher.Output:
    gen = torch.Generator().manual_seed(0)
    flow = torch.zeros((batch, 2, H, W))
    cov = torch.rand((batch, 3, H, W), generator=gen) * 0.1 + 0.05
    cov[:, 2] = 0.0
    return IMatcher.Output(flow=flow, cov=cov)


def test_mapping_selector_relative_filter():
    depth = make_depth()
    cfg = SimpleNamespace(max_depth=7.0, max_depth_cov=100.0, mask_width=MASK_W,
                          depth_cov_rel=1.0)
    MappingPointSelector.is_valid_config(cfg)
    pixels = MappingPointSelector(cfg).select_point(
        None, 200, depth, depth, None)  # type: ignore[arg-type]  # frame unused

    assert pixels.shape[0] > 0
    # (u, v) format: all selected pixels must come from the reliable left half
    assert bool((pixels[:, 0] < W // 2).all())

    # without the relative filter both halves pass the (loose) absolute gate
    cfg_off = SimpleNamespace(max_depth=7.0, max_depth_cov=100.0, mask_width=MASK_W)
    MappingPointSelector.is_valid_config(cfg_off)
    pixels_off = MappingPointSelector(cfg_off).select_point(
        None, 2000, depth, depth, None)  # type: ignore[arg-type]
    assert bool((pixels_off[:, 0] >= W // 2).any())


@pytest.mark.parametrize("match_batch", [1, 2])
def test_covaware_nodepth_relative_filter(match_batch: int):
    """match_batch=2 replicates window_length: 2 sequences, where the matcher
    output is batched over the frame pair while the depth cov has B=1."""
    depth = make_depth()
    match = make_match(batch=match_batch)
    cfg = SimpleNamespace(device="cpu", kernel_size=5, mask_width=MASK_W,
                          max_match_cov=100.0, depth_cov_rel=1.0)
    CovAwareSelector_NoDepth.is_valid_config(cfg)
    pixels = CovAwareSelector_NoDepth(cfg).select_point(
        None, 200, depth, depth, match)  # type: ignore[arg-type]

    assert pixels.shape[0] > 0
    assert bool((pixels[:, 0] < W // 2).all())

    # config without the optional key stays valid (backward compatibility)
    cfg_off = SimpleNamespace(device="cpu", kernel_size=5, mask_width=MASK_W,
                              max_match_cov=100.0)
    CovAwareSelector_NoDepth.is_valid_config(cfg_off)
    pixels_off = CovAwareSelector_NoDepth(cfg_off).select_point(
        None, 2000, depth, depth, match)  # type: ignore[arg-type]
    assert bool((pixels_off[:, 0] >= W // 2).any())
