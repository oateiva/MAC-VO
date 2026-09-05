"""
Tests for the sky-mask depth validity plumbing added for DepthAnythingV3:

* `dilate_invalid` (Module/Network/Depth/DepthAnythingV3/api.py) - grows the
  invalid (False) region of a validity mask by a pixel margin.
* `DepthAnything3.deepodo_initialize` config validation for the new
  `sky_margin_px` / `sky_prob_thresh` knobs.
"""
from types import SimpleNamespace

import pytest
import torch

from Module.Network.Depth.DepthAnythingV3.api import DepthAnything3, dilate_invalid


def test_dilate_invalid_margin_zero_is_identity():
    valid = torch.rand((1, 1, 9, 9)) > 0.5
    assert torch.equal(dilate_invalid(valid, 0), valid)


def test_dilate_invalid_all_valid_stays_all_valid():
    valid = torch.ones((1, 1, 11, 11), dtype=torch.bool)
    assert torch.equal(dilate_invalid(valid, 3), valid)


@pytest.mark.parametrize("margin", [1, 2, 4])
def test_dilate_invalid_grows_single_pixel_clipped_at_border(margin: int):
    h, w = 15, 15
    cy, cx = 7, 8
    valid = torch.ones((1, 1, h, w), dtype=torch.bool)
    valid[0, 0, cy, cx] = False

    out = dilate_invalid(valid, margin)

    y0, y1 = max(0, cy - margin), min(h - 1, cy + margin)
    x0, x1 = max(0, cx - margin), min(w - 1, cx + margin)
    expected = torch.ones((h, w), dtype=torch.bool)
    expected[y0:y1 + 1, x0:x1 + 1] = False

    assert torch.equal(out[0, 0], expected)


def _minimal_da3_config(**overrides: object) -> SimpleNamespace:
    config = SimpleNamespace(device="cpu", weight="dummy")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _model_stub() -> DepthAnything3:
    """A DepthAnything3 instance that skips actual weight loading / device
    transfer (both require real checkpoints/GPU), exercising only
    `deepodo_initialize`'s config validation and knob wiring."""
    model = object.__new__(DepthAnything3)
    model.model = SimpleNamespace()  # no da3_metric attribute: hook registration is skipped
    model._load_weight = lambda weight: None
    model.to = lambda device: model  # type: ignore[reportAttributeAccessIssue]
    return model


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_sky_margin_px_rejects_bad_values(bad: object):
    model = _model_stub()
    with pytest.raises(ValueError):
        model.deepodo_initialize(_minimal_da3_config(sky_margin_px=bad))


@pytest.mark.parametrize("bad", [True, "0.3", None])
def test_sky_prob_thresh_rejects_bad_values(bad: object):
    model = _model_stub()
    with pytest.raises(ValueError):
        model.deepodo_initialize(_minimal_da3_config(sky_margin_px=0, sky_prob_thresh=bad))


def test_sky_margin_px_default_off_and_valid_values_accepted():
    model = _model_stub()
    model.deepodo_initialize(_minimal_da3_config())
    assert model.sky_margin_px == 0
    assert model.sky_prob_thresh == 0.3

    model = _model_stub()
    model.deepodo_initialize(_minimal_da3_config(sky_margin_px=10, sky_prob_thresh=0.5))
    assert model.sky_margin_px == 10
    assert model.sky_prob_thresh == 0.5


def test_sky_margin_px_on_without_sky_head_skips_hook_registration():
    """No `da3_metric` attribute on the wrapped model: initialize must not raise,
    and `_last_sky` stays None (the hook that would populate it never gets installed)."""
    model = _model_stub()
    model.deepodo_initialize(_minimal_da3_config(sky_margin_px=10))
    assert model._last_sky is None
