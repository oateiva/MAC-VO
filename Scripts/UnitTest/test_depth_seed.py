"""Unit tests for the `depth_seed` knob on `MonocularFrontend`.

These tests build a `MonocularFrontend` without running its `__init__` (no model
weights, no GPU) and stub out `depth_model.deepodo_inference` with a fake object
that draws from torch's global RNG, so the tests only exercise the reseed-before-
inference behavior added in `Module/Frontend/Frontend.py`.
"""

from types import SimpleNamespace

import pytest
import torch

from Module.Frontend.Frontend import MonocularFrontend
from Module.Frontend.StereoDepth import IDepth


class _FakeDepthModel:
    def deepodo_inference(self, frame: object) -> IDepth.Output:
        return IDepth.Output(depth=torch.rand(1, 1, 4, 4))


def _make_frontend(depth_seed: int | None) -> MonocularFrontend:
    frontend = object.__new__(MonocularFrontend)
    frontend.config = SimpleNamespace(device="cpu", depth_seed=depth_seed)
    frontend.depth_model = _FakeDepthModel()
    frontend._device_depth = "cpu"
    frontend.depth_seed = depth_seed
    return frontend


def test_depth_seed_makes_estimate_depth_reproducible():
    frontend = _make_frontend(depth_seed=123)
    out1 = frontend.estimate_depth(object())
    out2 = frontend.estimate_depth(object())
    assert torch.equal(out1.depth, out2.depth)


def test_depth_seed_none_is_not_reproducible():
    frontend = _make_frontend(depth_seed=None)
    out1 = frontend.estimate_depth(object())
    out2 = frontend.estimate_depth(object())
    assert not torch.equal(out1.depth, out2.depth)


def _minimal_valid_config(**overrides: object) -> SimpleNamespace:
    config = SimpleNamespace(
        match=SimpleNamespace(type="GTMatcher", args=SimpleNamespace()),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_is_valid_config_accepts_depth_seed_zero():
    MonocularFrontend.is_valid_config(_minimal_valid_config(depth_seed=0))


def test_is_valid_config_rejects_negative_depth_seed():
    with pytest.raises(AssertionError):
        MonocularFrontend.is_valid_config(_minimal_valid_config(depth_seed=-1))


def test_is_valid_config_rejects_non_int_depth_seed():
    with pytest.raises(AssertionError):
        MonocularFrontend.is_valid_config(_minimal_valid_config(depth_seed="abc"))
