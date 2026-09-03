"""
Tests for the keyframe policies (Module/KeyframeTracker.py): each criterion's
switch rule and max_gap cap, EveryN's first-switch offset, AnyOf composition
(every wrapped policy is evaluated), and type/args config validation.
"""
from types import SimpleNamespace
from typing import Any

import pytest

from Module.KeyframeTracker import (
    AnyOf, BaselineRatio, Covisibility, EveryN, IKeyframePolicy, Parallax, TrackContext)


def ctx(**overrides) -> TrackContext:
    base: dict[str, Any] = dict(frames_since_kf=1, n_grid=100, n_kf_matches=100,
                                parallax_px=0.0, translation_m=0.0, median_depth_m=5.0)
    base.update(overrides)
    return TrackContext(**base)


def make(policy_type: str, **args) -> IKeyframePolicy:
    cfg = SimpleNamespace(type=policy_type, args=SimpleNamespace(**args))
    IKeyframePolicy.is_valid_config(cfg)
    return IKeyframePolicy.instantiate(cfg.type, cfg.args)


def test_every_n_and_first_offset():
    p = make("EveryN", n=3)
    assert [p.should_switch(ctx(frames_since_kf=k)) for k in (1, 2, 3)] == [False, False, True]
    p = make("EveryN", n=3, first=1)
    assert p.should_switch(ctx(frames_since_kf=1))          # first switch on the offset
    assert not p.should_switch(ctx(frames_since_kf=2))      # then every n
    assert p.should_switch(ctx(frames_since_kf=3))
    assert repr(p) == "every3@1"


def test_parallax_threshold_and_gap():
    p = make("Parallax", px=40.0, max_gap=10)
    assert not p.should_switch(ctx(parallax_px=39.9))
    assert p.should_switch(ctx(parallax_px=40.0))
    assert p.should_switch(ctx(parallax_px=0.0, frames_since_kf=10))
    assert isinstance(p, Parallax)
    default = make("Parallax", px=40.0)
    assert isinstance(default, Parallax) and default.max_gap == 20


def test_covisibility_fraction():
    p = make("Covisibility", frac=0.5)
    assert not p.should_switch(ctx(n_grid=100, n_kf_matches=50))
    assert p.should_switch(ctx(n_grid=100, n_kf_matches=49))
    assert p.should_switch(ctx(n_grid=0, n_kf_matches=0))
    assert p.should_switch(ctx(frames_since_kf=20))


def test_baseline_ratio():
    p = make("BaselineRatio", ratio=0.1)
    assert not p.should_switch(ctx(translation_m=0.49, median_depth_m=5.0))
    assert p.should_switch(ctx(translation_m=0.5, median_depth_m=5.0))
    assert p.should_switch(ctx(median_depth_m=0.0))


def test_any_of_evaluates_every_policy():
    p = make("AnyOf", policies=[
        SimpleNamespace(type="Parallax", args=SimpleNamespace(px=40.0)),
        SimpleNamespace(type="EveryN", args=SimpleNamespace(n=2)),
    ])
    assert isinstance(p, AnyOf)
    assert repr(p) == "parallax40|every2"
    # Parallax fires first; EveryN must still see the tick so its own clock advances.
    assert p.should_switch(ctx(parallax_px=100.0, frames_since_kf=2))
    every_n = p.policies[1]
    assert isinstance(every_n, EveryN) and every_n.switches == 1
    assert not p.should_switch(ctx(parallax_px=0.0, frames_since_kf=1))


def test_config_validation():
    with pytest.raises(KeyError):
        make("Parallax", px=40.0, unexpected=1)
    with pytest.raises(ValueError):
        make("Parallax", px=-1.0)
    with pytest.raises(ValueError):
        make("Covisibility", frac=1.5)
    with pytest.raises(ValueError):
        make("EveryN", n=0)
    with pytest.raises(ValueError):
        make("AnyOf", policies=[])
    with pytest.raises(ValueError):
        make("AnyOf", policies=[SimpleNamespace(type="Parallax", args=SimpleNamespace(px=0.0))])
    for cls in (Parallax, Covisibility, BaselineRatio, EveryN, AnyOf):
        assert IKeyframePolicy.get_class(cls.__name__) is cls
