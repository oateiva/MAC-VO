"""When should the odometry adopt a new keyframe?

Port of learningUAVO's gtsam_backend/keyframe_selection.py onto MAC-VO's
`type/args` config pattern. This is NOT the frame-subsampling gate in
Module/KeyframeSelector.py (`IKeyframeSelector` decides which frames enter the
map at all); a *keyframe policy* decides when the reference frame that every
new frame is additionally matched against (keyframe -> current optical flow,
see `MACVO._track_keyframe`) should move to the current frame.

Every criterion in the SLAM literature answers the same question - "has the
current frame drifted far enough from the keyframe that matching against it
is becoming unreliable?" - but each measures 'far' differently:

  EveryN         frames since the last keyframe (the trivial baseline)
  Parallax       median pixel displacement from keyframe to current frame
  Covisibility   fraction of the keyframe's keypoints still matchable
                 (the 'MVS score' / ORB-SLAM tracked-ratio criterion)
  BaselineRatio  translation since the keyframe divided by scene depth - the
                 triangulation-quality number: too small and the keyframe adds
                 no geometric information, too large and matching breaks
  AnyOf          switch when any wrapped policy fires

Each policy sees the same `TrackContext` and returns a bool, so the odometry
never needs to know which one it is holding.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace

from Utility.Extensions import ConfigTestableSubclass


@dataclass
class TrackContext:
    """Everything a policy may look at, measured for the frame just tracked."""
    frames_since_kf: int
    n_grid: int              # keypoint candidates the keyframe offers (its registered rows)
    n_kf_matches: int        # of those, how many survived kf -> cur
    parallax_px: float       # median |kp_cur - kp_kf| over surviving matches
    translation_m: float     # |t| of the estimated kf -> cur motion
    median_depth_m: float    # median depth of the surviving matches


class IKeyframePolicy(ABC, ConfigTestableSubclass):
    """Decide whether the keyframe should move to the frame just tracked."""
    # NOTE: `name()` is taken by SubclassRegistry (the config `type` key); the
    # human-readable policy label lives in `label`.
    label = "policy"

    def __init__(self, config: SimpleNamespace):
        self.config = config

    @abstractmethod
    def should_switch(self, ctx: TrackContext) -> bool: ...

    def __repr__(self) -> str:
        return self.label


def _positive_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _with_max_gap(config: SimpleNamespace | None, spec: dict) -> dict:
    if config is not None and hasattr(config, "max_gap"):
        spec["max_gap"] = lambda v: isinstance(v, int) and v >= 1
    return spec


class EveryN(IKeyframePolicy):
    """Switch every n frames. `first` shifts the whole schedule, which is the
    control needed to tell a real gain from a lucky keyframe placement."""

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.n: int = config.n
        first = getattr(config, "first", None)
        self.first: int = self.n if first is None else first
        self.switches = 0
        self.label = f"every{self.n}" + (f"@{first}" if first is not None else "")

    def should_switch(self, ctx: TrackContext) -> bool:
        limit = self.first if self.switches == 0 else self.n
        if ctx.frames_since_kf >= limit:
            self.switches += 1
            return True
        return False

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        spec: dict = {"n": lambda v: isinstance(v, int) and v >= 1}
        if config is not None and hasattr(config, "first"):
            spec["first"] = lambda v: isinstance(v, int) and v >= 1
        cls._enforce_config_spec(config, spec)


class Parallax(IKeyframePolicy):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.px: float = float(config.px)
        self.max_gap: int = getattr(config, "max_gap", 20)
        self.label = f"parallax{self.px:g}"

    def should_switch(self, ctx: TrackContext) -> bool:
        return ctx.parallax_px >= self.px or ctx.frames_since_kf >= self.max_gap

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, _with_max_gap(config, {"px": _positive_number}))


class Covisibility(IKeyframePolicy):
    """Switch when the keyframe's tracked fraction falls below `frac`."""

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.frac: float = float(config.frac)
        self.max_gap: int = getattr(config, "max_gap", 20)
        self.label = f"covis{self.frac:g}"

    def should_switch(self, ctx: TrackContext) -> bool:
        if ctx.n_grid == 0:
            return True
        tracked = ctx.n_kf_matches / ctx.n_grid
        return tracked < self.frac or ctx.frames_since_kf >= self.max_gap

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, _with_max_gap(config, {
            "frac": lambda v: isinstance(v, (int, float)) and 0. < v <= 1.
        }))


class BaselineRatio(IKeyframePolicy):
    """Switch when translation/depth exceeds `ratio` (triangulation quality)."""

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.ratio: float = float(config.ratio)
        self.max_gap: int = getattr(config, "max_gap", 20)
        self.label = f"baseline{self.ratio:g}"

    def should_switch(self, ctx: TrackContext) -> bool:
        if ctx.median_depth_m <= 0:
            return True
        return (ctx.translation_m / ctx.median_depth_m >= self.ratio
                or ctx.frames_since_kf >= self.max_gap)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, _with_max_gap(config, {"ratio": _positive_number}))


class AnyOf(IKeyframePolicy):
    """Switch when any wrapped policy fires. Every wrapped policy is evaluated
    (no short-circuit) so stateful policies such as EveryN keep their clocks."""

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.policies: list[IKeyframePolicy] = [
            IKeyframePolicy.instantiate(p.type, p.args) for p in config.policies
        ]
        self.label = "|".join(p.label for p in self.policies)

    def should_switch(self, ctx: TrackContext) -> bool:
        fired = [p.should_switch(ctx) for p in self.policies]
        return any(fired)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "policies": lambda ps: isinstance(ps, list) and len(ps) >= 1
        })
        for p in config.policies:
            IKeyframePolicy.is_valid_config(p)
