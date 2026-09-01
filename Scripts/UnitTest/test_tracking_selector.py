"""
Tests for TrackingCovAwareSelector (Module/KeypointSelector.py): cov-aware
seeding (two-region variance map on both axes, border exclusion, threshold
rule, spacing, seed cap, determinism), the bit-exact flow-carry mirror of
MACVO.run_pair, track gating across calls, and the no-covariance fallback.
"""
from types import SimpleNamespace

import pytest
import torch

from Module.Frontend.StereoDepth import IDepth
from Module.Frontend.Matching import IMatcher
from Module.KeypointSelector import TrackingCovAwareSelector, spaced_greedy
from Utility.Utils import retrieve_scalar_map_pixels

H, W = 80, 100
BORDER = 16


def dither() -> torch.Tensor:
    """Deterministic sub-1e-2 texture so NMS local minima are unique."""
    u = torch.arange(W).view(1, -1)
    v = torch.arange(H).view(-1, 1)
    return 1e-3 * ((u * 7 + v * 13) % 11).float()


def make_depth(value: float = 3.0) -> IDepth.Output:
    return IDepth.Output(depth=torch.full((1, 1, H, W), value))


def make_match(quality: torch.Tensor, batch: int = 1,
               flow_u: float = 0.0, flow_v: float = 0.0) -> IMatcher.Output:
    """IMatcher.Output whose batch-0 quality map (uu + vv - 2uv) equals `quality`.

    With batch=2, batch 1 carries the SPATIALLY FLIPPED map: a selector reading
    the wrong batch slot then fails the region assertions.
    """
    cov = torch.zeros((batch, 3, H, W))
    cov[0, 0] = quality * 0.5
    cov[0, 1] = quality * 0.5
    if batch > 1:
        cov[1:, 0] = torch.flip(quality, dims=[-1]) * 0.5
        cov[1:, 1] = torch.flip(quality, dims=[-1]) * 0.5
    flow = torch.zeros((batch, 2, H, W))
    flow[:, 0] = flow_u
    flow[:, 1] = flow_v
    return IMatcher.Output(flow=flow, cov=cov)


def make_cfg(**overrides) -> SimpleNamespace:
    cfg = SimpleNamespace(device="cpu", mask_width=BORDER, kernel_size=5,
                          max_match_cov=10.0, median_rel=1.5, seed_radius=4.0)
    cfg.__dict__.update(overrides)
    return cfg


def make_selector(**overrides) -> TrackingCovAwareSelector:
    cfg = make_cfg(**overrides)
    TrackingCovAwareSelector.is_valid_config(cfg)
    return TrackingCovAwareSelector(cfg)


@pytest.mark.parametrize("match_batch", [1, 2])
@pytest.mark.parametrize("axis", ["u", "v"])
def test_two_region_selection(axis: str, match_batch: int):
    """Only the low-variance half may be seeded; run mirrored on both axes since
    the [v, u] indexing vs (u, v) output order is the likeliest bug here."""
    half = W // 2 if axis == "u" else H // 2
    base = (torch.where(torch.arange(W).view(1, -1) < half, 0.05, 500.0) if axis == "u"
            else torch.where(torch.arange(H).view(-1, 1) < half, 0.05, 500.0))
    quality = base.expand(H, W) + dither()
    depth = make_depth()
    out = make_selector().select_point(
        None, 200, depth, depth, make_match(quality, batch=match_batch))  # type: ignore[arg-type]

    assert out.shape[0] > 0
    assert bool((out[:, 0 if axis == "u" else 1] < half).all())


def test_border_exclusion():
    """The global minimum sits inside the border and must not appear; nothing may
    come back within `mask_width` of any edge."""
    quality = 100.0 + dither()
    quality[3, 3] = 0.001
    quality[24, 30] = 0.002
    depth = make_depth()
    out = make_selector(max_match_cov=1e9).select_point(
        None, 200, depth, depth, make_match(quality))  # type: ignore[arg-type]

    assert out.shape[0] > 0
    assert not bool((out[:, 0] == 3).any())
    assert int(out[:, 0].min()) >= BORDER and int(out[:, 1].min()) >= BORDER
    assert int(out[:, 0].max()) <= W - 1 - BORDER and int(out[:, 1].max()) <= H - 1 - BORDER


def test_threshold_cap_and_order():
    """Absolute threshold admits only the planted dips; the seed cap keeps the
    lowest-Q rows, best-Q-first."""
    quality = 100.0 + dither()
    dips = [(20, 20), (30, 24), (40, 28), (44, 32)]
    for i, (u, v) in enumerate(dips):
        quality[v, u] = 0.01 * (i + 1)          # ascending: (20, 20) is the best
    depth = make_depth()
    out = make_selector(seed_radius=3.0, max_match_cov=1.0).select_point(
        None, 2, depth, depth, make_match(quality))  # type: ignore[arg-type]

    assert torch.equal(out, torch.tensor([[20, 20], [30, 24]], dtype=out.dtype))


def test_spaced_greedy_matches_naive_oracle():
    gen = torch.Generator().manual_seed(0)
    pts = torch.rand((200, 2), generator=gen) * 100
    live = torch.rand((25, 2), generator=gen) * 100

    fast = spaced_greedy(pts, live, 7.0)

    naive: list[torch.Tensor] = []
    for p in pts.double():
        if bool((((live.double() - p) ** 2).sum(-1) <= 49.0).any()):
            continue
        if naive and bool((((torch.stack(naive) - p) ** 2).sum(-1) <= 49.0).any()):
            continue
        naive.append(p)
    assert torch.allclose(fast.double(), torch.stack(naive))

    dips = torch.tensor([[20.0, 20.0], [40.0, 20.0], [60.0, 20.0], [20.0, 40.0]])
    assert spaced_greedy(dips, torch.empty((0, 2)), 7.0, cap=2).shape[0] == 2
    blocked = spaced_greedy(dips, dips[:1], 7.0)
    assert blocked.shape[0] == 3 and not bool((blocked == dips[0]).all(dim=1).any())
    assert spaced_greedy(dips, dips[:1] + torch.tensor([7.001, 0.0]), 7.0).shape[0] == 4


def test_determinism():
    quality = 100.0 + dither()
    quality[24, 30] = 0.01
    quality[40, 60] = 0.02
    depth = make_depth()
    match = make_match(quality)
    out_a = make_selector(max_match_cov=1.0).select_point(None, 50, depth, depth, match)  # type: ignore[arg-type]
    out_b = make_selector(max_match_cov=1.0).select_point(None, 50, depth, depth, match)  # type: ignore[arg-type]
    assert torch.equal(out_a, out_b)


def test_carry_mirrors_run_pair():
    """The stored track positions must equal `kp0 + retrieve_pixels(kp0, flow).T`
    bit for bit — the invariant the downstream integer association relies on."""
    quality = 100.0 + dither()
    quality[24, 30] = 0.01
    quality[40, 60] = 0.02
    depth = make_depth()
    match = make_match(quality, flow_u=5.25, flow_v=-2.5)
    selector = make_selector(max_match_cov=1.0)
    kp0 = selector.select_point(None, 50, depth, depth, match)  # type: ignore[arg-type]

    flow_at_kp = retrieve_scalar_map_pixels(kp0, match.flow)
    assert flow_at_kp is not None
    assert selector.track_uv is not None
    assert torch.equal(selector.track_uv, kp0.float() + flow_at_kp.T)


@pytest.mark.parametrize("match_batch", [1, 2])
def test_tracks_carried_across_calls(match_batch: int):
    """A seeded keypoint reappears flow-shifted in the next call's output, and no
    new seed lands within seed_radius of it."""
    quality = 100.0 + dither()
    quality[24, 30] = 0.01
    depth = make_depth()
    selector = make_selector(max_match_cov=1.0, seed_radius=8.0)

    first = selector.select_point(
        None, 50, depth, depth, make_match(quality, batch=match_batch, flow_u=5.0))  # type: ignore[arg-type]
    assert first.shape[0] == 1 and torch.equal(first[0], torch.tensor([30, 24], dtype=first.dtype))

    second = selector.select_point(
        None, 50, depth, depth, make_match(quality, batch=match_batch))  # type: ignore[arg-type]
    carried = second[0]
    assert torch.equal(carried, torch.tensor([35, 24], dtype=second.dtype))
    if second.shape[0] > 1:
        dist2 = ((second[1:].float() - carried.float()) ** 2).sum(dim=-1)
        assert bool((dist2 > 8.0 ** 2).all())


def test_track_killed_by_border_and_depth():
    quality = 100.0 + dither()
    quality[24, 30] = 0.01
    selector = make_selector(max_match_cov=1.0, seed_radius=8.0)

    # Flow pushes the only track outside the border: it must not reappear.
    selector.select_point(None, 50, make_depth(), make_depth(),  # type: ignore[arg-type]
                          make_match(quality, flow_u=float(W)))
    out = selector.select_point(None, 50, make_depth(), make_depth(), make_match(quality))  # type: ignore[arg-type]
    assert not bool((out[:, 0] > W - 1 - BORDER).any())

    # Invalid depth at the carried position kills the track.
    selector = make_selector(max_match_cov=1.0, seed_radius=8.0)
    selector.select_point(None, 50, make_depth(), make_depth(), make_match(quality, flow_u=5.0))  # type: ignore[arg-type]
    depth_hole = make_depth()
    depth_hole.depth[0, 0, 24, 35] = torch.nan
    out = selector.select_point(None, 50, depth_hole, depth_hole, make_match(quality))  # type: ignore[arg-type]
    assert not bool(((out[:, 0] == 35) & (out[:, 1] == 24)).any())


def test_fallback_without_covariance():
    """No covariance map -> grid seeding, still spaced against live tracks."""
    quality = 100.0 + dither()
    quality[24, 30] = 0.01
    depth = make_depth()
    selector = make_selector(max_match_cov=1.0, seed_radius=8.0)
    selector.select_point(None, 50, depth, depth, make_match(quality, flow_u=5.0))  # type: ignore[arg-type]

    frame_stub = SimpleNamespace(height=H, width=W)      # GridSelector fallback reads only these
    out = selector.select_point(frame_stub, 50, depth, depth, IMatcher.Output(flow=torch.zeros((1, 2, H, W))))  # type: ignore[arg-type]
    assert out.shape[0] > 1                          # carried track + grid seeds
    assert torch.equal(out[0], torch.tensor([35, 24], dtype=out.dtype))
    dist2 = ((out[1:].float() - out[0].float()) ** 2).sum(dim=-1)
    assert bool((dist2 > 8.0 ** 2).all())


def test_config_validation():
    TrackingCovAwareSelector.is_valid_config(make_cfg())
    with pytest.raises(KeyError):
        TrackingCovAwareSelector.is_valid_config(make_cfg(unexpected_knob=1))
    with pytest.raises(KeyError):
        incomplete = make_cfg()
        del incomplete.__dict__["seed_radius"]
        TrackingCovAwareSelector.is_valid_config(incomplete)
    with pytest.raises(ValueError):
        TrackingCovAwareSelector.is_valid_config(make_cfg(kernel_size=4))
