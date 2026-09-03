"""
Round-trip tests for VisualMap serialize/deserialize (Module/Map): field
recovery through real npz bytes, edge projections, and — the regression that
motivated the fix — that a deserialized map still accepts pushes with its
edges auto-growing (deserialize used to raise unconditionally, and would have
returned push-broken stores had it not).
"""
import io

import numpy as np
import pypose as pp
import pytest
import torch

from Module.Map import FrameNode, MatchObs, PointNode, VisualMap


def make_frame(i: int) -> FrameNode:
    return FrameNode.init({
        "pose"       : pp.identity_SE3(1).tensor(),
        "T_BS"       : pp.identity_SE3(1).tensor(),
        "need_interp": torch.tensor([i % 2 == 0]),
        "time_ns"    : torch.tensor([i], dtype=torch.long),
        "K"          : torch.eye(3).unsqueeze(0),
        "baseline"   : torch.tensor([0.1]),
    })


def make_matches(n: int, seed: int) -> MatchObs:
    gen = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.rand(shape, generator=gen)
    return MatchObs.init({
        "pixel1_uv": rand(n, 2) * 100, "pixel2_uv": rand(n, 2) * 100,
        "pixel1_d": rand(n, 1), "pixel2_d": rand(n, 1),
        "pixel1_disp": rand(n, 1), "pixel2_disp": rand(n, 1),
        "pixel1_disp_cov": rand(n, 1), "pixel2_disp_cov": rand(n, 1),
        "pixel1_uv_cov": rand(n, 3), "pixel2_uv_cov": rand(n, 3),
        "pixel1_d_cov": rand(n, 1), "pixel2_d_cov": rand(n, 1),
        "obs1_covTc": torch.eye(3).double().repeat(n, 1, 1),
        "obs2_covTc": torch.eye(3).double().repeat(n, 1, 1),
    })


def push_pair(vmap: VisualMap, prev_idx: torch.Tensor, n_kp: int, seed: int) -> torch.Tensor:
    """Mimic MACVO.run_pair's graph registration for one frame pair."""
    match_obs = make_matches(n_kp, seed)
    n_match_orig = len(vmap.match)
    point_idx = vmap.points.push(PointNode.init({
        "pos_Tw": torch.rand((n_kp, 3)),
        "cov_Tw": torch.eye(3).double().repeat(n_kp, 1, 1),
        "color" : torch.zeros((n_kp, 3), dtype=torch.uint8),
    }))
    frame_idx = vmap.frames.push(make_frame(int(prev_idx.item()) + 1))
    match_idx = vmap.match.push(match_obs)
    vmap.point2match.add(point_idx, match_idx)
    vmap.match2point.set(match_idx, point_idx)
    vmap.frame2match.add(prev_idx, torch.tensor([n_match_orig]), torch.tensor([n_kp]))
    vmap.frame2match.add(frame_idx, torch.tensor([n_match_orig]), torch.tensor([n_kp]))
    vmap.match2frame1.set(match_idx, torch.full((n_kp,), prev_idx.item(), dtype=torch.long))
    vmap.match2frame2.set(match_idx, torch.full((n_kp,), frame_idx.item(), dtype=torch.long))
    return frame_idx


def push_keyframe_rows(vmap: VisualMap, kf_idx: int, frame_idx: torch.Tensor, n_kp: int, seed: int) -> torch.Tensor:
    """Mimic MACVO._observe_keyframe: keyframe -> frame_idx rows pointing at the
    points born by the keyframe's own pair (the first n_kp points here)."""
    kf_obs = make_matches(n_kp, seed)
    n_orig = len(vmap.kf_match)
    kfm_idx = vmap.kf_match.push(kf_obs)
    vmap.kfmatch2point.set(kfm_idx, torch.arange(n_kp))
    vmap.kfmatch2frame1.set(kfm_idx, torch.full((n_kp,), kf_idx, dtype=torch.long))
    vmap.kfmatch2frame2.set(kfm_idx, torch.full((n_kp,), int(frame_idx.item()), dtype=torch.long))
    vmap.frame2kfmatch.add(frame_idx, torch.tensor([n_orig]), torch.tensor([n_kp]))
    return kfm_idx


def build_map(n_pairs: int = 3, n_kp: int = 4, keyframe_rows: bool = True) -> VisualMap:
    vmap = VisualMap()
    frame_idx = vmap.frames.push(make_frame(0))
    for p in range(n_pairs):
        frame_idx = push_pair(vmap, frame_idx, n_kp, seed=p)
        if keyframe_rows and p >= 1:
            push_keyframe_rows(vmap, 0, frame_idx, n_kp, seed=100 + p)
    return vmap


def roundtrip(vmap: VisualMap) -> VisualMap:
    """Through actual compressed npz bytes, as the Sandbox writes them."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **vmap.serialize())  # type: ignore[arg-type]  # str keys via **kwds
    buffer.seek(0)
    return VisualMap.deserialize(dict(np.load(buffer)))


def test_roundtrip_preserves_data():
    vmap = build_map()
    loaded = roundtrip(vmap)

    assert len(loaded.frames) == len(vmap.frames)
    assert len(loaded.match) == len(vmap.match)
    assert len(loaded.points) == len(vmap.points)
    assert len(loaded.kf_match) == len(vmap.kf_match) > 0
    raw = lambda t: t.tensor if hasattr(t, "tensor") else t
    for store in ("frames", "match", "points", "kf_match"):
        for key, tensor in getattr(vmap, store).data.items():
            expected, got = raw(tensor), raw(getattr(loaded, store).data[key])
            assert got.dtype == expected.dtype, f"{store}/{key} dtype"
            assert torch.equal(got, expected), f"{store}/{key}"


def test_roundtrip_preserves_edges():
    vmap = build_map()
    loaded = roundtrip(vmap)

    for k in range(len(vmap.frames)):
        idx = torch.tensor([k])
        original = vmap.get_frame2match(vmap.frames[idx])
        restored = loaded.get_frame2match(loaded.frames[idx])
        assert torch.equal(original.index, restored.index)
        assert torch.equal(
            vmap.get_match2point(original).index,
            loaded.get_match2point(restored).index)
    n_match = len(vmap.match)
    all_matches = torch.arange(n_match)
    assert torch.equal(vmap.match2frame2.project(all_matches),
                       loaded.match2frame2.project(all_matches))

    for k in range(len(vmap.frames)):
        idx = torch.tensor([k])
        original = vmap.get_frame2kfmatch(vmap.frames[idx])
        restored = loaded.get_frame2kfmatch(loaded.frames[idx])
        assert torch.equal(original.index, restored.index)
        assert torch.equal(vmap.get_kfmatch2point(original).index,
                           loaded.get_kfmatch2point(restored).index)
        assert torch.equal(vmap.get_kfmatch2frame1(original).index,
                           loaded.get_kfmatch2frame1(restored).index)


def test_deserialized_map_accepts_pushes():
    """Edges must be re-registered on the NEW stores, or pushes after loading
    silently stop auto-growing them."""
    loaded = roundtrip(build_map())
    last_frame = torch.tensor([len(loaded.frames) - 1])

    new_frame_idx = push_pair(loaded, last_frame, n_kp=4, seed=99)

    obs = loaded.get_frame2match(loaded.frames[new_frame_idx])
    assert len(obs) == 4
    assert len(loaded.get_match2point(obs)) == 4

    push_keyframe_rows(loaded, 0, new_frame_idx, n_kp=3, seed=7)
    kf_obs = loaded.get_frame2kfmatch(loaded.frames[new_frame_idx])
    assert len(kf_obs) == 3
    assert torch.equal(loaded.get_kfmatch2frame1(kf_obs).index, torch.zeros(3, dtype=torch.long))


def test_legacy_npz_without_keyframe_store_loads():
    """Maps written before kf_match existed carry no kf_match/ keys: they must
    load with an empty keyframe store that still accepts pushes."""
    vmap = build_map(keyframe_rows=False)
    legacy = {k: v for k, v in vmap.serialize().items()
              if not (k.startswith("kf_match/") or "kfmatch" in k)}
    loaded = VisualMap.deserialize(legacy)
    assert len(loaded.kf_match) == 0
    assert len(loaded.get_frame2kfmatch(loaded.frames[torch.tensor([1])])) == 0
    last = torch.tensor([len(loaded.frames) - 1])
    push_keyframe_rows(loaded, 0, last, n_kp=2, seed=3)
    assert len(loaded.get_frame2kfmatch(loaded.frames[last])) == 2


def test_deserialize_missing_prefix_raises():
    with pytest.raises(KeyError):
        VisualMap.deserialize({"unrelated/key": np.zeros(3)})
