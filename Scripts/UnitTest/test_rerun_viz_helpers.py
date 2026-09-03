"""
Tests for the pure (rerun/matplotlib-free) numeric helpers in
Utility/Visualize/Rerun_Visualize.py that back the ISAM2 debug visualizations:
`kf_link_segments` (Feature A: keyframe -> frame re-observation lines) and
`compute_track_ages` / `age_colors` (Feature B: landmark-age coloring).
"""
import numpy as np
import torch

from Utility.Visualize.Rerun_Visualize import (
    _KF_PALETTE,
    age_colors,
    compute_track_ages,
    filter_persistent_landmarks,
    kf_link_segments,
    landmark_color_array,
    update_landmark_colors,
    update_landmark_obs,
)


def make_positions(n: int) -> torch.Tensor:
    """Distinct, easily-identifiable (id, id, id) positions for id in [0, n)."""
    return torch.arange(n, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)


# --- kf_link_segments --------------------------------------------------------

def test_kf_link_segments_empty_input():
    positions = make_positions(5)
    segments, seg_colors, kf_pos, kf_colors = kf_link_segments(
        torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), positions
    )
    assert segments.shape == (0, 2, 3) and segments.dtype == np.float32
    assert seg_colors.shape == (0, 3) and seg_colors.dtype == np.uint8
    assert kf_pos.shape == (0, 3) and kf_pos.dtype == np.float32
    assert kf_colors.shape == (0, 3) and kf_colors.dtype == np.uint8


def test_kf_link_segments_duplicate_pair_collapse():
    """Repeated (kf, frame) rows collapse to a single segment."""
    positions = make_positions(10)
    kf_ids    = torch.tensor([1, 1, 1, 2])
    frame_ids = torch.tensor([5, 5, 5, 6])
    segments, seg_colors, kf_pos, kf_colors = kf_link_segments(kf_ids, frame_ids, positions)

    assert segments.shape == (2, 2, 3)
    assert seg_colors.shape == (2, 3)
    # one segment kf=1->frame=5, one kf=2->frame=6
    pairs = {(int(s[0, 0]), int(s[1, 0])) for s in segments}
    assert pairs == {(1, 5), (2, 6)}


def test_kf_link_segments_filters_out_of_range_and_negative_ids():
    positions = make_positions(5)   # valid ids: 0..4
    kf_ids    = torch.tensor([-1, 0, 0, 10])
    frame_ids = torch.tensor([2, 3, -1, 4])
    segments, seg_colors, kf_pos, kf_colors = kf_link_segments(kf_ids, frame_ids, positions)

    # only row index 1 (kf=0, frame=3) is fully valid
    assert segments.shape == (1, 2, 3)
    assert np.allclose(segments[0, 0], [0, 0, 0])
    assert np.allclose(segments[0, 1], [3, 3, 3])
    assert kf_pos.shape == (1, 3)
    assert np.allclose(kf_pos[0], [0, 0, 0])


def test_kf_link_segments_drops_self_pairs():
    positions = make_positions(5)
    kf_ids    = torch.tensor([2, 3])
    frame_ids = torch.tensor([2, 4])   # first row is a self-pair (kf == frame)
    segments, seg_colors, kf_pos, kf_colors = kf_link_segments(kf_ids, frame_ids, positions)

    assert segments.shape == (1, 2, 3)
    assert np.allclose(segments[0, 0], [3, 3, 3])
    assert np.allclose(segments[0, 1], [4, 4, 4])


def test_kf_link_segments_all_invalid_returns_empty():
    positions = make_positions(3)
    kf_ids    = torch.tensor([-1, 5])
    frame_ids = torch.tensor([0, 1])
    segments, seg_colors, kf_pos, kf_colors = kf_link_segments(kf_ids, frame_ids, positions)
    assert segments.shape == (0, 2, 3)
    assert kf_pos.shape == (0, 3)


def test_kf_link_segments_color_rank_stable_when_later_keyframe_appears():
    """Colors are assigned by rank among sorted unique keyframe ids. Since
    keyframe indices only grow over a run, adding a later (larger) keyframe id
    must not change the color already assigned to an earlier keyframe."""
    positions = make_positions(20)

    kf_ids_a    = torch.tensor([1, 1, 5, 5])
    frame_ids_a = torch.tensor([2, 3, 6, 7])
    _, seg_colors_a, kf_pos_a, kf_colors_a = kf_link_segments(kf_ids_a, frame_ids_a, positions)

    kf_ids_b    = torch.tensor([1, 1, 5, 5, 9, 9])
    frame_ids_b = torch.tensor([2, 3, 6, 7, 10, 11])
    _, seg_colors_b, kf_pos_b, kf_colors_b = kf_link_segments(kf_ids_b, frame_ids_b, positions)

    # kf=1 is rank 0 and kf=5 is rank 1 in both calls -> same colors
    # (kf_colors is aligned with kf_pos, one row per unique kf, sorted ascending).
    assert np.array_equal(kf_colors_a[0], kf_colors_b[0])   # kf=1 stays rank 0
    assert np.array_equal(kf_colors_a[1], kf_colors_b[1])   # kf=5 stays rank 1
    assert np.array_equal(kf_colors_a[0], _KF_PALETTE[0])
    assert np.array_equal(kf_colors_a[1], _KF_PALETTE[1])
    assert np.array_equal(kf_colors_b[2], _KF_PALETTE[2])   # kf=9 is the new rank-2 keyframe


def test_kf_link_segments_palette_wraps_mod_10():
    positions = make_positions(31)
    kf_ids    = torch.arange(11)         # 11 distinct keyframes -> rank 10 wraps to palette[0]
    frame_ids = kf_ids + 20
    _, _, kf_pos, kf_colors = kf_link_segments(kf_ids, frame_ids, positions)
    assert kf_colors.shape == (11, 3)
    assert np.array_equal(kf_colors[0], _KF_PALETTE[0])
    assert np.array_equal(kf_colors[10], _KF_PALETTE[0])   # rank 10 % 10 == 0


def test_kf_link_segments_drops_pair_with_nan_frame_endpoint():
    """A dense `pose_xyz`-style positions array can have NaN rows for graph
    frame indices whose pose key doesn't (yet/anymore) `exists()` in the
    estimate; any edge touching such a row must be dropped, not just
    NaN-propagated into the output."""
    positions = make_positions(5)
    positions[3] = float("nan")   # frame-end of the second pair goes NaN
    kf_ids    = torch.tensor([0, 1])
    frame_ids = torch.tensor([2, 3])
    segments, seg_colors, kf_pos, kf_colors = kf_link_segments(kf_ids, frame_ids, positions)

    assert segments.shape == (1, 2, 3)
    assert np.allclose(segments[0, 0], [0, 0, 0])
    assert np.allclose(segments[0, 1], [2, 2, 2])
    assert seg_colors.shape == (1, 3)


def test_kf_link_segments_drops_pair_with_nan_kf_endpoint():
    positions = make_positions(5)
    positions[1] = float("nan")   # kf-end of the first pair goes NaN
    kf_ids    = torch.tensor([1, 2])
    frame_ids = torch.tensor([3, 4])
    segments, seg_colors, kf_pos, kf_colors = kf_link_segments(kf_ids, frame_ids, positions)

    assert segments.shape == (1, 2, 3)
    assert np.allclose(segments[0, 0], [2, 2, 2])
    assert np.allclose(segments[0, 1], [4, 4, 4])
    # kf=1's own marker is also dropped (its position is NaN), kf=2's is kept
    assert kf_pos.shape == (1, 3)
    assert np.allclose(kf_pos[0], [2, 2, 2])


def test_kf_link_segments_all_nan_positions_returns_empty():
    positions = make_positions(5)
    positions[:] = float("nan")
    kf_ids    = torch.tensor([0, 1])
    frame_ids = torch.tensor([2, 3])
    segments, seg_colors, kf_pos, kf_colors = kf_link_segments(kf_ids, frame_ids, positions)

    assert segments.shape == (0, 2, 3)
    assert kf_pos.shape == (0, 3)


# --- compute_track_ages ------------------------------------------------------

def test_compute_track_ages_empty_input():
    ages = compute_track_ages(torch.empty((0, 2)), {})
    assert ages.shape == (0,)
    assert ages.dtype == np.int64


def test_compute_track_ages_missing_key_defaults_to_one():
    pixel2_uv = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    ages = compute_track_ages(pixel2_uv, {(10, 20): 7})
    assert ages.tolist() == [7, 1]


def test_compute_track_ages_uses_np_rint_half_even_rounding():
    """np.rint rounds .5 to the nearest EVEN integer, unlike Python's round()
    half-away/half-to-even ambiguity across versions - the tracker's own key
    convention (`np.rint(uv).astype(int)`) must be matched bit for bit."""
    pixel2_uv = torch.tensor([[2.5, 3.5], [0.5, 1.5]])
    # np.rint(2.5) == 2.0, np.rint(3.5) == 4.0, np.rint(0.5) == 0.0, np.rint(1.5) == 2.0
    n_obs_of = {(2, 4): 3, (0, 2): 9}
    ages = compute_track_ages(pixel2_uv, n_obs_of)
    assert ages.tolist() == [3, 9]


# --- age_colors ---------------------------------------------------------------

def test_age_colors_empty_input():
    colors = age_colors(np.zeros((0,), dtype=np.int64))
    assert colors.shape == (0, 3)
    assert colors.dtype == np.uint8


def test_age_colors_shape_and_dtype():
    ages = np.array([1, 5, 15, 30], dtype=np.int64)
    colors = age_colors(ages)
    assert colors.shape == (4, 3)
    assert colors.dtype == np.uint8


def test_age_colors_clamped_at_cap():
    """Ages above `cap` must clip to the same color as `cap` itself (Normalize
    clip=True), not wrap or extrapolate."""
    colors = age_colors(np.array([30, 1000], dtype=np.int64), cap=30)
    assert np.array_equal(colors[0], colors[1])


def test_age_colors_monotonic_with_age():
    """Higher age -> different color than the minimum age (plasma colormap is
    not constant), sanity-checking the Normalize(vmin=1, vmax=cap) mapping."""
    colors = age_colors(np.array([1, 15, 30], dtype=np.int64), cap=30)
    assert not np.array_equal(colors[0], colors[1])
    assert not np.array_equal(colors[1], colors[2])


# --- update_landmark_obs -----------------------------------------------------

def test_update_landmark_obs_max_merge_and_no_regress():
    """Higher counts win; a smaller count for an already-known landmark must
    not regress its stored n_obs (n_obs is monotonic per landmark within a run,
    but the merge is defensively a max, not an overwrite)."""
    lm_n_obs = {1: 5}
    result = update_landmark_obs(lm_n_obs, [(1, 7), (2, 3)])
    assert result == {1: 7, 2: 3}
    # now feed a smaller count for landmark 1 - must not regress
    result = update_landmark_obs(result, [(1, 4)])
    assert result[1] == 7


def test_update_landmark_obs_dead_landmark_retained_across_calls():
    """A landmark absent from a later `live` iterable (dropped from the
    tracker's live table) keeps its last known count rather than being
    dropped or reset."""
    lm_n_obs = update_landmark_obs({}, [(1, 3), (2, 5)])
    # landmark 2 goes "dead": absent from the next live update
    lm_n_obs = update_landmark_obs(lm_n_obs, [(1, 4)])
    assert lm_n_obs == {1: 4, 2: 5}


def test_update_landmark_obs_empty_live_iterable():
    lm_n_obs = update_landmark_obs({1: 3}, [])
    assert lm_n_obs == {1: 3}


# --- filter_persistent_landmarks ----------------------------------------------

def test_filter_persistent_landmarks_min_obs_boundary():
    """n_obs == min_obs is kept; n_obs == min_obs - 1 is dropped."""
    lm_n_obs = {1: 3, 2: 2}
    keys, counts = filter_persistent_landmarks(lm_n_obs, min_obs=3)
    assert keys == [1]
    assert counts.tolist() == [3]
    assert counts.dtype == np.int64


def test_filter_persistent_landmarks_sorted_keys_aligned_counts():
    lm_n_obs = {5: 10, 1: 4, 3: 7}
    keys, counts = filter_persistent_landmarks(lm_n_obs, min_obs=0)
    assert keys == [1, 3, 5]
    assert counts.tolist() == [4, 7, 10]
    assert counts.dtype == np.int64


def test_filter_persistent_landmarks_empty_dict():
    keys, counts = filter_persistent_landmarks({}, min_obs=3)
    assert keys == []
    assert counts.shape == (0,)
    assert counts.dtype == np.int64


# --- update_landmark_colors --------------------------------------------------

def test_update_landmark_colors_set_if_absent_does_not_overwrite():
    """A landmark's color is fixed on first sighting; a later sighting with a
    different RGB (e.g. lighting/exposure drift) must not overwrite it."""
    pixel2_uv = torch.tensor([[10.0, 20.0]])
    colors = torch.tensor([[1, 2, 3]], dtype=torch.uint8)
    lm_key_of = {(10, 20): 42}
    lm_color = update_landmark_colors({}, pixel2_uv, colors, lm_key_of)
    assert lm_color == {42: (1, 2, 3)}

    # same landmark, different pixel row/color on a later frame
    pixel2_uv2 = torch.tensor([[11.0, 21.0]])
    colors2 = torch.tensor([[9, 9, 9]], dtype=torch.uint8)
    lm_key_of2 = {(11, 21): 42}
    lm_color = update_landmark_colors(lm_color, pixel2_uv2, colors2, lm_key_of2)
    assert lm_color == {42: (1, 2, 3)}


def test_update_landmark_colors_uses_np_rint_half_even_rounding():
    """Must join via the same np.rint half-to-even convention as
    `compute_track_ages` - not Python's round()."""
    pixel2_uv = torch.tensor([[2.5, 3.5], [0.5, 1.5]])
    # np.rint(2.5) == 2.0, np.rint(3.5) == 4.0, np.rint(0.5) == 0.0, np.rint(1.5) == 2.0
    colors = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.uint8)
    lm_key_of = {(2, 4): 1, (0, 2): 2}
    lm_color = update_landmark_colors({}, pixel2_uv, colors, lm_key_of)
    assert lm_color == {1: (10, 20, 30), 2: (40, 50, 60)}


def test_update_landmark_colors_skips_rows_with_no_matching_track():
    pixel2_uv = torch.tensor([[10.0, 20.0], [99.0, 99.0]])
    colors = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8)
    lm_key_of = {(10, 20): 7}   # no entry for (99, 99)
    lm_color = update_landmark_colors({}, pixel2_uv, colors, lm_key_of)
    assert lm_color == {7: (1, 2, 3)}


def test_update_landmark_colors_empty_input_is_noop():
    lm_color = update_landmark_colors({1: (1, 1, 1)}, torch.empty((0, 2)), torch.empty((0, 3)), {})
    assert lm_color == {1: (1, 1, 1)}


# --- landmark_color_array ------------------------------------------------------

def test_landmark_color_array_ordering_and_fallback():
    lm_color = {1: (10, 20, 30), 3: (40, 50, 60)}
    colors = landmark_color_array([1, 2, 3], lm_color, fallback=(128, 128, 128))
    assert colors.shape == (3, 3)
    assert colors.dtype == np.uint8
    assert colors.tolist() == [[10, 20, 30], [128, 128, 128], [40, 50, 60]]


def test_landmark_color_array_empty_input():
    colors = landmark_color_array([], {1: (1, 2, 3)})
    assert colors.shape == (0, 3)
    assert colors.dtype == np.uint8


# --- Rerun_Visualizer import smoke (module still imports cleanly) -----------

def test_module_imports_cleanly():
    from Utility.Visualize.Rerun_Visualize import Rerun_Visualizer
    assert hasattr(Rerun_Visualizer, "log_kf_links")
    assert hasattr(Rerun_Visualizer, "log_scalar")
    assert hasattr(Rerun_Visualizer, "log_keypoints")
