"""
Baseline + pure-black degenerate case.

The consecutive-pair test here is the positive control for the whole directory:
without it, a suite in which every case reports "lost" would prove nothing.
"""
import pytest

from asset_trackloss import (
    MIN_NUM_POINT, black_like, build_system, report, requires_assets, run_pair,
    swap_image, teardown,
)


@pytest.mark.local
@requires_assets
def test_consecutive_pair_is_not_flagged(sequence, anchor_idx):
    """A genuine consecutive pair must track cleanly - the positive control."""
    system = build_system()
    try:
        r = run_pair(
            system, sequence[anchor_idx], sequence[anchor_idx + 1], "consecutive_pair",
        )
        report(r)
        assert not r.need_interp, (
            f"a real consecutive pair was flagged as lost track ({r.branch}, "
            f"n_obs={r.n_obs}); the degenerate cases below prove nothing if this fails"
        )
        assert r.n_obs >= MIN_NUM_POINT
    finally:
        teardown(system)


@pytest.mark.local
@requires_assets
def test_black_frame_is_flagged_lost(sequence, anchor_idx):
    """A pure black second frame carries no structure at all."""
    system = build_system()
    try:
        frame0 = sequence[anchor_idx]
        frame1 = swap_image(frame0, black_like(frame0))

        r = run_pair(system, frame0, frame1, "black_frame")
        report(r)
        assert r.need_interp, (
            f"MAC-VO accepted a pure black frame as a tracked pose "
            f"(n_obs={r.n_obs}, step_t={r.step_translation:.5f})"
        )
        assert r.n_obs < MIN_NUM_POINT
    finally:
        teardown(system)
