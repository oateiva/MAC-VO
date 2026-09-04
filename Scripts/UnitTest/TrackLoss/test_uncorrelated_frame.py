"""
The hardest case, and the one closest to a real failure: both frames are
genuine plane_nose seafloor imagery from the same camera, but 400 frames apart,
so they share no view. Nothing about the pair looks synthetic - if MAC-VO is
going to emit a confident pose for an untracked frame, it will be here.
"""
import pytest

from asset_trackloss import (
    GAP, MIN_NUM_POINT, build_system, report, requires_assets, run_pair, teardown,
)


@pytest.mark.local
@requires_assets
def test_distant_real_frames_flagged_lost(sequence, anchor_idx):
    system = build_system()
    try:
        r = run_pair(
            system, sequence[anchor_idx], sequence[anchor_idx + GAP], "distant_real_frames",
        )
        report(r)
        assert r.need_interp, (
            f"MAC-VO accepted two unrelated real frames {GAP} apart as a tracked "
            f"pose (n_obs={r.n_obs}, flow_cov_med={r.flow_cov_median}, "
            f"step_t={r.step_translation:.5f})"
        )
        assert r.n_obs < MIN_NUM_POINT
    finally:
        teardown(system)
