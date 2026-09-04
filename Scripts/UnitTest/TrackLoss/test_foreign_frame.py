"""
Foreign-image degenerate case: a well-exposed, richly textured photograph that
has nothing to do with the sequence. Unlike a black frame it gives the matcher
plenty to latch onto, so this probes whether flow covariance - the only real
defence under this config, since the outlier filter is just
CovarianceSanityFilter - actually rejects confident-but-wrong matches.
"""
import pytest

from asset_trackloss import (
    MIN_NUM_POINT, build_system, foreign_like, report, requires_assets, run_pair,
    swap_image, teardown,
)


@pytest.mark.local
@requires_assets
def test_foreign_frame_is_flagged_lost(sequence, anchor_idx):
    system = build_system()
    try:
        frame0 = sequence[anchor_idx]
        frame1 = swap_image(frame0, foreign_like(frame0))

        r = run_pair(system, frame0, frame1, "foreign_turbot")
        report(r)
        assert r.need_interp, (
            f"MAC-VO accepted an unrelated photograph as a tracked pose "
            f"(n_obs={r.n_obs}, flow_cov_med={r.flow_cov_median}, "
            f"step_t={r.step_translation:.5f})"
        )
        assert r.n_obs < MIN_NUM_POINT
    finally:
        teardown(system)
