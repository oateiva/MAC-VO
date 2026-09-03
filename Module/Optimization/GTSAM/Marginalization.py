"""Bounded per-frame cost: fixed-lag marginalization for the persistent iSAM2 graph.

Port of learningUAVO's gtsam_backend/marginalization.py. ISAM2Optimizer.py never
removes anything: a track's death is just "not carried into `surviving`" — its
landmark variable and every factor on it stay in the Bayes tree forever, so the
per-frame cost grows with the SEQUENCE, not with the config (replayed on
plane_nose: br 70 -> 867 ms, p2p 186 -> 1692 ms median over 565 frames).

Marginalization ELIMINATES a variable older than the lag: its information
survives as a dense linear factor over its Markov blanket. The front-end tracks
are untouched (no reseeding churn); only variables nothing observes any more
actually leave. What it costs is that the linearization point behind the window
is frozen and can never be revisited. The falsifiable prediction that follows:
the CHAIN readout should be nearly unaffected (it only ever reads p_k and
p_(k-1)); a large lag moving the chain means the implementation is wrong, not
the method.

Three gtsam landmines (observed on 4.2a9, still present on 4.3a2) are handled
here and nowhere else:

  * gtsam.ISAM2 does not expose marginalizeLeaves at all in this build, so the
    real elimination has to come from IncrementalFixedLagSmoother (gtsam core on
    4.3, gtsam_unstable on 4.2).

  * IncrementalFixedLagSmoother.getISAM2() returns a COPY. `getISAM2().update()`
    does not raise -- it moves the smoother's estimate by exactly 0.0. Wiring
    the extra-update loop that way silently deletes every extra update, which
    reads as "marginalization costs accuracy" and is really a bug. The bare
    extra pass MUST go through the smoother with empty arguments (an empty
    timestamp map does not advance the current time and expires nothing).

  * the smoother exposes only the untyped calculateEstimate() -- there is no
    calculateEstimatePose3. A typed read therefore costs one Values of the whole
    live window, so FixedLagIsam2 caches the Values per update.

Retention is nothing but re-stamping: FixedLagSmootherKeyTimestampMap.insert
takes a TUPLE and re-inserting an existing key REFRESHES its timestamp. Keys
with stamp < (current time - lag) are marginalized on the next update; a key
that is never stamped never expires (a leak, not a crash).
"""
from typing import Optional

import gtsam

# gtsam 4.3 promoted the fixed-lag smoothers into core gtsam; 4.2 ships them
# only in gtsam_unstable. Probe the attribute (not the version string -- gtsam
# has no __version__) so one import line serves both.
if hasattr(gtsam, "IncrementalFixedLagSmoother"):
    from gtsam import FixedLagSmootherKeyTimestampMap, IncrementalFixedLagSmoother
else:  # pragma: no cover - gtsam 4.2 fallback
    from gtsam_unstable import FixedLagSmootherKeyTimestampMap   # type: ignore
    from gtsam_unstable import IncrementalFixedLagSmoother       # type: ignore


class MarginalizationFailure(RuntimeError):
    """The smoother could not eliminate an expiring variable.

    Its own type because the underlying IndexError names an innocent pose and
    reads like data corruption, which sends you looking in the wrong place.
    """


class FixedLagIsam2:
    """gtsam.ISAM2's surface, fixed-lag semantics — a drop-in for `tracker.isam`.

    update() takes ONE extra argument, the frame's timestamp map, and that is
    the entire behavioural difference. It returns a FixedLagSmootherResult,
    which has no getNewFactorsIndices(): GNC cannot run against this object,
    and ISAM2FlowTracker rejects that combination at construction. (The third
    positional argument of gtsam.ISAM2.update is removeFactorIndices, which
    means something else entirely — safe only because the two objects are never
    both live.)
    """

    def __init__(self, params: gtsam.ISAM2Params, lag: int):
        self.smoother = IncrementalFixedLagSmoother(float(lag), params)
        self.lag = int(lag)
        self._est: Optional[gtsam.Values] = None

    def update(self, graph=None, values=None, stamps=None):
        """Empty arguments give an extra GN pass without advancing the clock.

        They do not expire anything NEW -- the smoother's current time only
        moves when a stamp map moves it -- but every call still re-evaluates
        the expiry set, so a structurally borderline marginalization can
        surface on an extra pass rather than on the frame's own update.
        """
        self._est = None                       # every update invalidates
        try:
            return self.smoother.update(
                graph if graph is not None else gtsam.NonlinearFactorGraph(),
                values if values is not None else gtsam.Values(),
                stamps if stamps is not None
                else FixedLagSmootherKeyTimestampMap())
        except IndexError as e:
            raise MarginalizationFailure(
                f"marginalization failed at lag {self.lag}: {e} "
                f"The named variable is the one being eliminated, not the "
                f"cause. This is gtsam 4.2a9 failing to reduce an expiring "
                f"pose to a Bayes-tree leaf when too much landmark structure "
                f"straddles the window boundary. It is CONFIGURATION- rather "
                f"than purely lag-dependent, and the safe lag SHRINKS with "
                f"sequence length (learningUAVO, plane_nose 566 frames: lag 5 "
                f"survived, lag 10 died at frame 92, lag 20 at frame 54). "
                f"Shorten `marg_lag` first (5 is the measured-safe value), "
                f"then lower `extra_updates`; do not silence this.") from e

    def calculateEstimate(self) -> gtsam.Values:
        est = self._est
        if est is None:
            est = self.smoother.calculateEstimate()
            self._est = est
        return est

    def calculateEstimatePose3(self, key) -> gtsam.Pose3:
        return self.calculateEstimate().atPose3(key)

    def getFactorsUnsafe(self):
        raise NotImplementedError("GNC re-weighting is not available under marg_lag (see class docstring)")


def timestamp_map(stamps: dict) -> FixedLagSmootherKeyTimestampMap:
    """{key: t} -> the smoother's map. insert() takes a TUPLE in 4.2a9, and
    re-inserting an existing key REFRESHES its timestamp."""
    ts = FixedLagSmootherKeyTimestampMap()
    for key, t in stamps.items():
        ts.insert((int(key), float(t)))
    return ts
