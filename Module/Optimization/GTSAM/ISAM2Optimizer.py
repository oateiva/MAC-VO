"""Persistent-graph iSAM2 backend: one gtsam.ISAM2 over the whole sequence.

Port of learningUAVO's gtsam_backend/isam2_tracker.py into MAC-VO's IOptimizer
contract. Every other backend in this repo solves a fresh window per frame pair
and chains the relative poses; here a single factor graph persists — a pose key
p_k per frame and landmark keys l_i that live as long as their track — and
iSAM2 updates the solution incrementally as each pair's factors arrive.

Landmarks carry no descriptors — identity is POSITIONAL, established by exact
integer pixel association with the stateful `TrackingCovAwareSelector`
(Module/KeypointSelector.py): the selector carries each keypoint by the optical
flow with a bit-for-bit mirror of `MACVO.run_pair`'s kp1 computation, so the new
pair's integer `pixel1_uv` equals `round(pixel2_uv)` of the previous pair for
every surviving track. A row whose pixel1 matches a stored track continues that
track (its pixel1 observation is SKIPPED — it re-quantizes the previous pair's
pixel2 observation and would double-count); an unmatched row births a new
landmark, whose pixel1 fields ARE its first (quantization-variance) observation.
Flow variance accumulates along a track (the position error after n steps is the
sum of n flow errors); `accumulate_fvar: false` reverts to per-step variance.

Robustness is either a per-factor M-estimator kernel (`kernel`, Huber by
default) or GNC-GM (`gnc_rounds` > 0, Yang et al. 2019), never both: GNC
REPLACES the kernel. The GNC annealing loop runs incrementally once per frame —
solve, take residuals, recompute the Geman-McClure weights
w = (mu c^2 / (r^2 + mu c^2))^2, then REPLACE this frame's point factors via
isam.update's removeFactorIndices, annealing mu toward 1. GNC runs under Dogleg
(`gnc_damped`, on by default): measured in learningUAVO, undamped GN plus GNC
down-weighting diverges to float64 overflow (surfacing as an
IndeterminantLinearSystemException naming whichever pose the NaN reached first);
a weight floor and a divergence rollback were both measured INSUFFICIENT alone.

`marg_lag` N > 0 (default 0 = off) swaps gtsam.ISAM2 for the fixed-lag smoother
in Marginalization.py: poses older than N frames and landmarks whose track died N
frames ago are ELIMINATED from the live graph, their information surviving as
dense marginal factors, so per-frame cost is bounded instead of growing with the
sequence. Retention is re-stamping: every frame stamps p_k, whatever was inserted
this frame, and every surviving track's landmark. The linearization point behind
the window is frozen; the chain readout (which only reads p_k and p_last) should
be nearly unaffected. Incompatible with GNC (the smoother's result carries no
factor indices). learningUAVO's ledger: lag 5 was the only lag that survived the
566-frame plane_nose; longer lags hit a MarginalizationFailure that is never
silenced — shorten the lag instead.

The failure was traced (backend replay of plane_nose, 2026-09-02) to one Bayes-tree
shape: the expiring pose is the TOP frontal of a clique that this update does not
re-eliminate (no new factor on its frontals, no child clique naming the pose in
its separator), sitting above frontals that stay — dead-but-unexpired landmarks,
or neighbouring poses merged into the same clique. The smoother then cuts the
frontal out in place and gtsam 4.2a9 leaves the kept frontals conditioned on the
removed key; the next solve raises "Requested variable 'pNN' is not in this
VectorValues". Whenever the clique IS re-eliminated, the smoother's constrained
ordering puts expiring keys at the bottom and the cut is clean (1100+ such
expiries observed, none failed). Two mitigations, both flags:
  marg_touch_sigma (default 1e4; 0 = off)  adds a prior of that sigma on every key
      about to expire, so its clique is always re-eliminated first. 1e4 m carries
      1e-8 of precision: readout unchanged to 2 mm at lag 5, and lags 10 and 20
      complete the 565 frames that died at 314 / 37 without it.
  marg_dead_at_birth (default false)  re-stamps a landmark to its BIRTH frame the
      frame its track dies, so it leaves with its first observer and never lingers
      above a later pose; also halves the largest separators (54 -> 21 at lag 20).

Keyframe rows (`ISAM2_GraphInput.kf`, filled from `VisualMap.kf_match` by the
odometry's keyframe tracker, Module/KeyframeTracker.py): the keyframe's pixel1
rows of pair (kf, kf+1) re-observed in frame k through a SECOND flow inference
kf -> k. Each row is associated to the landmark the same integer pixel resolved
to when pair (kf, kf+1) was stepped (`frame_lm[kf]`, a per-frame snapshot of the
pixel -> landmark table) and adds ONE more factor p_k -> l on that SAME key, with
variance quantization + one flow step (no accumulation: that is the point). The
chain observation of the same landmark at p_k is kept as well (an additional
connection, as designed); the two share the frame-k depth sample, so
`kf_cov_scale` (default 1) exists to inflate the keyframe covariance. Under
marg_lag the keyframe pose and every landmark it re-observes are re-stamped each
frame so they never expire while the keyframe is live; a landmark already
marginalized (chain-dead for marg_lag frames, or at once under
marg_dead_at_birth) is skipped, so keyframe rows then only reach chain-alive
tracks.

`final_lm` (default false) is the port of isam2_tracker.finalize(lm_polish=True)
and run_isam2.py's `--final-lm`: ONE offline batch Levenberg-Marquardt solve over
the ENTIRE accumulated graph from the iSAM2 estimate, after the last frame
(IOptimizer.finalize, called by the odometry's terminate() before poses.npy is
written). Every pose key's LM result replaces the online readout in the map, so
the saved trajectory is the smoothed one (the graph's own gauge, anchored by the
first pose's prior — not the chain's). LM's damping reaches the deeper robust
basin that the per-frame Gauss-Newton passes miss, and recovers the smoothed
accuracy when the online settings were deliberately lazy (large
relin_threshold, few extra_updates). The graph it solves is the SAME
measurement model the online solver ran (factor_type, kernel, GNC weights as
frozen): without marg_lag it is gtsam.ISAM2's own factor store
(getFactorsUnsafe, GNC-replaced slots included); under marg_lag the smoother has
eliminated old variables, so the tracker keeps a shadow NonlinearFactorGraph of
every nonlinear factor it ever added (touch priors excluded — they are
scaffolding, not measurements) and snapshots each key's last live estimate the
frame it expires, and LM runs over shadow + (live ∪ frozen) values. Cost is one
batch solve of the whole sequence — `final_lm_max_iters` (default 100) bounds it.

Deliberately NOT ported from isam2_tracker.py (see learningUAVO FINDINGS.md):
the gp depth-prior mode; obs_stride / obs_phase / add_budget (the winning online
arms run stride 1 / budget 0, and deferred insertion crashes the online
readout); max_age and max_step_cov (both measured harmful — track length is
monotonically good); landmark write-back into the map (the chain readout's gauge
and the live graph's gauge diverge after low-support stretches — map points stay
as the odometry registered them). One measured deviation: the observation depth
is the nearest-sampled `pixel2_d` (this repo's house convention), not the
kernel-weighted depth of learningUAVO's compose_observation.
"""
import time
import numpy as np
import torch
import gtsam
import pypose as pp

try:
    import gtsam_unstable
    # Wrapped only in wheels carrying MAC-VO's PoseToPointFactor wrapper patch
    # (Scripts/patches/gtsam-posetopoint-wrapper.patch); stock gtsam wheels
    # ship gtsam_unstable without it, hence getattr rather than an attribute
    # reference (keeps pyright quiet against unpatched stubs too).
    _NATIVE_P2P = getattr(gtsam_unstable, "PoseToPointFactorPose3Point3", None)
except ImportError:
    _NATIVE_P2P = None
from dataclasses import dataclass
from types import SimpleNamespace

from Module.Map import VisualMap
from Utility.GTSAM_Utils import make_pose_to_point_factor
from Utility.PrettyPrint import Logger
from Utility.Timer import Timer

from ..Interface import IOptimizer

_KERNELS = ("huber", "cauchy", "geman", "tukey", "welsch", "none")


def _check_marg_lag(marg_lag: int, gnc_rounds: int) -> None:
    """Cross-key rules for fixed-lag marginalization (see module docstring)."""
    if marg_lag == 0:
        return
    if marg_lag < 2:
        raise ValueError(f"marg_lag {marg_lag}: the readout reads p_(last) one frame "
                         f"after its own update, so the window must hold >= 2 poses")
    if gnc_rounds > 0:
        raise ValueError("marg_lag needs gnc_rounds 0: the fixed-lag smoother's result "
                         "has no getNewFactorsIndices(), which GNC re-weighting needs")


@dataclass
class ISAM2_KeyframeRows:
    """Keyframe -> frame_idx re-observations, straight from VisualMap.kf_match."""
    kf_idx       : int
    pixel1_uv    : torch.Tensor     # (M,2) integer-valued keyframe pixels (pixel1 of pair kf -> kf+1)
    pixel2_uv    : torch.Tensor     # (M,2) subpixel, flow-carried into frame_idx
    pixel2_d     : torch.Tensor     # (M,)
    pixel2_d_cov : torch.Tensor     # (M,)  -1 = unavailable
    pixel2_uv_cov: torch.Tensor     # (M,3) (sigma_uu, sigma_vv, sigma_uv) of the kf -> frame_idx flow


@dataclass
class ISAM2_GraphInput:
    frame_idx: int
    from_idx : int
    from_pose: torch.Tensor         # (7,)  SE3 pose of frame `from_idx` (map estimate)
    K        : torch.Tensor         # (3,3) camera intrinsics
    # Per-observation rows of the (from_idx -> frame_idx) pair, straight from MatchObs.
    pixel1_uv    : torch.Tensor     # (N,2) float32, integer-valued (selector output)
    pixel2_uv    : torch.Tensor     # (N,2) float32, subpixel (flow-carried)
    pixel1_d     : torch.Tensor     # (N,)
    pixel2_d     : torch.Tensor     # (N,)
    pixel1_d_cov : torch.Tensor     # (N,)  -1 = unavailable
    pixel2_d_cov : torch.Tensor     # (N,)  -1 = unavailable
    pixel1_uv_cov: torch.Tensor     # (N,3) (sigma_uu, sigma_vv, sigma_uv), -1 rows = unavailable
    pixel2_uv_cov: torch.Tensor     # (N,3)
    kf           : ISAM2_KeyframeRows | None = None


@dataclass
class ISAM2_GraphOutput:
    frame_idx    : int
    pose_estimate: torch.Tensor     # (7,) SE3, float32


@dataclass
class _TrackState:
    lm_key: int                     # gtsam.symbol("l", id) — never reused
    fvar  : np.ndarray              # (2,) accumulated flow variance (see module docstring)
    n_obs : int
    born  : int                     # frame index of the pose holding its first observation


def gnc_weights(r2: np.ndarray, mu: float, c: float, floor: float) -> np.ndarray:
    """Geman-McClure weights for one GNC round: w = (mu c^2 / (r^2 + mu c^2))^2,
    clipped from below at `floor`.

    `floor` bounds the damage a rejected factor can do: a factor rebuilt at
    cov/floor carries 1/floor times less information. 1e-4 is GM's own weight at
    r = 10c, so the clip only binds beyond ~10 inlier scales — the decades below
    that distinguished "outlier" from "worse outlier" at no benefit and produced
    silently-wrong solves in learningUAVO's probes.
    """
    mc2 = float(mu) * float(c) ** 2
    return np.maximum((mc2 / (np.asarray(r2, dtype=np.float64) + mc2)) ** 2, float(floor))


def robustify(base, kernel: str, delta: float):
    """Wrap a gtsam noise model in an M-estimator, or return it unchanged for 'none'."""
    match kernel:
        case "none"  : return base
        case "huber" : m = gtsam.noiseModel.mEstimator.Huber.Create(delta)
        case "cauchy": m = gtsam.noiseModel.mEstimator.Cauchy.Create(delta)
        case "geman" : m = gtsam.noiseModel.mEstimator.GemanMcClure.Create(delta)
        case "tukey" : m = gtsam.noiseModel.mEstimator.Tukey.Create(delta)
        case "welsch": m = gtsam.noiseModel.mEstimator.Welsch.Create(delta)
        case _: raise ValueError(f"unknown kernel '{kernel}', expected one of {_KERNELS}")
    return gtsam.noiseModel.Robust.Create(m, base)


def make_native_point_factor(pose_key: int, landmark_key: int, obs_Tc: np.ndarray,
                             cov: np.ndarray, kernel: str, delta: float):
    """BearingRangeFactor3D carrying the same information as the pose-to-point
    CustomFactor.

    (bearing, range) is a bijective reparametrization of the 3D observation; the
    3x3 covariance is propagated through the exact Jacobian at the measurement,
    J = [B^T (I - bb^T)/r ; b^T] with B the Unit3 tangent basis, so the
    Mahalanobis metric matches the 3D factor's to first order. The factor
    relinearizes in C++ — the Python callback is the dominant per-frame cost of
    a persistent graph.
    """
    m = np.asarray(obs_Tc, dtype=np.float64)
    r = float(np.linalg.norm(m))
    bearing = gtsam.Unit3(m)
    b = m / r
    J = np.vstack([bearing.basis().T @ (np.eye(3) - np.outer(b, b)) / r, b])
    cov_br = J @ cov @ J.T
    cov_br = 0.5 * (cov_br + cov_br.T)
    noise = robustify(gtsam.noiseModel.Gaussian.Covariance(cov_br), kernel, delta)
    return gtsam.BearingRangeFactor3D(pose_key, landmark_key, bearing, r, noise)


def require_native_p2p():
    """The wrapped C++ PoseToPointFactor class, or a hard error telling how to get it."""
    if _NATIVE_P2P is None:
        raise RuntimeError(
            "factor_type 'pose2point_native' needs gtsam_unstable.PoseToPointFactorPose3Point3, "
            "which stock gtsam wheels do not wrap. Build gtsam with "
            "Scripts/build_gtsam_windows.ps1 -PatchFile Scripts/patches/gtsam-posetopoint-wrapper.patch"
        )
    return _NATIVE_P2P


def make_native_pose_to_point_factor(pose_key: int, landmark_key: int, obs_Tc: np.ndarray,
                                     cov: np.ndarray, kernel: str, delta: float):
    """C++ PoseToPointFactor with the exact residual of make_pose_to_point_factor
    (transformTo(l_w) - obs, same key order and Jacobians) but relinearized in
    C++ — no Python callback per iSAM2 update."""
    noise = robustify(gtsam.noiseModel.Gaussian.Covariance(np.asarray(cov, dtype=np.float64)), kernel, delta)
    return require_native_p2p()(pose_key, landmark_key, np.asarray(obs_Tc, dtype=np.float64), noise)


def _covariance_2to3_full(var_u: np.ndarray, var_v: np.ndarray, var_d: np.ndarray,
                          u: np.ndarray, v: np.ndarray, d: np.ndarray,
                          K: np.ndarray) -> np.ndarray:
    """(N,3,3) float64 NED covariance of the back-projected point.

    Same closed form as Module/Covariance/Project2to3.py::Covariance_2to3_full
    with sigma_uv = 0, re-stated in numpy because that function materializes a
    float32 intermediate; the persistent graph wants float64 end to end.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    sigma_xx = ((u - cx) ** 2 * var_d + d ** 2 * var_u + var_u * var_d) / fx ** 2
    sigma_yy = ((v - cy) ** 2 * var_d + d ** 2 * var_v + var_v * var_d) / fy ** 2
    sigma_zz = var_d
    sigma_xy = ((u - cx) * (v - cy) * var_d) / (fx * fy)
    sigma_xz = var_d * (u - cx) / fx
    sigma_yz = var_d * (v - cy) / fy

    cov = np.empty((u.shape[0], 3, 3), dtype=np.float64)
    cov[:, 0, 0] = sigma_zz
    cov[:, 0, 1] = cov[:, 1, 0] = sigma_xz
    cov[:, 0, 2] = cov[:, 2, 0] = sigma_yz
    cov[:, 1, 1] = sigma_xx
    cov[:, 1, 2] = cov[:, 2, 1] = sigma_xy
    cov[:, 2, 2] = sigma_yy
    return cov


def _pixel2point_ned(uv: np.ndarray, d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N,3) float64 NED camera-frame points: [d, (u-cx)d/fx, (v-cy)d/fy]."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([d, (uv[:, 0] - cx) * d / fx, (uv[:, 1] - cy) * d / fy], axis=-1)


def _matrix_to_se3(T: np.ndarray) -> torch.Tensor:
    """(4,4) -> (7,) [t, qx, qy, qz, qw] float32, via gtsam (renormalizes R)."""
    pose = gtsam.Pose3(np.asarray(T, dtype=np.float64))
    q = pose.rotation().toQuaternion()      # (w, x, y, z) accessors
    t = pose.translation()
    return torch.tensor([t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w()], dtype=torch.float32)


def _se3_to_matrix(pose: torch.Tensor) -> np.ndarray:
    return pp.SE3(pose.detach().cpu().double()).matrix().numpy().reshape(4, 4)


class ISAM2FlowTracker:
    """The persistent iSAM2 graph + track table, fed one frame pair per step()."""
    FRAME_LM_KEEP = 100     # frames of pixel->landmark tables kept while no keyframe rows arrive

    def __init__(self, cfg: SimpleNamespace):
        self.cfg = cfg
        if getattr(cfg, "factor_type", "pose2point") == "pose2point_native":
            require_native_p2p()     # fail at construction, not mid-sequence
        self.gnc_rounds : int   = getattr(cfg, "gnc_rounds", 0)
        self.gnc_c      : float = getattr(cfg, "gnc_c", 1.0)
        self.gnc_mu_rate: float = getattr(cfg, "gnc_mu_rate", 1.4)
        self.gnc_w_floor: float = getattr(cfg, "gnc_w_floor", 1e-4)
        self.gnc_sanity_metres: float = getattr(cfg, "gnc_sanity_metres", 1e3)
        self.gnc_sanity_rel   : float = getattr(cfg, "gnc_sanity_rel", 50.0)
        self.prior_sigma: float = getattr(cfg, "prior_sigma", 1e-4)
        self.marg_lag   : int   = int(getattr(cfg, "marg_lag", 0))
        self.marg_dead_at_birth: bool = bool(getattr(cfg, "marg_dead_at_birth", False))
        self.marg_touch_sigma  : float = float(getattr(cfg, "marg_touch_sigma", 1e4))
        self.kf_cov_scale      : float = float(getattr(cfg, "kf_cov_scale", 1.0))
        self.final_lm          : bool  = bool(getattr(cfg, "final_lm", False))
        self.final_lm_max_iters: int   = int(getattr(cfg, "final_lm_max_iters", 100))
        _check_marg_lag(self.marg_lag, self.gnc_rounds)
        self._stamp_of: dict[int, float] = {}    # mirror of every stamp handed to the smoother
        # final_lm under marg_lag needs what the smoother forgets (see module docstring)
        self._shadow: gtsam.NonlinearFactorGraph | None = (
            gtsam.NonlinearFactorGraph() if self.final_lm and self.marg_lag else None)
        self._frozen: gtsam.Values = gtsam.Values()
        self.final_lm_stats: dict | None = None

        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(cfg.relin_threshold)
        params.relinearizeSkip = cfg.relin_skip
        # QR, not the default Cholesky: Huber-downweighted cliques go numerically
        # rank-deficient and Cholesky elimination throws
        # IndeterminantLinearSystemException (measured in learningUAVO).
        params.setFactorization("QR")
        if getattr(cfg, "dogleg", False) or (self.gnc_rounds > 0 and getattr(cfg, "gnc_damped", True)):
            params.setOptimizationParams(gtsam.ISAM2DoglegParams())
        else:
            gn = gtsam.ISAM2GaussNewtonParams()
            gn.setWildfireThreshold(getattr(cfg, "wildfire", 1e-3))
            params.setOptimizationParams(gn)
        if self.marg_lag:
            # Lazy: the default path never imports gtsam_unstable.
            from .Marginalization import FixedLagIsam2, timestamp_map
            self._timestamp_map = timestamp_map
            self.isam = FixedLagIsam2(params, self.marg_lag)
        else:
            self.isam = gtsam.ISAM2(params)

        self.stats: list[dict] = []     # per frame: frame, live_vars, n_tracks, n_in_graph, ms
        self.tracks: dict[tuple[int, int], _TrackState] = {}
        # frame -> {integer pixel1 -> landmark key}, the association keyframe rows look up
        self.frame_lm: dict[int, dict[tuple[int, int], int]] = {}
        self.n_kf_total  : int = 0
        self.next_lm_id  : int = 0
        self.pose_keys   : set[int] = set()
        self.last_out_idx: int | None = None
        self.T_rel_prev  : np.ndarray | None = None
        self.T_chain     : np.ndarray | None = None
        self.n_frames_done : int = 0
        self.n_gnc_rollback: int = 0
        self._gnc_extent   : float = 1.0

    # -- measurement preparation ---------------------------------------------

    def _step_variance(self, uv_cov: np.ndarray) -> np.ndarray:
        """(N,2) per-step flow variance from a MatchObs uv_cov block, with the
        -1 unavailable-sentinel replaced and the pixel-quantization floor applied."""
        var = uv_cov[:, :2].astype(np.float64).copy()
        var[var < 0] = self.cfg.match_cov_default
        return np.maximum(var, self.cfg.min_flow_cov ** 2)

    def _depth_variance(self, d_cov: np.ndarray) -> np.ndarray:
        # Scale before floor; the -1 sentinel scales negative and gets floored too.
        return np.maximum(d_cov.astype(np.float64) * self.cfg.depth_var_scale, self.cfg.min_depth_cov)

    # -- graph assembly helpers ----------------------------------------------

    def _sigmas(self, sigma: float) -> np.ndarray:
        """BetweenFactorPose3 sigmas: rotation (first 3) twice as loose as translation."""
        return np.r_[np.full(3, 2.0 * sigma), np.full(3, sigma)]

    def _ensure_from_pose(self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values,
                          data: ISAM2_GraphInput) -> np.ndarray:
        """Make sure p_{from_idx} exists; return its current (4,4) estimate/init."""
        p_f = gtsam.symbol("p", data.from_idx)
        if not self.pose_keys:
            T_f = _se3_to_matrix(data.from_pose)
            values.insert(p_f, gtsam.Pose3(T_f))
            graph.add(gtsam.PriorFactorPose3(     # the ONLY gauge prior, ever
                p_f, gtsam.Pose3(T_f),
                gtsam.noiseModel.Diagonal.Sigmas(np.full(6, self.prior_sigma))))
            self.pose_keys.add(data.from_idx)
            self.T_chain = T_f.copy()
            self.last_out_idx = data.from_idx
            return T_f
        if data.from_idx not in self.pose_keys:
            # Odometry lost track on `from_idx` and skipped its job: coast the pose in.
            assert self.last_out_idx is not None
            p_last = gtsam.symbol("p", self.last_out_idx)
            T_rel = self.T_rel_prev if self.T_rel_prev is not None else np.eye(4)
            T_f = self.isam.calculateEstimatePose3(p_last).matrix() @ T_rel
            values.insert(p_f, gtsam.Pose3(T_f))
            graph.add(gtsam.BetweenFactorPose3(
                p_last, p_f, gtsam.Pose3(T_rel),
                gtsam.noiseModel.Diagonal.Sigmas(self._sigmas(self.cfg.coast_sigma))))
            self.pose_keys.add(data.from_idx)
            return T_f
        return self.isam.calculateEstimatePose3(p_f).matrix()

    def _stamps(self, k: int, values: gtsam.Values, surviving: dict, dead: list,
                p_last: int | None, kf_keys: list[int]):
        """Frame k's timestamp map: what the smoother is allowed to forget.

        A key stamped k survives another marg_lag frames, and re-stamping
        OVERWRITES its stamp, so retention is just "stamp it again every frame":

          every key in `values`   inserted THIS frame: p_k, new landmarks, and
                                  p_from on the first frame or when coasted in
                                  (a never-stamped key never expires). p_(k-1)
                                  is deliberately NOT re-stamped in the normal
                                  case: the window is exactly marg_lag poses.
          surviving landmarks     every continuing track; dead tracks are not
                                  refreshed and expire marg_lag frames later.
                                  All of them are in the graph (new ones got
                                  their Values this frame).
          dead tracks             (marg_dead_at_birth only) stamped back to
                                  their birth frame the frame they die, so
                                  they leave with their first observer — see
                                  the module docstring for the failure shape
                                  this removes.
          p_last (gap only)       after a VOLostTrack gap the readout reads
                                  p_(last_out_idx) AFTER this update; its own
                                  stamp is older than the gap, so refresh it.
          kf_keys                 the keyframe pose and every landmark a keyframe
                                  row re-observed this frame.
        """
        stamps = {int(key): float(k) for key in values.keys()}
        for track in surviving.values():
            stamps[track.lm_key] = float(k)
        if self.marg_dead_at_birth:
            for track in dead:
                stamps[track.lm_key] = float(track.born)
        if p_last is not None:
            stamps[p_last] = float(k)
        for key in kf_keys:
            stamps[key] = float(k)
        self._stamp_of.update(stamps)
        return self._timestamp_map(stamps)

    def _touch_expiring(self, graph: gtsam.NonlinearFactorGraph, k: int) -> int:
        """Add a near-zero-information prior on every key this update will expire.

        The smoother only re-eliminates cliques it touches (new factor, or a
        child clique whose separator names the expiring key). An expiring pose
        sitting untouched as the TOP frontal of a clique above kept frontals is
        cut out of the clique in place, and gtsam 4.2a9 leaves the kept
        frontals conditioned on the removed key (the plane_nose failure at every
        lag). A new factor on the key forces its clique through the constrained
        re-elimination that puts expiring keys at the bottom, where the cut is
        clean. sigma = marg_touch_sigma; 1e4 m adds 1e-8 of precision.
        """
        est = self.isam.calculateEstimate()
        n = 0
        for key, t in self._stamp_of.items():
            if t >= k - self.marg_lag or not est.exists(key):
                continue
            if chr(gtsam.symbolChr(key)) == "p":
                graph.add(gtsam.PriorFactorPose3(key, est.atPose3(key),
                                                 gtsam.noiseModel.Isotropic.Sigma(6, self.marg_touch_sigma)))
            else:
                graph.add(gtsam.PriorFactorPoint3(key, est.atPoint3(key),
                                                  gtsam.noiseModel.Isotropic.Sigma(3, self.marg_touch_sigma)))
            n += 1
        return n

    def _freeze_expiring(self, k: int) -> None:
        """Snapshot the last live estimate of every key this update will expire
        (final_lm under marg_lag): the batch solve later needs a value for every
        key the shadow graph names, and the smoother deletes them."""
        est = self.isam.calculateEstimate()
        for key, t in self._stamp_of.items():
            if t >= k - self.marg_lag or not est.exists(key) or self._frozen.exists(key):
                continue
            if chr(gtsam.symbolChr(key)) == "p":
                self._frozen.insert(key, est.atPose3(key))
            else:
                self._frozen.insert(key, est.atPoint3(key))

    # -- keyframe re-observations ------------------------------------------------

    def _keyframe_factors(self, kf: ISAM2_KeyframeRows, K: np.ndarray, p_k: int,
                          add_point) -> tuple[int, list[int]]:
        """Add p_k -> l for every keyframe row whose integer keyframe pixel resolves
        to a live landmark; return (count, [p_kf, *landmark keys]) for stamping."""
        cfg = self.cfg
        table = self.frame_lm.get(kf.kf_idx)
        if table is None or kf.kf_idx not in self.pose_keys:
            return 0, []
        p_kf = gtsam.symbol("p", kf.kf_idx)
        est = self.isam.calculateEstimate() if self.marg_lag else None
        if est is not None and not est.exists(p_kf):
            return 0, []

        m = kf.pixel1_uv.shape[0]
        uv1 = kf.pixel1_uv.detach().cpu().numpy().astype(np.float64)
        uv2 = kf.pixel2_uv.detach().cpu().numpy().astype(np.float64)
        d2  = kf.pixel2_d.detach().cpu().numpy().astype(np.float64)
        var_d2 = self._depth_variance(kf.pixel2_d_cov.detach().cpu().numpy())
        fvar = cfg.match_cov_default + self._step_variance(kf.pixel2_uv_cov.detach().cpu().numpy())
        obs2 = _pixel2point_ned(uv2, d2, K)
        cov2 = _covariance_2to3_full(fvar[:, 0], fvar[:, 1], var_d2, uv2[:, 0], uv2[:, 1], d2, K) * self.kf_cov_scale
        valid = np.isfinite(d2) & (d2 > 0) & np.isfinite(cov2.reshape(m, -1)).all(axis=1)
        valid &= np.linalg.eigvalsh(np.where(valid[:, None, None], cov2, np.eye(3))).min(axis=1) > 0

        keys: list[int] = [p_kf]
        seen: set[int] = set()
        for j in range(m):
            if not valid[j]:
                continue
            lm = table.get((int(np.rint(uv1[j, 0])), int(np.rint(uv1[j, 1]))))
            if lm is None or lm in seen or (est is not None and not est.exists(lm)):
                continue
            add_point(p_k, lm, obs2[j], cov2[j])
            seen.add(lm)
            keys.append(lm)
        return len(seen), keys

    # -- the per-frame step ----------------------------------------------------

    def step(self, data: ISAM2_GraphInput) -> torch.Tensor:
        """Associate rows to tracks, add the pair's factors, isam.update(),
        return the (7,) readout pose of `frame_idx`."""
        t_start = time.perf_counter()
        cfg = self.cfg
        k = data.frame_idx
        K = data.K.detach().cpu().double().numpy().reshape(3, 3)

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        p_k = gtsam.symbol("p", k)
        p_f = gtsam.symbol("p", data.from_idx)
        T_f_est = self._ensure_from_pose(graph, values, data)
        # the readout reads p_(last_out_idx) even across a VOLostTrack gap: keep it stamped
        gap_last = (gtsam.symbol("p", self.last_out_idx)
                    if self.last_out_idx is not None and self.last_out_idx != data.from_idx else None)

        T_init = (T_f_est @ self.T_rel_prev
                  if cfg.motion_init == "cv" and self.T_rel_prev is not None else T_f_est)
        values.insert(p_k, gtsam.Pose3(T_init))
        self.pose_keys.add(k)

        if cfg.motion_prior_sigma > 0 and self.T_rel_prev is not None:
            graph.add(gtsam.BetweenFactorPose3(
                p_f, p_k, gtsam.Pose3(self.T_rel_prev),
                gtsam.noiseModel.Diagonal.Sigmas(self._sigmas(cfg.motion_prior_sigma))))

        # -- associate rows to tracks by exact integer pixel identity
        n = data.pixel1_uv.shape[0]
        uv1 = data.pixel1_uv.detach().cpu().numpy().astype(np.float64)
        uv2 = data.pixel2_uv.detach().cpu().numpy().astype(np.float64)
        d1  = data.pixel1_d.detach().cpu().numpy().astype(np.float64)
        d2  = data.pixel2_d.detach().cpu().numpy().astype(np.float64)
        var_d1 = self._depth_variance(data.pixel1_d_cov.detach().cpu().numpy())
        var_d2 = self._depth_variance(data.pixel2_d_cov.detach().cpu().numpy())
        step_var = self._step_variance(data.pixel2_uv_cov.detach().cpu().numpy())

        row_track: list[_TrackState | None] = []    # None = new landmark
        fvar2 = np.empty((n, 2))
        for i in range(n):
            key1 = (int(np.rint(uv1[i, 0])), int(np.rint(uv1[i, 1])))
            track = self.tracks.pop(key1, None)
            row_track.append(track)
            prev_fvar = track.fvar if track is not None else np.full(2, cfg.match_cov_default)
            fvar2[i] = prev_fvar + step_var[i] if cfg.accumulate_fvar else step_var[i]

        obs2 = _pixel2point_ned(uv2, d2, K)
        cov2 = _covariance_2to3_full(fvar2[:, 0], fvar2[:, 1], var_d2, uv2[:, 0], uv2[:, 1], d2, K)
        obs1 = _pixel2point_ned(uv1, d1, K)
        birth_var = np.full(n, cfg.match_cov_default)
        cov1 = _covariance_2to3_full(birth_var, birth_var, var_d1, uv1[:, 0], uv1[:, 1], d1, K)

        # A gate-failed row kills its track; a gate-failed new row leaves no graph residue.
        valid = np.isfinite(d2) & (d2 > 0) & np.isfinite(cov2.reshape(n, -1)).all(axis=1)
        valid &= np.linalg.eigvalsh(np.where(valid[:, None, None], cov2, np.eye(3))).min(axis=1) > 0
        is_new = np.array([t is None for t in row_track])
        new_valid = valid & is_new & np.isfinite(d1) & (d1 > 0)
        new_valid &= np.linalg.eigvalsh(np.where(new_valid[:, None, None], cov1, np.eye(3))).min(axis=1) > 0
        valid = np.where(is_new, new_valid, valid)

        gnc_on = self.gnc_rounds > 0
        kernel = "none" if gnc_on else cfg.kernel       # GNC replaces the kernel
        reweight: list[tuple] = []                      # (graph slot, pose key, lm key, obs, cov)

        def add_point(pose_key: int, lm_key: int, obs: np.ndarray, cov: np.ndarray):
            if gnc_on:
                reweight.append((graph.size(), pose_key, lm_key, obs, cov))
            if cfg.factor_type == "bearingrange":
                graph.add(make_native_point_factor(pose_key, lm_key, obs, cov, kernel, cfg.kernel_delta))
            elif cfg.factor_type == "pose2point_native":
                graph.add(make_native_pose_to_point_factor(pose_key, lm_key, obs, cov, kernel, cfg.kernel_delta))
            else:
                graph.add(make_pose_to_point_factor(
                    pose_key, lm_key, obs,
                    robustify(gtsam.noiseModel.Gaussian.Covariance(cov), kernel, cfg.kernel_delta)))

        surviving: dict[tuple[int, int], _TrackState] = {}
        dead: list[_TrackState] = list(self.tracks.values())    # un-popped = no row carried them
        lm_table: dict[tuple[int, int], int] = {}
        n_in_graph = 0
        for i in range(n):
            track = row_track[i]
            if not valid[i]:
                if track is not None:
                    dead.append(track)
                continue
            if track is None:
                lm_key = gtsam.symbol("l", self.next_lm_id)
                self.next_lm_id += 1
                values.insert(lm_key, gtsam.Point3(*(T_f_est[:3, :3] @ obs1[i] + T_f_est[:3, 3])))
                add_point(p_f, lm_key, obs1[i], cov1[i])
                track = _TrackState(lm_key=lm_key, fvar=fvar2[i].copy(), n_obs=2, born=data.from_idx)
            else:
                track = _TrackState(lm_key=track.lm_key, fvar=fvar2[i].copy(), n_obs=track.n_obs + 1,
                                    born=track.born)
            add_point(p_k, track.lm_key, obs2[i], cov2[i])
            n_in_graph += 1
            key1 = (int(np.rint(uv1[i, 0])), int(np.rint(uv1[i, 1])))
            lm_table.setdefault(key1, track.lm_key)
            key2 = (int(np.rint(uv2[i, 0])), int(np.rint(uv2[i, 1])))
            surviving.setdefault(key2, track)
        self.tracks = surviving
        self.frame_lm[data.from_idx] = lm_table

        # -- keyframe rows: extra p_k -> l on the keyframe's landmarks
        n_kf, kf_keys = 0, []
        if data.kf is not None:
            keep_from = data.kf.kf_idx
            n_kf, kf_keys = self._keyframe_factors(data.kf, K, p_k, add_point)
        else:
            keep_from = data.from_idx - self.FRAME_LM_KEEP
        self.frame_lm = {f: t for f, t in self.frame_lm.items() if f >= keep_from}
        self.n_kf_total += n_kf
        n_obs = n_in_graph + n_kf

        if n_obs < cfg.min_support:
            # Low-support coast — weak on purpose: a tight coast welds low-support
            # garbage into the chain; a loose one lets later evidence pull it straight.
            graph.add(gtsam.BetweenFactorPose3(
                p_f, p_k,
                gtsam.Pose3(self.T_rel_prev if self.T_rel_prev is not None else np.eye(4)),
                gtsam.noiseModel.Diagonal.Sigmas(self._sigmas(cfg.coast_sigma))))

        # -- update and read out
        if self.marg_lag:   # explicit branch: gtsam.ISAM2.update's 3rd positional arg is removeFactorIndices
            stamps = self._stamps(k, values, surviving, dead, gap_last, kf_keys)
            if self._shadow is not None:
                self._shadow.push_back(graph)       # measurements only: touch priors come after
                self._freeze_expiring(k)
            if self.marg_touch_sigma > 0:
                self._touch_expiring(graph, k)
            res = self.isam.update(graph, values, stamps)
            self._stamp_of = {key: t for key, t in self._stamp_of.items() if t >= k - self.marg_lag}
        else:
            res = self.isam.update(graph, values)
        n_extra = cfg.extra_updates + (cfg.warmup_extra if self.n_frames_done < cfg.warmup_frames else 0)
        for _ in range(n_extra):
            # Bare call on either object: an extra GN pass that expires nothing.
            self.isam.update()
        if gnc_on and reweight:
            # GNC needs the per-update factor indices, which only gtsam.ISAM2Result
            # carries; the marg_lag + GNC combination is rejected at construction.
            assert isinstance(res, gtsam.ISAM2Result), "GNC requires plain gtsam.ISAM2 (not marg_lag)"
            self._gnc_reweight(reweight, list(res.getNewFactorsIndices()), p_k)

        T_k = self.isam.calculateEstimatePose3(p_k).matrix()
        assert np.isfinite(T_k).all(), (
            f"frame {k}: pose estimate is non-finite. Under QR an "
            f"IndeterminantLinearSystemException means NaN in back-substitution, "
            f"i.e. the solve diverged and overflowed; the pose GTSAM names is "
            f"where the NaN surfaced, not the cause.")

        assert self.last_out_idx is not None and self.T_chain is not None
        if n_obs > 0:
            T_last_now = self.isam.calculateEstimatePose3(gtsam.symbol("p", self.last_out_idx)).matrix()
            T_rel_now = np.linalg.inv(T_last_now) @ T_k
            self.T_rel_prev = T_rel_now
        else:   # blind frame: constant-velocity coast, do not update T_rel_prev
            T_rel_now = self.T_rel_prev if self.T_rel_prev is not None else np.eye(4)
        self.T_chain = self.T_chain @ T_rel_now
        self.last_out_idx = k
        self.n_frames_done += 1

        # marg_lag: the Values is already cached by the readouts above; unbounded: nothing is ever removed
        live_vars = (self.isam.calculateEstimate().size() if self.marg_lag
                     else len(self.pose_keys) + self.next_lm_id)
        self.stats.append(dict(frame=k, live_vars=live_vars, n_tracks=len(surviving),
                               n_in_graph=n_in_graph, n_kf_obs=n_kf,
                               kf_idx=(data.kf.kf_idx if data.kf is not None else -1),
                               ms=(time.perf_counter() - t_start) * 1e3))

        return _matrix_to_se3(self.T_chain if cfg.readout == "chain" else T_k)

    # -- offline batch polish (final_lm) --------------------------------------------

    def _final_graph_and_values(self) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
        """The exact accumulated model and a value for every key it names."""
        if self._shadow is None:
            return self.isam.getFactorsUnsafe(), self.isam.calculateEstimate()
        graph = self._shadow
        est = gtsam.Values(self._frozen)
        live = self.isam.calculateEstimate()
        for key in live.keys():
            value = live.atPose3(key) if chr(gtsam.symbolChr(key)) == "p" else live.atPoint3(key)
            if est.exists(key):
                est.update(key, value)
            else:
                est.insert(key, value)
        missing = [key for key in graph.keyVector() if not est.exists(key)]
        assert not missing, (
            f"final_lm: {len(missing)} shadow-graph keys have no value (first: "
            f"{chr(gtsam.symbolChr(missing[0]))}{gtsam.symbolIndex(missing[0])}) -- "
            f"a key was marginalized without ever being snapshotted")
        return graph, est

    def final_lm_solve(self) -> dict[int, np.ndarray]:
        """Batch Levenberg-Marquardt over the entire accumulated graph from the
        iSAM2 estimate (see module docstring); returns {frame_idx: (4,4) pose}
        for every pose key and records `final_lm_stats`."""
        t0 = time.perf_counter()
        graph, initial = self._final_graph_and_values()
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.final_lm_max_iters)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()
        poses = {k: result.atPose3(gtsam.symbol("p", k)).matrix() for k in sorted(self.pose_keys)}
        for k, T in poses.items():
            assert np.isfinite(T).all(), f"final_lm: pose {k} is non-finite after the batch solve"
        self.final_lm_stats = dict(
            error_before=float(graph.error(initial)), error_after=float(graph.error(result)),
            iterations=int(optimizer.iterations()), n_factors=int(graph.nrFactors()),
            n_values=int(initial.size()), ms=(time.perf_counter() - t0) * 1e3)
        return poses

    # -- GNC-GM ------------------------------------------------------------------

    def _gnc_reweight(self, reweight: list, new_idx: list, p_k: int) -> None:
        """Anneal the GM surrogate over this frame's point factors (see module
        docstring). Earlier frames keep the weight they were frozen with."""
        rows = [(new_idx[slot], pk, lk, o, c) for slot, pk, lk, o, c in reweight]
        w = np.ones(len(rows))
        mu: float | None = None
        rollback = ""
        for _ in range(self.gnc_rounds):
            est = self.isam.calculateEstimate()
            fg = self.isam.getFactorsUnsafe()
            # the factor carries cov/w, so its error is w/2 * r^2 unweighted
            r2 = np.array([2.0 * fg.at(i).error(est) / w[j]
                           for j, (i, _, _, _, _) in enumerate(rows)])
            if mu is None:
                mu = max(2.0 * float(r2.max()) / self.gnc_c ** 2, 2.0)
            w = gnc_weights(r2, mu, self.gnc_c, self.gnc_w_floor)
            mu = max(mu / self.gnc_mu_rate, 1.0)
            rows = self._gnc_replace(rows, w)
            rollback = self._gnc_diverged(rows, p_k)
            if rollback:
                break
        if rollback:
            # Restore unit weights; the trust region keeps the values recoverable.
            self._gnc_replace(rows, np.ones(len(rows)))
            self.n_gnc_rollback += 1

    def _gnc_replace(self, rows: list, w: np.ndarray) -> list:
        """Rebuild this frame's point factors at covariance cov/w, in place."""
        g = gtsam.NonlinearFactorGraph()
        remove = []
        for j, (i, pk, lk, o, c) in enumerate(rows):
            remove.append(i)
            cw = c / w[j]
            if self.cfg.factor_type == "bearingrange":
                g.add(make_native_point_factor(pk, lk, o, cw, "none", 0.0))
            elif self.cfg.factor_type == "pose2point_native":
                g.add(make_native_pose_to_point_factor(pk, lk, o, cw, "none", 0.0))
            else:
                g.add(make_pose_to_point_factor(pk, lk, o, gtsam.noiseModel.Gaussian.Covariance(cw)))
        res = self.isam.update(g, gtsam.Values(), remove)
        assert isinstance(res, gtsam.ISAM2Result), "GNC requires plain gtsam.ISAM2 (not marg_lag)"
        fresh = list(res.getNewFactorsIndices())
        assert len(fresh) == len(rows), f"factor index drift {len(fresh)} != {len(rows)}"
        return [(fresh[j], pk, lk, o, c) for j, (_, pk, lk, o, c) in enumerate(rows)]

    def _gnc_diverged(self, rows: list, p_k: int) -> str:
        """Reason string if this frame's estimate has run away, else ''.

        Only the keys this frame touches are inspected — divergence shows up
        there first, and a full estimate scan per round is not free.
        """
        if self.gnc_sanity_metres <= 0 and self.gnc_sanity_rel <= 0:
            return ""
        est = self.isam.calculateEstimate()
        worst = 0.0
        for key in {p_k} | {r[1] for r in rows}:
            t = np.asarray(est.atPose3(key).translation())
            if not np.isfinite(t).all():
                return "nonfinite"
            worst = max(worst, float(np.abs(t).max()))
        for key in {r[2] for r in rows}:
            point = np.asarray(est.atPoint3(key))
            if not np.isfinite(point).all():
                return "nonfinite"
            worst = max(worst, float(np.abs(point).max()))
        if self.gnc_sanity_metres > 0 and worst > self.gnc_sanity_metres:
            return "absolute"
        if self.gnc_sanity_rel > 0 and worst > self.gnc_sanity_rel * self._gnc_extent:
            return "relative"
        self._gnc_extent = max(self._gnc_extent, worst)
        return ""


class ISAM2_Graph(IOptimizer[ISAM2_GraphInput, dict, ISAM2_GraphOutput]):
    """IOptimizer wrapper around ISAM2FlowTracker (see the module docstring).

    Sequential-only (`parallel: false` enforced): the tracker context is
    stateful, and the parallel path's timeout-abandonment would desynchronize
    the job/result pairing the persistent graph depends on.
    """
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config)
        self._last_written_frame: int = -1

    @torch.no_grad()
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> ISAM2_GraphInput:
        frame2opt = global_map.frames[frame_idx]
        obs = global_map.get_frame2match(frame2opt)
        kf_rows = global_map.frame2kfmatch.project(frame2opt.index)
        kf: ISAM2_KeyframeRows | None = None
        if kf_rows.numel() > 0:
            kf_obs = global_map.kf_match[kf_rows]
            kf = ISAM2_KeyframeRows(
                kf_idx=int(global_map.kfmatch2frame1.project(kf_rows[:1])[0].item()),
                pixel1_uv=kf_obs.data["pixel1_uv"].clone(),
                pixel2_uv=kf_obs.data["pixel2_uv"].clone(),
                pixel2_d=kf_obs.data["pixel2_d"].reshape(-1).clone(),
                pixel2_d_cov=kf_obs.data["pixel2_d_cov"].reshape(-1).clone(),
                pixel2_uv_cov=kf_obs.data["pixel2_uv_cov"].clone(),
            )
        return ISAM2_GraphInput(
            frame_idx=int(frame_idx.item()),
            from_idx=int(frame_idx.item()) - 1,
            from_pose=global_map.frames.data["pose"][frame_idx - 1].reshape(7).clone(),
            K=frame2opt.data["K"][0].clone(),
            pixel1_uv=obs.data["pixel1_uv"].clone(),
            pixel2_uv=obs.data["pixel2_uv"].clone(),
            pixel1_d=obs.data["pixel1_d"].reshape(-1).clone(),
            pixel2_d=obs.data["pixel2_d"].reshape(-1).clone(),
            pixel1_d_cov=obs.data["pixel1_d_cov"].reshape(-1).clone(),
            pixel2_d_cov=obs.data["pixel2_d_cov"].reshape(-1).clone(),
            pixel1_uv_cov=obs.data["pixel1_uv_cov"].clone(),
            pixel2_uv_cov=obs.data["pixel2_uv_cov"].clone(),
            kf=kf,
        )

    @staticmethod
    def init_context(config) -> dict:
        return {"tracker": ISAM2FlowTracker(config)}

    @staticmethod
    def _optimize(context: dict, graph_data: ISAM2_GraphInput) -> tuple[dict, ISAM2_GraphOutput]:
        with Timer.CPUTimingContext("ISAM2Graph"):
            tracker: ISAM2FlowTracker = context["tracker"]
            pose = tracker.step(graph_data)
        return context, ISAM2_GraphOutput(frame_idx=graph_data.frame_idx, pose_estimate=pose)

    def write_graph_data(self, result: ISAM2_GraphOutput | None, global_map: VisualMap) -> None:
        if result is None:
            return
        # Sequential mode re-delivers the last result after a frame whose job was
        # skipped (VOLostTrack) — writing it twice is harmless but masks the skip.
        if result.frame_idx == self._last_written_frame:
            return
        self._last_written_frame = result.frame_idx
        global_map.frames.data["pose"][result.frame_idx] = result.pose_estimate

    def finalize(self, global_map: VisualMap) -> None:
        """`final_lm`: one batch LM over the whole accumulated graph, every pose key
        written back over its online readout (see module docstring). No-op otherwise."""
        tracker = self._tracker()
        if tracker is None or not tracker.final_lm or not tracker.pose_keys:
            return
        poses = tracker.final_lm_solve()
        moved = 0.0
        for k, T in poses.items():
            polished = _matrix_to_se3(T)
            moved = max(moved, float((polished[:3] - global_map.frames.data["pose"][k][:3]).norm()))
            global_map.frames.data["pose"][k] = polished
        s = tracker.final_lm_stats
        assert s is not None
        Logger.write("info",
            f"ISAM2_Graph final_lm: {s['n_factors']} factors / {s['n_values']} values, "
            f"error {s['error_before']:.4g} -> {s['error_after']:.4g} in {s['iterations']} LM "
            f"iterations, {s['ms'] / 1e3:.1f} s (offline) | {len(poses)} poses rewritten, "
            f"largest translation change vs online readout {moved:.3f} m")

    def _tracker(self) -> "ISAM2FlowTracker | None":
        return self.context["tracker"] if isinstance(self.context, dict) else None

    def frame_stats(self) -> dict[str, np.ndarray] | None:
        """Per-frame backend stats as column arrays (frame, live_vars, n_tracks,
        n_in_graph, ms), or None before the first step."""
        tracker = self._tracker()
        if tracker is None or not tracker.stats:
            return None
        return {key: np.array([s[key] for s in tracker.stats]) for key in tracker.stats[0]}

    def terminate(self):
        super().terminate()
        tracker = self._tracker()
        if tracker is None or not tracker.stats:
            return
        lv = np.array([s["live_vars"] for s in tracker.stats])
        ms = np.array([s["ms"] for s in tracker.stats])
        # the plateau is the result: unbounded climbs with the frame index, marginalized levels off
        Logger.write("info",
            f"ISAM2_Graph marg_lag={tracker.marg_lag}: live variables peak {lv.max()}, "
            f"last {lv[-1]}, median over the last half {np.median(lv[len(lv) // 2:]):.0f} | "
            f"step ms median {np.median(ms):.0f}, p90 {np.percentile(ms, 90):.0f}, "
            f"max {ms.max():.0f} | {tracker.next_lm_id} landmarks minted over {len(lv)} frames"
            f" | {tracker.n_kf_total} keyframe re-observations")

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        spec: dict = {
            "device"            : lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "parallel"          : lambda b: b is False,     # see class docstring
            "factor_type"       : lambda s: s in {"pose2point", "pose2point_native", "bearingrange"},
            "kernel"            : lambda s: s in _KERNELS,
            "kernel_delta"      : lambda v: isinstance(v, (int, float)) and v >= 0.,
            "relin_threshold"   : lambda v: isinstance(v, (int, float)) and v > 0.,
            "relin_skip"        : lambda v: isinstance(v, int) and v >= 1,
            "extra_updates"     : lambda v: isinstance(v, int) and v >= 0,
            "warmup_frames"     : lambda v: isinstance(v, int) and v >= 0,
            "warmup_extra"      : lambda v: isinstance(v, int) and v >= 0,
            "depth_var_scale"   : lambda v: isinstance(v, (int, float)) and v > 0.,
            "accumulate_fvar"   : lambda b: isinstance(b, bool),
            "motion_init"       : lambda s: s in {"static", "cv"},
            "motion_prior_sigma": lambda v: isinstance(v, (int, float)) and v >= 0.,
            "coast_sigma"       : lambda v: isinstance(v, (int, float)) and v > 0.,
            "min_support"       : lambda v: isinstance(v, int) and v >= 0,
            "readout"           : lambda s: s in {"chain", "online"},
            "min_flow_cov"      : lambda v: isinstance(v, (int, float)) and v > 0.,
            "min_depth_cov"     : lambda v: isinstance(v, (int, float)) and v > 0.,
            "match_cov_default" : lambda v: isinstance(v, (int, float)) and v > 0.,
        }
        optional_spec: dict = {
            "prior_sigma"       : lambda v: isinstance(v, (int, float)) and v > 0.,
            "wildfire"          : lambda v: isinstance(v, (int, float)) and v > 0.,
            "dogleg"            : lambda b: isinstance(b, bool),
            "gnc_rounds"        : lambda v: isinstance(v, int) and v >= 0,
            "gnc_c"             : lambda v: isinstance(v, (int, float)) and v > 0.,
            "gnc_mu_rate"       : lambda v: isinstance(v, (int, float)) and v > 1.,
            "gnc_w_floor"       : lambda v: isinstance(v, (int, float)) and 0. < v <= 1.,
            "gnc_sanity_metres" : lambda v: isinstance(v, (int, float)),
            "gnc_sanity_rel"    : lambda v: isinstance(v, (int, float)),
            "gnc_damped"        : lambda b: isinstance(b, bool),
            "marg_lag"          : lambda v: isinstance(v, int) and v >= 0,
            "marg_dead_at_birth": lambda b: isinstance(b, bool),
            "marg_touch_sigma"  : lambda v: isinstance(v, (int, float)) and v >= 0.,
            "kf_cov_scale"      : lambda v: isinstance(v, (int, float)) and v > 0.,
            "final_lm"          : lambda b: isinstance(b, bool),
            "final_lm_max_iters": lambda v: isinstance(v, int) and v >= 1,
        }
        for key, check in optional_spec.items():
            if hasattr(config, key):
                spec[key] = check
        cls._enforce_config_spec(config, spec)
        _check_marg_lag(getattr(config, "marg_lag", 0), getattr(config, "gnc_rounds", 0))
