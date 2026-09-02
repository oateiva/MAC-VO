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

Deliberately NOT ported from isam2_tracker.py (see learningUAVO FINDINGS.md):
the gp depth-prior mode; obs_stride / obs_phase / add_budget (the winning online
arms run stride 1 / budget 0, and deferred insertion crashes the online
readout); max_age and max_step_cov (both measured harmful — track length is
monotonically good); the exact-model shadow graph + offline LM polish (offline-
only readout); landmark write-back into the map (the chain readout's gauge and
the live graph's gauge diverge after low-support stretches — map points stay as
the odometry registered them). One measured deviation: the observation depth is
the nearest-sampled `pixel2_d` (this repo's house convention), not the
kernel-weighted depth of learningUAVO's compose_observation.
"""
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
from Utility.Timer import Timer

from ..Interface import IOptimizer

_KERNELS = ("huber", "cauchy", "geman", "tukey", "welsch", "none")


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


@dataclass
class ISAM2_GraphOutput:
    frame_idx    : int
    pose_estimate: torch.Tensor     # (7,) SE3, float32


@dataclass
class _TrackState:
    lm_key: int                     # gtsam.symbol("l", id) — never reused
    fvar  : np.ndarray              # (2,) accumulated flow variance (see module docstring)
    n_obs : int


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
        self.isam = gtsam.ISAM2(params)

        self.tracks: dict[tuple[int, int], _TrackState] = {}
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

    # -- the per-frame step ----------------------------------------------------

    def step(self, data: ISAM2_GraphInput) -> torch.Tensor:
        """Associate rows to tracks, add the pair's factors, isam.update(),
        return the (7,) readout pose of `frame_idx`."""
        cfg = self.cfg
        k = data.frame_idx
        K = data.K.detach().cpu().double().numpy().reshape(3, 3)

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        p_k = gtsam.symbol("p", k)
        p_f = gtsam.symbol("p", data.from_idx)
        T_f_est = self._ensure_from_pose(graph, values, data)

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
        n_in_graph = 0
        for i in range(n):
            if not valid[i]:
                continue
            track = row_track[i]
            if track is None:
                lm_key = gtsam.symbol("l", self.next_lm_id)
                self.next_lm_id += 1
                values.insert(lm_key, gtsam.Point3(*(T_f_est[:3, :3] @ obs1[i] + T_f_est[:3, 3])))
                add_point(p_f, lm_key, obs1[i], cov1[i])
                track = _TrackState(lm_key=lm_key, fvar=fvar2[i].copy(), n_obs=2)
            else:
                track = _TrackState(lm_key=track.lm_key, fvar=fvar2[i].copy(), n_obs=track.n_obs + 1)
            add_point(p_k, track.lm_key, obs2[i], cov2[i])
            n_in_graph += 1
            key2 = (int(np.rint(uv2[i, 0])), int(np.rint(uv2[i, 1])))
            surviving.setdefault(key2, track)
        self.tracks = surviving

        if n_in_graph < cfg.min_support:
            # Low-support coast — weak on purpose: a tight coast welds low-support
            # garbage into the chain; a loose one lets later evidence pull it straight.
            graph.add(gtsam.BetweenFactorPose3(
                p_f, p_k,
                gtsam.Pose3(self.T_rel_prev if self.T_rel_prev is not None else np.eye(4)),
                gtsam.noiseModel.Diagonal.Sigmas(self._sigmas(cfg.coast_sigma))))

        # -- update and read out
        res = self.isam.update(graph, values)
        n_extra = cfg.extra_updates + (cfg.warmup_extra if self.n_frames_done < cfg.warmup_frames else 0)
        for _ in range(n_extra):
            self.isam.update()
        if gnc_on and reweight:
            self._gnc_reweight(reweight, list(res.getNewFactorsIndices()), p_k)

        T_k = self.isam.calculateEstimatePose3(p_k).matrix()
        assert np.isfinite(T_k).all(), (
            f"frame {k}: pose estimate is non-finite. Under QR an "
            f"IndeterminantLinearSystemException means NaN in back-substitution, "
            f"i.e. the solve diverged and overflowed; the pose GTSAM names is "
            f"where the NaN surfaced, not the cause.")

        assert self.last_out_idx is not None and self.T_chain is not None
        if n_in_graph > 0:
            T_last_now = self.isam.calculateEstimatePose3(gtsam.symbol("p", self.last_out_idx)).matrix()
            T_rel_now = np.linalg.inv(T_last_now) @ T_k
            self.T_rel_prev = T_rel_now
        else:   # blind frame: constant-velocity coast, do not update T_rel_prev
            T_rel_now = self.T_rel_prev if self.T_rel_prev is not None else np.eye(4)
        self.T_chain = self.T_chain @ T_rel_now
        self.last_out_idx = k
        self.n_frames_done += 1

        return _matrix_to_se3(self.T_chain if cfg.readout == "chain" else T_k)

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
        }
        for key, check in optional_spec.items():
            if hasattr(config, key):
                spec[key] = check
        cls._enforce_config_spec(config, spec)
