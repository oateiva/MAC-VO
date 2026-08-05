"""
Correlated log-depth prior for the GTSAM pose2point graph (Phase 0: stationary kernel).

The pose2point graph has one unconstrained along-ray degree of freedom per landmark:
`Sigma_p` is strongly elongated along the viewing ray (~30:1 at typical focal/depth), so
each landmark slides along its own ray nearly for free and a global depth-scale error
never reaches the pose (see ProgressReports/2026-07-07_gtsam-alignment-sweep.md for the
sim3 null result this explains). The factor built here couples those freedoms: per
spatially-local block B of landmarks, residual

    r_b = Linv @ (y_B - yhat_B - s * 1),   y_j = log(e^T T^{-1} l_j),  yhat_j = log d_j

whitened by the inverse Cholesky factor of a Matern-3/2 kernel over pixel coordinates,
with one scalar `s` per frame shared by all of that frame's blocks. `s` subtracts the
global depth scale so the prior asserts only the *shape* of the depth field; sharing it
across blocks is what supplies the rank-1 global coupling of the marginalized
scale-free prior (it is the Schur-complement representation, not an approximation).

Everything here is depth-module agnostic: inputs are pixel coordinates and metric depths
from the map schema (`MatchObs.pixel*_uv/pixel*_d`), never a specific network's output.
Phase 1 swaps the kernel through `KERNEL_BUILDERS` without touching the factor.
"""
import numpy as np
import gtsam
import typing as typ

from typing import Callable, Optional

from Utility.PrettyPrint import Logger

if typ.TYPE_CHECKING:
    from types import SimpleNamespace
    from .Augmentations import BuildContext
    from .Graphs import GTSAM_Pose2Point, GTSAM_GraphInput

# The camera-frame depth axis. Camera points in this codebase are NED with component 0
# the optical/forward axis: `pixel2point_NED` (Utility/Point.py) rolls pypose's EDN
# output so index 0 == depth exactly (unnormalized ray d * K^-1 [u,v,1]), and
# `Covariance_2to3_full` (Module/Covariance/Project2to3.py) places sigma_zz (the depth
# variance) at row/col 0 of the same 3x3. Never index the depth axis inline — use this.
OPTICAL_AXIS: int = 0


def matern32_kernel(uv: np.ndarray, ell: float, sigma_f: float,
                    sigma_n: "float | np.ndarray") -> np.ndarray:
    """
    Isotropic Matern-3/2 over pixel coordinates, nugget included:

        K[m,n] = sigma_f^2 (1 + sqrt(3) rho) exp(-sqrt(3) rho) + sigma_n^2 delta_mn,
        rho    = ||x_m - x_n|| / ell

    `ell` is in pixels; `sigma_f`/`sigma_n` are in log-depth units. The ratio
    sigma_f/sigma_n is what makes the prior correlated rather than a bag of independent
    per-point pulls. `sigma_n` may be a per-point (n,) array — the heteroscedastic
    nugget used when the prior owns the depth measurement and the network reports
    per-point uncertainty (gp_prior_nugget: measured).
    """
    uv = np.asarray(uv, dtype=np.float64)
    d = np.linalg.norm(uv[:, None, :] - uv[None, :, :], axis=-1)
    rho = (np.sqrt(3.0) / ell) * d
    K = (sigma_f ** 2) * (1.0 + rho) * np.exp(-rho)
    K[np.diag_indices_from(K)] += np.square(np.asarray(sigma_n, dtype=np.float64))
    return K


# Kernel registry: Phase 1 (learned, e.g. DepthCov-derived) kernels are added here; the
# factor, blocking and graph wiring are kernel-agnostic. Config validation accepts
# exactly these names.
KERNEL_BUILDERS: dict[str, Callable[..., np.ndarray]] = {
    "matern32": matern32_kernel,
}


def depth_free_covariance(uv: np.ndarray, d: np.ndarray, uv_cov: np.ndarray,
                          K_intr: np.ndarray, slide_rel: float) -> np.ndarray:
    """
    Point-factor covariance with the depth measurement moved OUT (Phase 1: the GP
    prior owns depth). NED [depth, east, down]:

        Sigma_df = L + (slide_rel * d)^2 * outer(r, r),   r = [1, (u-cx)/fx, (v-cy)/fy]

    where L is the pixel-noise lateral block (d^2 sigma_uu/fx^2 etc., zero on the
    depth axis) and the second term is a large slide along the VIEWING RAY, making
    the along-ray direction nearly uninformative. Zeroing sigma_dd instead would
    claim the measured depth is exact (the observation d*ray embeds it) — the
    opposite of intent; and a huge sigma_dd inside Covariance_2to3_full blows up
    the lateral directions through its second-order sigma_uu*sigma_dd term. The
    slide term also keeps the matrix full-rank SPD: it is the rank floor the
    depth-free factor needs.

    `uv_cov` columns are (sigma_uu, sigma_vv, sigma_uv), the MatchObs layout.
    Consequence for robust weighting: Huber on these factors gates on lateral/flow
    consistency only — the along-ray residual is whitened by ~1/(slide_rel*d).
    """
    uv = np.asarray(uv, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    uv_cov = np.asarray(uv_cov, dtype=np.float64)
    fx, fy = float(K_intr[0, 0]), float(K_intr[1, 1])
    cx, cy = float(K_intr[0, 2]), float(K_intr[1, 2])

    rho_x = (uv[:, 0] - cx) / fx
    rho_y = (uv[:, 1] - cy) / fy
    rays = np.stack([np.ones_like(rho_x), rho_x, rho_y], axis=-1)          # (N,3)

    n = uv.shape[0]
    cov = (slide_rel * d)[:, None, None] ** 2 * (rays[:, :, None] * rays[:, None, :])
    d2 = d ** 2
    # tiny relative jitter keeps the lateral block SPD if a matcher ever emits a
    # degenerate pixel covariance (sigma_uv^2 == sigma_uu*sigma_vv)
    eps = 1e-9 * d2
    cov[:, 1, 1] += d2 * uv_cov[:, 0] / fx ** 2 + eps
    cov[:, 2, 2] += d2 * uv_cov[:, 1] / fy ** 2 + eps
    cov[:, 1, 2] += d2 * uv_cov[:, 2] / (fx * fy)
    cov[:, 2, 1] += d2 * uv_cov[:, 2] / (fx * fy)
    assert cov.shape == (n, 3, 3)
    return cov


def partition_blocks(uv: np.ndarray, block_size: int,
                     indices: Optional[np.ndarray] = None) -> list[np.ndarray]:
    """
    Partition points into spatially local blocks of at most `block_size` by recursive
    median splits along the wider pixel-extent axis (kd-style). Every input point lands
    in exactly one block (no dropped remainder); resulting blocks have sizes in roughly
    (block_size/2, block_size]. Returned arrays contain ORIGINAL row indices (aligned
    with `gtsam.symbol('l', i)` numbering), which is what the factor keys need — never
    positions within a masked subset.
    """
    uv = np.asarray(uv, dtype=np.float64)
    if indices is None:
        indices = np.arange(uv.shape[0])
    indices = np.asarray(indices)
    assert uv.shape[0] == indices.shape[0]

    if indices.shape[0] == 0:
        return []
    if indices.shape[0] <= block_size:
        return [indices]

    extent = uv.max(axis=0) - uv.min(axis=0)
    axis = int(np.argmax(extent))
    order = np.argsort(uv[:, axis], kind="stable")
    half = indices.shape[0] // 2
    lo, hi = order[:half], order[half:]
    return (partition_blocks(uv[lo], block_size, indices[lo])
            + partition_blocks(uv[hi], block_size, indices[hi]))


def chol_inv_lower(K: np.ndarray, base_jitter: float) -> Optional[np.ndarray]:
    """
    `Linv = L^-1` with `K = L L^T`, via Cholesky + triangular solve against identity
    (never a direct inverse of K). On failure the nugget is escalated: retry with
    `K + (10^i - 1) * base_jitter * I` for i = 1..3, then give up (caller drops the
    block and logs). `base_jitter` should be sigma_n^2 — the kernel's own nugget scale.
    """
    from scipy.linalg import solve_triangular
    n = K.shape[0]
    for i in range(4):
        try:
            L = np.linalg.cholesky(K + ((10.0 ** i) - 1.0) * base_jitter * np.eye(n))
            return solve_triangular(L, np.eye(n), lower=True)
        except np.linalg.LinAlgError:
            continue
    return None


def make_gp_depth_prior_factor(pose_key: int, s_key: int, landmark_keys: list[int],
                               Linv: np.ndarray, yhat: np.ndarray, z_min: float,
                               counters: dict) -> gtsam.CustomFactor:
    """
    The block factor. Keys in this exact order: [pose, s] + landmarks, so H[0] is
    (n_b x 6), H[1] is (n_b x 1), H[2+k] is (n_b x 3). Noise model is Unit(n_b): the
    whitening is explicit in Linv. No robust kernel — a robust loss on a whitened block
    residual would down-weight the whole block, which is not the wanted semantics.

    Jacobians (e = optical axis, z_j = e^T T^-1 l_j, from Pose3.transformTo's H_self /
    H_point):

        d r / d xi   = Linv @ J_xi,        J_xi[k,:] = (e^T / z_k) @ H_self_k
        d r / d s    = -Linv @ 1
        d r / d l_j  = outer(Linv[:,k], (e^T / z_j) @ H_point_j)      -- rank 1 by
                       construction: the prior constrains each landmark only along its
                       own ray, leaving lateral directions to the flow-driven factors.

    `z` is clamped at `z_min` (landmark behind/near the camera plane during iteration);
    a clamped row's y is constant so its pose/landmark Jacobian rows are ZEROED — the
    unclamped formula at z=z_min would disagree with the cost and stall LM. Clamps are
    counted into `counters["z_clamps"]`; frequent clamping means something upstream is
    wrong and must be reported, not tuned around.
    """
    n_b = len(landmark_keys)
    Linv = np.asarray(Linv, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(n_b)
    keys = [pose_key, s_key] + list(landmark_keys)
    ones = np.ones(n_b, dtype=np.float64)
    dr_ds = (-Linv @ ones).reshape(n_b, 1)

    def error_func(this_factor, values, H) -> np.ndarray:
        pose: gtsam.Pose3 = values.atPose3(this_factor.keys()[0])
        s = float(values.atVector(this_factor.keys()[1])[0])

        y = np.empty(n_b, dtype=np.float64)
        if H is not None:
            J_xi = np.zeros((n_b, 6), dtype=np.float64)
            # transformTo's optional Jacobian buffers must be float64 F-contiguous in
            # this binding (C-order raises TypeError); one reusable pair per call.
            H_self = np.zeros((3, 6), dtype=np.float64, order='F')
            H_point = np.zeros((3, 3), dtype=np.float64, order='F')
            for k in range(n_b):
                l_w = values.atPoint3(this_factor.keys()[2 + k])
                p_c = pose.transformTo(l_w, H_self, H_point)
                z = float(p_c[OPTICAL_AXIS])
                if z < z_min:
                    counters["z_clamps"] = counters.get("z_clamps", 0) + 1
                    y[k] = np.log(z_min)
                    H[2 + k] = np.zeros((n_b, 3), dtype=np.float64)
                else:
                    y[k] = np.log(z)
                    J_xi[k, :] = H_self[OPTICAL_AXIS, :] / z
                    H[2 + k] = np.outer(Linv[:, k], H_point[OPTICAL_AXIS, :] / z)
            H[0] = Linv @ J_xi
            H[1] = dr_ds
        else:
            for k in range(n_b):
                l_w = values.atPoint3(this_factor.keys()[2 + k])
                p_c = pose.transformTo(l_w)
                z = float(p_c[OPTICAL_AXIS])
                if z < z_min:
                    counters["z_clamps"] = counters.get("z_clamps", 0) + 1
                    z = z_min
                y[k] = np.log(z)

        return Linv @ (y - yhat - s * ones)

    return gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(n_b), keys, error_func)


def _sane_uv_cov(t, n: int) -> np.ndarray:
    """
    Pixel covariance (sigma_uu, sigma_vv, sigma_uv) sanitized for factor use:
    the -1 "not available" sentinel (or an absent field) becomes the 1 px^2
    default, and cross terms that would make the 2x2 indefinite are zeroed.
    """
    if t is None:
        return np.tile([1.0, 1.0, 0.0], (n, 1))
    c = t.detach().cpu().double().numpy().copy()
    bad = ~((c[:, 0] > 0) & (c[:, 1] > 0))
    c[bad] = (1.0, 1.0, 0.0)
    indefinite = np.square(c[:, 2]) >= c[:, 0] * c[:, 1]
    c[indefinite, 2] = 0.0
    return c


class CorrelatedDepthPrior:
    """
    The correlated log-depth prior as a `GraphAugmentation` (see
    Augmentations.py for the lifecycle contract). Consumes only the
    frontend-agnostic map schema — `MatchObs.pixel*_uv` / `pixel*_d`, row-aligned
    with `gtsam.symbol('l', i)` — so it works unchanged under any depth module.

    `cfg` is the resolved namespace built by `GTSAM_Graph.init_context`:
    frames, block_size, kernel, length_scale_px, sigma_f, sigma_n,
    scale_prior_sigma, z_min, own_depth, slide_rel, nugget. Mutual exclusion
    with a non-se3 alignment is enforced at config validation (both occupy the
    global depth-scale direction).

    Phase 1 (`own_depth: true`): the point factors' covariances are replaced by
    `depth_free_covariance` — the prior becomes the SOLE per-point depth
    measurement model, removing the double count that made Phase 0 a structural
    null (the prior's yhat duplicated the sigma_dd already in Sigma_p). With
    `nugget: measured` the per-point nugget is the network's own log-depth
    variance, `pixel*_d_cov / d^2` floored at sigma_n^2 (for DepthCov this is
    exactly its GP's Var(log d)).
    """
    def __init__(self, cfg: "SimpleNamespace"):
        self.cfg = cfg
        self._uv: dict[str, np.ndarray] = {}
        self._d: dict[str, np.ndarray] = {}
        self._nugget: dict[str, "float | np.ndarray"] = {}
        self._info: dict = {}
        self._ctx: "BuildContext | None" = None

    def on_parse(self, graph: "GTSAM_Pose2Point", data: "GTSAM_GraphInput") -> None:
        # "prev" = frame from_idx (pixel1_*), "curr" = frame frame_idx (pixel2_*).
        cfg = self.cfg
        obs_data = data.current_graph_data.observations.data
        self._uv = {
            "prev": obs_data["pixel1_uv"].detach().cpu().double().numpy(),
            "curr": obs_data["pixel2_uv"].detach().cpu().double().numpy(),
        }
        self._d = {
            "prev": obs_data["pixel1_d"].detach().cpu().double().numpy().reshape(-1),
            "curr": obs_data["pixel2_d"].detach().cpu().double().numpy().reshape(-1),
        }

        if getattr(cfg, "nugget", "fixed") == "measured":
            # Per-point nugget = the network's log-depth std, sigma_n as floor.
            # `pixel*_d_cov` can hold the -1 "not available" sentinel (or be
            # absent from a pared-down bundle) — those rows keep the floor.
            for role, key in (("prev", "pixel1_d_cov"), ("curr", "pixel2_d_cov")):
                d = self._d[role]
                var_log = np.full_like(d, cfg.sigma_n ** 2)
                if key in obs_data:
                    d_cov = obs_data[key].detach().cpu().double().numpy().reshape(-1)
                    ok = (d > 0) & (d_cov > 0)
                    var_log[ok] = np.maximum(d_cov[ok] / np.square(d[ok]), cfg.sigma_n ** 2)
                self._nugget[role] = np.sqrt(var_log)
        else:
            self._nugget = {"prev": cfg.sigma_n, "curr": cfg.sigma_n}

        if getattr(cfg, "own_depth", False):
            # Replace the point factors' noise with the depth-free covariance.
            # graph.obs*_covTc are numpy COPIES of the map data, so the stored
            # MatchObs covariances (and everything downstream of the map) are
            # untouched; the frame k-2 re-observation factor reads the map
            # tensor directly and deliberately keeps sigma_dd — the prior does
            # not cover frame k-2, so depth-freeing it would discard that
            # frame's depth measurement outright.
            K_intr = data.current_graph_data.images_intrinsic.detach().cpu().double().numpy()
            n = self._uv["curr"].shape[0]
            graph.obs1_covTc = depth_free_covariance(
                self._uv["prev"], self._d["prev"],
                _sane_uv_cov(obs_data.get("pixel1_uv_cov"), n), K_intr, cfg.slide_rel)
            graph.obs2_covTc = depth_free_covariance(
                self._uv["curr"], self._d["curr"],
                _sane_uv_cov(obs_data.get("pixel2_uv_cov"), n), K_intr, cfg.slide_rel)

    def on_build(self, graph: "GTSAM_Pose2Point", factor_graph, initial_estimate,
                 ctx: "BuildContext") -> None:
        """
        Block factors per enabled frame plus one per-frame scale variable `s`
        (symbol 's'). Block membership is computed ONCE from the CURRENT frame's
        pixel coordinates and reused for both frames, so the prev/curr ablation
        compares identical point groups; each frame's kernel is built from its
        own coordinates. Points with nonpositive measured depth are excluded
        from the prior entirely (never clamped).
        """
        cfg = self.cfg
        kernel = KERNEL_BUILDERS[cfg.kernel]
        self._ctx = ctx
        info: dict = {
            "factors": {"prev": [], "curr": []}, "scale_priors": {}, "s_keys": {},
            "counters": {"z_clamps": 0}, "dropped_nonpos_d": 0,
            "blocks_dropped": 0, "n_blocks": 0, "valid_idx": np.empty(0, dtype=np.int64),
        }
        self._info = info

        valid = (self._d["prev"] > 0) & (self._d["curr"] > 0)
        info["dropped_nonpos_d"] = int((~valid).sum())
        if info["dropped_nonpos_d"] > 0:
            Logger.write("warn", f"GP depth prior: {info['dropped_nonpos_d']} points "
                                 "excluded for nonpositive measured depth")
        idx = np.nonzero(valid)[0]
        info["valid_idx"] = idx
        if idx.size < 2:
            return

        blocks = partition_blocks(self._uv["curr"][idx], cfg.block_size, indices=idx)
        info["n_blocks"] = len(blocks)

        roles = {"prev": (ctx.pose_1_key, int(graph.from_idx.cpu().item())),
                 "curr": (ctx.pose_2_key, int(graph.frame_idx.cpu().item()))}
        for role in cfg.frames:
            pose_key, role_frame_idx = roles[role]
            s_key = gtsam.symbol('s', role_frame_idx)
            uv_f, d_f = self._uv[role], self._d[role]
            nugget = self._nugget.get(role, cfg.sigma_n)
            role_factors = []
            for block in blocks:
                block_nugget = nugget[block] if isinstance(nugget, np.ndarray) else nugget
                K_b = kernel(uv_f[block], cfg.length_scale_px, cfg.sigma_f, block_nugget)
                Linv = chol_inv_lower(K_b, base_jitter=cfg.sigma_n ** 2)
                if Linv is None:
                    info["blocks_dropped"] += 1
                    Logger.write("warn", "GP depth prior: block dropped "
                                         "(Cholesky failed after jitter retries)")
                    continue
                factor = make_gp_depth_prior_factor(
                    pose_key, s_key, [ctx.landmark_keys[int(i)] for i in block],
                    Linv, np.log(d_f[block]), cfg.z_min, info["counters"])
                factor_graph.add(factor)
                role_factors.append(factor)
            if role_factors:
                # `s` enters the graph only when a block factor references it, and
                # always with its prior: an unconstrained variable breaks Marginals
                # and (in degenerate cases) the solve itself.
                initial_estimate.insert(s_key, np.zeros(1, dtype=np.float64))
                scale_prior = gtsam.PriorFactorVector(
                    s_key, np.zeros(1, dtype=np.float64),
                    gtsam.noiseModel.Isotropic.Sigma(1, cfg.scale_prior_sigma))
                factor_graph.add(scale_prior)
                info["factors"][role] = role_factors
                info["scale_priors"][role] = scale_prior
                info["s_keys"][role] = s_key

    def on_solved(self, graph: "GTSAM_Pose2Point", factor_graph, result,
                  pose_1, pose_2, landmarks_np: np.ndarray) -> dict:
        info, ctx, cfg = self._info, self._ctx, self.cfg
        if not info or ctx is None:
            return {}
        fields: dict = {
            "dropped_nonpos_d": info["dropped_nonpos_d"],
            "z_clamps": int(info["counters"].get("z_clamps", 0)),
            "blocks_dropped": info["blocks_dropped"],
            "n_blocks": info["n_blocks"],
        }

        # Cost split (factor.error == 0.5 * whitened-norm^2, Huber-rho for the
        # robust point factors). If the prior share is negligible, a null sweep
        # result means nothing — this is the first diagnostic to check.
        fields["cost_points"] = float(sum(f.error(result) for f in ctx.point_factors))
        fields["cost_prior"] = float(sum(
            f.error(result) for fs in info["factors"].values() for f in fs))
        fields["cost_scale_prior"] = float(
            sum(f.error(result) for f in info["scale_priors"].values()))

        marginals = None
        try:
            marginals = gtsam.Marginals(factor_graph, result)
        except Exception as e:  # gtsam raises plain RuntimeError/IndexError here
            Logger.write("warn", f"GP depth prior: Marginals unavailable ({e})")

        idx = info["valid_idx"]
        poses = {"prev": pose_1, "curr": pose_2}
        for role, s_key in info["s_keys"].items():
            s_val = float(result.atVector(s_key)[0])
            fields[f"s_{role}"] = s_val
            if marginals is not None:
                try:
                    fields[f"s_sigma_{role}"] = float(
                        np.sqrt(marginals.marginalCovariance(s_key)[0, 0]))
                except Exception:
                    pass
            # RMS of y - yhat - s over the prior's points: the landmarks' departure
            # from the network's depth field after removing global scale — the
            # direct measure of how much work the prior is doing.
            P = poses[role]
            R = P.rotation().matrix()
            t = np.asarray(P.translation(), dtype=np.float64).reshape(3)
            z = ((landmarks_np[idx] - t) @ R)[:, OPTICAL_AXIS]
            y = np.log(np.clip(z, cfg.z_min, None))
            resid = y - np.log(self._d[role][idx]) - s_val
            fields[f"rms_{role}"] = float(np.sqrt(np.mean(resid ** 2)))
        return fields
