"""
Batched per-cube GMM fitting for the online G-EDF mapper.

Pure functions, ported from the G-EDF reference implementation:
  * training targets  <- PURE-mode EDT. In the reference this is a per-voxel
    kd-tree nearest-neighbor distance (`edt_generator.hpp::generateEDT_Pure`);
    here it is computed *exactly at the sample points* with a masked cdist,
    which is equivalent (min distance to the local cloud) and skips the grid.
  * initialization    <- `gaussian_trainer.hpp::initializeSmartGaussians`
    (NMS over local extrema of the distance grid; positive gaussians at maxima,
    negative at minima; random fill as fallback).
  * residual/Jacobian <- `solver/solver.hpp::DynamicGMMCostFunction`
    with q_a = p_a^2 + eps and lambda_a = q_a^2 (the stored "sigma" p is a root):
        pred = sum_k w_k exp(-0.5 * sum_a v_a^2 / q_a^2),   r = (pred - d) * imp
        dr/dw = e * imp;  dr/dmu = w e v / q^2 * imp;  dr/dp = 2 w e v^2 p / q^3 * imp
  * solver            <- hand-rolled batched Levenberg-Marquardt (the reference
    uses Ceres LM with DENSE_NORMAL_CHOLESKY; the normal matrix here is only
    (7K x 7K) per cube, batched over cubes).

Shapes: B = cubes in batch, S = samples, K = gaussians, P = padded local cloud.
Parameter layout per gaussian: [mu_x, mu_y, mu_z, p_x, p_y, p_z, w].
"""
import torch

from .Config import GEDFConfig

_EPS_Q = 1e-9   # matches DynamicGMMCostFunction's eps


# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #
def edt_targets(samples: torch.Tensor, points_pad: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """
    Exact PURE-mode EDT at the sample locations: min distance to the local cloud.
    samples (B,S,3), points_pad (B,P,3), mask (B,P) -> (B,S).
    """
    D = torch.cdist(samples, points_pad)                     # (B, S, P)
    D = D.masked_fill(~mask.unsqueeze(1), torch.inf)
    return D.amin(-1)


def sample_training_data(points_pad: torch.Tensor, mask: torch.Tensor,
                         origins: torch.Tensor, cfg: GEDFConfig,
                         generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw training samples per cube: half jittered around cloud points inside the
    training box (near-surface), half uniform over the training box
    [origin - margin, origin + cube_size + margin]. Returns (X (B,S,3), d (B,S)).
    """
    B = points_pad.shape[0]
    S = cfg.sample_points
    dev, dt = points_pad.device, points_pad.dtype
    box_lo = origins - cfg.margin                            # (B, 3)
    box_hi = origins + cfg.cube_size + cfg.margin

    n_surf = int(S * cfg.surface_sample_frac)
    n_unif = S - n_surf

    # All random draws happen on the CPU generator and are then moved, so the
    # sampling is identical across devices (CPU/CUDA RNG streams differ).
    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(shape, dtype=dt, generator=generator).to(dev)

    uniform = box_lo.unsqueeze(1) + (box_hi - box_lo).unsqueeze(1) * rand(B, n_unif, 3)

    # Surface samples: random source points inside the training box, jittered.
    in_box = mask & (points_pad >= box_lo.unsqueeze(1)).all(-1) \
                  & (points_pad <= box_hi.unsqueeze(1)).all(-1)          # (B, P)
    weights = in_box.cpu().to(dt)
    has_src = in_box.any(-1)                                             # (B,)
    weights = torch.where(has_src.cpu().unsqueeze(-1), weights,
                          torch.ones_like(weights))  # placeholder rows, replaced below
    pick = torch.multinomial(weights, n_surf, replacement=True, generator=generator).to(dev)
    src = torch.gather(points_pad, 1, pick.unsqueeze(-1).expand(-1, -1, 3))
    jitter = (rand(B, n_surf, 3) * 2.0 - 1.0) * cfg.surface_jitter
    surface = (src + jitter).clamp(box_lo.unsqueeze(1), box_hi.unsqueeze(1))
    # Cubes with no in-box source point fall back to uniform samples.
    fallback = box_lo.unsqueeze(1) + (box_hi - box_lo).unsqueeze(1) * rand(B, n_surf, 3)
    surface = torch.where(has_src.view(B, 1, 1), surface, fallback)

    X = torch.cat([surface, uniform], dim=1)                             # (B, S, 3)
    d = edt_targets(X, points_pad, mask)
    return X, d


# --------------------------------------------------------------------------- #
# Initialization (NMS over a coarse distance grid; cold cubes only)
# --------------------------------------------------------------------------- #
def init_nms(points_pad: torch.Tensor, mask: torch.Tensor, origins: torch.Tensor,
             cfg: GEDFConfig, generator: torch.Generator) -> torch.Tensor:
    """
    Port of `initializeSmartGaussians`: build a coarse local distance grid over
    [origin - halo, origin + cube + halo], find local maxima (positive gaussians)
    and minima (negative gaussians) restricted to the cube interior, greedily
    NMS-select K/2 of each with Chebyshev suppression, random-fill the rest.
    Returns theta0 (B, K, 7).
    """
    B = points_pad.shape[0]
    K = cfg.num_gaussians
    dev, dt = points_pad.device, points_pad.dtype
    vox = cfg.init_voxel_size
    side = cfg.cube_size + 2.0 * cfg.halo
    G = max(2 * cfg.nms_radius + 2, int(round(side / vox)))
    r = cfg.nms_radius

    idx1d = torch.arange(G, device=dev)
    iz, iy, ix = torch.meshgrid(idx1d, idx1d, idx1d, indexing="ij")
    coords = torch.stack([ix, iy, iz], dim=-1).reshape(-1, 3)             # (G^3, 3) int voxel idx
    centers_local = (coords.to(dt) + 0.5) * vox - cfg.halo                # (G^3, 3) [x fastest]
    centers = origins.unsqueeze(1) + centers_local.unsqueeze(0)           # (B, G^3, 3)

    dgrid = edt_targets(centers, points_pad, mask)                        # (B, G^3)
    vol = dgrid.reshape(B, 1, G, G, G)                                    # (B,1,z,y,x)

    # Local extrema via max pooling (window 2r+1); ties count as extrema, as in C++.
    kmax = torch.nn.functional.max_pool3d(vol, 2 * r + 1, stride=1, padding=r)
    kmin = -torch.nn.functional.max_pool3d(-vol, 2 * r + 1, stride=1, padding=r)
    is_max = (vol >= kmax).reshape(B, -1)
    is_min = (vol <= kmin).reshape(B, -1)

    # Restrict candidate scan to the cube interior (the C++ scans only the cube).
    interior = ((centers_local >= 0.0) & (centers_local < cfg.cube_size)).all(-1)  # (G^3,)
    finite = torch.isfinite(dgrid)
    pos_cand = is_max & interior & finite & (dgrid > 1e-3)
    neg_cand = is_min & interior & finite

    n_pos = K // 2
    theta0 = torch.zeros((B, K, 7), device=dev, dtype=dt)
    theta0[..., 3:6] = cfg.init_sigma_param

    for b in range(B):
        chosen: list[tuple[int, float]] = []                              # (flat idx, weight)

        def nms_select(cand_mask: torch.Tensor, descending: bool, count: int) -> list[int]:
            idx = torch.nonzero(cand_mask[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                return []
            vals = dgrid[b, idx]
            order = torch.argsort(vals, descending=descending)
            idx = idx[order].tolist()
            picked: list[int] = []
            picked_coords: list[torch.Tensor] = []
            for i in idx:
                if len(picked) >= count:
                    break
                c = coords[i]
                if any(bool((c - pc).abs().max() <= r) for pc in picked_coords):
                    continue
                picked.append(i)
                picked_coords.append(c)
            return picked

        pos_sel = nms_select(pos_cand, descending=True, count=n_pos)
        neg_sel = nms_select(neg_cand, descending=False, count=K - n_pos)

        # Random fill from remaining voxels, mirroring the C++ fallback.
        def random_fill(pool_mask: torch.Tensor, need: int) -> list[int]:
            pool = torch.nonzero(pool_mask[b], as_tuple=False).flatten()
            if pool.numel() == 0 or need <= 0:
                return []
            sel = torch.randint(0, pool.numel(), (need,), generator=generator).to(dev)
            return pool[sel].tolist()

        # Random fill mirrors the C++ fallback: positives from any voxel with
        # d > 0, negatives from remaining minima (near-surface in PURE mode).
        pos_sel += random_fill(finite & interior & (dgrid > 0), n_pos - len(pos_sel))
        neg_sel += random_fill(neg_cand, (K - n_pos) - len(neg_sel))
        neg_sel += random_fill(finite & interior, (K - n_pos) - len(neg_sel))

        chosen = [(i, cfg.init_weight) for i in pos_sel[:n_pos]] \
               + [(i, -cfg.init_weight) for i in neg_sel[:K - n_pos]]
        if len(chosen) == 0:
            continue
        base = list(chosen)
        while len(chosen) < K:                                            # degenerate cubes
            chosen.append(base[len(chosen) % len(base)])

        flat = torch.tensor([i for i, _ in chosen], device=dev, dtype=torch.long)
        w = torch.tensor([wv for _, wv in chosen], device=dev, dtype=dt)
        jit = (torch.rand((K, 3), dtype=dt, generator=generator).to(dev)
               * 0.2 - 0.1) * vox
        theta0[b, :, :3] = centers[b, flat] + jit
        theta0[b, :, 6] = w
    return theta0


# --------------------------------------------------------------------------- #
# Residual / Jacobian / batched LM
# --------------------------------------------------------------------------- #
def gmm_predict(theta: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Field prediction of the fitting model. theta (B,K,7), X (B,S,3) -> (B,S)."""
    mu, p, w = theta[..., :3], theta[..., 3:6], theta[..., 6]
    q = p * p + _EPS_Q                                                    # (B, K, 3)
    v = X.unsqueeze(2) - mu.unsqueeze(1)                                  # (B, S, K, 3)
    z = v / q.unsqueeze(1)
    e = torch.exp(-0.5 * (z * z).sum(-1))                                 # (B, S, K)
    return (w.unsqueeze(1) * e).sum(-1)


def _residual_jacobian(theta: torch.Tensor, X: torch.Tensor, d: torch.Tensor,
                       imp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns r (B,S) and J (B,S,7K) - the analytic Jacobian of solver.hpp."""
    B, S = X.shape[0], X.shape[1]
    K = theta.shape[1]
    mu, p, w = theta[..., :3], theta[..., 3:6], theta[..., 6]
    q = p * p + _EPS_Q                                                    # (B, K, 3)
    v = X.unsqueeze(2) - mu.unsqueeze(1)                                  # (B, S, K, 3)
    z = v / q.unsqueeze(1)
    e = torch.exp(-0.5 * (z * z).sum(-1))                                 # (B, S, K)
    term = w.unsqueeze(1) * e                                             # (B, S, K)

    r = (term.sum(-1) - d) * imp

    a = z / q.unsqueeze(1)                                                # v / q^2
    J_mu = term.unsqueeze(-1) * a                                         # (B, S, K, 3)
    J_p = term.unsqueeze(-1) * a * z * (2.0 * p.unsqueeze(1))             # (B, S, K, 3)
    J_w = e.unsqueeze(-1)                                                 # (B, S, K, 1)
    J = torch.cat([J_mu, J_p, J_w], dim=-1) * imp.view(B, S, 1, 1)
    return r, J.reshape(B, S, K * 7)


def lm_fit(theta0: torch.Tensor, X: torch.Tensor, d: torch.Tensor,
           iters: int, cfg: GEDFConfig) -> torch.Tensor:
    """
    Batched Levenberg-Marquardt with per-cube damping. theta0 (B,K,7),
    X (B,S,3), d (B,S) -> optimized theta (B,K,7).
    """
    B, K = theta0.shape[0], theta0.shape[1]
    n = K * 7
    imp = torch.exp(-5.0 * d.abs()) if cfg.importance_weighting else torch.ones_like(d)

    theta = theta0.clone()
    lam = torch.full((B,), cfg.lm_lambda_init, device=theta.device, dtype=theta.dtype)
    eye = torch.eye(n, device=theta.device, dtype=theta.dtype)

    r, J = _residual_jacobian(theta, X, d, imp)
    cost = (r * r).mean(-1)                                               # (B,)

    for _ in range(iters):
        H = J.transpose(1, 2) @ J                                         # (B, n, n)
        g = J.transpose(1, 2) @ r.unsqueeze(-1)                           # (B, n, 1)
        Hd = H + lam.view(B, 1, 1) * torch.diag_embed(H.diagonal(dim1=1, dim2=2)) \
               + 1e-12 * eye

        L, info = torch.linalg.cholesky_ex(Hd)
        delta = torch.cholesky_solve(-g, L).squeeze(-1)                   # (B, n)
        if bool((info != 0).any()):                                       # singular rows -> lstsq
            bad = info != 0
            delta[bad] = torch.linalg.lstsq(Hd[bad], -g[bad]).solution.squeeze(-1)

        theta_try = theta + delta.reshape(B, K, 7)
        r_try = (gmm_predict(theta_try, X) - d) * imp
        cost_try = (r_try * r_try).mean(-1)

        accept = (cost_try < cost) & torch.isfinite(cost_try)
        theta = torch.where(accept.view(B, 1, 1), theta_try, theta)
        cost = torch.where(accept, cost_try, cost)
        lam = torch.where(accept, lam / cfg.lm_lambda_down, lam * cfg.lm_lambda_up)
        lam = lam.clamp(1e-9, 1e8)

        converged = accept & (delta.abs().amax(-1) < cfg.lm_step_tol)
        if bool(converged.all()):
            break
        r, J = _residual_jacobian(theta, X, d, imp)
    return theta


def fit_batch(points_pad: torch.Tensor, mask: torch.Tensor, origins: torch.Tensor,
              theta_warm: torch.Tensor | None, cfg: GEDFConfig,
              seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit one batch of cubes. `theta_warm` (B,K,7) warm-starts the fit (skipping NMS
    init and using the shorter iteration budget); pass None for cold cubes.

    Returns (theta (B,K,7), mae (B,), std (B,), usable (B,)) in the fit dtype.
    """
    fit_dtype = torch.float64 if cfg.fit_dtype == "float64" else torch.float32
    points_pad = points_pad.to(fit_dtype)
    origins = origins.to(points_pad.device, fit_dtype)

    # CPU generator regardless of compute device: keeps sampling identical
    # across CPU and CUDA (their native RNG streams differ for the same seed).
    generator = torch.Generator()
    generator.manual_seed(seed)

    X, d = sample_training_data(points_pad, mask, origins, cfg, generator)

    if theta_warm is None:
        theta0 = init_nms(points_pad, mask, origins, cfg, generator)
        iters = cfg.lm_iters_cold
    else:
        theta0 = theta_warm.to(fit_dtype)
        iters = cfg.lm_iters_warm

    theta = lm_fit(theta0, X, d, iters, cfg)

    err = (gmm_predict(theta, X) - d).abs()
    mae = err.mean(-1)
    std = err.std(-1)
    usable = torch.isfinite(theta).all(-1).all(-1) & torch.isfinite(mae) \
             & (mae <= cfg.mae_threshold_max)
    return theta, mae, std, usable
