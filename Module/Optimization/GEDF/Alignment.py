"""
Alignment axis for the GEDF factor graphs.

The camera-to-world mapping is modeled as `SE3 o Warp`, where Warp acts on the
camera-frame points and is one of:

  * se3  - identity (today's behavior, bit-identical),
  * sim3 - uniform scaling `exp(log_s) * p` (absorbs per-frame monocular
           depth-scale bias),
  * sl4  - a small projective warp `dehomog(matrix_exp(sum_i x_i E_i) @ homog(p))`
           over a 9-dim complement basis of sl(4) disjoint from se(3)
           (additionally absorbs affine/projective depth bias).

Only the SE(3) factor is ever written back to the map (the pose schema is a
7-float SE3); the warp parameters are estimated jointly during the solve and
reported through `GEDF_GraphOutput` for diagnostics. For sim3 the estimated
scale is additionally fed forward by GEDF_PGO into the NEXT call's landmark
insertion (scaling about the previous camera center), keeping the online map
scale-consistent with the corrected registration - see Optimizer.py, step 0.

The warp parameters carry a quadratic prior (rows appended to the graph
residual) pulling them toward identity; `prior_weight` is the information
(1 / sigma^2) of that prior.

NOTE on parameter ordering: `pose2opt` MUST be registered before `extras`.
pypose's `modjac` (Jacobian column order) and `_Optimizer.update_parameter`
(step split by numel) both follow nn.Module registration order - swapping them
would silently mis-route the LM step.
"""
import typing as typ
from types import SimpleNamespace

import torch
import torch.nn as nn
import pypose as pp


def _sl4_complement_basis() -> torch.Tensor:
    """
    A basis of the 9-dim complement of se(3) inside sl(4) (traceless 4x4):
    E0..E2 shears (symmetric off-diagonal in the 3x3 block), E3..E4 anisotropic
    volume-preserving scalings, E5 isotropic scale (diag(1,1,1,-3); the
    dehomogenized scale is exp(4*x5)), E6..E8 projective elations (bottom row).
    Together with se(3)'s 6 generators (skew 3x3 + translation column) these
    span all of sl(4).
    """
    E = torch.zeros((9, 4, 4), dtype=torch.float64)
    for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):    # shears
        E[k, i, j] = 1.0
        E[k, j, i] = 1.0
    E[3, 0, 0], E[3, 1, 1] = 1.0, -1.0                        # anisotropic scalings
    E[4, 0, 0], E[4, 1, 1], E[4, 2, 2] = 1.0, 1.0, -2.0
    E[5, 0, 0] = E[5, 1, 1] = E[5, 2, 2] = 1.0                # isotropic scale
    E[5, 3, 3] = -3.0
    for k, i in enumerate([0, 1, 2]):                         # projective elations
        E[6 + k, 3, i] = 1.0
    return E


class SE3Alignment(nn.Module):
    """Identity warp: plain SE(3) pose optimization (default)."""
    n_extra: int = 0

    def __init__(self, init_motion: pp.LieTensor, prior_weight: float = 100.0) -> None:
        super().__init__()
        # pose2opt registered FIRST - ordering is load-bearing (module docstring)
        self.pose2opt = pp.Parameter(pp.SE3(init_motion))
        self.prior_weight = float(prior_weight)

    # ------------------------------------------------------------- hot path
    def warp(self, points_Tc: torch.Tensor) -> torch.Tensor:
        return points_Tc

    def act(self, points_Tc: torch.Tensor, edges_index: torch.Tensor) -> torch.Tensor:
        frame_pose = typ.cast(pp.LieTensor, self.pose2opt[edges_index])
        return frame_pose.Act(self.warp(points_Tc))

    @torch.no_grad()
    def rotate_covariance(self, cov_Tc: torch.Tensor, edges_index: torch.Tensor) -> torch.Tensor:
        frame_pose = typ.cast(pp.LieTensor, self.pose2opt[edges_index])
        R = frame_pose.rotation().matrix()
        return self._cov_scale() * (R @ cov_Tc @ R.transpose(-2, -1))

    def _cov_scale(self) -> float | torch.Tensor:
        return 1.0

    # ---------------------------------------------------------- prior rows
    def _extras(self) -> torch.Tensor | None:
        return None

    def prior_residual(self) -> torch.Tensor:
        """(P,) differentiable prior rows pulling the warp toward identity."""
        extras = self._extras()
        if extras is None:
            return self.pose2opt.new_zeros((0,))
        return extras

    def prior_variance(self) -> torch.Tensor:
        """(P,) variances of the prior rows (= 1 / prior_weight)."""
        extras = self._extras()
        if extras is None:
            return self.pose2opt.new_zeros((0,))
        return torch.full_like(extras.detach(), 1.0 / self.prior_weight)

    # ----------------------------------------------------------- reporting
    def se3(self) -> pp.LieTensor:
        """The SE(3) component written back to the map."""
        return self.pose2opt

    def extra_state(self) -> torch.Tensor | None:
        extras = self._extras()
        if extras is None:
            return None
        return extras.detach().float().cpu().clone()

    @torch.no_grad()
    def load_extra_state(self, state: torch.Tensor | None) -> None:
        """Seed the warp parameters from another alignment's extra_state()
        (the ICP->field warp hand-off of graph_type "icp->gedf")."""
        extras = self._extras()
        if state is None or extras is None:
            return
        extras.copy_(state.to(dtype=extras.dtype, device=extras.device))

    def scale(self) -> float | None:
        return None


class Sim3Alignment(SE3Alignment):
    """Uniform scale warp: `act(p) = SE3.Act(exp(log_s) * p)`.

    A monocular depth over-estimate by factor k is recovered as log_s = -log k;
    the reported `scale()` = exp(log_s) is the multiplicative correction applied
    to the measured camera points."""
    n_extra = 1

    def __init__(self, init_motion: pp.LieTensor, prior_weight: float = 100.0) -> None:
        super().__init__(init_motion, prior_weight)
        self.extras = nn.Parameter(torch.zeros(1, dtype=torch.float64))   # log_s

    def warp(self, points_Tc: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.extras.to(points_Tc.dtype)) * points_Tc

    def _cov_scale(self) -> torch.Tensor:
        # Sigma_obs is measured on the raw camera points; the residual acts on
        # s * p, so the covariance pushforward picks up s^2.
        return torch.exp(2.0 * self.extras).detach()

    def _extras(self) -> torch.Tensor | None:
        return self.extras

    def scale(self) -> float | None:
        return float(torch.exp(self.extras.detach()))


class SL4Alignment(SE3Alignment):
    """Projective warp over the 9-dim sl(4) complement of se(3) (experimental).

    Covariance rotation uses the SE(3) rotation only (the warp Jacobian is
    I + O(|x|) under the prior - a documented approximation)."""
    n_extra = 9

    basis: torch.Tensor

    def __init__(self, init_motion: pp.LieTensor, prior_weight: float = 100.0) -> None:
        super().__init__(init_motion, prior_weight)
        self.register_buffer("basis", _sl4_complement_basis())
        self.extras = nn.Parameter(torch.zeros(9, dtype=torch.float64))

    def warp(self, points_Tc: torch.Tensor) -> torch.Tensor:
        basis = self.basis.to(points_Tc.dtype)
        W = torch.matrix_exp((self.extras.to(points_Tc.dtype).view(9, 1, 1) * basis).sum(0))
        ph = torch.cat([points_Tc, torch.ones_like(points_Tc[:, :1])], dim=-1) @ W.mT
        # w ~= 1 near the (prior-held) operating point; the clamp never bites in
        # normal operation but keeps a degenerate step from producing NaN loss
        # (which would corrupt the LM accept/reject comparison).
        return ph[:, :3] / ph[:, 3:4].clamp_min(0.25)

    def _extras(self) -> torch.Tensor | None:
        return self.extras

    def scale(self) -> float | None:
        # E5 = diag(1,1,1,-3): dehomogenized isotropic scale = exp(4 * x5)
        return float(torch.exp(4.0 * self.extras.detach()[5]))


_ALIGNMENTS: dict[str, type[SE3Alignment]] = {
    "se3": SE3Alignment,
    "sim3": Sim3Alignment,
    "sl4": SL4Alignment,
}


def make_alignment(alignment_cfg: SimpleNamespace | None,
                   init_motion: pp.LieTensor) -> SE3Alignment:
    a_type = getattr(alignment_cfg, "type", "se3") if alignment_cfg is not None else "se3"
    prior_weight = float(getattr(alignment_cfg, "prior_weight", 100.0)) \
        if alignment_cfg is not None else 100.0
    if a_type not in _ALIGNMENTS:
        raise ValueError(f"Unknown alignment type '{a_type}'; expected one of {sorted(_ALIGNMENTS)}")
    return _ALIGNMENTS[a_type](init_motion, prior_weight=prior_weight)
