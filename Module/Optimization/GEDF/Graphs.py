"""
Factor graphs registering a frame's keypoints against a G-EDF map.

Residual formulation follows G-EDF-Loc (`include/solver/solver_analitic.hpp`):
each 3D point back-projected from the current frame contributes the scalar
residual r_i = d_hat(T . p_i) - the blended field value at the pose-transformed
point. Out-of-map points (field value >= oob_value_threshold) get a constant
residual (`oob_residual`) with zero Jacobian. The analytic Jacobian in PyPose's
7-wide SE3 layout is  J = [ g^T | -g^T [q]_x | 0 ]  with q = T . p and
g = grad d_hat(q).

`GEDF_ICP` fuses this with the covariance-weighted ICP residual of
`ICP_TwoframePGO` into a 4-row-per-point hybrid graph. While the online map is
not ready (`field.is_ready` false) the field row is inert (zero residual and
Jacobian, unit variance), so the hybrid degrades to pure ICP without any shape
changes mid-run.

The graphs run in the WORLD frame only (the map is world-anchored); there is no
`Local_`-frame variant.

Alignment axis: the camera-to-world mapping is `SE3 o Warp` where the warp is
selected by config (`se3` identity / `sim3` scale / `sl4` projective, see
Alignment.py). The warp absorbs monocular depth bias during the solve; its
parameters get quadratic prior rows appended to the residual, and only the
SE(3) component is written back. sim3/sl4 require the autodiff path (the
`Analytic_*` graphs are SE3-only).
"""
import math
import typing as typ
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import pypose as pp

from Utility.Point import pixel2point_NED
from ..PyposeOptimizers import AnalyticModule, FactorGraph
from ..TwoFramePGO.Graphs import GraphInput, GraphOutput
from .Alignment import SE3Alignment, make_alignment
from .Mapper import GEDFMapProtocol


@dataclass
class GEDF_GraphInput(GraphInput):
    # Optional extra insertion payload (e.g. the previous frame's dense mapping
    # points); the sparse landmarks travel in `points` as usual.
    map_insert_pos_Tw: torch.Tensor | None = None    # (M, 3)
    map_insert_cov_Tw: torch.Tensor | None = None    # (M, 3, 3)
    # Set by get_graph_data (parent side) when a Rerun map snapshot is wanted
    # for this frame; the child attaches it to GEDF_GraphOutput.
    want_map_snapshot: bool = False


@dataclass
class GEDF_GraphOutput(GraphOutput):
    # Near-surface sample of the online map (GEDFMapper.sample_surface), CPU
    # float32 so it pickles cheaply across the parallel-worker result queue.
    # None unless the corresponding GEDF_GraphInput requested a snapshot.
    map_points: torch.Tensor | None = None           # (M, 3)
    map_dist: torch.Tensor | None = None             # (M,)
    # Alignment diagnostics ("estimate + report"): the warp parameters
    # estimated jointly with the pose. The pose in `motion` is always the pure
    # SE(3) component.
    alignment_type: str = "se3"
    alignment_state: torch.Tensor | None = None      # (1,) log_s | (9,) sl4 coeffs, CPU f32
    scale: float | None = None                       # exp(log_s) / exp(4*x5)


class GEDF_Registration(FactorGraph):
    """Pure point-to-distance-field registration (autodiff residual)."""

    def __init__(self, graph_data: GEDF_GraphInput, field: GEDFMapProtocol,
                 field_cfg: SimpleNamespace,
                 alignment_cfg: SimpleNamespace | None = None) -> None:
        super().__init__()
        self.field = field                      # plain attribute: .to() must not re-cast the map
        self.field_cfg = field_cfg
        self.from_idx: torch.Tensor = graph_data.from_idx
        self.frame_idx: torch.Tensor = graph_data.frame_idx
        self.init_motion: pp.LieTensor = graph_data.init_motion

        self.alignment = make_alignment(alignment_cfg, self.init_motion)
        self.edges_index = graph_data.edges_index

        self.points_Tc: torch.Tensor
        self.obs_covTc: torch.Tensor
        self.register_buffer("points_Tc", pixel2point_NED(
            graph_data.observations.data["pixel2_uv"],
            graph_data.observations.data["pixel2_d"].squeeze(-1),
            graph_data.images_intrinsic))
        self.register_buffer("obs_covTc", graph_data.observations.data["obs2_covTc"])

    # -------------------------------------------------------------- #
    def _points_world(self) -> torch.Tensor:
        return self.alignment.act(typ.cast(torch.Tensor, self.points_Tc), self.edges_index)

    def _field_residual(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Field residual (N,) and OOB mask (N,), differentiable in q."""
        cfg = self.field_cfg
        if not self.field.is_ready:
            zeros = q.sum(-1) * 0.0             # keeps the autograd graph shape intact
            return zeros, torch.zeros_like(zeros, dtype=torch.bool)
        dist = self.field.query(q)
        oob = (dist >= cfg.oob_value_threshold).detach()
        r = torch.where(oob, torch.full_like(dist, cfg.oob_residual), dist)
        return r, oob

    @torch.no_grad()
    def _oob_mask(self, q: torch.Tensor) -> torch.Tensor:
        if not self.field.is_ready:
            return torch.zeros(q.shape[0], device=q.device, dtype=torch.bool)
        dist, _ = self.field.query_with_grad(q)
        return dist >= self.field_cfg.oob_value_threshold

    @torch.no_grad()
    def _field_variance(self, q: torch.Tensor, oob: torch.Tensor) -> torch.Tensor:
        """Variance of the scalar field residual per point (N,), in m^2."""
        cfg = self.field_cfg
        if not self.field.is_ready:
            # Field rows are inert during cold start: unit variance.
            return torch.ones(q.shape[:1], device=q.device, dtype=q.dtype)
        sigma_map = self.field.sigma
        floor = max(sigma_map if math.isfinite(sigma_map) else 0.0, cfg.sigma)
        if cfg.weighting == "fixed":
            var = torch.full(q.shape[:1], floor ** 2, device=q.device, dtype=q.dtype)
        else:
            # Mahalanobis: propagate the 3D observation covariance through the
            # field gradient, plus the map-error floor.
            _, g = self.field.query_with_grad(q)
            cov_w = self.alignment.rotate_covariance(
                typ.cast(torch.Tensor, self.obs_covTc).to(q.dtype), self.edges_index)
            var = torch.einsum("ni,nij,nj->n", g, cov_w, g) + floor ** 2
        # OOB rows carry a constant residual and a zero Jacobian; their weight
        # only rescales a constant loss offset.
        return torch.where(oob, torch.ones_like(var), var)

    @torch.no_grad()
    def _field_jacobian(self, q: torch.Tensor, oob: torch.Tensor) -> torch.Tensor:
        """Analytic Jacobian of the field residual, (N, 1, 7)."""
        cfg = self.field_cfg
        N = q.shape[0]
        J = torch.zeros((N, 1, 7), device=q.device, dtype=q.dtype)
        if not self.field.is_ready:
            return J
        _, g = self.field.query_with_grad(q)
        norm = g.norm(dim=-1, keepdim=True)
        g = torch.where(norm > cfg.max_grad_norm, g * (cfg.max_grad_norm / norm), g)
        J[:, 0, 0:3] = g
        J[:, 0, 3:6] = -(g.unsqueeze(-2) @ pp.vec2skew(q)).squeeze(-2)
        J[oob] = 0.0
        return J

    # -------------------------------------------------------------- #
    def forward(self) -> torch.Tensor:
        q = self._points_world()
        r, _ = self._field_residual(q)
        r = r.unsqueeze(-1)                     # (N, 1)
        prior = self.alignment.prior_residual()
        if prior.numel() == 0:
            return r
        return torch.cat([r, prior.unsqueeze(-1).to(r.dtype)], dim=0)   # (N+P, 1)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        q = self._points_world()
        var = self._field_variance(q, self._oob_mask(q)).view(-1, 1, 1)
        prior_var = self.alignment.prior_variance()
        if prior_var.numel() == 0:
            return var
        return torch.cat([var, prior_var.view(-1, 1, 1).to(var.dtype)], dim=0)  # (N+P, 1, 1)

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        return GraphOutput(motion=self.alignment.se3(),
                           frame_idx=self.frame_idx, from_idx=self.from_idx)


class Analytic_GEDF_Registration(GEDF_Registration, AnalyticModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert type(self.alignment) is SE3Alignment, \
            "Analytic GEDF graphs are SE3-only; use autodiff: true for sim3/sl4 alignment"

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        q = self._points_world()
        return self._field_jacobian(q, self._oob_mask(q)).view(-1, 7)


class GEDF_ICP(GEDF_Registration):
    """Hybrid graph: covariance-weighted ICP rows + one field-registration row."""

    def __init__(self, graph_data: GEDF_GraphInput, field: GEDFMapProtocol,
                 field_cfg: SimpleNamespace,
                 alignment_cfg: SimpleNamespace | None = None) -> None:
        super().__init__(graph_data, field, field_cfg, alignment_cfg)
        self.points_Tw: torch.Tensor
        self.pts_covTw: torch.Tensor
        self.register_buffer("points_Tw", graph_data.points.data["pos_Tw"])
        self.register_buffer("pts_covTw", graph_data.points.data["cov_Tw"])

    def forward(self) -> torch.Tensor:
        q = self._points_world()
        icp_r = q - typ.cast(torch.Tensor, self.points_Tw)               # (N, 3)
        field_r, _ = self._field_residual(q)                             # (N,)
        r = torch.cat([icp_r, field_r.unsqueeze(-1)], dim=-1)            # (N, 4)
        prior = self.alignment.prior_residual()                          # (P,)
        if prior.numel() == 0:
            return r
        pad = r.new_zeros((prior.shape[0], 4))
        pad = torch.cat([prior.unsqueeze(-1).to(r.dtype), pad[:, 1:]], dim=-1)
        return torch.cat([r, pad], dim=0)                                # (N+P, 4)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        cov_icp = self.alignment.rotate_covariance(
            typ.cast(torch.Tensor, self.obs_covTc), self.edges_index) \
            + typ.cast(torch.Tensor, self.pts_covTw)                     # (N, 3, 3)

        q = self._points_world()
        var_f = self._field_variance(q, self._oob_mask(q))               # (N,)

        N = q.shape[0]
        cov = torch.zeros((N, 4, 4), device=q.device, dtype=cov_icp.dtype)
        cov[:, :3, :3] = cov_icp
        cov[:, 3, 3] = var_f.to(cov_icp.dtype)

        prior_var = self.alignment.prior_variance()
        if prior_var.numel() == 0:
            return cov
        # Prior blocks diag(1/w, 1, 1, 1): the unit-variance pad entries carry
        # zero residual and zero Jacobian, so they contribute exactly nothing.
        P = prior_var.shape[0]
        pad = torch.eye(4, device=cov.device, dtype=cov.dtype).expand(P, 4, 4).clone()
        pad[:, 0, 0] = prior_var.to(cov.dtype)
        return torch.cat([cov, pad], dim=0)                              # (N+P, 4, 4)


class Analytic_GEDF_ICP(GEDF_ICP, AnalyticModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert type(self.alignment) is SE3Alignment, \
            "Analytic GEDF graphs are SE3-only; use autodiff: true for sim3/sl4 alignment"

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        q = self._points_world()
        N = q.shape[0]

        J_icp = torch.zeros((N, 3, 7), device=q.device, dtype=q.dtype)
        J_icp[..., 0:3] = torch.eye(3, device=q.device, dtype=q.dtype).unsqueeze(0)
        J_icp[..., 3:6] = -pp.vec2skew(q)

        J_field = self._field_jacobian(q, self._oob_mask(q))             # (N, 1, 7)

        return torch.cat([J_icp, J_field], dim=1).view(-1, 7)
