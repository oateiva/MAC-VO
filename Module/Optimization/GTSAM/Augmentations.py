"""
Graph augmentations for GTSAM_Pose2Point.

`GTSAM_Pose2Point` builds exactly the pose-to-point two-frame graph and nothing
else; every optional extension (the G-EDF field factor, the correlated log-depth
prior, per-solve diagnostics) is an object implementing the `GraphAugmentation`
protocol, injected by `GTSAM_Graph.init_context` and invoked at three fixed
lifecycle points:

    on_parse   after parse_graph_data      — stash whatever the extension needs
    on_build   after the landmark loop     — add factors/variables, pre-LM
    on_solved  after the LM solve          — return plain-scalar diagnostics

With no augmentations the built graph is byte-identical to the bare pose2point
path (regression-gated). Augmentation list ORDER is factor insertion order —
preserve it when composing. `on_solved` returns a flat dict of floats/ints that
is merged across augmentations into `GTSAM_GraphOutput.aug_diag` (it crosses an
mp.Queue in parallel mode, so scalars only); key collisions are a bug in the
composition, not handled here.

The protocol is structural (PEP 544): implementations need no base class, and
pyright checks conformance at the injection site.
"""
import math
import torch
import gtsam
import numpy as np
import pypose as pp
import typing as typ
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from Utility.PrettyPrint import Logger
from Utility.GTSAM_Utils import make_gedf_field_factor, pypose_to_pose3

if typ.TYPE_CHECKING:
    from types import SimpleNamespace
    from ..GEDF.Mapper import GEDFMapProtocol
    from .Graphs import GTSAM_Pose2Point, GTSAM_GraphInput


@dataclass
class BuildContext:
    """Keys and bookkeeping the base graph exposes to `on_build`/`on_solved`."""
    pose_1_key: int
    pose_2_key: int
    pose_0_key: Optional[int]
    landmark_keys: list
    point_factors: list      # the base's own pose->point factor handles, in order


@runtime_checkable
class GraphAugmentation(Protocol):
    def on_parse(self, graph: "GTSAM_Pose2Point", data: "GTSAM_GraphInput") -> None: ...
    def on_build(self, graph: "GTSAM_Pose2Point", factor_graph, initial_estimate,
                 ctx: BuildContext) -> None: ...
    def on_solved(self, graph: "GTSAM_Pose2Point", factor_graph, result,
                  pose_1, pose_2, landmarks_np: np.ndarray) -> dict: ...


def make_field_eval(field: "GEDFMapProtocol", field_cfg: "SimpleNamespace"):
    """
    numpy adapter around the G-EDF field for `make_gedf_field_factor`:
    `(N,3) world points -> (residual (N,), gradient (N,3))` with the same
    semantics as the pypose GEDF graphs — OOB points (d_hat >=
    oob_value_threshold) get the constant `oob_residual` and a zero gradient,
    and the per-point gradient norm is clamped to `max_grad_norm`.
    """
    def field_eval(q_np: np.ndarray):
        q = torch.from_numpy(np.ascontiguousarray(q_np, dtype=np.float64))
        dist, grad = field.query_with_grad(q)
        oob = dist >= field_cfg.oob_value_threshold
        r = torch.where(oob, torch.full_like(dist, field_cfg.oob_residual), dist)
        norm = grad.norm(dim=-1, keepdim=True)
        grad = torch.where(norm > field_cfg.max_grad_norm,
                           grad * (field_cfg.max_grad_norm / norm), grad)
        grad = torch.where(oob.unsqueeze(-1), torch.zeros_like(grad), grad)
        return (r.detach().cpu().numpy(), grad.detach().cpu().numpy())
    return field_eval


class GEDFField:
    """
    G-EDF scan-to-map factor (graph_type "pose2point+gedf"): one batched unary
    factor on the CURRENT pose, residual d_hat(T . p_i) per keypoint, joining
    the pose->point ("GTSAM ICP") solve. Inert while the map is not ready. The
    field factor acts on the RAW camera points (no alignment warp) — the
    Optimizer enforces se3-only for this graph type.
    """
    def __init__(self, field: "GEDFMapProtocol", field_cfg: "SimpleNamespace"):
        self.field = field
        self.field_cfg = field_cfg

    def on_parse(self, graph: "GTSAM_Pose2Point", data: "GTSAM_GraphInput") -> None:
        pass

    def on_build(self, graph: "GTSAM_Pose2Point", factor_graph, initial_estimate,
                 ctx: BuildContext) -> None:
        if not self.field.is_ready:
            return
        field_eval = make_field_eval(self.field, self.field_cfg)
        map_sigma = self.field.sigma
        floor = max(map_sigma if math.isfinite(map_sigma) else 0.0,
                    float(self.field_cfg.sigma))
        if getattr(self.field_cfg, "weighting", "fixed") == "mahalanobis":
            # Per-point variance via the field gradient at the initial pose
            # (linearization-point approximation of the pypose backend's
            # per-iteration reweighting).
            R0 = graph.init_pose.rotation().matrix()
            t0 = np.asarray(graph.init_pose.translation(), dtype=np.float64).reshape(3)
            _, g0 = field_eval(graph.obs_Tc_2 @ R0.T + t0)
            cov_w = np.einsum("ij,njk,lk->nil", R0, graph.obs2_covTc, R0)
            var = np.einsum("ni,nij,nj->n", g0, cov_w, g0) + floor ** 2
            field_noise = gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(var))
        else:
            field_noise = gtsam.noiseModel.Isotropic.Sigma(len(graph.obs_Tc_2), floor)
        factor_graph.add(make_gedf_field_factor(
            ctx.pose_2_key, graph.obs_Tc_2, field_eval, field_noise))

    def on_solved(self, graph: "GTSAM_Pose2Point", factor_graph, result,
                  pose_1, pose_2, landmarks_np: np.ndarray) -> dict:
        return {}


class MotionPrior:
    """
    Soft constant-velocity prior: one BetweenFactorPose3 from pose_1 to pose_2
    whose measurement is the PREVIOUS pair's optimized relative motion. A soft
    factor, deliberately not an extrapolated initialization — the learningUAVO
    ledger measured the soft form at −35 % APE while velocity-coasting through
    measurement gaps was 4–8x worse than freezing. Skipped on the first pair
    (no k−2 pose to derive a velocity from).

    `trans_sigma` is in the VO's own translation units (mono: scale-inflated
    metres); rotation sigma = rot_mult * trans_sigma in radians (their sweep
    found rot_mult != 2 regresses).
    """
    def __init__(self, trans_sigma: float, rot_mult: float = 2.0):
        self.trans_sigma = float(trans_sigma)
        self.rot_mult = float(rot_mult)
        self._pred: "gtsam.Pose3 | None" = None
        self._factor = None

    def on_parse(self, graph: "GTSAM_Pose2Point", data: "GTSAM_GraphInput") -> None:
        self._pred, self._factor = None, None
        if int(data.previous_graph_data.from_idx) < 0:
            return
        p_km2 = pp.SE3(data.previous_graph_data.from_pose.detach().cpu())
        p_km1 = pp.SE3(data.current_graph_data.from_pose.detach().cpu())
        self._pred = pypose_to_pose3(typ.cast(pp.LieTensor, p_km2.Inv() @ p_km1))

    def on_build(self, graph: "GTSAM_Pose2Point", factor_graph, initial_estimate,
                 ctx: BuildContext) -> None:
        if self._pred is None:
            return
        sigmas = np.array([self.rot_mult * self.trans_sigma] * 3
                          + [self.trans_sigma] * 3, dtype=np.float64)
        self._factor = gtsam.BetweenFactorPose3(
            ctx.pose_1_key, ctx.pose_2_key, self._pred,
            gtsam.noiseModel.Diagonal.Sigmas(sigmas))
        factor_graph.add(self._factor)

    def on_solved(self, graph: "GTSAM_Pose2Point", factor_graph, result,
                  pose_1, pose_2, landmarks_np: np.ndarray) -> dict:
        if self._factor is None:
            return {}
        return {"cost_motion_prior": float(self._factor.error(result))}


class SolveDiagnostics:
    """
    Per-solve observability diagnostics, independent of any prior being active:
    median parallax angle between the two observation rays (at the optimized
    poses), the number of points the Huber kernels down-weight at the solution,
    and the point count. Attached whenever a diagnostics CSV is requested, so
    prior-OFF arms of an on/off comparison stratify identically to prior-ON.
    """
    def on_parse(self, graph: "GTSAM_Pose2Point", data: "GTSAM_GraphInput") -> None:
        pass

    def on_build(self, graph: "GTSAM_Pose2Point", factor_graph, initial_estimate,
                 ctx: BuildContext) -> None:
        pass

    def on_solved(self, graph: "GTSAM_Pose2Point", factor_graph, result,
                  pose_1, pose_2, landmarks_np: np.ndarray) -> dict:
        n = graph.obs_Tc_1.shape[0]
        if n == 0:
            return {"n_points": 0}

        R1 = pose_1.rotation().matrix()
        R2 = pose_2.rotation().matrix()
        t1 = np.asarray(pose_1.translation(), dtype=np.float64).reshape(3)
        t2 = np.asarray(pose_2.translation(), dtype=np.float64).reshape(3)

        # Low parallax = the along-ray position is set by the depth measurement
        # alone; this is the stratification axis for depth-prior experiments.
        r1_w = graph.obs_Tc_1 @ R1.T
        r2_w = graph.obs_Tc_2 @ R2.T
        cos = (r1_w * r2_w).sum(-1) / (
            np.linalg.norm(r1_w, axis=-1) * np.linalg.norm(r2_w, axis=-1) + 1e-300)
        parallax = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

        # Whitened norm above delta on either observation factor. (Robust noise
        # models expose no accessors in this binding — recomputed from the
        # stashed covariances.)
        res1 = (landmarks_np - t1) @ R1 - graph.obs_Tc_1
        res2 = (landmarks_np - t2) @ R2 - graph.obs_Tc_2
        m1 = np.sqrt(np.einsum("ni,ni->n", res1,
                               np.linalg.solve(graph.obs1_covTc, res1[..., None])[..., 0]))
        m2 = np.sqrt(np.einsum("ni,ni->n", res2,
                               np.linalg.solve(graph.obs2_covTc, res2[..., None])[..., 0]))
        return {
            "n_points": int(n),
            "median_parallax_deg": float(np.median(parallax)),
            "huber_rejects": int(((m1 > graph.huber_delta) | (m2 > graph.huber_delta)).sum()),
        }
