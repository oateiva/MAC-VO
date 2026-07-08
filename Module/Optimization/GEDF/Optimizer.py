"""
`GEDF_PGO`: pose optimization by registering each frame's keypoints against a
G-EDF map that is built online from MAC-VO's own landmarks (or loaded pre-built
from a GDF1 .bin for validation).

Per `_optimize` call (runs in the spawned worker when `parallel: true`):
  1. insert the frame pair's new landmarks into the map (they are anchored at
     the PREVIOUS, already-optimized keyframe pose, so inserting before the
     solve is safe and never double-inserts). With `alignment: sim3` the scale
     estimated by the PREVIOUS solve is applied first (about the previous
     camera center) - those landmarks come from the previous frame's depth,
     whose scale correction is exactly what that solve estimated. This keeps
     the map (and the ICP rows, which share the same tensors) scale-consistent
     with the poses being written back; without it, monocular depth-scale
     drift accumulates in the map as double surfaces.
  2. refit a budgeted number of dirty map cubes,
  3. solve the pose with a two-stage (coarse -> fine robust kernel) LM over the
     selected factor graph, mirroring G-EDF-Loc's two-stage Ceres solve.

Runs in the WORLD frame only (the map is world-anchored) - do not wrap this
optimizer in a `Local_`-style frame transform.
"""
import contextlib
import math
import typing as typ
from types import SimpleNamespace

import torch
import pypose as pp

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion

from Module.Map import VisualMap
from Utility.Timer import Timer
# NOTE: Utility.Visualize imports Module.Map, which imports Module.Optimization;
# importing rr_plt at module level here would close an import cycle. It is
# imported lazily in the (parent-side) methods that need it instead.

from ..Interface import IOptimizer
from ..PyposeOptimizers import AnalyticModule, FactorGraph, LM_analytic
from .Config import GEDFConfig
from .Graphs import (
    Analytic_GEDF_ICP, Analytic_GEDF_Registration,
    GEDF_GraphInput, GEDF_GraphOutput, GEDF_ICP, GEDF_Registration,
)
from .Mapper import GEDFMapper


# Sim(3) feed-forward controller (step 0 of `_optimize`): the scale applied to
# incoming landmarks is a gated, damped state - never a single solve's raw
# estimate. Feeding raw estimates forward is unstable: one diverged solve (or a
# genuine depth-scale transient, e.g. plane_nose[128:140]) shrinks the next
# insertion about the camera center, the shrunken ICP targets pull the next
# estimate lower, and within ~10 frames the map and warp collapse to a point
# (scale -> 1e-3, frozen trajectory).
_ALIGN_FF_ACCEPT = (0.5, 2.0)   # reject solve estimates outside this range
_ALIGN_FF_ALPHA = 0.3           # log-space EMA step for accepted estimates


def _update_align_scale_state(state: float | None, estimate: float | None) -> float | None:
    """Advance the feed-forward scale state with one solve's estimate.

    Non-finite or out-of-accept-range estimates leave the state untouched
    (death-spiral guard); accepted ones move it by a log-space EMA step, so the
    applied correction tracks sustained depth-scale drift but a short transient
    only nudges it."""
    if estimate is None or not math.isfinite(estimate):
        return state
    lo, hi = _ALIGN_FF_ACCEPT
    if not (lo <= estimate <= hi):
        from Utility.PrettyPrint import Logger
        Logger.write("warn", f"GEDF_PGO: sim3 scale estimate {estimate:.4f} outside "
                             f"accept range [{lo}, {hi}]; feed-forward state kept")
        return state
    prev = 1.0 if state is None else state
    return math.exp((1.0 - _ALIGN_FF_ALPHA) * math.log(prev)
                    + _ALIGN_FF_ALPHA * math.log(estimate))


class GEDF_PGO(IOptimizer[GEDF_GraphInput, dict, GEDF_GraphOutput]):
    @torch.no_grad()
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None,
                       edges: torch.Tensor | None = None) -> GEDF_GraphInput:
        frame2opt = global_map.frames[frame_idx]

        obs = global_map.get_frame2match(frame2opt)
        pts = global_map.get_match2point(obs)
        im_intrinsics = frame2opt.data["K"][0]

        lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())
        P1_last = global_map.frames.data["pose"][frame_idx - 1]
        init_motion = pp.SE3(frame2opt.data["pose"])
        baseline = frame2opt.data["baseline"]

        dense_pos: torch.Tensor | None = None
        dense_cov: torch.Tensor | None = None
        if self.config.map.insert_dense and bool((frame_idx > 0).all()):
            prev = global_map.frames[frame_idx - 1]
            dense = global_map.get_frame2map(prev)
            if len(dense) > 0:
                dense_pos = dense.data["pos_Tw"]
                dense_cov = dense.data["cov_Tw"]

        # Rerun map snapshot cadence (parent side knows whether --useRR is on;
        # keeps the child free of any snapshot cost when visualization is off).
        from Utility.Visualize import rr_plt
        viz_every = self.config.viz.every
        want_snapshot = (rr_plt.default_mode == "rerun" and viz_every > 0
                         and int(frame_idx.flatten()[0]) % viz_every == 0)

        return GEDF_GraphInput(
            frame_idx, frame_idx - 1, pp.SE3(P1_last), init_motion, baseline, obs, pts,
            im_intrinsics, edges_idx, "cpu",
            map_insert_pos_Tw=dense_pos, map_insert_cov_Tw=dense_cov,
            want_map_snapshot=want_snapshot,
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        spec: dict = {
            "graph_type": lambda s: s in {"gedf", "gedf+icp"},
            "device": lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "vectorize": lambda b: isinstance(b, bool),
            "parallel": lambda b: isinstance(b, bool),
            "autodiff": lambda b: isinstance(b, bool),
            "map": {
                "source": lambda s: s in {"prebuilt", "online"},
                # "" or None = unset (the yaml loader maps `null` to an empty
                # namespace, so configs should use "" instead). Existence is
                # deliberately NOT checked here (CI sweeps all configs);
                # init_context raises a clear error instead.
                "path": lambda p: p is None or isinstance(p, str),
                "insert_keypoints": lambda b: isinstance(b, bool),
                "insert_dense": lambda b: isinstance(b, bool),
                "min_gaussians": lambda n: isinstance(n, int) and n >= 0,
                # Mapper parameters; fully validated by GEDFConfig.from_namespace
                # in init_context (all keys optional, unknown keys raise).
                "online": lambda ns: ns is None or isinstance(ns, SimpleNamespace),
            },
            "field": {
                "weighting": lambda s: s in {"fixed", "mahalanobis"},
                "sigma": lambda v: isinstance(v, (int, float)) and v > 0,
                "oob_value_threshold": lambda v: isinstance(v, (int, float)),
                "oob_residual": lambda v: isinstance(v, (int, float)),
                "max_grad_norm": lambda v: isinstance(v, (int, float)) and v > 0,
            },
            "solver": {
                "coarse_kernel_delta": lambda v: isinstance(v, (int, float)) and v > 0,
                "coarse_steps": lambda n: isinstance(n, int) and n > 0,
                "fine_kernel_delta": lambda v: isinstance(v, (int, float)) and v > 0,
                "fine_steps": lambda n: isinstance(n, int) and n > 0,
            },
            "viz": {
                "every": lambda n: isinstance(n, int) and n >= 0,   # 0 = disabled
                "iso": lambda v: isinstance(v, (int, float)) and v > 0,
                "resolution": lambda v: isinstance(v, (int, float)) and v > 0,
                "max_points": lambda n: isinstance(n, int) and n > 0,
            },
        }
        # Optional ellipsoid-layer keys (absent = points-only, current behavior).
        viz_ns = getattr(config, "viz", None) if config is not None else None
        if viz_ns is not None:
            optional_viz = {
                "gaussians": lambda b: isinstance(b, bool),
                "n_sigma": lambda v: isinstance(v, (int, float)) and v > 0,
                "max_gaussians": lambda n: isinstance(n, int) and n > 0,
            }
            for key, check in optional_viz.items():
                if hasattr(viz_ns, key):
                    spec["viz"][key] = check
        # Optional alignment block (default se3; existing configs stay valid).
        if config is not None and hasattr(config, "alignment"):
            align_spec: dict = {"type": lambda s: s in {"se3", "sim3", "sl4"}}
            if hasattr(config.alignment, "prior_weight"):
                align_spec["prior_weight"] = lambda v: isinstance(v, (int, float)) and v > 0
            spec["alignment"] = align_spec
        cls._enforce_config_spec(config, spec)

        a_type = getattr(getattr(config, "alignment", None), "type", "se3") \
            if config is not None else "se3"
        if a_type != "se3" and config is not None and not config.autodiff:
            raise ValueError(f"GEDF_PGO alignment '{a_type}' requires autodiff: true "
                             "(analytic Jacobians are SE3-only)")

    @staticmethod
    def init_context(config) -> dict:
        match (config.autodiff, config.graph_type):
            case (True, "gedf"):
                PoseGraphClass = GEDF_Registration
            case (True, "gedf+icp"):
                PoseGraphClass = GEDF_ICP
            case (False, "gedf"):
                PoseGraphClass = Analytic_GEDF_Registration
            case (False, "gedf+icp"):
                PoseGraphClass = Analytic_GEDF_ICP
            case _:
                raise ValueError(f"Graph type of {config.graph_type} is not supported")

        match config.map.source:
            case "prebuilt":
                if not config.map.path:
                    raise ValueError("GEDF_PGO with map.source=prebuilt requires map.path")
                gedf_map = GEDFMapper.from_gdf1(
                    config.map.path,
                    GEDFConfig.from_namespace(getattr(config.map, "online", None)),
                    dtype=torch.float64)
            case "online":
                gedf_map = GEDFMapper(
                    GEDFConfig.from_namespace(getattr(config.map, "online", None)))
            case _:
                raise ValueError(f"Unknown map source {config.map.source}")
        gedf_map.ready_min_gaussians = max(1, config.map.min_gaussians)

        def stage(delta: float, steps: int) -> dict:
            return {
                "optimizer_cfg": {
                    "kernel": Huber(delta=delta),
                    "solver": PINV(),
                    "strategy": TrustRegion(radius=1e3),
                    "corrector": FastTriggs(Huber(delta=delta)),
                    "vectorize": config.vectorize,
                },
                "steps": steps,
            }

        align_ns = getattr(config, "alignment", None)
        alignment_cfg = SimpleNamespace(
            type=getattr(align_ns, "type", "se3") if align_ns is not None else "se3",
            prior_weight=float(getattr(align_ns, "prior_weight", 100.0))
            if align_ns is not None else 100.0)
        if alignment_cfg.type != "se3" and not config.autodiff:
            # defense in depth for programmatically-built configs
            raise ValueError(f"GEDF_PGO alignment '{alignment_cfg.type}' requires autodiff: true "
                             "(analytic Jacobians are SE3-only)")

        return {
            "device": config.device,
            "pose_graph_class": PoseGraphClass,
            "graph_type": config.graph_type,
            "field_cfg": config.field,
            "alignment_cfg": alignment_cfg,
            # Gated + damped sim3 feed-forward state (_update_align_scale_state);
            # applied to each call's landmarks before they reach the map / ICP
            # rows (step 1 of the module docstring). None until the first
            # accepted sim3 estimate.
            "align_scale_prev": None,
            # Normalized so the worker never needs getattr defaults.
            "viz_cfg": SimpleNamespace(
                every=config.viz.every, iso=config.viz.iso,
                resolution=config.viz.resolution, max_points=config.viz.max_points,
                gaussians=bool(getattr(config.viz, "gaussians", False)),
                n_sigma=float(getattr(config.viz, "n_sigma", 1.0)),
                max_gaussians=int(getattr(config.viz, "max_gaussians", 20_000))),
            "map": gedf_map,
            "insert_keypoints": config.map.insert_keypoints,
            "insert_dense": config.map.insert_dense,
            "stages": [
                stage(config.solver.coarse_kernel_delta, config.solver.coarse_steps),
                stage(config.solver.fine_kernel_delta, config.solver.fine_steps),
            ],
        }

    @staticmethod
    def _optimize(context: dict, graph_data: GEDF_GraphInput) -> tuple[dict, GEDF_GraphOutput]:
        gpu_ctx = Timer.GPUTimingContext("GEDF_PGO", torch.cuda.current_stream()) \
            if context["device"] != "cpu" else contextlib.nullcontext()
        with Timer.CPUTimingContext("GEDF_PGO"), gpu_ctx:
            gedf_map: GEDFMapper = context["map"]

            # 0. Apply the previous solve's sim3 scale to this call's landmarks
            #    (module docstring, step 1). They were back-projected from the
            #    previous frame's depth at its optimized pose, so the correction
            #    is a uniform scaling about that camera center. Rebinding the
            #    dict entries is local: `points` comes out of fancy indexing in
            #    VisualMap.get_match2point, never a view of the global map.
            s_prev: float | None = context.get("align_scale_prev")
            if s_prev is not None and context["alignment_cfg"].type == "sim3":
                anchor = pp.SE3(graph_data.from_pose).translation()      # (1, 3)
                if len(graph_data.points) > 0:
                    pos = graph_data.points.data["pos_Tw"]
                    graph_data.points.data["pos_Tw"] = \
                        anchor.to(pos.dtype) + s_prev * (pos - anchor.to(pos.dtype))
                    graph_data.points.data["cov_Tw"] = \
                        (s_prev * s_prev) * graph_data.points.data["cov_Tw"]
                if graph_data.map_insert_pos_Tw is not None:
                    pos = graph_data.map_insert_pos_Tw
                    graph_data.map_insert_pos_Tw = \
                        anchor.to(pos.dtype) + s_prev * (pos - anchor.to(pos.dtype))
                    if graph_data.map_insert_cov_Tw is not None:
                        graph_data.map_insert_cov_Tw = \
                            (s_prev * s_prev) * graph_data.map_insert_cov_Tw

            # 1. Feed the map (landmarks are anchored at the previous, already
            #    optimized pose - see module docstring).
            if not gedf_map.frozen:
                if context["insert_keypoints"] and len(graph_data.points) > 0:
                    gedf_map.insert(graph_data.points.data["pos_Tw"],
                                    graph_data.points.data["cov_Tw"])
                if context["insert_dense"] and graph_data.map_insert_pos_Tw is not None:
                    gedf_map.insert(graph_data.map_insert_pos_Tw,
                                    graph_data.map_insert_cov_Tw)
                # 2. Budgeted refit, prioritizing cubes near the camera.
                cam_pos = pp.SE3(graph_data.init_motion).tensor().reshape(-1)[:3]
                gedf_map.refit(camera_pos=cam_pos)

            # Rerun visualization snapshot (requested by the parent process)
            viz = context["viz_cfg"]
            snap_pts: torch.Tensor | None = None
            snap_dist: torch.Tensor | None = None
            gauss_kw: dict = {}
            if graph_data.want_map_snapshot and gedf_map.is_ready:
                snap_pts, snap_dist = gedf_map.sample_surface(
                    resolution=viz.resolution, iso=viz.iso, max_points=viz.max_points)
                if viz.gaussians:
                    g_mu, g_sig, g_w, g_mae = gedf_map.gaussians(max_gaussians=viz.max_gaussians)
                    gauss_kw = dict(map_gauss_means=g_mu, map_gauss_sigmas=g_sig,
                                    map_gauss_weights=g_w, map_gauss_mae=g_mae)

            # 3. Cold start / degenerate input guards.
            no_obs = len(graph_data.observations) == 0
            if no_obs or (context["graph_type"] == "gedf" and not gedf_map.is_ready):
                return context, GEDF_GraphOutput(motion=pp.SE3(graph_data.init_motion),
                                                 frame_idx=graph_data.frame_idx,
                                                 from_idx=graph_data.from_idx,
                                                 map_points=snap_pts, map_dist=snap_dist,
                                                 alignment_type=context["alignment_cfg"].type,
                                                 **gauss_kw)

            # 4. Two-stage robust LM over the same graph module.
            graph: FactorGraph = context["pose_graph_class"](
                graph_data, field=gedf_map, field_cfg=context["field_cfg"],
                alignment_cfg=context["alignment_cfg"]) \
                .to(device=torch.device(context["device"]), dtype=torch.double)
            assert isinstance(graph, FactorGraph)

            for stage in context["stages"]:
                if isinstance(graph, AnalyticModule):
                    optimizer = LM_analytic(graph, min=1e-6, **stage["optimizer_cfg"])
                else:
                    optimizer = LM(graph, min=1e-6, **stage["optimizer_cfg"])
                scheduler = StopOnPlateau(optimizer, steps=stage["steps"],
                                          patience=2, decreasing=1e-5, verbose=False)
                while scheduler.continual():
                    cov_inv = torch.pinverse(
                        graph.covariance_array().to(context["device"]).double())
                    if not isinstance(graph, AnalyticModule) and cov_inv.shape[-1] == 1:
                        # pypose's RobustModel.normalize_RWJ treats the weight of a
                        # residual with last dim 1 as per-scalar entries - an (N,N)
                        # block-diag matrix would explode into N^2 1x1 blocks there.
                        weight = cov_inv.view(-1, 1)
                    else:
                        weight = torch.block_diag(*cov_inv)
                    loss = optimizer.step(input=(), weight=weight)
                    scheduler.step(loss)

        out = graph.write_back()
        align = typ.cast("GEDF_Registration", graph).alignment
        if context["alignment_cfg"].type == "sim3":
            # feed-forward for the next call's landmark insertion (step 0),
            # gated + damped by _update_align_scale_state; sl4's scale() is a
            # diagnostic, not a uniform warp - never stored.
            context["align_scale_prev"] = _update_align_scale_state(
                context.get("align_scale_prev"), align.scale())
        return context, GEDF_GraphOutput(motion=out.motion, frame_idx=out.frame_idx,
                                         from_idx=out.from_idx,
                                         map_points=snap_pts, map_dist=snap_dist,
                                         alignment_type=context["alignment_cfg"].type,
                                         alignment_state=align.extra_state(),
                                         scale=align.scale(),
                                         **gauss_kw)

    def write_graph_data(self, result: GEDF_GraphOutput | None, global_map: VisualMap) -> None:
        if result is None:
            return
        to_pose = pp.SE3(result.motion[0].data.double().cpu())
        global_map.frames.data["pose"][result.frame_idx] = to_pose.float()

        # Rerun visualization (parent process owns the recording; no-op unless
        # --useRR switched rr_plt into rerun mode).
        from Utility.Visualize import rr_plt
        if rr_plt.default_mode == "rerun" and \
                (result.map_points is not None or result.map_gauss_means is not None
                 or result.scale is not None):
            import rerun as rr
            rr.set_time("frame_idx", sequence=int(result.frame_idx.flatten()[0]))
            if result.map_points is not None:
                rr_plt.log_gedf_map("/world/gedf_map", result.map_points, result.map_dist)
            if result.map_gauss_means is not None:
                rr_plt.log_gedf_gaussians(
                    "/world/gedf_map/gaussians",
                    result.map_gauss_means, result.map_gauss_sigmas,
                    result.map_gauss_weights, result.map_gauss_mae,
                    n_sigma=float(getattr(self.config.viz, "n_sigma", 1.0)))
            if result.scale is not None:
                rr.log("/world/gedf_alignment/scale", rr.Scalars(result.scale))
