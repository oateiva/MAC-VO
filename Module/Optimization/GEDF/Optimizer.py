"""
`GEDF_PGO`: pose optimization by registering each frame's keypoints against a
G-EDF map that is built online from MAC-VO's own landmarks (or loaded pre-built
from a GDF1 .bin for validation).

Per `_optimize` call (runs in the spawned worker when `parallel: true`):
  1. insert the frame pair's new landmarks into the map (they are anchored at
     the PREVIOUS, already-optimized keyframe pose, so inserting before the
     solve is safe and never double-inserts),
  2. refit a budgeted number of dirty map cubes,
  3. solve the pose with a two-stage (coarse -> fine robust kernel) LM over the
     selected factor graph, mirroring G-EDF-Loc's two-stage Ceres solve.

Runs in the WORLD frame only (the map is world-anchored) - do not wrap this
optimizer in a `Local_`-style frame transform.
"""
import contextlib
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

from ..Interface import IOptimizer
from ..PyposeOptimizers import AnalyticModule, FactorGraph, LM_analytic
from ..TwoFramePGO.Graphs import GraphOutput
from .Config import GEDFConfig
from .Graphs import (
    Analytic_GEDF_ICP, Analytic_GEDF_Registration,
    GEDF_GraphInput, GEDF_ICP, GEDF_Registration,
)
from .Mapper import GEDFMapper


class GEDF_PGO(IOptimizer[GEDF_GraphInput, dict, GraphOutput]):
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

        return GEDF_GraphInput(
            frame_idx, frame_idx - 1, pp.SE3(P1_last), init_motion, baseline, obs, pts,
            im_intrinsics, edges_idx, "cpu",
            map_insert_pos_Tw=dense_pos, map_insert_cov_Tw=dense_cov,
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
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
        })

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

        return {
            "device": config.device,
            "pose_graph_class": PoseGraphClass,
            "graph_type": config.graph_type,
            "field_cfg": config.field,
            "map": gedf_map,
            "insert_keypoints": config.map.insert_keypoints,
            "insert_dense": config.map.insert_dense,
            "stages": [
                stage(config.solver.coarse_kernel_delta, config.solver.coarse_steps),
                stage(config.solver.fine_kernel_delta, config.solver.fine_steps),
            ],
        }

    @staticmethod
    def _optimize(context: dict, graph_data: GEDF_GraphInput) -> tuple[dict, GraphOutput]:
        gpu_ctx = Timer.GPUTimingContext("GEDF_PGO", torch.cuda.current_stream()) \
            if context["device"] != "cpu" else contextlib.nullcontext()
        with Timer.CPUTimingContext("GEDF_PGO"), gpu_ctx:
            gedf_map: GEDFMapper = context["map"]

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

            # 3. Cold start / degenerate input guards.
            no_obs = len(graph_data.observations) == 0
            if no_obs or (context["graph_type"] == "gedf" and not gedf_map.is_ready):
                return context, GraphOutput(motion=pp.SE3(graph_data.init_motion),
                                            frame_idx=graph_data.frame_idx,
                                            from_idx=graph_data.from_idx)

            # 4. Two-stage robust LM over the same graph module.
            graph: FactorGraph = context["pose_graph_class"](
                graph_data, field=gedf_map, field_cfg=context["field_cfg"]) \
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

        return context, graph.write_back()

    def write_graph_data(self, result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None:
            return
        to_pose = pp.SE3(result.motion[0].data.double().cpu())
        global_map.frames.data["pose"][result.frame_idx] = to_pose.float()
