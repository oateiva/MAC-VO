import torch
import pypose as pp
from types import SimpleNamespace

from Module.Map import VisualMap
from Utility.Timer import Timer

from ..Interface import IOptimizer
from ..PyposeOptimizers import FactorGraph
from ..TwoFramePGO.Graphs import GraphInput
from .Graphs import GTSAM_GraphInput, GTSAM_GraphOutput, GTSAM_Pose2Point, ISAM


class GTSAM_Optimizer():
    def __init__(self, model: FactorGraph):
        self.model = model

    @torch.no_grad()
    def step(self, input=None, target=None):
        # Run GTSAM optimization
        return self.model.run_gtsam_optimization()


class GTSAM_Graph(IOptimizer[GTSAM_GraphInput, dict, GTSAM_GraphOutput]):
    def __init__(self, config):
        super().__init__(config)
        self.window_size = 2
    def connect_graphs(self, previous_graph_data: GraphInput, current_graph_data: GraphInput) -> GTSAM_GraphInput:
        matches_prev = previous_graph_data.observations
        matches_curr = current_graph_data.observations

        match_atol = float(getattr(self.config, "match_atol", 1.0))
        # Vectorized all-pairs association (same semantics as the original
        # per-keypoint loop: for each current pixel1_uv, the FIRST index in
        # matches_prev whose pixel2_uv is within atol on both axes).
        indexes_prev_curr = []
        prev_uv = matches_prev.data["pixel2_uv"]   # (Np, 2)
        curr_uv = matches_curr.data["pixel1_uv"]   # (Nc, 2)
        if prev_uv.shape[0] > 0 and curr_uv.shape[0] > 0:
            close = torch.isclose(
                prev_uv.unsqueeze(0), curr_uv.unsqueeze(1), atol=match_atol
            ).all(dim=-1)                          # (Nc, Np)
            has_match = close.any(dim=1)
            first_prev = torch.argmax(close.to(torch.uint8), dim=1)
            indexes_prev_curr = [
                (int(p), int(c)) for c, p in
                zip(torch.nonzero(has_match, as_tuple=True)[0], first_prev[has_match])
            ]

        return GTSAM_GraphInput(previous_graph_data,
            current_graph_data,
            indexes_prev_curr,
        )

    @torch.no_grad()
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> GTSAM_GraphInput:
        frame2opt_last = global_map.frames[frame_idx]

        # Last frame
        obs_last = global_map.get_frame2match(frame2opt_last)
        pts_last = global_map.get_match2point(obs_last)
        im_intrinsics = frame2opt_last.data["K"][0]

        lengths_last = global_map.frame2match.ranges[frame2opt_last.index, :, 1].flatten()
        lengths_last = lengths_last[lengths_last >= 0]
        edges_idx_last = torch.repeat_interleave(torch.arange(lengths_last.size(0)), lengths_last.long())
        init_motion_last = pp.SE3(frame2opt_last.data["pose"])
        baseline = frame2opt_last.data["baseline"]

        P1_last = global_map.frames.data["pose"][frame_idx - 1]

        GI_last = GraphInput(
            frame_idx,
            frame_idx - 1,
            P1_last,
            init_motion_last,
            baseline,
            obs_last,
            pts_last,
            im_intrinsics,
            edges_idx_last,
            "cpu"
            )

        # Previous frame
        frame2opt_prev = global_map.frames[frame_idx - 1]
        obs_prev = global_map.get_frame2match(frame2opt_prev)
        pts_prev = global_map.get_match2point(obs_prev)
        lengths_prev = global_map.frame2match.ranges[frame2opt_prev.index, :, 1].flatten()
        lengths_prev = lengths_prev[lengths_prev >= 0]
        edges_idx_prev = torch.repeat_interleave(torch.arange(lengths_prev.size(0)), lengths_prev.long())
        init_motion_prev = pp.SE3(frame2opt_prev.data["pose"])

        P1_prev = global_map.frames.data["pose"][frame_idx - 2]

        GI_prev = GraphInput(
            frame_idx - 1,
            frame_idx - 2,
            P1_prev,
            init_motion_prev,
            baseline,
            obs_prev,
            pts_prev,
            im_intrinsics,
            edges_idx_prev,
            "cpu"
            )

        gi = self.connect_graphs(GI_prev, GI_last)
        if self.config.graph_type == "pose2point+gedf":
            # Rerun map snapshot cadence (parent side; mirrors GEDF_PGO)
            from Utility.Visualize import rr_plt
            viz_every = int(getattr(getattr(self.config.gedf, "viz", None), "every", 0))
            gi.want_map_snapshot = (rr_plt.default_mode == "rerun" and viz_every > 0
                                    and int(frame_idx.flatten()[0]) % viz_every == 0)
        return gi


    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        spec: dict = {
            "graph_type": lambda s: s in {"pose2point", "isam", "pose2point+gedf"},
            "device": lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "vectorize": lambda b: isinstance(b, bool),
            "parallel": lambda b: isinstance(b, bool),
            "autodiff": lambda b: isinstance(b, bool)
        }
        # G-EDF hybrid: the "gedf" block configures the online map and the
        # field factor (mirrors GEDF_PGO's map/field/viz sub-configs, minus
        # insert_dense — the GTSAM input carries no dense payload).
        if config is not None and hasattr(config, "gedf"):
            gedf_spec: dict = {
                "map": {
                    "source": lambda s: s in {"prebuilt", "online"},
                    "path": lambda p: p is None or isinstance(p, str),
                    "insert_keypoints": lambda b: isinstance(b, bool),
                    "min_gaussians": lambda n: isinstance(n, int) and n >= 0,
                    "online": lambda ns: ns is None or isinstance(ns, SimpleNamespace),
                },
                "field": {
                    "weighting": lambda s: s in {"fixed", "mahalanobis"},
                    "sigma": lambda v: isinstance(v, (int, float)) and v > 0,
                    "oob_value_threshold": lambda v: isinstance(v, (int, float)),
                    "oob_residual": lambda v: isinstance(v, (int, float)),
                    "max_grad_norm": lambda v: isinstance(v, (int, float)) and v > 0,
                },
            }
            if hasattr(config.gedf, "viz"):
                viz_spec: dict = {
                    "every": lambda n: isinstance(n, int) and n >= 0,
                    "iso": lambda v: isinstance(v, (int, float)) and v > 0,
                    "resolution": lambda v: isinstance(v, (int, float)) and v > 0,
                    "max_points": lambda n: isinstance(n, int) and n > 0,
                }
                optional_viz = {
                    "gaussians": lambda b: isinstance(b, bool),
                    "n_sigma": lambda v: isinstance(v, (int, float)) and v > 0,
                    "max_gaussians": lambda n: isinstance(n, int) and n > 0,
                    "max_sigma": lambda v: isinstance(v, (int, float)),
                    "cubes": lambda b: isinstance(b, bool),
                }
                for key, check in optional_viz.items():
                    if hasattr(config.gedf.viz, key):
                        viz_spec[key] = check
                gedf_spec["viz"] = viz_spec
            spec["gedf"] = gedf_spec
        if config is not None and config.graph_type == "pose2point+gedf" \
                and not hasattr(config, "gedf"):
            raise ValueError("graph_type pose2point+gedf requires a 'gedf' config block")
        # Optional pose2point hyperparameters (defaults in GTSAM_Pose2Point /
        # connect_graphs reproduce the historical hardcoded values).
        optional_spec = {
            "huber_delta"     : lambda v: isinstance(v, (int, float)) and v > 0.,
            "huber_delta_prev": lambda v: isinstance(v, (int, float)) and v > 0.,
            "prior_sigma"     : lambda v: isinstance(v, (int, float)) and v > 0.,
            "max_iterations"  : lambda n: isinstance(n, int) and n > 0,
            "match_atol"      : lambda v: isinstance(v, (int, float)) and v > 0.,
        }
        for key, check in optional_spec.items():
            if config is not None and hasattr(config, key):
                spec[key] = check
        # Optional alignment block (default se3; mirrors GEDF_PGO's axis).
        if config is not None and hasattr(config, "alignment"):
            align_spec: dict = {"type": lambda s: s in {"se3", "sim3", "sl4"}}
            if hasattr(config.alignment, "prior_weight"):
                align_spec["prior_weight"] = lambda v: isinstance(v, (int, float)) and v > 0
            spec["alignment"] = align_spec
        cls._enforce_config_spec(config, spec)

        a_type = getattr(getattr(config, "alignment", None), "type", "se3") \
            if config is not None else "se3"
        if a_type != "se3" and config is not None and config.graph_type != "pose2point":
            # pose2point+gedf included: the field factor acts on the RAW camera
            # points, so a warp would desynchronize the two factor families.
            raise ValueError(f"GTSAM_Graph alignment '{a_type}' is only supported by "
                             "graph_type: pose2point (isam and pose2point+gedf stay SE3-only)")

    @staticmethod
    def init_context(config) -> dict:
        align_ns = getattr(config, "alignment", None)
        alignment_type = getattr(align_ns, "type", "se3") if align_ns is not None else "se3"
        alignment_prior_weight = float(getattr(align_ns, "prior_weight", 100.0)) \
            if align_ns is not None else 100.0
        if alignment_type != "se3" and config.graph_type != "pose2point":
            # defense in depth for programmatically-built configs
            raise ValueError(f"GTSAM_Graph alignment '{alignment_type}' is only supported by "
                             "graph_type: pose2point (isam stays SE3-only)")

        # G-EDF map + field config for the hybrid (mirrors GEDF_PGO.init_context)
        gedf_map = None
        gedf_cfg = None
        if config.graph_type == "pose2point+gedf":
            from ..GEDF.Config import GEDFConfig
            from ..GEDF.Mapper import GEDFMapper
            gedf_cfg = config.gedf
            match gedf_cfg.map.source:
                case "prebuilt":
                    if not gedf_cfg.map.path:
                        raise ValueError("pose2point+gedf with map.source=prebuilt requires map.path")
                    gedf_map = GEDFMapper.from_gdf1(
                        gedf_cfg.map.path,
                        GEDFConfig.from_namespace(getattr(gedf_cfg.map, "online", None)),
                        dtype=torch.float64)
                case "online":
                    gedf_map = GEDFMapper(
                        GEDFConfig.from_namespace(getattr(gedf_cfg.map, "online", None)))
                case _:
                    raise ValueError(f"Unknown map source {gedf_cfg.map.source}")
            gedf_map.ready_min_gaussians = max(1, gedf_cfg.map.min_gaussians)

        match (config.graph_type):
            case "pose2point" | "pose2point+gedf":
                PoseGraphClass = lambda: GTSAM_Pose2Point(
                    huber_delta=float(getattr(config, "huber_delta", 0.1)),
                    huber_delta_prev=float(getattr(config, "huber_delta_prev", 1.0)),
                    prior_sigma=float(getattr(config, "prior_sigma", 1e-4)),
                    max_iterations=int(getattr(config, "max_iterations", 20)),
                    alignment_type=alignment_type,
                    alignment_prior_weight=alignment_prior_weight,
                    field=gedf_map,
                    field_cfg=gedf_cfg.field if gedf_cfg is not None else None,
                )
            case ("isam"):
                PoseGraphClass = ISAM
            case _:
                raise ValueError(f"Graph type of {config.graph_type} is not supported")

        with Timer.CPUTimingContext("GTSAM_Graph"):
            # initialize the graph instance
            graph: FactorGraph = PoseGraphClass().to(device=torch.device(config.device), dtype=torch.double)
            assert isinstance(graph, FactorGraph)

            viz_ns = getattr(gedf_cfg, "viz", None) if gedf_cfg is not None else None
            context = {
            "device": config.device,
            "graph": graph,
            # G-EDF hybrid state (None / inert for the other graph types)
            "gedf_map": gedf_map,
            "gedf_insert": bool(gedf_cfg.map.insert_keypoints) if gedf_cfg is not None else False,
            "gedf_viz": SimpleNamespace(
                every=int(getattr(viz_ns, "every", 0)),
                iso=float(getattr(viz_ns, "iso", 0.10)),
                resolution=float(getattr(viz_ns, "resolution", 0.10)),
                max_points=int(getattr(viz_ns, "max_points", 100_000)),
                gaussians=bool(getattr(viz_ns, "gaussians", False)),
                n_sigma=float(getattr(viz_ns, "n_sigma", 1.0)),
                max_gaussians=int(getattr(viz_ns, "max_gaussians", 20_000)),
                max_sigma=(float(viz_ns.max_sigma)
                           if viz_ns is not None and hasattr(viz_ns, "max_sigma") else None),
                cubes=bool(getattr(viz_ns, "cubes", False))),
            }

        if config.device != "cpu":
            # Only record GPU event if the graph lives on cuda device, otherwise the GPU event will be meaningless and may cause overhead
            # Warm up GPU and CUDA context by running a dummy optimization step (to avoid including CUDA initialization time in the first real optimization)
            with Timer.GPUTimingContext("GTSAM_Graph", torch.cuda.current_stream()):
                pass

        return context

    @staticmethod
    def _optimize(context: dict, graph_data: GTSAM_GraphInput) -> tuple[dict, GTSAM_GraphOutput]:

        # G-EDF hybrid: feed and refit the online map before the solve (the
        # landmarks are anchored at the previous, already-optimized pose —
        # same pre-solve-insertion argument as GEDF_PGO). Landmark positions
        # later refined by the solve are NOT retro-fitted into the map.
        gedf_map = context.get("gedf_map")
        if gedf_map is not None and not gedf_map.frozen:
            pts = graph_data.current_graph_data.points
            if context["gedf_insert"] and len(pts) > 0:
                gedf_map.insert(pts.data["pos_Tw"], pts.data["cov_Tw"])
            cam_pos = pp.SE3(graph_data.current_graph_data.init_motion) \
                .tensor().reshape(-1)[:3]
            gedf_map.refit(camera_pos=cam_pos)

        graph = context["graph"]

        # Incorporate new measurements
        graph.parse_graph_data(graph_data)

        # Step optimizer
        graph.run_gtsam_optimization()

        # Export result (+ optional G-EDF map snapshot for the parent's Rerun)
        out = graph.write_back()
        if graph_data.want_map_snapshot and gedf_map is not None:
            viz = context["gedf_viz"]
            if gedf_map.is_ready:
                out.gedf_points, out.gedf_dist = gedf_map.sample_surface(
                    resolution=viz.resolution, iso=viz.iso, max_points=viz.max_points)
                if viz.gaussians:
                    (out.gedf_gauss_means, out.gedf_gauss_sigmas,
                     out.gedf_gauss_weights, out.gedf_gauss_mae) = gedf_map.gaussians(
                        max_gaussians=viz.max_gaussians, max_sigma=viz.max_sigma)
            if viz.cubes and gedf_map.num_cubes > 0:
                (out.gedf_cube_centers, out.gedf_cube_valid,
                 out.gedf_cube_mae) = gedf_map.cubes()
                out.gedf_cube_size = float(gedf_map.cube_size)
        return context, out

    def write_graph_data(self, result: GTSAM_GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return

        # to_pose     = pp.SE3(result.pose_estimate[0].data.double().cpu())
        # global_map.frames.data["pose"][result.frame_idx] = to_pose.float()
        for frame_idx, pose_estimate in zip(result.frame_idexes, result.pose_estimates):
            global_map.frames.data["pose"][frame_idx] = pose_estimate.float()

        if result.need_interp is not None:
            global_map.frames.data["need_interp"][result.frame_idexes[-1]] = result.need_interp

        if result.map_points is not None and result.landmark_indexes is not None:
            idx = torch.tensor(result.landmark_indexes, dtype=torch.long, device=result.map_points.device)
            global_map.map_points.data["pos_Tw"][idx] = result.map_points.to(dtype=torch.float32, device=global_map.map_points.data["pos_Tw"].device)

        # Alignment scale diagnostic (parity with GEDF_PGO): no-op without --useRR.
        from Utility.Visualize import rr_plt
        if result.scale is not None and rr_plt.default_mode == "rerun":
            import rerun as rr
            rr.set_time("frame_idx", sequence=int(result.frame_idexes[-1]))
            rr.log("/world/gtsam_alignment/scale", rr.Scalars(result.scale))

        # G-EDF map snapshot (pose2point+gedf): same entities as GEDF_PGO so
        # the viewer setup is identical across backends.
        if rr_plt.default_mode == "rerun" and \
                (result.gedf_points is not None or result.gedf_gauss_means is not None
                 or result.gedf_cube_centers is not None):
            import rerun as rr
            rr.set_time("frame_idx", sequence=int(result.frame_idexes[-1]))
            if result.gedf_points is not None:
                rr_plt.log_gedf_map("/world/gedf_map", result.gedf_points, result.gedf_dist)
            if result.gedf_gauss_means is not None and result.gedf_gauss_sigmas is not None \
                    and result.gedf_gauss_weights is not None and result.gedf_gauss_mae is not None:
                rr_plt.log_gedf_gaussians(
                    "/world/gedf_map/gaussians",
                    result.gedf_gauss_means, result.gedf_gauss_sigmas,
                    result.gedf_gauss_weights, result.gedf_gauss_mae,
                    n_sigma=float(getattr(getattr(getattr(self.config, "gedf", None),
                                                  "viz", None), "n_sigma", 1.0)))
            if result.gedf_cube_centers is not None and result.gedf_cube_valid is not None \
                    and result.gedf_cube_mae is not None and result.gedf_cube_size is not None:
                rr_plt.log_gedf_cubes(
                    "/world/gedf_map/cubes",
                    result.gedf_cube_centers, result.gedf_cube_valid,
                    result.gedf_cube_mae, cube_size=result.gedf_cube_size)
