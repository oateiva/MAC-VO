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

        matches_prev_persistent = []
        matches_curr_persistent = []
        indexes_prev_curr = []
        for i in range(matches_curr.data["pixel1_uv"].shape[0]):
            matches_curr_i = matches_curr.data["pixel1_uv"][i]
            # Find indices in matches_prev where pixel2_uv matches matches_curr_i
            mask = torch.isclose(matches_prev.data["pixel2_uv"], matches_curr_i, atol=1.).all(dim=-1)
            if mask.any():
                # Append corresponding values to persistent lists
                matches_prev_persistent.append(matches_prev.data["pixel2_uv"][mask])
                matches_curr_persistent.append(matches_curr.data["pixel1_uv"][i].unsqueeze(0))
                indexes_prev_curr.append((torch.where(mask)[0][0].item(), i))

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

        return self.connect_graphs(GI_prev, GI_last)


    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "graph_type": lambda s: s in {"pose2point", "isam"},
            "device": lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "vectorize": lambda b: isinstance(b, bool),
            "parallel": lambda b: isinstance(b, bool),
            "autodiff": lambda b: isinstance(b, bool)
        })

    @staticmethod
    def init_context(config) -> dict:
        match (config.graph_type):
            case ("pose2point"):
                PoseGraphClass = GTSAM_Pose2Point
            case ("isam"):
                PoseGraphClass = ISAM
            case _:
                raise ValueError(f"Graph type of {config.graph_type} is not supported")

        with Timer.CPUTimingContext("GTSAM_Graph"):
            # initialize the graph instance
            graph: FactorGraph = PoseGraphClass().to(device=torch.device(config.device), dtype=torch.double)
            assert isinstance(graph, FactorGraph)

            context = {
            "device": config.device,
            "graph": graph,
            }

        if config.device != "cpu":
            # Only record GPU event if the graph lives on cuda device, otherwise the GPU event will be meaningless and may cause overhead
            # Warm up GPU and CUDA context by running a dummy optimization step (to avoid including CUDA initialization time in the first real optimization)
            with Timer.GPUTimingContext("GTSAM_Graph", torch.cuda.current_stream()):
                pass

        return context

    @staticmethod
    def _optimize(context: dict, graph_data: GTSAM_GraphInput) -> tuple[dict, GTSAM_GraphOutput]:

        graph = context["graph"]

        # Incorporate new measurements
        graph.parse_graph_data(graph_data)

        # Step optimizer
        graph.run_gtsam_optimization()

        # Export result
        return context, graph.write_back()

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
