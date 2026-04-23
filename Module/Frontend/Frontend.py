"""
What is Frontend?
    - up to now (2024/06) it's just a combination of StereoDepth and Matcher

Why we need Frontend?
    - Sometime the depth estimation and matching are tightly coupled, so we need a way to combine them.

      For instance, if depth (using disparity) and matching uses the same network with same weight, instead of
      inference twice in sequential mannor, we can compose a batch with size of 2 and inference once.

How to use this?
    - If there's no specific need (e.g. for performance improvement mentioned above), just use the `FrontendCompose`
      to combine an IStereoDepth and an IMatcher. This should work just fine.

    - Otherwise implement a new IFrontend and plug it in the pipeline.
"""

from __future__ import annotations

import torch
import time
from pathlib import Path
from types import SimpleNamespace
from typing import overload, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass

from DataLoader import CameraData
from Utility.PrettyPrint import Logger
from Utility.Timer import Timer
from Utility.Extensions import ConfigTestableSubclass
from Utility.Utils import reflect_torch_dtype

from .StereoDepth import IDepth, disparity_to_depth, disparity_to_depth_cov
from .Matching    import IMatcher

from Module.Network.ModelSelector import build_depth_model
from Module.Network.Depth.base import DepthModelProtocol

# Frontend interface ###
class IFrontend(ABC, ConfigTestableSubclass):
    """
    Jointly estimate dense depth map, dense matching and potentially their covariances given two pairs of stereo images.

    `IFrontend(frame_t1: CameraData, frame_t2: CameraData) -> IStereoDepth.Output, IMatcher.Output`

    Given two frames with imageL, imageR with shape of Bx3xHxW, return `output` where

    * [0] - IStereoDepth.Output, the predicted depth (and potentially depth covariance & validity mask)
    * [1] - IMatcher.Output or None, the predicted flow (potentially flow covariance & mask)

    If frame_t1 is None, return only `IStereoDepth.Output` and leave [1] as None.

    #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
    """

    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config

    @property
    @abstractmethod
    def provide_cov(self) -> tuple[bool, bool]: ...

    @abstractmethod
    def estimate_pair(self, frame_t1: CameraData, frame_t2: CameraData) -> tuple[IDepth.Output, IMatcher.Output]:
        """
        Given two frames with imageL, imageR with shape of Bx3xHxW, return `output` of
        -   [0] - IStereoDepth output of stereo frame from time t2
        -   [1] - IMatcher     output of left camera of t1 -> t2.

        #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
        """
        ...

    @abstractmethod
    def estimate_depth(self, frame: CameraData) -> IDepth.Output:
        """
        Given stereo frames with imageL, imageR with shape of Bx3xHxW, return IStereoDepth `output` of stereo frame

        #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
        """
        ...

    def estimate_triplet(self, frame_t1: CameraData, frame_t2: CameraData) -> tuple[IDepth.Output, IDepth.Output, IMatcher.Output]:
        """
        Given two frames with imageL, imageR with shape of Bx3xHxW, return `output` of
        -   [0] - IStereoDepth output of stereo frame from time t1
        -   [1] - IStereoDepth output of stereo frame from time t2
        -   [2] - IMatcher     output of left camera of t1 -> t2.

        #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
        """
        # Here is a simple yet less efficient sequential implementation, feel free to override with a more efficient (e.g. batched inference)
        # approach!
        depth_t1 = self.estimate_depth(frame_t1)
        depth_t2, match_t12 = self.estimate_pair(frame_t1, frame_t2)
        return depth_t1, depth_t2, match_t12

    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor, interpolate: bool=False) -> torch.Tensor: ...
    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: None, interpolate: bool=False) -> None: ...

    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor | None, interpolate: bool=False) -> torch.Tensor | None:
        """
        Given a pixel_uv (Nx2) tensor, retrieve the pixel values (CxN) from scalar_map (BxCxHxW).

        #### Note that the pixel_uv is in (x, y) format, not (row, col) format.

        #### Note that only first sample of scalar_map is used. (Batch idx=0)
        """
        if scalar_map is None: return None

        if interpolate:
            raise NotImplementedError("Not implemented yet")
        else:
            values = scalar_map[0, ..., pixel_uv[..., 1].long(), pixel_uv[..., 0].long()]
            return values

# End #######################

@dataclass
class CUDAGraphHandler:
    graph: torch.cuda.CUDAGraph
    shape: torch.Size
    static_input: dict[str, torch.Tensor]
    static_ouput: dict[str, torch.Tensor]

# Implementations

class FrontendCompose(IFrontend):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.depth = IDepth.instantiate(self.config.depth.type, self.config.depth.args)
        self.match = IMatcher.instantiate(self.config.match.type, self.config.match.args)

    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return self.depth.provide_cov, self.match.provide_cov

    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    def estimate_pair(self, frame_t1: CameraData, frame_t2: CameraData) -> tuple[IDepth.Output, IMatcher.Output]:
        return (
            self.depth.estimate(frame_t2),
            self.match.estimate(frame_t1, frame_t2)
        )

    def estimate_depth(self, frame: CameraData) -> IDepth.Output:
        return self.depth.estimate(frame)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.match)
        IDepth.is_valid_config(config.depth)


class FlowFormerCovFrontend(IFrontend):
    TENSOR_RT_AOT_RESULT_PATH = Path("./cache/FlowFormerCov_TRTCache")
    T_SUPPORT_DTYPE = Literal["fp32", "bf16", "fp16"]

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)

        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer

        cfg = get_cfg()
        cfg.latentcostformer.decoder_depth = self.config.decoder_depth
        model = build_flowformer(cfg, reflect_torch_dtype(config.enc_dtype), reflect_torch_dtype(config.dec_dtype))
        ckpt  = torch.load(self.config.weight, map_location=self.config.device, weights_only=True)

        model.eval()
        model.to(self.config.device)
        model.load_ddp_state_dict(ckpt)
        self.model = model

    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return True, True

    @staticmethod
    def inference_2_depth(flow_12: torch.Tensor, cov_12: torch.Tensor, frame: CameraData, enforce_positive_disparity: bool) -> IDepth.Output:
        disparity, disparity_cov = flow_12[:, :1].abs(), cov_12[:, :1]
        depth_map = disparity_to_depth(disparity, frame.frame_baseline, frame.fx)
        depth_cov = disparity_to_depth_cov(disparity, disparity_cov, frame.frame_baseline, frame.fx)

        if enforce_positive_disparity:
            bad_mask = flow_12[:, :1] <= 0
        else:
            bad_mask = None

        return IDepth.Output(depth=depth_map, cov=depth_cov, disparity=disparity, disparity_uncertainty=disparity_cov, mask=bad_mask)

    @staticmethod
    def inference_2_match(flow_12: torch.Tensor, cov_12: torch.Tensor) -> IMatcher.Output:
        match_map, match_cov = flow_12, cov_12
        match_mask = None
        return IMatcher.Output.from_partial_cov(flow=match_map, cov=match_cov, mask=match_mask)

    @torch.inference_mode()
    def estimate_depth(self, frame: CameraData) -> IDepth.Output:
        input_A, input_B = frame.imageL, frame.imageR
        input_A = input_A.to(device=self.config.device)
        input_B = input_B.to(device=self.config.device)

        est_flow, est_cov = self.model.inference(input_A, input_B)

        est_flow: torch.Tensor = est_flow.float()
        est_cov : torch.Tensor = est_cov.float()

        return self.inference_2_depth(est_flow, est_cov, frame, self.config.enforce_positive_disparity)

    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    @torch.inference_mode()
    def estimate_pair(self, frame_t1: CameraData, frame_t2: CameraData) -> tuple[IDepth.Output, IMatcher.Output]:
        input_A = torch.cat([frame_t2.imageL, frame_t1.imageL], dim=0)
        input_B = torch.cat([frame_t2.imageR, frame_t2.imageL], dim=0)

        input_A = input_A.to(device=self.config.device)
        input_B = input_B.to(device=self.config.device)
        est_flow, est_cov = self.model.inference(input_A, input_B)

        est_flow: torch.Tensor = est_flow.float()
        est_cov : torch.Tensor = est_cov.float()

        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], frame_t2, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[1:2], est_cov[1:2])
        )

    @torch.inference_mode()
    def estimate_triplet(self, frame_t1: CameraData, frame_t2: CameraData) -> tuple[IDepth.Output, IDepth.Output, IMatcher.Output]:
        input_A = torch.cat([frame_t1.imageL, frame_t2.imageL, frame_t1.imageL], dim=0)
        input_B = torch.cat([frame_t1.imageL, frame_t2.imageR, frame_t2.imageL], dim=0)

        input_A = input_A.to(device=self.config.device)
        input_B = input_B.to(device=self.config.device)
        est_flow, est_cov = self.model.inference(input_A, input_B)

        est_flow: torch.Tensor = est_flow.float()
        est_cov : torch.Tensor = est_cov.float()

        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], frame_t1, self.config.enforce_positive_disparity),
            self.inference_2_depth(est_flow[1:2], est_cov[1:2], frame_t2, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[2:3], est_cov[2:3])
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight"    : lambda s: isinstance(s, str), # Model Checkpoint path
            "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            "dec_dtype" : lambda b: isinstance(b, str) and b in ("fp32", "fp16", "bf16"),
            "enc_dtype" : lambda b: isinstance(b, str) and b in ("fp32", "fp16", "bf16"),
            "enforce_positive_disparity": lambda b: isinstance(b, bool),
            "decoder_depth" : lambda v: isinstance(v, int)
        })


class CUDAGraph_FlowFormerCovFrontend(FlowFormerCovFrontend):
    """
    FlowformerCov Frontend, but using CUDAGraph acceleration to improve inference speed.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)

        self.cuda_graph: CUDAGraphHandler | None = None
        assert "cuda" in self.config.device.lower(), "CUDAGraph_FlowFormerCovFrontend can only run on CUDA device."

        torch.backends.cuda.matmul.allow_tf32 = True    # Allow tensor cores
        torch.backends.cudnn.allow_tf32 = True          # Allow tensor cores
        torch.set_float32_matmul_precision("medium")    # Reduced precision for higher throughput
        torch.backends.cuda.preferred_linalg_library = "cusolver"   # For faster linalg ops

    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    def estimate_pair(self, frame_t1: CameraData, frame_t2: CameraData) -> tuple[IDepth.Output, IMatcher.Output]:
        # Joint inference
        input_A = torch.cat([frame_t2.imageL, frame_t1.imageL], dim=0)
        input_B = torch.cat([frame_t2.imageR, frame_t2.imageL], dim=0)

        input_A = input_A.to(device=self.config.device)
        input_B = input_B.to(device=self.config.device)

        est_flow, est_cov = self.cuda_graph_estimate(input_A, input_B)
        time.sleep(0.0) # Hint OS scheduler for context switch

        est_flow = est_flow.float()
        est_cov  = est_cov.float()

        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], frame_t2, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[1:2], est_cov[1:2])
        )

    def cuda_graph_estimate(self, inp_A: torch.Tensor, inp_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        If does not exist a cuda graph
            build one and run inference through it.
            Store the resulted graph in frontend for future use.
        If does exist a cuda graph
            Launch graph with new input.
        """
        if self.cuda_graph is None:
            Logger.write("info", "Building CUDAGraph for FlowFormerCovFrontend")
            static_input_A, static_input_B   = torch.empty_like(inp_A, device='cuda'), torch.empty_like(inp_A, device='cuda')

            static_input_A.copy_(inp_A)
            static_input_B.copy_(inp_B)

            output_val: None | torch.Tensor = None
            output_cov: None | torch.Tensor = None

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())  #type: ignore
            with torch.cuda.stream(s):                  #type: ignore
                for _ in range(3):
                    output_val, output_cov = self.model.inference(static_input_A, static_input_B)
            torch.cuda.current_stream().wait_stream(s)
            assert output_val is not None and output_cov is not None

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output, static_output_cov = self.model.inference(static_input_A, static_input_B)

            self.cuda_graph = CUDAGraphHandler(
                graph, inp_A.shape,
                static_input={"input_A": static_input_A, "input_B": static_input_B},
                static_ouput={"flow": static_output, "flow_cov": static_output_cov}
            )
            Logger.write("info", "CUDAGraph Built. Will use CUDAGraph for accelerated inference.")

            return output_val, output_cov
        else:
            g_context = self.cuda_graph

            assert inp_A.shape == g_context.shape, f"Input shape mismatch for CUDAGraph replay: {inp_A.shape} != {g_context.shape}"

            g_context.static_input["input_A"].copy_(inp_A)
            g_context.static_input["input_B"].copy_(inp_B)

            g_context.graph.replay()
            time.sleep(0.0) # Hint OS scheduler for context switch

            result_val = g_context.static_ouput["flow"].clone()
            result_cov = g_context.static_ouput["flow_cov"].clone()

        return result_val, result_cov


class MonocularFrontend(IFrontend):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)

        self.match: IMatcher = IMatcher.instantiate(config.match.type, config.match.args)

        monodepth_model: DepthModelProtocol = build_depth_model(
            self.config.monodepth.type, **vars(self.config.monodepth.args)
        )
        monodepth_model.deepodo_initialize(self.config.monodepth.args)
        monodepth_model.to(self.config.device)
        self.depth_model: DepthModelProtocol = monodepth_model

    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    @torch.inference_mode()
    def estimate_pair(self, frame_t1: CameraData, frame_t2: CameraData) -> tuple[IDepth.Output, IMatcher.Output]:
        return (
            self.estimate_depth(frame_t2),
            self.estimate_flowcov(frame_t1, frame_t2)
        )

    def estimate_depth(self, frame: CameraData) -> IDepth.Output:
        return self.depth_model.deepodo_inference(frame)

    def estimate_flowcov(self, frame_t1: CameraData, frame_t2: CameraData) -> IMatcher.Output:
        return self.match.estimate(frame_t1, frame_t2)

    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return self.depth_model.provide_cov, self.match.provide_cov

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.match)


class CUDAGraph_MonocularFrontend(MonocularFrontend):
    """
    MonocularFrontend with CUDAGraph acceleration on the flow estimation model.
    Requires match to be FlowFormerCovMatcher. Monodepth is not CUDAGraphed.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        from .Matching import FlowFormerCovMatcher
        assert isinstance(self.match, FlowFormerCovMatcher), \
            "CUDAGraph_MonocularFrontend requires match.type = FlowFormerCovMatcher"
        self._flow_model = self.match.model

        self.cuda_graph_flow: CUDAGraphHandler | None = None
        assert "cuda" in self.config.device.lower(), "CUDAGraph_MonocularFrontend requires a CUDA device."

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.preferred_linalg_library = "cusolver"

    def _cuda_graph_flow_estimate(self, inp_A: torch.Tensor, inp_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cuda_graph_flow is None:
            Logger.write("info", "Building CUDAGraph for MonocularFrontend flow model")
            static_A = torch.empty_like(inp_A, device="cuda")
            static_B = torch.empty_like(inp_B, device="cuda")
            static_A.copy_(inp_A)
            static_B.copy_(inp_B)

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())  # type: ignore
            with torch.cuda.stream(s):                  # type: ignore
                for _ in range(3):
                    out_val, out_cov = self._flow_model.inference(static_A, static_B)
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out, static_out_cov = self._flow_model.inference(static_A, static_B)

            self.cuda_graph_flow = CUDAGraphHandler(
                graph, inp_A.shape,
                static_input={"input_A": static_A, "input_B": static_B},
                static_ouput={"flow": static_out, "flow_cov": static_out_cov},
            )
            Logger.write("info", "CUDAGraph Built for MonocularFrontend flow model.")
            return out_val, out_cov
        else:
            g = self.cuda_graph_flow
            assert inp_A.shape == g.shape, f"CUDAGraph input shape mismatch: {inp_A.shape} != {g.shape}"
            g.static_input["input_A"].copy_(inp_A)
            g.static_input["input_B"].copy_(inp_B)
            g.graph.replay()
            time.sleep(0.0)
            return g.static_ouput["flow"].clone(), g.static_ouput["flow_cov"].clone()

    def estimate_flowcov(self, frame_t1: CameraData, frame_t2: CameraData) -> IMatcher.Output:
        image_t1_left = frame_t1.imageL.to(self.config.device)[0:1, :, :, :]
        image_t2_left = frame_t2.imageL.to(self.config.device)[0:1, :, :, :]
        est_flow, est_cov = self._cuda_graph_flow_estimate(image_t1_left, image_t2_left)
        return IMatcher.Output.from_partial_cov(flow=est_flow.float(), cov=est_cov.float())
