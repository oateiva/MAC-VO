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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import overload, Literal, Callable, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass

_A = TypeVar("_A")
_B = TypeVar("_B")

from DataLoader import CameraData
from Utility.PrettyPrint import Logger
from Utility.Timer import Timer
from Utility.Extensions import ConfigTestableSubclass
from Utility.Utils import reflect_torch_dtype, retrieve_scalar_map_pixels

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

    def estimate_match(self, frame_a: CameraData, frame_b: CameraData) -> IMatcher.Output:
        """
        Optical flow (and covariance) of the left camera a -> b ONLY, for an arbitrary,
        possibly non-consecutive frame pair - the keyframe -> current inference of
        `MACVO._track_keyframe`. Must be called after `estimate_pair` has returned
        (CUDA-graph frontends replay into shared static buffers).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement estimate_match")

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
        return retrieve_scalar_map_pixels(pixel_uv, scalar_map, interpolate)

# End #######################

@dataclass
class CUDAGraphHandler:
    graph: torch.cuda.CUDAGraph
    shape: torch.Size
    static_input: dict[str, torch.Tensor]
    static_ouput: dict[str, torch.Tensor]
    stream: torch.cuda.Stream  # capture stream — must replay on same stream


class CUDAGraphMixin:
    def _setup_cuda_backends(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.preferred_linalg_library = "cusolver"

    def _build_cuda_graph(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        inp_A: torch.Tensor,
        inp_B: torch.Tensor,
        log_name: str,
    ) -> tuple[CUDAGraphHandler, torch.Tensor, torch.Tensor]:
        Logger.write("info", f"Building CUDAGraph for {log_name}")
        capture_stream = torch.cuda.Stream()
        static_A = torch.empty_like(inp_A, device="cuda")
        static_B = torch.empty_like(inp_B, device="cuda")
        static_A.copy_(inp_A)
        static_B.copy_(inp_B)

        out_val: torch.Tensor | None = None
        out_cov: torch.Tensor | None = None
        capture_stream.wait_stream(torch.cuda.current_stream())  # type: ignore
        with torch.cuda.stream(capture_stream):                  # type: ignore
            for _ in range(3):
                out_val, out_cov = model_fn(static_A, static_B)
        torch.cuda.current_stream().wait_stream(capture_stream)
        assert out_val is not None and out_cov is not None

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):    # type: ignore
            static_out, static_out_cov = model_fn(static_A, static_B)

        handler = CUDAGraphHandler(
            graph, inp_A.shape,
            static_input={"input_A": static_A, "input_B": static_B},
            static_ouput={"flow": static_out, "flow_cov": static_out_cov},
            stream=capture_stream,
        )
        Logger.write("info", f"CUDAGraph Built for {log_name}.")
        return handler, out_val, out_cov

    def _replay_cuda_graph(
        self,
        handler: CUDAGraphHandler,
        inp_A: torch.Tensor,
        inp_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inp_A.shape == handler.shape, f"CUDAGraph input shape mismatch: {inp_A.shape} != {handler.shape}"
        handler.static_input["input_A"].copy_(inp_A)
        handler.static_input["input_B"].copy_(inp_B)
        with torch.cuda.stream(handler.stream):  # type: ignore
            handler.graph.replay()
        torch.cuda.current_stream().wait_stream(handler.stream)  # type: ignore
        return handler.static_ouput["flow"].clone(), handler.static_ouput["flow_cov"].clone()


class ParallelEstimateMixin:
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        # One long-lived stream per slot. The CUDA caching allocator keeps a
        # separate block pool per stream, so creating a fresh Stream on every
        # frame strands that frame's inference transients in a pool no later
        # frame can reuse — reserved VRAM then grows by the full inference
        # footprint every frame (on Windows/WDDM it silently spills into
        # shared memory and slows ~30x instead of raising OOM).
        self._streams: tuple | None = (
            (torch.cuda.Stream(), torch.cuda.Stream()) if torch.cuda.is_available() else None
        )

    @staticmethod
    def _run_on_stream(s, fn: Callable[[], object]) -> object:
        s.wait_stream(torch.cuda.current_stream())  # type: ignore
        with torch.cuda.stream(s):                  # type: ignore
            result = fn()
        s.synchronize()
        return result

    def _on_match_stream(self, fn: Callable[[], _B], parallel: bool) -> _B:
        """Run a flow-only inference on the stream the matcher's transients already live
        on: the `_parallel` matcher slot when the frontend runs parallel, the current
        stream otherwise. The caching allocator pools blocks per stream, so a flow
        inference on any OTHER stream adds a pool the size of the flow model's
        transients (measured: reserved VRAM 24 -> 37 GB on a 24 GB card, WDDM paging)."""
        if self._streams is None or not parallel:
            return fn()
        return self._run_on_stream(self._streams[1], fn)  # type: ignore[return-value]

    def _parallel(self, fn_a: Callable[[], _A], fn_b: Callable[[], _B]) -> tuple[_A, _B]:
        """
        Run fn_a and fn_b concurrently on two threads, each on its own (reused)
        CUDA stream. Blocks until both complete. Re-raises any exception from
        either thread. Safe for inference-only model calls sharing weights (no
        gradient state mutated). Falls back to sequential execution when CUDA
        is unavailable.
        """
        if self._streams is None:
            return fn_a(), fn_b()
        fut_a = self._thread_pool.submit(self._run_on_stream, self._streams[0], fn_a)
        fut_b = self._thread_pool.submit(self._run_on_stream, self._streams[1], fn_b)
        return fut_a.result(), fut_b.result()  # type: ignore[return-value]


# Implementations

class FrontendCompose(ParallelEstimateMixin, IFrontend):
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
        if getattr(self.config, "parallel", False):
            return self._parallel(
                lambda: self.depth.estimate(frame_t2),
                lambda: self.match.estimate(frame_t1, frame_t2),
            )
        return self.depth.estimate(frame_t2), self.match.estimate(frame_t1, frame_t2)

    def estimate_depth(self, frame: CameraData) -> IDepth.Output:
        return self.depth.estimate(frame)

    @Timer.cpu_timeit("Frontend.estimate_match")
    @Timer.gpu_timeit("Frontend.estimate_match")
    def estimate_match(self, frame_a: CameraData, frame_b: CameraData) -> IMatcher.Output:
        return self._on_match_stream(lambda: self.match.estimate(frame_a, frame_b),
                                     getattr(self.config, "parallel", False))

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.match)
        IDepth.is_valid_config(config.depth)
        if hasattr(config, "parallel"):
            assert isinstance(config.parallel, bool), "parallel must be bool"


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

    @Timer.cpu_timeit("Frontend.estimate_match")
    @Timer.gpu_timeit("Frontend.estimate_match")
    @torch.inference_mode()
    def estimate_match(self, frame_a: CameraData, frame_b: CameraData) -> IMatcher.Output:
        # Eager B=1 path (the CUDAGraph subclass captures the B=2 estimate_pair batch only).
        input_A = frame_a.imageL.to(device=self.config.device)
        input_B = frame_b.imageL.to(device=self.config.device)
        est_flow, est_cov = self.model.inference(input_A, input_B)
        return self.inference_2_match(est_flow.float(), est_cov.float())

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


class CUDAGraph_FlowFormerCovFrontend(CUDAGraphMixin, FlowFormerCovFrontend):
    """
    FlowformerCov Frontend, but using CUDAGraph acceleration to improve inference speed.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.cuda_graph: CUDAGraphHandler | None = None
        assert "cuda" in self.config.device.lower(), "CUDAGraph_FlowFormerCovFrontend can only run on CUDA device."
        self._setup_cuda_backends()

    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    def estimate_pair(self, frame_t1: CameraData, frame_t2: CameraData) -> tuple[IDepth.Output, IMatcher.Output]:
        input_A = torch.cat([frame_t2.imageL, frame_t1.imageL], dim=0)
        input_B = torch.cat([frame_t2.imageR, frame_t2.imageL], dim=0)

        input_A = input_A.to(device=self.config.device)
        input_B = input_B.to(device=self.config.device)

        est_flow, est_cov = self.cuda_graph_estimate(input_A, input_B)
        est_flow = est_flow.float()
        est_cov  = est_cov.float()

        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], frame_t2, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[1:2], est_cov[1:2])
        )

    def cuda_graph_estimate(self, inp_A: torch.Tensor, inp_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cuda_graph is None:
            self.cuda_graph, out_val, out_cov = self._build_cuda_graph(
                self.model.inference, inp_A, inp_B, "FlowFormerCovFrontend"
            )
            return out_val, out_cov
        return self._replay_cuda_graph(self.cuda_graph, inp_A, inp_B)


class MonocularFrontend(ParallelEstimateMixin, IFrontend):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)

        self.match: IMatcher = IMatcher.instantiate(config.match.type, config.match.args)

        self._device_depth: str = getattr(config, 'device_depth', config.device)
        monodepth_model: DepthModelProtocol = build_depth_model(
            self.config.monodepth.type, **vars(self.config.monodepth.args)
        )
        monodepth_model.deepodo_initialize(self.config.monodepth.args)
        monodepth_model.to(self._device_depth)
        if hasattr(monodepth_model, 'device'):
            monodepth_model.device = self._device_depth  # type: ignore[assignment]
        self.depth_model: DepthModelProtocol = monodepth_model

        self.depth_seed: int | None = getattr(config, "depth_seed", None)
        """Reseeds torch right before each depth inference, so unseeded RNG draws inside the
        monodepth backbone become reproducible and configs sharing the same seed see identical
        depth (also determinizes a `torch.randperm`-based selector run right after, e.g. `CovAwareSelector_NoDepth`)."""

    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    def estimate_pair(self, frame_t1: CameraData, frame_t2: CameraData) -> tuple[IDepth.Output, IMatcher.Output]:
        if getattr(self.config, "parallel", True):
            return self._parallel(
                lambda: self.estimate_depth(frame_t2),
                lambda: self.estimate_flowcov(frame_t1, frame_t2),
            )
        return self.estimate_depth(frame_t2), self.estimate_flowcov(frame_t1, frame_t2)

    def estimate_depth(self, frame: CameraData) -> IDepth.Output:
        if self.depth_seed is not None:
            torch.manual_seed(self.depth_seed)
            torch.cuda.manual_seed_all(self.depth_seed)
        out = self.depth_model.deepodo_inference(frame)
        if self._device_depth != self.config.device:
            return out.to(self.config.device)
        return out

    def estimate_flowcov(self, frame_t1: CameraData, frame_t2: CameraData) -> IMatcher.Output:
        return self.match.estimate(frame_t1, frame_t2)

    @Timer.cpu_timeit("Frontend.estimate_match")
    @Timer.gpu_timeit("Frontend.estimate_match")
    def estimate_match(self, frame_a: CameraData, frame_b: CameraData) -> IMatcher.Output:
        # via estimate_flowcov: CUDAGraph_MonocularFrontend's captured B=1 flow graph applies
        return self._on_match_stream(lambda: self.estimate_flowcov(frame_a, frame_b),
                                     getattr(self.config, "parallel", True))

    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return self.depth_model.provide_cov, self.match.provide_cov

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.match)
        if hasattr(config, "parallel"):
            assert isinstance(config.parallel, bool), "parallel must be bool"
        if hasattr(config, "device_depth"):
            assert isinstance(config.device_depth, str) and "cuda" in config.device_depth, \
                "device_depth must be a CUDA device string (e.g. 'cuda:1')"
        if hasattr(config, "depth_seed"):
            assert isinstance(config.depth_seed, int) and not isinstance(config.depth_seed, bool) and config.depth_seed >= 0, \
                "depth_seed must be an int >= 0"


class CUDAGraph_MonocularFrontend(CUDAGraphMixin, MonocularFrontend):
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
        self._setup_cuda_backends()

    def _cuda_graph_flow_estimate(self, inp_A: torch.Tensor, inp_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cuda_graph_flow is None:
            self.cuda_graph_flow, out_val, out_cov = self._build_cuda_graph(
                self._flow_model.inference, inp_A, inp_B, "MonocularFrontend flow model"
            )
            return out_val, out_cov
        return self._replay_cuda_graph(self.cuda_graph_flow, inp_A, inp_B)

    def estimate_flowcov(self, frame_t1: CameraData, frame_t2: CameraData) -> IMatcher.Output:
        image_t1_left = frame_t1.imageL.to(self.config.device)[0:1, :, :, :]
        image_t2_left = frame_t2.imageL.to(self.config.device)[0:1, :, :, :]
        est_flow, est_cov = self._cuda_graph_flow_estimate(image_t1_left, image_t2_left)
        return IMatcher.Output.from_partial_cov(flow=est_flow.float(), cov=est_cov.float())
