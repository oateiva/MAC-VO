"""
DepthCrafter monocular depth source for MAC-VO.

DepthCrafter (https://github.com/Tencent/DepthCrafter) is a video-diffusion depth
model: it expects a temporal clip and produces temporally-consistent, *relative*
(affine-invariant) depth. MAC-VO's `MonocularFrontend` asks for depth one frame at
a time, so this wrapper keeps a sliding window of the most recent frames, runs the
diffusion pipeline on that window each step, and returns the depth for the newest
frame.

NOTE on scale: DepthCrafter output is relative and min-max normalized per window, so
it is **not** metric and the scale can drift between windows. We convert it to a
usable depth with a configurable `scale_factor` / `invert` (the same trick
`DepthAnythingV2` uses with `scale_factor / idepth`). Good enough to exercise the
pipeline end-to-end; not tuned for metric accuracy.

The heavy `diffusers`-based imports are deferred to `deepodo_initialize` so that a
missing/broken `diffusers` install does not break the shared depth-model registry
(and therefore the other depth models) at import time.
"""
from __future__ import annotations

from types import SimpleNamespace
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from DataLoader import CameraData
from Module.Frontend.StereoDepth import IDepth


def _round_to_multiple(value: float, multiple: int = 64) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


class DepthCrafter(nn.Module):
    """
    Sliding-window adapter around the DepthCrafter video-diffusion pipeline that
    conforms to MAC-VO's `DepthModelProtocol`
    (`provide_cov`, `deepodo_initialize`, `deepodo_inference`).
    """

    def __init__(self, **kwargs):
        super().__init__()
        # build_depth_model() forwards the whole `args` namespace as kwargs; we keep
        # them and (re)read everything in deepodo_initialize, mirroring DepthAnythingV3.
        self._init_kwargs = kwargs
        self.device: str | torch.device = kwargs.get("device", "cuda")
        self.pipe = None
        self.buffer: deque[torch.Tensor] = deque()

    @property
    def provide_cov(self) -> bool:
        return True

    def deepodo_initialize(self, config: Any) -> None:
        # Defer the diffusers-dependent imports so the depth registry imports even
        # without diffusers installed.
        from .unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
        from .depth_crafter_ppl import DepthCrafterPipeline

        def cfg(name: str, default):
            return getattr(config, name, default)

        self.device = cfg("device", self.device)
        self.unet_path: str = cfg("unet_path", "tencent/DepthCrafter")
        self.pretrain_path: str = cfg("pretrain_path", "stabilityai/stable-video-diffusion-img2vid-xt")
        self.window_size: int = int(cfg("window_size", 5))
        self.overlap: int = int(cfg("overlap", 2))
        self.num_inference_steps: int = int(cfg("num_inference_steps", 5))
        self.guidance_scale: float = float(cfg("guidance_scale", 1.0))
        self.max_res: int = int(cfg("max_res", 512))
        # DepthCrafter output is relative (affine-invariant), per-window min-max
        # normalized to [0, 1]. Linearly remap that into a metric window so depths
        # populate the keypoint selector's valid range. `invert=True` follows
        # DepthCrafter's disparity-like convention (high value = near).
        self.near_depth: float = float(cfg("near_depth", 0.5))
        self.far_depth: float = float(cfg("far_depth", 7.0))
        self.invert: bool = bool(cfg("invert", True))
        self.cov_scale: float = float(cfg("cov_scale", 0.1))
        self.cpu_offload: str | None = cfg("cpu_offload", None)

        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            self.unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        pipe = DepthCrafterPipeline.from_pretrained(
            self.pretrain_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.set_progress_bar_config(disable=True)

        if self.cpu_offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif self.cpu_offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        self.pipe = pipe
        self.buffer = deque(maxlen=self.window_size)

    def to(self, *args, **kwargs):  # type: ignore[override]
        # The diffusers pipeline is not a registered submodule, so move it manually.
        device = args[0] if args else kwargs.get("device", None)
        if device is not None:
            self.device = device
            if self.pipe is not None and self.cpu_offload is None:
                self.pipe.to(device)
        return self

    def _target_size(self, height: int, width: int) -> tuple[int, int]:
        scale = min(1.0, self.max_res / max(height, width))
        return _round_to_multiple(height * scale), _round_to_multiple(width * scale)

    @torch.inference_mode()
    def deepodo_inference(self, input: CameraData) -> IDepth.Output:
        assert self.pipe is not None, "DepthCrafter.deepodo_initialize must run first"

        image = input.imageL  # (B, 3, H, W) RGB in [0, 1]
        if image.dim() == 5:  # tolerate (B, N, 3, H, W)
            image = image[:, 0]
        image = image[:1].float().cpu()  # keep buffer on CPU, pipe moves to device
        _, _, H0, W0 = image.shape

        self.buffer.append(image[0])  # (3, H, W)
        clip = torch.stack(list(self.buffer), dim=0)  # (t, 3, H0, W0)

        th, tw = self._target_size(H0, W0)
        clip = F.interpolate(clip, size=(th, tw), mode="bilinear", align_corners=False)
        clip = clip.clamp(0.0, 1.0)  # pipeline expects [0, 1]

        frames = self.pipe(
            clip,
            height=th,
            width=tw,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            window_size=self.window_size,
            overlap=self.overlap,
            output_type="np",
        ).frames[0]  # (t, th, tw, c) in [0, 1]

        depth_norm = torch.from_numpy(frames).to(self.device).mean(dim=-1)  # (t, th, tw)
        dmin, dmax = depth_norm.min(), depth_norm.max()
        depth_norm = (depth_norm - dmin) / (dmax - dmin + 1e-6)
        last = depth_norm[-1]  # (th, tw), newest frame

        # rel: 0 = near, 1 = far. DepthCrafter's value is disparity-like (high = near),
        # so invert maps it back to a near->far parameter.
        rel = (1.0 - last) if self.invert else last
        depth = self.near_depth + rel * (self.far_depth - self.near_depth)

        depth = depth.view(1, 1, th, tw)
        depth = F.interpolate(depth, size=(H0, W0), mode="bilinear", align_corners=False)
        depth = depth.float()

        covariance = depth * self.cov_scale

        return IDepth.Output(
            depth=depth,
            disparity=None,
            cov=covariance,
            disparity_uncertainty=None,
        )


__all__ = ["DepthCrafter"]
