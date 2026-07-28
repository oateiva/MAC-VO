from __future__ import annotations

"""
CachedDepth — a MAC-VO monocular depth model that serves *precomputed* depth maps.

Motivation
----------
The external depth networks we benchmark (DGGT, Pixel-Perfect Depth, Flow3r,
GemDepth, GGPT, ...) pin mutually-incompatible dependency stacks (different
torch versions, flash-attn, mmcv, spconv/pointcept custom CUDA ops) and several
are video / multi-view models that are degenerate on a single frame. They cannot
share MAC-VO's Python environment nor run cleanly one-frame-at-a-time.

So instead of live in-process inference, each network is run *offline* in its own
isolated conda env over the exact frames MAC-VO uses (dumped by
``Scripts/Depth/dump_frames.py``), writing a per-frame depth map to a cache. This
class loads that cache at run time and hands MAC-VO an ``IDepth.Output`` — fully
config-driven, exactly like ``DepthAnythingV3``.

Cache layout
------------
``<cache_root>/<net>/<frame_ns>.npz`` with keys:
  * ``depth`` : float32 ``HxW``  (metric metres, or up-to-scale for relative nets)
  * ``conf``  : float32 ``HxW``  (optional per-pixel confidence; higher = more certain)

The key ``frame_ns`` is ``CameraData.frame_ns`` (nanosecond timestamp), which the
frame-dumper and every precompute script derive identically, so alignment with the
image MAC-VO feeds the optical-flow matcher is guaranteed.
"""

from types import SimpleNamespace
from typing import Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from DataLoader import CameraData
from Module.Frontend.StereoDepth import IDepth


def frame_key(frame_ns) -> str:
    """Canonical, notation-stable cache key from a (possibly float) ns timestamp.
    MUST match Scripts/Depth/dump_frames.py and every precompute_*.py."""
    return str(int(round(float(frame_ns))))


class CachedDepth(nn.Module):
    """Serve precomputed per-frame depth (+ optional confidence) from disk.

    Config (``monodepth.args``) keys:
      * ``cache_root`` (str)   : root dir holding ``<net>/<frame_ns>.npz``.
      * ``net`` (str)          : subdir name identifying which network's cache to read.
      * ``device`` (str)       : torch device for the returned tensors.
      * ``provide_cov`` (bool) : whether to emit a ``cov`` map (default True).
      * ``cov_from_conf`` (bool): if True and a ``conf`` map exists, cov = cov_scale / conf.
                                  Otherwise fall back to the DAv2 heuristic cov = depth * rel_cov.
      * ``cov_scale`` (float)  : scale applied to the confidence->variance conversion (default 1.0).
      * ``rel_cov`` (float)    : heuristic relative variance when no confidence (default 0.1).
      * ``scale_factor`` (float): global multiplier on depth for up-to-scale/relative nets (default 1.0).
      * ``max_depth`` (float)  : clamp upper bound in metres, 0 disables (default 0.0).
      * ``missing`` (str)      : behaviour when a frame's cache file is absent —
                                 ``"error"`` (raise) or ``"nan"`` (return a nan depth map).
    """

    def __init__(self, cache_root: str = "", net: str = "", provide_cov: bool = True,
                 cov_from_conf: bool = True, cov_scale: float = 1.0, rel_cov: float = 0.1,
                 scale_factor: float = 1.0, max_depth: float = 0.0, missing: str = "error",
                 **kwargs) -> None:
        super().__init__()
        self.cache_root = cache_root
        self.net = net
        self._provide_cov = provide_cov
        self.cov_from_conf = cov_from_conf
        self.cov_scale = cov_scale
        self.rel_cov = rel_cov
        self.scale_factor = scale_factor
        self.max_depth = max_depth
        self.missing = missing
        self.device = "cuda"
        self._dir: Path | None = None

    @property
    def provide_cov(self) -> bool:
        return self._provide_cov

    def deepodo_initialize(self, config: Any) -> None:
        self.device = getattr(config, "device", self.device)
        cache_root = getattr(config, "cache_root", self.cache_root)
        net = getattr(config, "net", self.net)
        if not cache_root or not net:
            raise ValueError("CachedDepth requires `cache_root` and `net` in monodepth.args")
        # allow per-run overrides to reach attributes set from args
        self.provide_cov  # touch property
        self._dir = Path(cache_root) / net
        if not self._dir.is_dir():
            raise FileNotFoundError(
                f"CachedDepth cache dir not found: {self._dir}. "
                f"Run Scripts/Depth/dump_frames.py then the net's precompute script first."
            )

    def _load(self, frame_ns) -> tuple[np.ndarray, np.ndarray | None]:
        assert self._dir is not None
        f = self._dir / f"{frame_key(frame_ns)}.npz"
        if not f.is_file():
            if self.missing == "nan":
                return np.full((1, 1), np.nan, dtype=np.float32), None
            raise FileNotFoundError(
                f"CachedDepth: no cached depth for frame_ns={frame_ns} at {f}. "
                f"The precompute run for net='{self.net}' is missing this frame."
            )
        data = np.load(f)
        depth = data["depth"].astype(np.float32)
        conf = data["conf"].astype(np.float32) if "conf" in data.files else None
        return depth, conf

    @torch.inference_mode()
    def deepodo_inference(self, input: CameraData) -> IDepth.Output:
        image = input.imageL
        if image.dim() == 5:            # mono may arrive as B x N x 3 x H x W
            image = image[:, 0]
        H, W = int(image.shape[-2]), int(image.shape[-1])

        depth_np, conf_np = self._load(input.frame_ns)
        depth = torch.from_numpy(depth_np).to(self.device).float()[None, None]  # 1x1xhxw
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
        if self.scale_factor != 1.0:
            depth = depth * self.scale_factor
        if self.max_depth > 0:
            depth = depth.clamp(max=self.max_depth)

        cov: torch.Tensor | None = None
        if self._provide_cov:
            if self.cov_from_conf and conf_np is not None:
                conf = torch.from_numpy(conf_np).to(self.device).float()[None, None]
                if conf.shape[-2:] != (H, W):
                    conf = F.interpolate(conf, size=(H, W), mode="bilinear", align_corners=False)
                cov = (self.cov_scale / conf.clamp_min(1e-6))
            else:
                cov = depth * self.rel_cov     # DAv2-style heuristic variance

        return IDepth.Output(depth=depth, disparity=None, cov=cov, disparity_uncertainty=None)
