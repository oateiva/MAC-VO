"""
Integration wrapper for the depth-completion network (ntnu-frl/depth-completion).

The upstream model is a DINOv2 + DPT network with a 4-channel input (RGB plus a 1-channel
sparse inverse-depth prior) that predicts a dense metric depth map. This module adapts it to
MAC-VO's monocular depth paradigm by implementing `DepthModelProtocol`
(`deepodo_initialize` / `deepodo_inference`), mirroring how `DepthAnythingV2`
(`Module/Network/Depth/DepthAnythingV2/dpt.py`) is integrated.

The sparse prior is read from `CameraData.depth_prior` (Bx1xHxW metric depth, 0 = no prior),
which `Odometry/MACVO.py` populates by projecting the running map's landmarks into the frame.
When no prior is available (e.g. cold-start with an empty map) the model runs with a zeros
prior, degrading gracefully to pure monocular prediction.

This is a torch port of the upstream `DepthCompletionModel.predict` preprocessing so that the
whole path stays on-device and tensor-native; the vendored network weights are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from DataLoader import CameraData
from Module.Frontend.StereoDepth import IDepth

from .dpt import DepthAnythingV2, model_configs
from .utils import invert_depth_repr

# The network was trained on ImageNet-normalized RGB (see upstream _preprocess_image).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class DepthCompletion(nn.Module):
    """
    Depth-completion network integrated as a monocular depth estimator.

    Consumes `CameraData.imageL` (RGB, [0, 1]) and `CameraData.depth_prior` (sparse metric
    depth, 0 = no prior) and predicts a dense metric depth map of shape Bx1xHxW.
    """
    def __init__(
        self,
        encoder: str = "vits",
        weight: str | None = None,
        depth_min: float = 0.5,
        depth_max: float = 80.0,
        focal_length_canonical: float = 900.0,
        input_resolution: tuple[int, int] | list[int] = (630, 476),
        cov_scale: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # 4-channel input (RGB + prior), 1-channel Sigmoid output — see model/dpt.py defaults.
        self.net = DepthAnythingV2(**model_configs[encoder])

        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.focal_length_canonical = float(focal_length_canonical)
        # input_resolution is (width, height); both must be multiples of 14.
        self.input_w, self.input_h = int(input_resolution[0]), int(input_resolution[1])
        self.cov_scale = float(cov_scale)

        self.device: str = "cpu"
        if weight is not None:
            self._load_weight(weight)

    def _load_weight(self, weight: str) -> None:
        # weights_only=False: upstream checkpoints may pickle non-tensor objects (e.g. Path);
        # the checkpoint is a trusted local file.
        ckpt = torch.load(weight, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.net.load_state_dict(state_dict, strict=False)

    @property
    def provide_cov(self) -> bool:
        # Upstream emits depth only; we synthesize a covariance proportional to depth,
        # following the DepthAnythingV2 convention (cov = depth * cov_scale).
        return True

    def deepodo_initialize(self, config) -> None:
        self.device = config.device
        self.cov_scale = float(getattr(config, "cov_scale", self.cov_scale))
        self.to(self.device)
        self.eval()

    def _build_prior_channel(self, depth_prior: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Convert a sparse metric depth prior (Bx1xHxW, 0 = empty, original resolution) into the
        1-channel inverse-depth prior expected by the network at the model input resolution.
        """
        H, W = depth_prior.shape[-2], depth_prior.shape[-1]
        dp = depth_prior.to(self.device).float()

        # Resize the sparse prior to the model input resolution, preserving sparse values.
        prior_in = torch.zeros((1, 1, self.input_h, self.input_w), device=self.device)
        nz = torch.nonzero(dp[0, 0] > 0, as_tuple=False)  # (M, 2) -> (y, x)
        if nz.numel() > 0:
            ys, xs = nz[:, 0], nz[:, 1]
            new_x = torch.clamp((xs.float() * (self.input_w / W)).long(), 0, self.input_w - 1)
            new_y = torch.clamp((ys.float() * (self.input_h / H)).long(), 0, self.input_h - 1)
            prior_in[0, 0, new_y, new_x] = dp[0, 0, ys, xs]

        # Block-fill: spread each prior point across its 14x14 patch (the patch-embed cell),
        # matching the upstream 14x14 fill. max-pool + nearest-upsample is the vectorized form.
        pooled = F.max_pool2d(prior_in, kernel_size=14, stride=14)
        prior_block = F.interpolate(pooled, scale_factor=14, mode="nearest")

        # Inverse-depth representation, scaled by the resized focal length.
        focal = ((K[0, 0] * (self.input_w / W)) + (K[1, 1] * (self.input_h / H))) / 2.0
        return invert_depth_repr(prior_block, float(focal), self.focal_length_canonical, self.depth_min)

    @torch.inference_mode()
    def deepodo_inference(self, input: CameraData) -> IDepth.Output:
        H, W = input.height, input.width
        K = input.frame_K

        image = input.imageL.to(self.device).float()
        if image.shape[0] > 1:
            image = image[:1]

        # Resize + ImageNet-normalize the RGB image to the model input resolution.
        image_in = F.interpolate(image, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  device=self.device).view(1, 3, 1, 1)
        image_in = (image_in - mean) / std

        # Build the sparse inverse-depth prior channel (zeros prior when none is available).
        if input.depth_prior is not None:
            prior_in = self._build_prior_channel(input.depth_prior, K)
        else:
            prior_in = torch.zeros((1, 1, self.input_h, self.input_w), device=self.device)

        # 4-channel input -> Sigmoid inverse-depth -> metric depth at original resolution.
        pred = self.net(torch.cat([image_in, prior_in], dim=1))

        focal = (K[0, 0] + K[1, 1]) / 2.0
        depth = invert_depth_repr(pred.float(), float(focal), self.focal_length_canonical, self.depth_min)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

        cov = depth * self.cov_scale
        return IDepth.Output(depth=depth, cov=cov)
