"""
Integration wrapper for DepthCov (Dexheimer & Davison, "Learning a Depth Covariance
Function", CVPR 2023, https://github.com/edexheim/DepthCov).

DepthCov is not a standalone depth estimator: its UNet maps an RGB image to per-pixel
Gaussian-process kernel parameters, and dense depth exists only as the GP posterior
conditioned on sparse depth anchors. This module bridges it to MAC-VO's monocular depth
paradigm (`DepthModelProtocol`: `deepodo_initialize` / `deepodo_inference`) by conditioning
on `CameraData.depth_prior` (Bx1xHxW metric depth, 0 = no prior), which `Odometry/MACVO.py`
populates by projecting the running map's landmarks into the frame — the same mechanism the
`DepthCompletion` adapter uses. The GP posterior over natural-log depth is converted to
metric depth and metric depth variance (delta method), so the backend's 2D->3D covariance
projection consumes DepthCov's learned, image-adaptive uncertainty directly.

When no usable prior exists (cold start with an empty map, or fewer than `min_points`
anchors, or a numerically failed GP solve) the model degrades gracefully to a constant
depth map at `init_depth` with a deliberately huge variance `init_cov`, so the optimizer
heavily downweights those observations.

The vendored code under `depth_cov/` is licensed by Imperial College London for
NON-COMMERCIAL, internal or academic research purposes only (see `depth_cov/LICENSE`).
This integration exists for internal research experiments and must not ship in a
commercial product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from DataLoader import CameraData
from Module.Frontend.StereoDepth import IDepth

from .depth_cov.core.NonstationaryGpModule import NonstationaryGpModule
from .depth_cov.utils.utils import normalize_coordinates


def log_depth_to_metric(
    log_depth: torch.Tensor, log_var: torch.Tensor, cov_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a GP posterior over natural-log depth to metric depth and metric depth
    variance via the delta method: depth = exp(mu), Var(depth) = exp(2*mu) * sigma^2.
    Both outputs carry the metric units the backend's `Covariance_2to3_full` expects.
    """
    depth = torch.exp(log_depth)
    cov = depth.square() * log_var * cov_scale
    return depth, cov


class DepthCov(nn.Module):
    """
    DepthCov GP depth + covariance integrated as a monocular depth estimator.

    Consumes `CameraData.imageL` (RGB, [0, 1]) and `CameraData.depth_prior` (sparse metric
    depth anchors, 0 = no prior) and predicts a dense metric depth map with a per-pixel
    metric depth variance, both of shape Bx1xHxW.
    """
    def __init__(
        self,
        weight: str | None = None,
        inference_size: tuple[int, int] | list[int] = (192, 256),  # (rows, cols) = trained size
        max_points: int = 256,
        min_points: int = 4,
        init_depth: float = 3.0,
        init_cov: float = 10.0,
        cov_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = NonstationaryGpModule()

        # DepthCov's kernel lengthscales are learned in normalized coordinates at 192x256;
        # inference runs at this resolution and outputs are upsampled to the frame size.
        # The UNet encoder halves the grid 5 times, so both dims must divide by 32.
        self.inference_h, self.inference_w = int(inference_size[0]), int(inference_size[1])
        if self.inference_h % 32 != 0 or self.inference_w % 32 != 0:
            raise ValueError(f"DepthCov inference_size must be divisible by 32, got {inference_size}")
        self.max_points = int(max_points)
        self.min_points = int(min_points)
        self.init_depth = float(init_depth)
        self.init_cov = float(init_cov)
        self.cov_scale = float(cov_scale)

        self.device: str = "cpu"
        if weight is not None:
            self._load_weight(weight)

    def _load_weight(self, weight: str) -> None:
        # weights_only=False: the upstream Lightning checkpoint pickles hyperparameters
        # alongside the state_dict; the checkpoint is a trusted local file.
        ckpt = torch.load(weight, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=True)

    @property
    def provide_cov(self) -> bool:
        return True

    def deepodo_initialize(self, config) -> None:
        self.device = config.device
        self.max_points = int(getattr(config, "max_points", self.max_points))
        self.min_points = int(getattr(config, "min_points", self.min_points))
        self.init_depth = float(getattr(config, "init_depth", self.init_depth))
        self.init_cov = float(getattr(config, "init_cov", self.init_cov))
        self.cov_scale = float(getattr(config, "cov_scale", self.cov_scale))
        self.to(self.device)
        self.eval()

    def _extract_anchors(
        self, depth_prior: torch.Tensor, H: int, W: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Turn the sparse metric prior (Bx1xHxW, 0 = empty) into GP conditioning inputs:
        normalized (row, col) anchor coordinates (1xMx2), natural-log anchor depths
        (1xMx1), and the scalar mean log-depth. Returns None when fewer than `min_points`
        usable anchors exist (caller falls back to the constant prior).

        Coordinates are normalized at frame resolution; `normalize_coordinates` maps pixel
        centers to [-1, 1], so the result is valid at the (different) inference resolution.
        """
        dp = depth_prior[0, 0]                                    # (H, W)
        nz = torch.nonzero(dp > 0, as_tuple=False)                # (M_all, 2) -> (row, col)
        d = dp[nz[:, 0], nz[:, 1]]
        keep = torch.isfinite(d)
        nz, d = nz[keep], d[keep]

        if int(nz.shape[0]) < self.min_points:
            return None
        if int(nz.shape[0]) > self.max_points:
            # Deterministic spread over the row-major anchor ordering: spatially even
            # without touching global RNG state (inference runs on a worker thread).
            idx = torch.linspace(0, nz.shape[0] - 1, self.max_points, device=nz.device).long()
            nz, d = nz[idx], d[idx]

        coords = nz.to(self.device).float().unsqueeze(0)          # (1, M, 2) (row, col)
        coords_norm = normalize_coordinates(coords, (H, W))
        sparse_log_depth = d.log().view(1, -1, 1).to(self.device)
        mean_log_depth = sparse_log_depth.mean()
        return coords_norm, sparse_log_depth, mean_log_depth

    def _fallback_output(self, H: int, W: int) -> IDepth.Output:
        depth = torch.full((1, 1, H, W), self.init_depth, dtype=torch.float32, device=self.device)
        cov = torch.full((1, 1, H, W), self.init_cov, dtype=torch.float32, device=self.device)
        return IDepth.Output(depth=depth, cov=cov)

    @torch.inference_mode()
    def deepodo_inference(self, input: CameraData) -> IDepth.Output:
        H, W = input.height, input.width

        image = input.imageL.to(self.device).float()
        if image.shape[0] > 1:
            image = image[:1]

        anchors = None
        if input.depth_prior is not None:
            anchors = self._extract_anchors(input.depth_prior, H, W)
        if anchors is None:
            return self._fallback_output(H, W)
        coords_norm, sparse_log_depth, mean_log_depth = anchors

        # Raw [0,1] RGB — the UNet applies ImageNet normalization internally.
        rgb_in = F.interpolate(
            image, size=(self.inference_h, self.inference_w), mode="bilinear", align_corners=False
        )
        gaussian_covs = self.model(rgb_in)
        log_depth, log_var, _, _ = self.model.condition_level(
            gaussian_covs, -1, coords_norm, sparse_log_depth, mean_log_depth,
            (self.inference_h, self.inference_w),
        )

        # Upstream continues past a failed Cholesky (cholesky_ex with check_errors=False)
        # and can emit non-positive posterior variances; guard both.
        if not bool(torch.isfinite(log_depth).all()) or not bool(torch.isfinite(log_var).all()):
            return self._fallback_output(H, W)
        log_var = log_var.clamp_min(1e-8)

        depth, cov = log_depth_to_metric(log_depth, log_var, self.cov_scale)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
        cov = F.interpolate(cov, size=(H, W), mode="bilinear", align_corners=False)
        return IDepth.Output(depth=depth.float(), cov=cov.float())
