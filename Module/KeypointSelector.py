import math
from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch


from .Frontend.StereoDepth import IDepth
from .Frontend.Matching    import IMatcher
from DataLoader import CameraData

from Utility.Extensions import ConfigTestableSubclass
from Utility.Timer import Timer
from Utility.Utils import retrieve_scalar_map_pixels



class IKeypointSelector(ABC, ConfigTestableSubclass):
    """
    This module selects keypoint given current frame and (optionally) estimated depth, depth_cov, and flow_cov.

    The selector also receives an argument `numPoint` as hint to how many keypoints to select. This hint may *not* be
    followed strictly.
    """
    def __init__(self, config: SimpleNamespace):
        self.config = config

    @abstractmethod
    def select_point(
        self,
        frame   : CameraData,
        numPoint: int,
        depth0_est: IDepth.Output,
        depth1_est: IDepth.Output,
        match_est: IMatcher.Output | None,
    ) -> torch.Tensor:
        """
        Select keypoint for tracking using given frame, (optionally) estimated depth, depth_cov, and flow_cov.

        Return keypoint as a FloatTensor with shape (N, 2) where keypoints are arranged in (u, v) format.

        ## NOTE

        this means that you need to output the index of keypoints in *different* coordinate system as pytorch.

        Use `image[kp[..., 1], kp[..., 0]]` to read value of image on all u-v coords of keypoints.
        The default output of this function is (0x2 torch.Tensor) which means no keypoints are selected.
        """
        return torch.zeros((0, 2), dtype=torch.long, device=self.config.device)


class SelectorCompose(IKeypointSelector):
    """
    Given multiple keypoint selectors and their weight, distribute keypoint selection
    requirement to these according to the provided weight.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.selectors = [IKeypointSelector.instantiate(arg.type, arg.args) for arg in self.config.selector_args]

        self.weight = torch.tensor(self.config.weight)
        self.weight = self.weight / self.weight.sum()

    def select_point(self, frame: CameraData, numPoint: int, depth0_est: IDepth.Output, depth1_est: IDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        keypoints = []
        for selector, weight in zip(self.selectors, self.weight):
            keypoints.append(selector.select_point(frame, int(numPoint * weight), depth0_est, depth1_est, match_est))
        return torch.cat(keypoints, dim=0)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        for arg in config.selector_args:
            IKeypointSelector.is_valid_config(arg)
        assert isinstance(config.weight, list)
        for val in config.weight: assert isinstance(val, (int, float))


class MappingPointSelector(IKeypointSelector):
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        spec: dict = {
            "max_depth": lambda v: isinstance(v, float),
            "max_depth_cov": lambda v: isinstance(v, float),
            "mask_width": lambda v: isinstance(v, int)
        }
        # Optional median-relative depth-cov filter (see select_point).
        if config is not None and hasattr(config, "depth_cov_rel"):
            spec["depth_cov_rel"] = lambda v: isinstance(v, (int, float)) and v > 0.
        return cls._enforce_config_spec(config, spec)

    def select_point(self, frame: CameraData, numPoint: int, depth0_est: IDepth.Output, depth1_est: IDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        assert depth0_est.cov is not None
        depth_mask     = depth0_est.depth < self.config.max_depth
        border_mask = torch.zeros_like(depth_mask, dtype=torch.bool)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = True

        # Depth-cov threshold: absolute cap, optionally tightened to a
        # median-relative cut (same pattern as CovAwareSelector). The relative
        # cut adapts to the depth source's covariance scale, so unreliable
        # pixels are rejected even when all absolute covariances are large
        # (e.g. monocular depth).
        depth_cov_thresh = float(self.config.max_depth_cov)
        depth_cov_rel: float | None = getattr(self.config, "depth_cov_rel", None)
        if depth_cov_rel is not None:
            candidate_cov = depth0_est.cov[depth_mask & border_mask]
            if candidate_cov.numel() > 0:
                depth_cov_thresh = min(depth_cov_thresh,
                                       candidate_cov.nanmedian().item() * depth_cov_rel)
        depth_cov_mask = depth0_est.cov < depth_cov_thresh

        candidates     = depth_mask & depth_cov_mask & border_mask
        selected_points = torch.nonzero(candidates, as_tuple=False)
        perm = torch.randperm(selected_points.size(0))[:numPoint]
        pixels = selected_points[perm][..., 2:].roll(shifts=1, dims=1)
        return pixels


class RandomSelector(IKeypointSelector):
    """
    Uniformly random select keypoints within the scope of [mask_width : -mask_width]
    """
    def select_point(self, frame: CameraData, numPoint: int, depth0_est: IDepth.Output, depth1_est: IDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        h_indices = torch.randint(self.config.mask_width, frame.height - self.config.mask_width, (numPoint, 1), device=self.config.device)
        w_indices = torch.randint(self.config.mask_width, frame.width  - self.config.mask_width, (numPoint, 1), device=self.config.device)
        kps = torch.cat([w_indices, h_indices], dim=1)
        return kps

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "mask_width": lambda m: isinstance(m, int) and m >= 0,
            "device": lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu"))
        })


class GradientSelector(IKeypointSelector):
    """
    Select keypoint based on gradient information. Will random select points with
    local image gradient > config.grad_std.
    """
    def select_point(self, frame: CameraData, numPoint: int, depth0_est: IDepth.Output, depth1_est: IDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        image = frame.imageL[0]

        image_grad = torch.nn.functional.conv2d(
            image.unsqueeze(0),
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            .float()
            .expand((1, 3, 3, 3)),
            padding=1,
        )[0].abs()
        image_grad_avg = image_grad.mean(dim=(1, 2), keepdim=True)
        image_grad_std = image_grad.std(dim=(1, 2), keepdim=True)
        # Positions with sufficient gradient (feature) > +3std
        points = image_grad > (image_grad_avg + self.config.grad_std * image_grad_std)

        # Positions that are not too close to the edge of image
        border_mask = torch.zeros_like(points)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = 1.0
        points = points * border_mask
        selected_points = torch.nonzero(points, as_tuple=False)

        # Randomly select points
        perm = torch.randperm(selected_points.shape[0])[:numPoint]
        return selected_points[perm][..., 1:].roll(shifts=1, dims=1)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "mask_width": lambda m: isinstance(m, int) and m >= 0,
            "grad_std"  : lambda g: isinstance(g, (int, float)) and g > 0.
        })


class SparseGradienSelector(IKeypointSelector):
    """
    Select keypoint based on gradient information. Will random select points with
    local image gradient > config.grad_std.

    Ensured sparsity of keypoint by applying non-maximum suppresion (NMS) on image gradient
    of keypoint candidates.
    """
    def select_point(self, frame: CameraData, numPoint: int, depth0_est: IDepth.Output, depth1_est: IDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        image = frame.imageL[0]

        image_grad = torch.nn.functional.conv2d(
            image.unsqueeze(0),
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            .float()
            .expand((1, 3, 3, 3)),
            padding=1,
        )[0].abs()
        image_grad_avg = image_grad.mean(dim=(1, 2), keepdim=True)
        image_grad_std = image_grad.std(dim=(1, 2), keepdim=True)
        # Positions with sufficient gradient (feature) > +3std
        points = image_grad > (image_grad_avg + self.config.grad_std * image_grad_std)

        # Positions that are not too close to the edge of image
        border_mask = torch.zeros_like(points)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = 1.0
        points = points * border_mask

        # Positions that are sufficiently far away (sparse)
        image_grad_erode = torch.nn.functional.max_pool2d(
            image_grad,
            kernel_size=self.config.nms_size,
            stride=1,
            padding=(self.config.nms_size // 2),
        )
        image_grad_nms = image_grad == image_grad_erode
        points = points * image_grad_nms

        selected_points = torch.nonzero(points, as_tuple=False)

        # Randomly select points
        perm = torch.randperm(selected_points.shape[0])[:numPoint]
        return selected_points[perm][..., 1:].roll(shifts=1, dims=1)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "mask_width": lambda m: isinstance(m, int) and m >= 0,
            "grad_std"  : lambda g: isinstance(g, (int, float)) and g > 0.,
            "nms_size"  : lambda k: isinstance(k, int) and k >= 0 and (k % 2 == 1),
        })


class GridSelector(IKeypointSelector):
    """
    Select keypoint following the grid - strictly uniform across the entire image.

    The requested `numPoint` will be used to estimate the spacing between keypoints, but the
    selector may not generate exactly `numPoint` amount of keypoints.
    """
    def select_point(self, frame: CameraData, numPoint: int, depth0_est: IDepth.Output, depth1_est: IDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        h, w = frame.height, frame.width
        h -= 2 * self.config.mask_width
        w -= 2 * self.config.mask_width

        unit = max(1, int(math.sqrt(numPoint // 2)))

        mesh_u, mesh_v = torch.meshgrid(
            torch.arange(0, h, h // unit, device=self.config.device),
            torch.arange(0, w, w // (unit * 2), device=self.config.device),
            indexing="ij",
        )
        mesh_u, mesh_v = mesh_u.flatten(), mesh_v.flatten()

        points = torch.stack([mesh_v, mesh_u], dim=1)
        points += self.config.mask_width

        return points

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "mask_width": lambda m: isinstance(m, int) and m >= 0,
            "device": lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu"))
        })


class CovAwareSelector(IKeypointSelector):
    """
    The keypoint selector used by the MAC-VO.

    Selecting keypoints based on estimated depth, depth_cov, and flow_cov. See sect III.B
    of paper for detail.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self._max_depth: float | None = None

    @Timer.cpu_timeit("KPSelector.select")
    @Timer.gpu_timeit("KPSelector.select")
    @torch.inference_mode()
    def select_point(self, frame: CameraData, numPoint: int, depth0_est: IDepth.Output, depth1_est: IDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        assert depth0_est.cov is not None
        assert depth1_est.cov is not None
        if self._max_depth is None and self.config.max_depth == "auto":
            self._max_depth = frame.fx * frame.frame_baseline
        max_depth = self._max_depth if self._max_depth is not None else self.config.max_depth

        depth0_map     = depth0_est.depth.to(self.config.device)
        depth0_cov_map = depth0_est.cov.to(self.config.device)
        depth1_map     = depth1_est.depth.to(self.config.device)
        depth1_cov_map = depth1_est.cov.to(self.config.device)

        if match_est is not None and match_est.cov is not None:
            # A paired-window matcher returns B=2 (one flow per window slot) while the
            # depth maps are B=1, so combining them below needs one batch convention.
            # Batch 0 is the only correct choice: every downstream sample of the
            # selected pixels goes through `IDepth.retrieve_pixels`, which reads batch
            # 0 of the flow / depth / covariance maps. Selecting on batch 1 would pair
            # keypoints found in one flow field with values read from another.
            flow_cov_map = match_est.cov[:1].to(self.config.device)
        else:
            flow_cov_map = None

        # Derive quality map
        quality_map = depth0_cov_map + depth1_cov_map
        if flow_cov_map is not None:
            flow_cov_map = (flow_cov_map[:, 0] + flow_cov_map[:, 1] - 2 * flow_cov_map[:, 2]).unsqueeze(1)
            quality_map *= flow_cov_map

        # Apply NMS on quality map
        quality_map_erode = -torch.nn.functional.max_pool2d(
            -quality_map,
            kernel_size=self.config.kernel_size,
            stride=1,
            padding=(self.config.kernel_size // 2),
        )
        quality_nms = torch.logical_and(quality_map == quality_map_erode, ~quality_map.isnan())

        # Positions that are not too close to the edge of image
        border_mask = torch.zeros_like(quality_nms, dtype=torch.bool)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = True

        # Positions that are sufficiently close to camera.
        depth_mask = (depth0_map < max_depth) & (depth1_map < max_depth)

        depth0_cov_thresh = min(self.config.max_depth_cov, depth0_cov_map[quality_nms].nanmedian().item() * 1.5)
        # depth1_cov_thresh = min(self.config.max_depth_cov, depth1_cov_map[quality_nms].nanmedian().item() * 2.0)
        depth0_cov_mask = depth0_cov_map < depth0_cov_thresh
        # depth1_cov_mask = depth1_cov_map < depth1_cov_thresh

        # Positions that has sufficiently small flow_cov
        if flow_cov_map is not None:
            flow_cov_thresh = min(self.config.max_match_cov, flow_cov_map[quality_nms].nanmedian().item() * 1.5)
            flow_cov_mask = flow_cov_map < flow_cov_thresh
        else:
            flow_cov_mask = None

        point_mask = torch.logical_and(quality_nms, border_mask)
        point_mask = torch.logical_and(point_mask, depth_mask)
        point_mask = torch.logical_and(point_mask, depth0_cov_mask)

        if flow_cov_mask is not None:
            point_mask = torch.logical_and(point_mask, flow_cov_mask)

        if depth0_est.mask is not None:
            point_mask = torch.logical_and(point_mask, depth0_est.mask.to(point_mask.device))

        if match_est is not None and match_est.mask is not None:
            # Batch 0 for the same reason as the covariance above: an out-of-place
            # logical_and against a B=2 mask would broadcast point_mask back up to
            # B=2 and emit every pixel twice from `nonzero` below.
            point_mask = torch.logical_and(point_mask, match_est.mask[:1].to(point_mask.device))

        # Select points
        # NOTE: potential performance bottleneck
        # this will trigger host-device sync and hang the CPU until CUDA stream finishes.
        selected_points = torch.nonzero(point_mask, as_tuple=False)
        # end

        # Randomly select points
        perm = torch.randperm(selected_points.size(0))[:numPoint]
        pixels = selected_points[perm][..., 2:].roll(shifts=1, dims=1)

        return pixels

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "device"        : lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu")),
            "mask_width"    : lambda m: isinstance(m, int) and m >= 0,
            "max_depth"     : lambda dist: (dist == "auto") or (isinstance(dist, (int, float)) and dist > 0.),
            "kernel_size"   : lambda k: isinstance(k, int) and k > 0 and (k % 2 == 1),
            "max_depth_cov" : lambda c: isinstance(c, (int, float)) and c > 0.,
            "max_match_cov" : lambda c: isinstance(c, (int, float)) and c > 0.
        })


def spaced_greedy(uv: torch.Tensor, live_uv: torch.Tensor, radius: float, cap: int = 0) -> torch.Tensor:
    """
    Accept rows of `uv` in the given order, keeping every accepted point and every
    point of `live_uv` strictly more than `radius` away; stop at `cap` (0 = no cap).

    Bucketed at a pitch of exactly `radius`, so only the 3x3 cell neighbourhood can
    hold a conflict (cells two apart already differ by more than `radius` in that
    axis). Rejection is `d^2 <= radius^2`. Port of learningUAVO
    gtsam_backend/keypoint_selector.py::spaced_greedy.

    Returns the accepted rows as an (M, 2) tensor with `uv`'s dtype, on CPU.
    """
    r = float(radius)
    r2 = r * r
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def stamp(u: float, v: float) -> None:
        buckets.setdefault((int(u // r), int(v // r)), []).append((u, v))

    for p in live_uv.detach().cpu().double().reshape(-1, 2):
        stamp(float(p[0]), float(p[1]))

    uv_cpu = uv.detach().cpu()
    keep: list[int] = []
    for row, p in enumerate(uv_cpu.double().reshape(-1, 2)):
        u, v = float(p[0]), float(p[1])
        cu, cv = int(u // r), int(v // r)
        clash = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for qu, qv in buckets.get((cu + dx, cv + dy), ()):
                    if (u - qu) ** 2 + (v - qv) ** 2 <= r2:
                        clash = True
                        break
                if clash: break
            if clash: break
        if clash: continue
        keep.append(row)
        stamp(u, v)
        if cap and len(keep) >= cap:
            break
    return uv_cpu[torch.tensor(keep, dtype=torch.long)] if keep else uv_cpu[:0]


class TrackingCovAwareSelector(IKeypointSelector):
    """
    Persistent-track covariance-aware selector: a stateful port of learningUAVO's
    gtsam_backend/keypoint_selector.py::CovAwareSelector into MAC-VO's per-pair loop.

    Unlike every other selector in this file, this one REMEMBERS its output: each
    selected keypoint is carried into the next frame by following the optical flow
    (`kp1 = kp0 + flow(kp0)`, mirrored bit-for-bit from `MACVO.run_pair`), so a
    physical scene point keeps being re-selected at its flow-tracked position frame
    after frame. A persistent-graph backend (`ISAM2_Graph`) can then rebuild track
    identity downstream by exact integer pixel association — `pixel1_uv` of the new
    pair equals `round(pixel2_uv)` of the previous pair for a surviving track.

    Per call (pair frame0 -> frame1, `match_est` = the OUTGOING flow of frame0):
      1. Carried tracks are gated at frame0: inside the `mask_width` border, on
         finite positive depth, and (when the depth model provides one) on the
         depth validity mask. No max-age and no per-step-cov kill (both measured
         harmful in learningUAVO's FINDINGS.md; identical gate sets for tracks and
         seeds also keep the downstream association airtight).
      2. New tracks are seeded where no live track sits within `seed_radius`:
         local minima (NMS) of the flow-variance quality map
         Q = sigma_uu + sigma_vv - 2*sigma_uv, cut at
         min(max_match_cov, median(Q[candidates]) * median_rel), depth-gated, then
         accepted best-Q-first with greedy spacing (deterministic — no randperm).
      3. `numPoint` caps the SEEDS only; carried tracks are never dropped to
         honor it (deviation from the "hint" semantics of this interface).

    The emitted keypoints are integer pixels (carried positions rounded), keeping
    the odometry's color indexing and nearest-pixel retrieval conventions exact.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        # (L, 2) float32 — flow-carried positions of live tracks in the frame the
        # NEXT select_point call will see as frame0. None before the first call.
        self.track_uv: torch.Tensor | None = None
        self.fallback_grid_selector = GridSelector(SimpleNamespace(mask_width=self.config.mask_width, device=self.config.device))

    def _gate_carried(self, depth0_est: IDepth.Output) -> torch.Tensor:
        """Round carried tracks to pixels, kill those at the border, on invalid depth,
        or (when provided) on an invalid depth-mask pixel."""
        depth0_map = depth0_est.depth.to(self.config.device)
        height, width = depth0_map.shape[-2:]
        border = int(self.config.mask_width)
        if self.track_uv is None or self.track_uv.numel() == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=depth0_map.device)

        kp = self.track_uv.to(depth0_map.device).round().long()
        inbound = (
            (kp[:, 0] >= border) & (kp[:, 0] <= width  - 1 - border) &
            (kp[:, 1] >= border) & (kp[:, 1] <= height - 1 - border)
        )
        kp = kp[inbound]
        depth = depth0_map[0, 0, kp[:, 1], kp[:, 0]]
        valid = torch.isfinite(depth) & (depth > 0)
        if depth0_est.mask is not None:
            valid = valid & depth0_est.mask.to(depth0_map.device)[0, 0, kp[:, 1], kp[:, 0]]
        return kp[valid]

    def _seed(self, carried: torch.Tensor, numPoint: int, depth0_est: IDepth.Output,
              match_est: IMatcher.Output) -> torch.Tensor:
        """Cov-aware seeds, spaced against `carried` and each other at seed_radius."""
        assert match_est.cov is not None
        depth0_map = depth0_est.depth.to(self.config.device)
        # Batch 0 only: a paired-window matcher returns B=2 while depth is B=1, and
        # every downstream sample of the selected pixels reads batch 0. Same
        # convention as CovAwareSelector / CovAwareSelector_NoDepth.
        flow_cov_map = match_est.cov[:1].to(self.config.device)
        quality_map = (flow_cov_map[:, 0] + flow_cov_map[:, 1] - 2 * flow_cov_map[:, 2]).unsqueeze(1)

        quality_map_erode = -torch.nn.functional.max_pool2d(
            -quality_map,
            kernel_size=self.config.kernel_size,
            stride=1,
            padding=(self.config.kernel_size // 2),
        )
        quality_nms = torch.logical_and(quality_map == quality_map_erode, ~quality_map.isnan())

        border_mask = torch.zeros_like(quality_nms, dtype=torch.bool)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = True

        candidates = quality_nms & border_mask
        candidate_q = quality_map[candidates]
        if candidate_q.numel() == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=self.config.device)

        flow_cov_thresh = min(self.config.max_match_cov, candidate_q.nanmedian().item() * self.config.median_rel)
        point_mask = candidates & (quality_map < flow_cov_thresh)

        # Same depth + mask gate as _gate_carried (identical gate sets).
        point_mask = point_mask & torch.isfinite(depth0_map) & (depth0_map > 0)
        if depth0_est.mask is not None:
            point_mask &= depth0_est.mask[:1].to(point_mask.device)

        if match_est.mask is not None:
            point_mask = torch.logical_and(point_mask, match_est.mask[:1].to(point_mask.device))

        selected = torch.nonzero(point_mask, as_tuple=False)    # (M, 4) [b, c, v, u]
        if selected.size(0) == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=self.config.device)

        # Best-Q-first (deterministic), then greedy spacing against live tracks.
        order = torch.argsort(quality_map[point_mask], stable=True)
        uv = selected[order][..., 2:].roll(shifts=1, dims=1)    # (M, 2) (u, v)
        seeds = spaced_greedy(uv, carried.float(), float(self.config.seed_radius), cap=numPoint)
        return seeds.to(device=self.config.device, dtype=torch.long)

    @Timer.cpu_timeit("KPSelector.select")
    @Timer.gpu_timeit("KPSelector.select")
    @torch.inference_mode()
    def select_point(self, frame: CameraData, numPoint: int, depth0_est: IDepth.Output, depth1_est: IDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        carried = self._gate_carried(depth0_est)

        if match_est is None or match_est.cov is None:
            # No covariance map: grid fallback, still spaced against live tracks.
            grid = self.fallback_grid_selector.select_point(frame, numPoint, depth0_est, depth1_est, match_est)
            seeds = spaced_greedy(grid, carried.float(), float(self.config.seed_radius), cap=numPoint)
            seeds = seeds.to(device=self.config.device, dtype=torch.long)
        else:
            seeds = self._seed(carried, numPoint, depth0_est, match_est)

        keypoints = torch.cat([carried, seeds], dim=0)

        # Carry to frame1 — a bit-for-bit mirror of MACVO.run_pair's
        # `kp1_uv = kp0_uv + retrieve_pixels(kp0_uv, match01.flow).T`, so the next
        # pair's pixel1_uv (this rounded) equals this pair's pixel2_uv exactly.
        if match_est is not None and keypoints.size(0) > 0:
            flow_at_kp = retrieve_scalar_map_pixels(keypoints, match_est.flow)
            assert flow_at_kp is not None
            self.track_uv = keypoints.float() + flow_at_kp.T
        else:
            self.track_uv = keypoints.float()

        return keypoints

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "device"        : lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu")),
            "mask_width"    : lambda m: isinstance(m, int) and m >= 0,
            "kernel_size"   : lambda k: isinstance(k, int) and k > 0 and (k % 2 == 1),
            "max_match_cov" : lambda c: isinstance(c, (int, float)) and c > 0.,
            "median_rel"    : lambda c: isinstance(c, (int, float)) and c > 0.,
            "seed_radius"   : lambda r: isinstance(r, (int, float)) and r > 0.,
        })


class CovAwareSelector_NoDepth(IKeypointSelector):
    """
    Selecting keypoints based on estimated flow_cov.

    The main difference with CovAwareSelector is dropping filters related with depth (i.e. max_depth and depth_cov).
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.fallback_grid_selector = GridSelector(SimpleNamespace(mask_width = self.config.mask_width, device=self.config.device))

    @Timer.cpu_timeit("KPSelector.select")
    @Timer.gpu_timeit("KPSelector.select")
    @torch.inference_mode()
    def select_point(self, frame: CameraData, numPoint: int, depth0_est: IDepth.Output, depth1_est: IDepth.Output, match_est: IMatcher.Output | None) -> torch.Tensor:
        if match_est is None or match_est.cov is None:
            return self.fallback_grid_selector.select_point(frame, numPoint, depth0_est, depth1_est, match_est)
        else:
            # Batch 0 only: a paired-window matcher returns B=2 (one flow per window
            # slot), but every downstream sample of the selected pixels goes through
            # `IDepth.retrieve_pixels`, which reads batch 0. Selecting across both
            # slots yields keypoints found in one flow field whose flow, depth and
            # covariance are then read from the other. Same convention as
            # CovAwareSelector.
            flow_cov_map = match_est.cov[:1].to(self.config.device)

        # Derive quality map
        quality_map = (flow_cov_map[:, 0] + flow_cov_map[:, 1] - 2 * flow_cov_map[:, 2]).unsqueeze(1)
        flow_cov_map = quality_map

        # Apply NMS on quality map
        quality_map_erode = -torch.nn.functional.max_pool2d(
            -quality_map,
            kernel_size=self.config.kernel_size,
            stride=1,
            padding=(self.config.kernel_size // 2),
        )
        quality_nms = torch.logical_and(quality_map == quality_map_erode, ~quality_map.isnan())

        # Positions that are not too close to the edge of image
        border_mask = torch.zeros_like(quality_nms, dtype=torch.bool)
        border_mask[
            ..., self.config.mask_width : -self.config.mask_width, self.config.mask_width : -self.config.mask_width
        ] = True

        # Positions that has sufficiently small flow_cov
        flow_cov_thresh = min(self.config.max_match_cov, flow_cov_map[quality_nms].median().item() * 1.5)
        flow_cov_mask = flow_cov_map < flow_cov_thresh

        point_mask = torch.logical_and(quality_nms, border_mask)
        point_mask = torch.logical_and(point_mask, flow_cov_mask)

        # Optional median-RELATIVE depth-cov filter: no absolute depth-cov
        # threshold (that is the point of the NoDepth variant - absolute scales
        # are unusable for monocular depth), but the relative ordering of the
        # depth covariance still identifies the reliable scene elements.
        depth_cov_rel: float | None = getattr(self.config, "depth_cov_rel", None)
        if depth_cov_rel is not None and depth0_est.cov is not None:
            depth_cov_map = depth0_est.cov.to(self.config.device)
            # Both are B=1 (the flow cov was sliced to batch 0 above, and the depth
            # cov is B=1); expand_as stays as a guard should either gain a batch.
            cand_mask = quality_nms & border_mask
            candidate_cov = depth_cov_map.expand_as(cand_mask)[cand_mask]
            if candidate_cov.numel() > 0:
                depth_cov_thresh = candidate_cov.nanmedian().item() * depth_cov_rel
                point_mask = torch.logical_and(point_mask, depth_cov_map < depth_cov_thresh)

        # Optional far-range gate. Beyond some range this footage's depth AND
        # flow are systematically BIASED (featureless open water), and bias
        # cannot be down-weighted by covariance - only gated (measured in
        # learningUAVO/gtsam_backend/FINDINGS.md). `max_depth` is absolute in
        # the depth map's own (mono, scale-inflated) units; `max_depth_rel`
        # is scale-free, a multiple of the median candidate depth.
        max_depth: float | None = getattr(self.config, "max_depth", None)
        max_depth_rel: float | None = getattr(self.config, "max_depth_rel", None)
        if max_depth is not None or max_depth_rel is not None:
            depth_map = depth0_est.depth.to(self.config.device)
            thresh = max_depth if max_depth is not None else float("inf")
            if max_depth_rel is not None:
                cand_mask = quality_nms & border_mask
                candidate_d = depth_map.expand_as(cand_mask)[cand_mask]
                if candidate_d.numel() > 0:
                    thresh = min(thresh, candidate_d.nanmedian().item() * max_depth_rel)
            point_mask = torch.logical_and(point_mask, depth_map < thresh)

        if match_est is not None and match_est.mask is not None:
            # Batch 0, matching the covariance above: an out-of-place logical_and
            # against a B=2 mask would broadcast point_mask back up to B=2 and emit
            # every pixel twice from `nonzero` below.
            point_mask = torch.logical_and(point_mask, match_est.mask[:1].to(point_mask.device))

        # Select points
        # NOTE: potential performance bottleneck
        # this will trigger host-device sync and hang the CPU until CUDA stream finishes.
        selected_points = torch.nonzero(point_mask, as_tuple=False)
        # end

        # Randomly select points
        perm = torch.randperm(selected_points.size(0))[:numPoint]
        pixels = selected_points[perm][..., 2:].roll(shifts=1, dims=1)

        return pixels

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        spec: dict = {
            "device"        : lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu")),
            "mask_width"    : lambda m: isinstance(m, int) and m >= 0,
            "kernel_size"   : lambda k: isinstance(k, int) and k > 0 and (k % 2 == 1),
            "max_match_cov" : lambda c: isinstance(c, (int, float)) and c > 0.
        }
        # Optional median-relative depth-cov filter and far-range gates
        # (see select_point).
        for key in ("depth_cov_rel", "max_depth", "max_depth_rel"):
            if config is not None and hasattr(config, key):
                spec[key] = lambda v: isinstance(v, (int, float)) and v > 0.
        cls._enforce_config_spec(config, spec)
