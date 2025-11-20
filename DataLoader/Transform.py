import torch
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal, get_args
from types import SimpleNamespace

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, center_crop

from .Interface import Frame, T_Data, CameraData
from Utility.Extensions import ConfigTestableSubclass
from Utility.Config     import build_dynamic_config, DynamicConfigSpec


T_from = TypeVar("T_from")
T_to   = TypeVar("T_to")

# -------------------- Infrastructure (base transform abstractions) --------------------
# This section defines foundational abstractions (generic transform interface)
# that concrete data transforms (scaling, cropping, noise, dtype casting, etc.) extend.


class IDataTransform(Generic[T_from, T_to], ABC, ConfigTestableSubclass, torch.nn.Module):
    def __init__(self, config: SimpleNamespace | None | DynamicConfigSpec) -> None:
        super().__init__()
        if config is None:
            self.config = SimpleNamespace()
        elif isinstance(config, SimpleNamespace):
            self.config = config
        else:
            self.config, _ = build_dynamic_config(config)

    @abstractmethod
    def forward(self, frame: T_from) -> T_to: ...


class NoTransform(IDataTransform[T_Data, T_Data]):
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        return

    def forward(self, frame: T_Data) -> T_Data:
        return frame


# -------------------- Helpers (mono+stereo) --------------------

def _resize_images(imgs: list[torch.Tensor], size_hw: list[int], mode: InterpolationMode) -> list[torch.Tensor]:
    return [resize(img, size_hw, interpolation=mode) for img in imgs]

# -------------------- Transforms --------------------

class ScaleFrame(IDataTransform[Frame, Frame]):
    """
    Scale the image & ground truths on u and v direction and modify the camera intrinsic accordingly.
    Works for mono (len(images)==1) and stereo (==2).
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "scale_u": lambda v: isinstance(v, (float, int)) and v > 0,
            "scale_v": lambda v: isinstance(v, (float, int)) and v > 0,
            "interp" : lambda v: v in {"nearest", "bilinear"}
        })

    @staticmethod
    def scale_camera(camera: CameraData, scale_u: float, scale_v: float, interpolate: Literal["nearest", "bilinear"]) -> CameraData:
        match interpolate:
            case "bilinear": interp = InterpolationMode.BILINEAR
            case "nearest" : interp = InterpolationMode.NEAREST_EXACT

        raw_height = camera.height
        raw_width  = camera.width

        target_h   = int(raw_height / scale_v)
        target_w   = int(raw_width  / scale_u)

        round_scale_v = raw_height / target_h
        round_scale_u = raw_width  / target_w

        camera.K = camera.K.clone()
        camera.height = target_h
        camera.width  = target_w
        camera.K[:, 0] /= round_scale_u
        camera.K[:, 1] /= round_scale_v

        camera.images = _resize_images(camera.images, [target_h, target_w], mode=interp)

        if camera.gt_flow is not None:
            camera.gt_flow = resize(camera.gt_flow, [target_h, target_w], interpolation=interp)
            camera.gt_flow[:, 0] /= round_scale_u
            camera.gt_flow[:, 1] /= round_scale_v

        if camera.flow_mask is not None and camera.flow_mask.numel()!=0:
            print(camera.flow_mask.size())
            camera.flow_mask = resize(camera.flow_mask, [target_h, target_w], interpolation=interp)

        if camera.gt_depth is not None:
            camera.gt_depth = resize(camera.gt_depth, [target_h, target_w], interpolation=interp)

        return camera

    def forward(self, frame: Frame) -> Frame:
        frame.camera = self.scale_camera(
            frame.camera,
            scale_u=self.config.scale_u,
            scale_v=self.config.scale_v,
            interpolate=self.config.interp
        )
        return frame


class CenterCropFrame(IDataTransform[Frame, Frame]):
    """
    Center crop the image and modify ground truth & camera intrinsic accordingly.
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "height": lambda v: isinstance(v, int) and v > 0,
            "width": lambda v: isinstance(v, int) and v > 0
        })

    @staticmethod
    def crop_camera(camera: CameraData, target_h: int, target_w: int) -> CameraData:
        orig_h, orig_w = camera.height, camera.width

        camera.images = [center_crop(img, [target_h, target_w]) for img in camera.images]

        if camera.gt_flow is not None:
            camera.gt_flow   = center_crop(camera.gt_flow, [target_h, target_w])
        if camera.flow_mask is not None:
            camera.flow_mask = center_crop(camera.flow_mask, [target_h, target_w])
        if camera.gt_depth is not None:
            camera.gt_depth  = center_crop(camera.gt_depth, [target_h, target_w])

        camera.K = camera.K.clone()
        camera.K[:, 0, 2] -= (orig_w - target_w) / 2.
        camera.K[:, 1, 2] -= (orig_h - target_h) / 2.

        camera.height = target_h
        camera.width  = target_w

        return camera

    def forward(self, frame: Frame) -> Frame:
        frame.camera = self.crop_camera(
            frame.camera, target_h=self.config.height, target_w=self.config.width
        )
        return frame


class AddImageNoise(IDataTransform[Frame, Frame]):
    """
    Add noise to image color. Note that the `stdv` is on scale of [0-255] image instead of
    on the scale of [0-1]. (That is, we will divide stdv by 255 when applying noise on image)
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "stdv": lambda v: isinstance(v, (int, float)) and v > 0
        })

    def forward(self, frame: Frame) -> Frame:
        sigma = (self.config.stdv / 255.0)
        frame.camera.images = [(img + sigma * torch.randn_like(img)).clamp(0.0, 1.0)
                               for img in frame.camera.images]
        return frame


class CastDataType(IDataTransform[Frame, Frame]):
    T_SUPPORT = Literal["fp16", "fp32", "bf16"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.dtype: torch.dtype = self.cast_dtype(self.config.dtype)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "dtype": lambda v: v in get_args(CastDataType.T_SUPPORT)
        })

    @staticmethod
    def cast_dtype(dtype: T_SUPPORT) -> torch.dtype:
        match dtype:
            case "bf16": return torch.bfloat16
            case "fp16": return torch.float16
            case "fp32": return torch.float32

    def forward(self, frame: Frame) -> Frame:
        cam = frame.camera
        cam.images = [img.to(dtype=self.dtype) for img in cam.images]
        cam.K      = cam.K.to(dtype=self.dtype)
        if cam.gt_flow is not None  : cam.gt_flow   = cam.gt_flow.to(dtype=self.dtype)
        if cam.gt_depth is not None : cam.gt_depth  = cam.gt_depth.to(dtype=self.dtype)
        if cam.flow_mask is not None: cam.flow_mask = cam.flow_mask.to(dtype=self.dtype)

        frame.camera = cam
        return frame


class SmartResizeFrame(IDataTransform[Frame, Frame]):
    """
    Automatically resize and crop the frame to target height and width to
    maximize the fov of resulted frame while achieving target shape.

    This process will maintein the aspect ratio of the image (i.e. the image
    will not be stretched)
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "height": lambda v: isinstance(v, int) and v > 0,
            "width": lambda v: isinstance(v, int) and v > 0,
            "interp" : lambda v: v in {"nearest", "bilinear"},
        })

    def forward(self, frame: Frame) -> Frame:
        cam = frame.camera
        orig_height, orig_width = cam.height, cam.width
        targ_height, targ_width = self.config.height, self.config.width

        scale_factor = min(orig_height / targ_height, orig_width / targ_width)
        cam = ScaleFrame.scale_camera(
            cam, scale_u=scale_factor, scale_v=scale_factor, interpolate=self.config.interp
        )
        cam = CenterCropFrame.crop_camera(
            cam, target_h=targ_height, target_w=targ_width
        )

        frame.camera = cam
        return frame
