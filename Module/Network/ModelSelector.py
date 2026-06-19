from collections.abc import Callable
from Module.Network.Depth.base import DepthModelProtocol
from typing import Type
import torch.nn as nn
from Module.Network.Depth.DepthAnythingV2.dpt import DepthAnythingV2
from Module.Network.Depth.DepthAnythingV2.metric.dpt import DepthAnythingV2 as metric_DAV2
from Module.Network.Depth.DepthAnythingV3.api import DepthAnything3
from Module.Network.Depth.DepthCompletion import DepthCompletion


DEPTH_MODELS: dict[str, Callable[..., DepthModelProtocol]] = {
    "DepthAnythingV2": DepthAnythingV2,
    "MetricDepthAnythingV2": metric_DAV2,
    "DepthAnythingV3": DepthAnything3,
    "DepthCompletion": DepthCompletion,
}


def build_depth_model(name: str, **kwargs) -> DepthModelProtocol:
    model_cls = DEPTH_MODELS[name]

    if model_cls is None:
        raise ValueError(f"Model '{name}' not found in registry")

    return model_cls(**kwargs)
