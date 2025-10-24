from types import SimpleNamespace
from typing import Type, Any
import torch.nn as nn

class ModelSelector:
    from Module.Network.DepthAnythingV2.dpt import DepthAnythingV2

    _model_registry: dict[str, Type[nn.Module]] = {
        "DepthAnythingV2": DepthAnythingV2,
    }

    @staticmethod
    def get(config: SimpleNamespace) -> nn.Module:
        model_name = config.type
        model_cls = ModelSelector._model_registry.get(model_name)

        if model_cls is None:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        args = vars(config.args)
        return model_cls(
            **args
        )