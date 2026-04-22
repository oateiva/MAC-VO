from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Any
from DataLoader import CameraData

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from Module.Frontend.StereoDepth import IDepth

class DepthModelProtocol(Protocol):
    '''
    Protocol for depth estimation models.
    We use Protocols here to allow for structural sub typing,
    so any class that implements these methods can be considered a DepthModelProtocol,
    regardless of its inheritance.
    This is a minimal interface; it allows for the depth models
    to be as untouched as possible.
    '''

    def deepodo_initialize(self, config: Any) -> None:
        ...

    def deepodo_inference(self, input: CameraData) -> IDepth.Output:
        '''
        Custom depth inference method for deepodo models.
        This method serves as a bridge to maintain compatibility
        with the deepodo framework.
        '''
        ...
