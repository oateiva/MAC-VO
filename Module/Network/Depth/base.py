from typing import Protocol, Any
from DataLoader import CameraData
from Module.Frontend.StereoDepth import IDepth

import torch
import torch.nn as nn

class DepthModelProtocol(Protocol):
    '''
    Protocol for depth estimation models.
    We use Protocols here to allow for structural sub typing,
    so any class that implements these methods can be considered a DepthModelProtocol,
    regardless of its inheritance.
    This is a minimal interface; it allows for the depth models
    to be as untouched as possible.
    '''

    def deepodo_inference(self, input: CameraData) -> IDepth:
        '''
        Custom depth inference method for deepodo models.
        This method serves as a bridge to maintain compatibility
        with the deepodo framework.
        '''
        ...
    def eval(self) -> None:
        ...
    def to(self, device: torch.device) -> None:
        ...
