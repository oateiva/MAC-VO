import cv2
import torch
import numpy as np
import pypose as pp
import roma
from types import SimpleNamespace
from typing import Any, cast
from pathlib import Path
from torch.utils.data import Dataset
from Utility.PrettyPrint import Logger

from DataLoader.Transform import IDataTransform
from ..SequenceBase import SequenceBase
from ..Interface    import StereoFrame, StereoData


class EIVA_StereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "EIVA_NoIMU"

    def __init__(self, config: SimpleNamespace | dict[str, Any]):
        cfg = self.config_dict2ns(config)

        # Metadata
        self.lcam_T_BS = pp.identity_SE3(1)
        self.lcam_K    = torch.tensor([
            [1847.5905420747683, 0.0, 1391.3],
            [0.0, 1847.5905420747683, 1407.177],
            [0.0, 0.0, 1.0]]).unsqueeze(0)
        self.baseline  = 0.17007674086397787
        self.width     = 2816
        self.height    = 2816
        # End

        # Stereo Loader
        self.lcam_loader = EIVAMonocularDataset(Path(cfg.root, "processed", "left"))
        self.rcam_loader = EIVAMonocularDataset(Path(cfg.root, "processed", "right"))

        cam_time_file_path = Path(cfg.root, "imu", "cam_time.npy")
        if cam_time_file_path.exists():
            self.lcam_time = (np.load(str(cam_time_file_path)) * 1_000_000_000).astype(np.int64)
        else:
            # Fake data, assume 10Hz image
            self.lcam_time = (np.arange(len(self.lcam_loader)) * 0.1 * 1_000_000_000).astype(np.int64)

        # Pose Loader
        if cfg.gtPose:
            # gt_poses is originally on left camera sensor frame, need to convert to body frame
            self.gt_poses = self.lcam_T_BS @ loadEIVAGT(Path(cfg.root, "pose_gt.txt")) @ self.lcam_T_BS.Inv()
        else:
            self.gt_poses = None

        self.length = len(self.lcam_loader)

        super().__init__(self.length)

    def __getitem__(self, local_index: int) -> StereoFrame:
        index   = self.get_index(local_index)
        return StereoFrame(
            idx=[local_index],
            stereo=StereoData(
                T_BS      = self.lcam_T_BS,
                K         = self.lcam_K,
                baseline  = torch.tensor([self.baseline]),
                time_ns   = [self.lcam_time[index].item()],  # Fake data, assume 10Hz image
                height    = 2816,
                width     = 2816,
                imageL    = self.lcam_loader[index],
                imageR    = self.rcam_loader[index],

                # Ground truth and labels
                gt_depth  = None,
                gt_flow   = None,
                flow_mask = None,
            ),
            time_ns   = [self.lcam_time[index].item()],  # Fake data, assume 10Hz image
            gt_pose   = cast(pp.LieTensor, self.gt_poses[index].unsqueeze(0)) if (self.gt_poses is not None) else None,
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root": lambda s: isinstance(s, str),
            "compressed": lambda b: isinstance(b, bool),
            "gtFlow"  : lambda b: isinstance(b, bool),
            "gtDepth" : lambda b: isinstance(b, bool),
            "gtPose"  : lambda b: isinstance(b, bool),
        })


# Specific dataset for a single sensor

class EIVAMonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    Return the image in shape (1, 3, H, W) with dtype=float32 
    and normalized (image in [0, 1])
    """
    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        assert self.directory.exists(), f"Monocular image directory {self.directory} does not exist"

        self.file_names = [f for f in self.directory.iterdir() if f.suffix == ".png" or f.suffix == ".jpg"]
        self.file_names.sort()
        self.length = len(self.file_names)
        assert self.length > 0, f"No flow with '.png' suffix is found under {self.directory}"

    @staticmethod
    def load_png_format(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Output image tensor in shape of (1, C, H, W)
        result = self.load_png_format(self.file_names[index])
        result = torch.tensor(result, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        result /= 255.

        return result


def loadEIVAGT(path: Path) -> pp.LieTensor:
    # Parse poses from the Zephyr Voyis format
    pose_list = []
    with open(str(path), 'r') as file:
        for line in file:
            parts = line.split(' ')
            position = torch.tensor([float(x) for x in parts[1:4]], dtype=torch.float32)
            so3_matrix = np.array([float(x) for x in parts[4:]])
            so3_matrix = torch.tensor(so3_matrix, dtype=torch.float32).reshape(3, 3)
            se3_matrix = torch.eye(4, dtype=torch.float32)
            se3_matrix[:3, :3] = so3_matrix
            se3_matrix[:3, 3] = position
            se3_matrix = se3_matrix.inverse()
            quat = roma.rotmat_to_unitquat(se3_matrix[:3, :3])
            t = se3_matrix[:3, 3]
            se3_data = np.concatenate([t, quat])
            pose_list.append(se3_data)
    poses = torch.tensor(pose_list, dtype=torch.float32)
    se3_data = pp.SE3(poses)

    return se3_data
