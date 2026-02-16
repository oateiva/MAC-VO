import cv2
import re
import torch
import numpy as np
import pypose as pp
import roma
import datetime
import csv
from types import SimpleNamespace
from typing import Any, cast
from pathlib import Path
from torch.utils.data import Dataset
from Utility.PrettyPrint import Logger
from scipy.spatial.transform import Rotation as R
from django.db.models import QuerySet, OuterRef, Subquery

from ..SequenceBase import SequenceBase
from ..Interface    import Frame, CameraData
from ..Django_Sequence import DjangoORMSequence, ensure_django


def load_poses_from_txt_eiffeltower(file_name: str):
    """Load absolute camera poses from txt file (eiffeltower-style) as pp.SE3.

    Expected fields (in line[7:14]):
        x y z qx qy qz qw

    Returns:
        poses (dict): dictionary of poses, each pose is a pp.SE3 object
    """
    poses = {}

    with open(file_name, "r", newline="") as f:
        datareader = csv.reader(f)
        for i, line in enumerate(datareader):
            if i == 0:
                continue

            x, y, z, qx, qy, qz, qw = map(float, line[7:14])

            # pp.SE3 expects: [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
            poses[i - 1] = pp.SE3([x, y, z, qx, qy, qz, qw])

    # Normalize so first pose is identity (left-multiply by inv(pose_0))
    pose_0 = poses[next(iter(poses))]
    inv_pose_0 = pose_0.Inv()
    for k in list(poses.keys()):
        poses[k] = inv_pose_0 @ poses[k]

    return poses


class EiffelTowerSequence(SequenceBase[Frame]):
    @classmethod
    def name(cls) -> str: return "EiffelTower"

    def __init__(self, config: SimpleNamespace | dict[str, Any]):
        cfg = self.config_dict2ns(config)

        # Metadata (common)
        self.lcam_T_BS = pp.identity_SE3(1)
        self.lcam_K    = torch.tensor([
            [1847.5905420747683, 0.0, 1391.3],
            [0.0, 1847.5905420747683, 1407.177],
            [0.0, 0.0, 1.0]]).unsqueeze(0)
        self.width     = 2704
        self.height    = 1520
        # End

        self.is_stereo = cfg.is_stereo if hasattr(cfg, "is_stereo") else False
        self.window_length = cfg.window_length if hasattr(cfg, "window_length") else 1
        self.step_size = cfg.step_size if hasattr(cfg, "step_size") else 1

        # Loaders
        self.lcam_loader = EiffelTowerMonocularDataset(Path(cfg.root, "Cam0_images"), window_length=self.window_length, step_size=self.step_size)

        if self.is_stereo:
            assert ("EiffelTower does not provide stereo images, please set is_stereo to false in config if you wish to use this dataset")
        else:
            self.rcam_loader = None
            self.baseline    = -1.

        cam_time_file_path = Path(cfg.root, "Cam0_images")
        # list all files in dir
        cam_files = list(cam_time_file_path.glob("*.jpg")) + list(cam_time_file_path.glob("*.png"))
        cam_files.sort()
        # parse GT and timestamps
        poses_ts = load_poses_from_txt_EiffelTower(Path(cfg.root, "EstimatedState.csv"))

        self.lcam_time = [values for values in poses_ts.keys()]
        self.gt_poses = [values for values in poses_ts.values()]

        self.length = len(self.lcam_loader)

        super().__init__(self.length)

    def __getitem__(self, local_index: int) -> Frame:
        index   = self.get_index(local_index)
        index  = index + self.step_size-1 if index != 0 else index
        window_slice = slice(index, index + self.window_length)
        window_index_list = list(range(window_slice.start, window_slice.stop, window_slice.step or 1))
        return Frame(
            idx=window_index_list,
            camera=CameraData.from_mono(
                T_BS      = self.lcam_T_BS,
                K         = self.lcam_K,
                baseline  = torch.tensor([self.baseline]),
                time_ns   = [self.lcam_time[index]],
                height    = self.height,
                width     = self.width,
                images    = self.lcam_loader[index],
                # Ground truth and labels
                gt_depth  = None,
                gt_flow   = None,
                flow_mask = None,
            ),
            time_ns   = [self.lcam_time[index]],
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

class EiffelTowerMonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    Return the image in shape (1, 3, H, W) with dtype=float32
    and normalized (image in [0, 1])
    """
    def __init__(self, directory: Path, window_length: int, step_size: int = 1) -> None:
        super().__init__()
        self.window_length = window_length
        self.step_size = step_size
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
        return int(self.length/self.step_size - self.window_length + 1)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Output image tensor in shape of (N, C, H, W)
        result = []
        for i in range(self.window_length):
             img = self.load_png_format(self.file_names[index])
             img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
             img_tensor /= 255.
             result.append(img_tensor)
        result = torch.cat(result, dim=0)  # (N, C, H, W)

        return result


def loadEiffelTowerGT(path: Path) -> pp.LieTensor:
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
