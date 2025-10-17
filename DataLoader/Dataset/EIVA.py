import cv2
import re
import torch
import numpy as np
import pypose as pp
import roma
import datetime
from types import SimpleNamespace
from typing import Any, cast
from pathlib import Path
from torch.utils.data import Dataset
from Utility.PrettyPrint import Logger
from django.db.models import QuerySet, OuterRef, Subquery

from ..SequenceBase import SequenceBase
from ..Interface    import Frame, CameraData
from ..Django_Sequence import DjangoORMSequence, ensure_django


def zephyr_filename_to_ns(filename):
    # Extract timestamp from filename
    match1 = re.search(r'(\d{4}-\d{2}-\d{2}T\d{6}\.\d+)', filename)
    match2 = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d+)', filename)
    if match1:
        timestamp_str = match1.group(0)
        timestamp_dt = datetime.datetime.strptime(
            timestamp_str,
            '%Y-%m-%dT%H%M%S.%f'
        )
        nanoseconds = float(timestamp_dt.timestamp() * 1e9)
    elif match2:
        timestamp_str = match2.group(0)
        timestamp_dt = datetime.datetime.strptime(
            timestamp_str,
            '%Y-%m-%dT%H-%M-%S-%f'
        )
        nanoseconds = float(timestamp_dt.timestamp() * 1e9)
    else:
        raise ValueError(f"Could not extract timestamp from filename: {filename}")  # noqa

    return nanoseconds


class EIVA_StereoSequenceORM(DjangoORMSequence[Frame]):
    def __init__(self, config: dict | Any):
        # Metadata
        self.lcam_T_BS = pp.identity_SE3(1)
        self.lcam_K    = torch.tensor([
            [1847.5905420747683, 0.0, 1391.3],
            [0.0, 1847.5905420747683, 1407.177],
            [0.0, 0.0, 1.0]]).unsqueeze(0)
        self.baseline  = 0.17007674086397787

        super().__init__(config)

    # ---------- queryset definition ----------
    def build_queryset(self, config: dict | Any) -> QuerySet:
        """
        Returns rows that uniquely define a stereo pair in order.
        """
        ensure_django()
        from annotationserver.sequence_data.models import SequenceImage

        seq_names = getattr(config, "sequence_camera_names", None)
        nav_names = getattr(config, "navigation_names", None)

        qs: QuerySet = SequenceImage.objects.all()

        if nav_names:
            qs = qs.filter(sequence__navigation__name__in=nav_names)
        if seq_names:
            # qs = qs.filter(sequence__name__in=seq_names)
            qs_left = qs.filter(sequence__name__in=[seq_names[0]]).order_by("datetime", "pk")
            qs_right = qs.filter(sequence__name__in=[seq_names[1]]).order_by("datetime", "pk") if len(seq_names) > 1 else None

        # Make sure your model relates left/right images or you can derive a pair
        # qs = qs.exclude(image__isnull=True).order_by("datetime", "pk")
        if qs_left is not None and qs_right is not None:
            # Annotate each left image with all right images having the same timestamp
            paired_qs = (
                qs_left
                .annotate(
                    right_id=Subquery(
                        qs_right.filter(datetime=OuterRef("datetime")).values("pk")[:1]
                    ),
                    right_image_path=Subquery(
                        qs_right.filter(datetime=OuterRef("datetime")).values("image__path")[:1]
                    ),
                )
                .filter(right_id__isnull=False)
                .order_by("datetime", "pk")
            )
            qs = paired_qs
        return qs

    def default_ordering(self) -> str:
        return "datetime"

    def optimize_fetch(self, queryset: QuerySet) -> QuerySet:
        # Pull only what you need and follow FKs in one query
        return (queryset
                .select_related("image")
                .only("pk", "datetime", "image__path"))

    # ---------- mapping ----------
    def record_to_frame(self, row: "SequenceImage", *, local_index: int, original_index: int) -> Frame:
        # timestamps (ensure ns)
        t = row.datetime  # datetime.time
        ns = ((t.hour * 3600) + (t.minute * 60) + t.second) * int(1e9) + t.microsecond * 1000

        imgL = self._read_path_to_tensor(row.image.path)
        imgR = self._read_path_to_tensor(row.right_image_path)

        B, C, H, W = imgL.shape
        return Frame(
            idx=[local_index],
            camera=CameraData.from_stereo(
                T_BS=self.lcam_T_BS,
                K=self.lcam_K,
                baseline=torch.tensor([self.baseline], dtype=torch.float32),
                time_ns=[ns],
                height=H, width=W,
                imageL=imgL,  # (1,C,H,W)
                imageR=imgR,
                gt_depth=None, gt_flow=None, flow_mask=None,
            ),
            time_ns=[ns],
            gt_pose=None,  # fill from your pose table if available
        )

    # ---------- helpers ----------
    @staticmethod
    def _read_path_to_tensor(path: str) -> torch.Tensor:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ten = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        ten /= 255.
        return ten


class EIVA_StereoSequence(SequenceBase[Frame]):
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

        cam_time_file_path = Path(cfg.root, "processed", "left")
        # list all files in dir
        cam_files = list(cam_time_file_path.glob("*.jpg"))
        cam_files.sort()
        self.lcam_time = [zephyr_filename_to_ns(f.name) for f in cam_files]

        # Pose Loader
        if cfg.gtPose:
            # gt_poses is originally on left camera sensor frame, need to convert to body frame
            self.gt_poses = self.lcam_T_BS @ loadEIVAGT(Path(cfg.root, "pose_gt.txt")) @ self.lcam_T_BS.Inv()
        else:
            self.gt_poses = None

        self.length = len(self.lcam_loader)

        super().__init__(self.length)

    def __getitem__(self, local_index: int) -> Frame:
        index   = self.get_index(local_index)
        return Frame(
            idx=[local_index],
            camera=CameraData.from_stereo(
                T_BS      = self.lcam_T_BS,
                K         = self.lcam_K,
                baseline  = torch.tensor([self.baseline]),
                time_ns   = [self.lcam_time[index]],
                height    = 2816,
                width     = 2816,
                imageL    = self.lcam_loader[index],
                imageR    = self.rcam_loader[index],

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
