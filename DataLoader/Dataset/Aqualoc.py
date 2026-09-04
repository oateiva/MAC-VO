from __future__ import annotations

import re
import cv2
import yaml
import torch
import numpy as np
import pypose as pp

from types import SimpleNamespace
from typing import Any, cast, Optional
from pathlib import Path
from torch.utils.data import Dataset

from Utility.PrettyPrint import Logger
from Utility.Math import interpolate_pose  # you already use this in EuRoC

from ..Interface import Frame, CameraData
from ..SequenceBase import SequenceBase


# Canonical definition lives in Utility.Point.EDN2NED / NED2EDN; duplicated here to avoid
# refactoring this already-correct dataloader.
# Keep the same convention you used in EuRoC, even if Aqualoc is already in some ENU/EDN.
# If you later discover Aqualoc's coordinate convention, you can swap this to the correct transform.
EDN2NED = pp.from_matrix(torch.tensor([
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
]), pp.SE3_type)
NED2EDN = EDN2NED.Inv()


class Aqualoc_MonoSequence(SequenceBase[Frame]):
    @classmethod
    def name(cls) -> str: return "Aqualoc_NoIMU"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)

        self.window_length = cfg.window_length if hasattr(cfg, "window_length") else 1

        self.root = Path(cfg.root)
        assert self.root.exists(), f"Aqualoc root does not exist: {self.root}"

        # Determine site variant. Use explicit config field if provided, otherwise
        # infer from the root path substring (legacy behaviour preserved for existing configs).
        if hasattr(cfg, "site") and cfg.site in {"harbor", "archaeo"}:
            self.site = cfg.site
        else:
            self.site = "harbor" if "harbor" in cfg.root else "archaeo"
        self.seq_number = self._extract_sequence_number(cfg.root)


        # Resolve sequence root
        # Remove the last two directories from the root path to get seqRoot
        self.seqRoot = self.root.parent.parent
        assert self.seqRoot.exists(), f"Aqualoc sequence path does not exist: {self.seqRoot}"

        # Intrinsics yaml (your importer uses a "sequence_x" placeholder and substitutes from navigation name)
        self.intrinsics_file = self._resolve_intrinsics_file(self.seqRoot, self.site, self.seq_number)
        assert self.intrinsics_file.exists(), f"Intrinsics file not found: {self.intrinsics_file}"

        # Image CSV (your importer uses a per-sequence img CSV living at .. / raw_data / img_sequence_X.csv)
        self.image_csv = self._resolve_image_csv(self.root, self.site, self.seq_number)
        assert self.image_csv.exists(), f"Image CSV not found: {self.image_csv}"

        # Image folder
        # In your importer: image_folder = <...>/raw_data/<images_sequence_# or harbor_images_sequence_#>
        # self.image_dir = self._resolve_image_dir(self.seqRoot, self.site, self.seq_number)
        self.image_dir = self.root
        assert self.image_dir.exists(), f"Image dir not found: {self.image_dir}"

        # Load camera model / intrinsics / distortion / resolution
        self.K, self.distort, self.width, self.height, self.camera_model_type = self._load_intrinsics(self.intrinsics_file)

        # Dataset providing mono images and camera timestamps
        self.Image = AqualocMonocularDataset(
            image_dir=self.image_dir,
            window_length = self.window_length,
            image_csv=self.image_csv,
            K=self.K,
            distort=self.distort,
            resolution=(self.width, self.height),
        )

        self.T_BS = pp.identity_SE3(1)  # (1,4,4)

        # Ground truth pose (optional): your importer reads txt traj files, then matches frames->timestamps via image CSV.
        self.gt_pose_data: Optional[pp.LieTensor] = None
        if hasattr(cfg, "gt_pose") and cfg.gt_pose:
            pose_txt = self._resolve_pose_file(self.seqRoot, self.site, self.seq_number, self.seqRoot)
            assert pose_txt.exists(), f"Pose file not found: {pose_txt}"
            self.gt_pose_data = load_AqualocGTPose(
                pose_txt,
                self.image_csv,
                self.Image.cam_timestamps
            )

        # Package the camera intrinsics tensor in the shape you use elsewhere
        self.K_t = torch.tensor(self.K, dtype=torch.float32).unsqueeze(0)

        super().__init__(len(self.gt_pose_data) if self.gt_pose_data is not None else len(self.Image))

    def __getitem__(self, local_index: int) -> Frame:
        index = self.get_index(local_index)
        t = int(self.Image.cam_timestamps[index].item())

        return Frame(
            idx=[local_index],
            time_ns=[t],
            camera=CameraData.from_mono(
                T_BS=self.T_BS,
                K=self.K_t,
                baseline=torch.tensor([0.0]),  # mono
                time_ns=[t],
                height=self.height,
                width=self.width,
                images=self.Image[index],
                gt_depth=None,
                gt_flow=None,
                flow_mask=None,
            ),
            gt_pose=None if self.gt_pose_data is None else cast(pp.LieTensor, self.gt_pose_data[index].unsqueeze(0)),
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root": lambda v: isinstance(v, str),
            "navigation_name": lambda v: isinstance(v, str),
            "navigation_path": lambda v: isinstance(v, str),
            "intrinsics_path": lambda v: isinstance(v, str),
            "pose_path": lambda v: isinstance(v, str),
            "gt_pose": lambda b: isinstance(b, bool),
            # Optional overrides
            # "sequence_camera_path": lambda v: isinstance(v, str),
            # "camera_dir_name": lambda v: isinstance(v, str),
        })

    @staticmethod
    def _extract_sequence_number(navigation_name: str) -> int:
        # Extracts the sequence number from navigation_name or path.
        # For archaeo: ..._1, ..._2, etc.
        # For harbor: ..._01, ..._02, etc.
        m = re.search(r"(?:_)(\d{1,2})$", navigation_name)
        if not m:
            raise ValueError(f"Could not extract sequence number from: {navigation_name}")
        return int(m.group(1).lstrip("0") or "0")

    @staticmethod
    def _resolve_intrinsics_file(intrinsics_path: Path, site: str, seq: int) -> Path:
        # Your importer replaces sequence_x inside the provided intrinsics_path based on navigation name.
        # intrinsics_path may be relative to seqRoot; support both.
        p = intrinsics_path
        if site == "archaeo":
            # If intrinsics_path is just the root, append the calibration folder for archaeo
            p = p / "archaeo_calibration_files" / "archaeo_camera_calib.yaml"
        else:
            p = p / "harbor_calibration_files" / "harbor_camera_calib.yaml"
        return p

    @staticmethod
    def _resolve_image_csv(seqRoot: Path, site: str, seq: int) -> Path:
        # seqRoot = .../<nav>/<raw_data>
        # CSV lives in .../<nav>/raw_data/../raw_data/<img_sequence_X.csv> per your importer,
        # effectively: .../<nav>/raw_data/../raw_data/<file> == .../<nav>/raw_data/<file>
        # In your code you used: join(*image_folder.split('\\')[:-1], '..', <csv>)
        # We'll resolve robustly.
        if site == "harbor":
            return (seqRoot.parent / f"harbor_img_sequence_{seq:02d}.csv").resolve()
        else:
            return (seqRoot.parent / f"img_sequence_{seq}.csv").resolve()

    @staticmethod
    def _resolve_image_dir(seqRoot: Path, site: str, seq: int) -> Path:
        # In your importer, camera dir naming is:
        #   harbor_images_sequence_x  (harbor)
        #   images_sequence_x         (archaeo)
        # located inside raw_data/
        if site == "harbor":
            return (seqRoot / f"harbor_images_sequence_{seq}").resolve()
        else:
            return (seqRoot / f"images_sequence_{seq}").resolve()

    @staticmethod
    def _resolve_pose_file(pose_path: Path, site: str, seq: int, seqRoot: Path) -> Path:
        # Your importer:
        #  - harbor: ..\harbor_groundtruth_files\new_harbor_colmap_traj_sequence_x.txt
        #  - archaeo: ..\new_archaeo_colmap_traj_sequence_x.txt (and you had a 0-padding bug)
        #
        # We treat pose_path as relative to seqRoot unless it's absolute.
        p = pose_path
        if not p.is_absolute():
            p = (seqRoot / p).resolve()

        # Replace placeholder
        if site == "archaeo":
            p = p / "archaeo_groundtruth_files" / f"new_archaeo_colmap_traj_sequence_{seq:02d}.txt"
        else:            p = p / "harbor_groundtruth_files" / f"new_harbor_colmap_traj_sequence_{seq:02d}.txt"
        return p

    @staticmethod
    def _load_intrinsics(intrinsics_file: Path) -> tuple[np.ndarray, np.ndarray, int, int, str]:
        with open(intrinsics_file, "r") as f:
            sensor = yaml.safe_load(f)
        cam0 = sensor["cam0"]

        fx, fy, cx, cy = cam0["intrinsics"]
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        # Aqualoc yaml stores distortion_coeffs as [k1, k2, p1, p2] for radtan in many setups.
        distort = np.array(cam0.get("distortion_coeffs", [0, 0, 0, 0]), dtype=np.float64)

        # Note: your importer swaps resolution ordering:
        # resolution = [sensor['resolution'][1], sensor['resolution'][0]]
        # We'll keep width/height as (W,H) in variables.
        res = cam0["resolution"]
        width, height = int(res[0]), int(res[1])

        cam_model = cam0.get("camera_model", "pinhole")
        dist_model = cam0.get("distortion_model", "radtan")
        if cam_model == "pinhole" and dist_model == "radtan":
            model_type = "PinholeRadTan"
        elif cam_model == "pinhole" and dist_model == "equidistant":
            model_type = "Equidistant"
        else:
            model_type = f"{cam_model}_{dist_model}"

        return K, distort, width, height, model_type


class AqualocMonocularDataset(Dataset):
    """
    Aqualoc mono image dataset using the per-sequence CSV:
        <timestamp_ns>,<relative_image_path_or_filename>
    and images living under a folder such as:
        raw_data/images_sequence_{k}/frame000001.png
    """

    def __init__(
        self,
        image_dir: Path,
        window_length:int,
        image_csv: Path,
        K: np.ndarray,
        distort: np.ndarray,
        resolution: tuple[int, int],
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.image_csv = image_csv
        self.K = K
        self.distort = distort
        self.width, self.height = resolution
        self.window_length = window_length

        assert self.image_dir.exists(), f"Aqualoc image dir does not exist: {self.image_dir}"
        assert self.image_csv.exists(), f"Aqualoc image CSV does not exist: {self.image_csv}"

        self.cam_timestamps, self.file_names = self._read_csv(self.image_csv, self.image_dir)
        self.length = len(self.file_names)
        assert self.length > 0, f"No images listed in {self.image_csv}"

        # For now, do NOT undistort/rectify automatically (Aqualoc is mono).
        # If you want undistortion, implement cv2.initUndistortRectifyMap similar to EuRoC.
        self.undistort_map: None | tuple[np.ndarray, np.ndarray] = None

    @staticmethod
    def _read_csv(csv_path: Path, image_dir: Path) -> tuple[torch.Tensor, list[Path]]:
        # Skip header; column0 timestamp (ns), column1 image filename
        lines = csv_path.read_text().splitlines()[1:]
        ts = []
        files: list[Path] = []
        for ln in lines:
            parts = ln.split(",")
            if len(parts) < 2:
                continue
            t_ns = int(float(parts[0]))
            fn = parts[1].strip()
            ts.append(t_ns)
            files.append((image_dir / fn).resolve())
        return torch.tensor(ts, dtype=torch.long), files

    def __len__(self) -> int:
        return int(self.length - self.window_length + 1)

    def __getitem__(self, index: int) -> torch.Tensor:
        result = []
        for i in range(self.window_length):
            img = cv2.imread(str(self.file_names[index]), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {self.file_names[index]}")
            # BGR->RGB (optional). If your pipeline expects BGR like EuRoC code currently uses, remove this.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Undistort image if distortion coefficients are nonzero
            if np.any(self.distort != 0):
                img = cv2.undistort(img, self.K, self.distort)

            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            result.append(t)
        result = torch.cat(result, dim=0)
        return result


def load_AqualocGTPose(
    pose_txt_path: Path,
    image_csv_path: Path,
    cam_time: torch.Tensor
) -> pp.LieTensor:
    """
    Your importer’s pose file format:
        frame px py pz qx qy qz qw
    and timestamps are recovered by matching the frameXXXXXX.png entries
    inside the image CSV.

    Returns:
        pose aligned/interpolated onto cam_time (one pose per cam frame).
    """
    # 1) Build frame->timestamp map from the image CSV
    # CSV format: timestamp_ns, filename
    lines = image_csv_path.read_text().splitlines()[1:]
    frame_time = {}
    for ln in lines:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        t_ns = int(float(parts[0]))
        fn = parts[1].strip()
        m = re.search(r"frame(\d+)\.png", fn)
        if m:
            frame_idx = int(m.group(1))
            frame_time[frame_idx] = t_ns

    # 2) Read pose txt -> (pose_time, pose_SE3)
    pose_lines = pose_txt_path.read_text().splitlines()
    pose_t = []
    pose_vec = []  # [tx,ty,tz,qx,qy,qz,qw] in xyzw for pp.SE3 later
    missing = 0

    for ln in pose_lines:
        if not ln.strip():
            continue
        parts = ln.strip().split()
        if len(parts) < 8:
            continue
        frame = int(float(parts[0]))

        if frame not in frame_time:
            missing += 1
            continue

        tx, ty, tz = map(float, parts[1:4])
        qx, qy, qz, qw = map(float, parts[4:8])

        pose_t.append(frame_time[frame])
        # pp.SE3 expects [x y z qx qy qz qw] in xyzw
        pose_vec.append([tx, ty, tz, qx, qy, qz, qw])

    if missing > 0:
        Logger.write("warn", f"Aqualoc GT pose: {missing} pose frames missing timestamps in image CSV")

    pose_t = torch.tensor(pose_t, dtype=torch.long)
    pose_vec = torch.tensor(pose_vec, dtype=torch.float64)
    pose_SE3 = pp.SE3(pose_vec)

    # 3) Interpolate onto cam_time (cam_time is torch.long already)
    # Mask camera times to pose valid range to avoid extrapolation
    cam_np = cam_time.squeeze().numpy()
    t0 = int(pose_t[0].item())
    t1 = int(pose_t[-1].item())
    cam_mask = (cam_np > t0) & (cam_np < t1)

    bodyPose_SE3, _ = interpolate_pose(pose_SE3, pose_t, cam_time[cam_mask].double())
    # Re-expand to full length by dropping invalid front/back (matches EuRoC behavior with masking).
    # If you prefer to keep full length and set None for invalid, do it at sequence-level instead.
    if cam_mask.sum() != cam_time.numel():
        Logger.write("warn", "Aqualoc: camera timestamps were masked to match GT pose validity range. "
                             "If you need full-length frames, handle invalid GT outside.")
    return cast(pp.LieTensor, bodyPose_SE3)
