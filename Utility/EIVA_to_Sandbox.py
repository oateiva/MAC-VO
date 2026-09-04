
import re
from pathlib import Path
import argparse
import numpy as np
import datetime
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from Utility.Point import NED2EDN

# Mirrors the body-axis fix in DataLoader/Dataset/EIVA.py::loadEIVAGT: this script is a
# second, standalone producer of EIVA ground-truth poses (ref_poses.npy) from the same
# source data, so it needs the same EDN->NED camera/body-axis rebase. Precompute the 4x4
# homogeneous matrix once; right-multiplying a camera->world pose by it only replaces the
# rotation block (NED2EDN is a pure rotation), so translations are left bit-identical.
# Check the output convention with Scripts/AdHoc/audit_gt_convention.py.
_NED2EDN_MAT = NED2EDN.matrix().double().numpy()


def detect_delimiter(sample_line: str) -> str | None:
    if "," in sample_line:
        return ","
    return None  # use .split() for whitespace


def parse_slice(spec: str, ncols_hint: Optional[int] = None) -> tuple[Optional[int], Optional[int]]:
    if ":" not in spec:
        raise ValueError(f"Invalid slice spec '{spec}'. Expected 'start:end'.")
    start_s, end_s = spec.split(":", 1)
    start = int(start_s) if start_s != "" else 0
    end = int(end_s) if end_s != "" else ncols_hint
    return start, end


def parse_cols_arg(arg: str, ncols_hint: Optional[int] = None) -> list[int]:
    arg = arg.strip()
    if ":" in arg and all(part.strip().replace("-", "").isdigit() or part.strip() == "" for part in arg.split(":", 1)):
        start, end = parse_slice(arg, ncols_hint)
        if end is None:
            raise ValueError("When using a slice without an end, please provide --ncols-hint.")
        return list(range(start, end))
    return [int(x.strip()) for x in arg.split(",") if x.strip() != ""]


def to_time_ns(x: float, unit: str) -> int:
    if unit == "ns":
        return int(round(x))
    if unit == "us":
        return int(round(x * 1e3))
    if unit == "ms":
        return int(round(x * 1e6))
    if unit == "s":
        return int(round(x * 1e9))
    raise ValueError(f"Unsupported time unit: {unit}")


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 transformation matrix to its inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def rotation_to_quaternion(Rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (x,y,z,w)."""
    r = R.from_matrix(Rot)
    q = r.as_quat(scalar_first=False)  # (x,y,z,w) -- this is what pp.SE3 and the downstream loaders expect
    return np.array(q, dtype=np.float64)


def zephyr_filename_to_ns( filename):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt", type=Path)
    ap.add_argument("--out", type=Path, default=Path("ref_poses.npy"))
    ap.add_argument("--time-col", type=int, default=0)
    ap.add_argument("--time-unit", type=str, choices=["s","ms","us","ns"], default="s")
    ap.add_argument("--skip-header", type=int, default=0)
    ap.add_argument("--comment", type=str, default="#")
    args = ap.parse_args()

    poses = []
    with args.txt.open("r") as f:
        for _ in range(args.skip_header):
            next(f, None)
        for line in f:
            line = line.strip()
            if not line or (args.comment and line.startswith(args.comment)):
                continue
            parts = line.split()
            # filename = parts[args.time_col]
            # nanoseconds = zephyr_filename_to_ns(filename)
            # t_ns = to_time_ns(nanoseconds, args.time_unit)
            t_ns = parts[0]

            # tx, ty, tz = map(float, parts[1:4])
            # R_vals = list(map(float, parts[4:13]))
            # R = np.array(R_vals).reshape(3,3)
            # T = np.eye(4)
            # T[:3, :3] = R
            # T[:3, 3] = [tx, ty, tz]
            T = np.array(list(map(float, parts[1:17]))).reshape(4, 4)
            T_inv = inverse_transform(T)
            T_inv = T_inv @ _NED2EDN_MAT  # rebase camera/body axes EDN -> NED (see Utility.Point.NED2EDN)

            R_inv = T_inv[:3, :3]
            t_inv = T_inv[:3, 3]

            quat = rotation_to_quaternion(R_inv)  # [qx,qy,qz,qw]
            poses.append([t_ns, *t_inv, *quat])

    poses = np.array(poses, dtype=np.float64)
    np.save(args.out, poses)
    print(f"Saved poses with shape {poses.shape} to {args.out}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(poses[:,1], poses[:,2], poses[:,3], marker='o')
    ax.set_xlabel('tx')
    ax.set_ylabel('ty')
    plt.title('Trajectory: tx, ty, tz')
    plt.show()


if __name__ == "__main__":
    main()
