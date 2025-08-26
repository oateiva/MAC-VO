
import re
from pathlib import Path
import argparse
import numpy as np
import datetime
from typing import List, Optional, Tuple


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


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z)."""
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    if qw < 1e-8:  # handle degenerate cases
        qw = 0
    qx = (R[2,1] - R[1,2]) / (4*qw) if abs(qw) > 1e-8 else 0
    qy = (R[0,2] - R[2,0]) / (4*qw) if abs(qw) > 1e-8 else 0
    qz = (R[1,0] - R[0,1]) / (4*qw) if abs(qw) > 1e-8 else 0
    return np.array([qw, qx, qy, qz])


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
            filename = parts[args.time_col]
            nanoseconds = zephyr_filename_to_ns(filename)
            t_ns = to_time_ns(nanoseconds, args.time_unit)

            tx, ty, tz = map(float, parts[1:4])
            R_vals = list(map(float, parts[4:13]))
            R = np.array(R_vals).reshape(3,3)

            quat = rotation_to_quaternion(R)  # [qw,qx,qy,qz]
            poses.append([t_ns, tx, ty, tz, *quat])

    poses = np.array(poses, dtype=np.float64)
    np.save(args.out, poses)
    print(f"Saved poses with shape {poses.shape} to {args.out}")


if __name__ == "__main__":
    main()
