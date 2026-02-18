import numpy as np
import rerun as rr
import torch
import pypose as pp

DTYPE = torch.float64

# --- Mirror (reflection) ---
MIRROR3 = torch.diag(torch.tensor([-1.0, 1.0, -1.0], dtype=DTYPE))
T_MIRROR = torch.eye(4, dtype=DTYPE)
T_MIRROR[:3, :3] = MIRROR3

R_y90 = pp.euler2SO3(torch.tensor([0.0, -np.pi / 2.0, -np.pi / 2.0], dtype=DTYPE)).matrix()
T_RY90 = torch.eye(4, dtype=DTYPE)
T_RY90[:3, :3] = R_y90

# Compose: mirror THEN rotate
T_MIRROR = T_RY90 @ T_MIRROR

# Global origin transform (will be set from the first pose)
T_ORIGIN = None  # type: torch.Tensor | None


def _to_homogeneous_points(pts_xyz: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((pts_xyz.shape[0], 1), dtype=pts_xyz.dtype, device=pts_xyz.device)
    return torch.cat([pts_xyz, ones], dim=1)


def _apply_T_to_points(T: torch.Tensor, pts_xyz: torch.Tensor) -> torch.Tensor:
    pts_h = _to_homogeneous_points(pts_xyz)      # (N,4)
    pts_h2 = (T @ pts_h.T).T                     # (N,4)
    return pts_h2[:, :3]


def load_vslam_track(file_path: str, *, entity: str = "world/vslam", entity_name: str = "vslam_track") -> None:
    global T_ORIGIN

    rr.init(entity_name, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    positions = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            easting = float(parts[2])
            northing = float(parts[3])
            depth = float(parts[4])
            roll_deg = float(parts[5])
            pitch_deg = float(parts[6])
            yaw_deg = float(parts[7])

            t = torch.tensor([easting, northing, -depth], dtype=DTYPE)

            rpy = torch.tensor(
                [np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)],
                dtype=DTYPE,
            )
            # Match: Rz(yaw) @ Ry(pitch) @ Rx(roll)
            R = pp.euler2SO3(rpy).matrix()

            T = torch.eye(4, dtype=DTYPE)
            T[:3, :3] = R
            T[:3, 3] = t

            if T_ORIGIN is None:
                T_ORIGIN = torch.linalg.inv(T)

            T_fix = T_MIRROR @ T_ORIGIN
            T_out = T_fix @ T

            R_out = T_out[:3, :3]
            t_out = T_out[:3, 3]

            positions.append(t_out.detach().cpu().numpy())

            rr.log(
                f"{entity}/pose",
                rr.Transform3D(
                    translation=t_out.detach().cpu().numpy().tolist(),
                    mat3x3=R_out.detach().cpu().numpy(),
                ),
            )

    if positions:
        rr.log(f"{entity}/traj", rr.LineStrips3D([np.stack(positions, axis=0)]))


def load_vslam_pointcloud(
    file_path: str,
    *,
    entity: str = "world/vslam/point_cloud",
    entity_name: str = "vslam_track",
) -> None:
    global T_ORIGIN

    rr.init(entity_name, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    if T_ORIGIN is None:
        raise RuntimeError("T_ORIGIN is not set. Call load_vslam_track(...) first.")

    points = []
    colors = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 6:
                continue

            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            r = int(parts[3])
            g = int(parts[4])
            b = int(parts[5])

            points.append([x, y, -z])
            colors.append([r, g, b])

    if points:
        pts = torch.tensor(points, dtype=DTYPE)  # (N,3)

        T_fix = T_MIRROR @ T_ORIGIN
        pts_out = _apply_T_to_points(T_fix, pts)

        rr.log(
            entity,
            rr.Points3D(
                pts_out.detach().cpu().numpy(),
                colors=np.array(colors),
                radii=0.05,
            ),
        )
