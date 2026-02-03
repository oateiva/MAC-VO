import torch
import gtsam
import numpy as np
import pypose as pp
from typing import List,Optional

def skew(p):
    return np.array([
        [0.0, -p[2],  p[1]],
        [p[2],  0.0, -p[0]],
        [-p[1], p[0], 0.0]
    ], dtype=np.float64)

def convert_macvo_to_gtsam_coords(points):
    """Convert MACVO points to GTSAM coordinates"""
    gtsam_points = []
    for pt in points:
        pt_gtsam = np.array([pt[1], pt[2], pt[0]], dtype=np.float64).reshape(3,)
        gtsam_points.append(pt_gtsam)
    return gtsam_points

def make_pose_to_point_factor(pose_key, landmark_key, obs_Tc_k_i, noise_model):
    obs_Tc_k_i = np.asarray(obs_Tc_k_i, dtype=np.float64).reshape(3,)

    keys = [pose_key, landmark_key]

    def error_func(this_factor, values,  H: Optional[List[np.ndarray]]):
        H_c: gtsam.Pose3 = values.atPose3(this_factor.keys()[0])
        Tw_k_i: gtsam.Point3 = values.atPoint3(this_factor.keys()[1])

        if H is not None:
            H[0] = np.zeros((3, 6), dtype=np.float64)
            H[1] = np.zeros((3, 3), dtype=np.float64)

            pred_Tc_k_i = H_c.transformTo(Tw_k_i, H[0], H[1])

        else:
            pred_Tc_k_i = H_c.transformTo(Tw_k_i)  # (3,)

        r = pred_Tc_k_i - obs_Tc_k_i  # (3,)
        return r

    return gtsam.CustomFactor(noise_model, keys, error_func)

def pypose_to_pose3(se3: pp.SE3) -> gtsam.Pose3:
    T = se3.matrix().detach().cpu().double().numpy()
    if T.ndim == 3: T = T[0]
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    return gtsam.Pose3(R, t)

def pose3_to_pypose(p: gtsam.Pose3) -> pp.SE3:
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = torch.from_numpy(p.rotation().matrix())
    T[:3, 3] = torch.tensor([p.x(), p.y(), p.z()], dtype=torch.float)
    T = T.to(dtype=torch.float)
    T_SE3 = pp.from_matrix(T.unsqueeze(0), pp.SE3_type)
    # # Permutation matrix to convert from GTSAM (X-forward, Y-left, Z-up) to MACVO NED (Z-down, X-forward, Y-right)
    # P = torch.tensor([[0, 1, 0, 0],
    #                   [0, 0, 1, 0],
    #                   [1, 0, 0, 0],
    #                   [0, 0, 0, 1]], dtype=torch.float)
    # T_permuted = P @ T @ P.T
    # T_SE3 = pp.from_matrix(T_permuted.unsqueeze(0), pp.SE3_type)
    return T_SE3
