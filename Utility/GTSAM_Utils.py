import torch
import gtsam
import numpy as np
import pypose as pp
from typing import Optional

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

    # H is left unannotated: gtsam >= 4.3 types it as JacobianVector, older
    # stubs as Optional[List[ndarray]] — an annotation can't satisfy both.
    def error_func(this_factor, values, H):
        H_c: gtsam.Pose3 = values.atPose3(this_factor.keys()[0])
        Tw_k_i: np.ndarray = values.atPoint3(this_factor.keys()[1])

        if H is not None:
            H[0] = np.zeros((3, 6), dtype=np.float64)
            H[1] = np.zeros((3, 3), dtype=np.float64)

            pred_Tc_k_i = H_c.transformTo(Tw_k_i, H[0], H[1])

        else:
            pred_Tc_k_i = H_c.transformTo(Tw_k_i)  # (3,)

        r = pred_Tc_k_i - obs_Tc_k_i  # (3,)
        return r

    return gtsam.CustomFactor(noise_model, keys, error_func)

_SL4_BASIS_NP: Optional[np.ndarray] = None


def _sl4_basis_np() -> np.ndarray:
    """The 9-dim sl(4) complement basis, shared with the GEDF backend.

    Imported lazily to avoid an import cycle (Utility <- Module at load time)."""
    global _SL4_BASIS_NP
    if _SL4_BASIS_NP is None:
        from Module.Optimization.GEDF.Alignment import _sl4_complement_basis
        _SL4_BASIS_NP = _sl4_complement_basis().numpy()
    return _SL4_BASIS_NP


def make_alignment_warp(alignment_type: str):
    """
    Returns `warp(x, p) -> (warped_p (3,), d_warped_dx (3, E))` applying the
    per-frame alignment correction to a measured camera-frame point, with the
    analytic Jacobian w.r.t. the E alignment parameters.

    sim3 (E=1): warped = exp(x0) * p.
    sl4  (E=9): warped = dehomog(expm(sum_i x_i E_i) @ homog(p)); the Jacobian
    uses the exact matrix-exponential Frechet derivative (scipy). The expensive
    per-x quantities (W and dW_i) are cached, since one frame's factors all
    share the same x at each linearization point.
    """
    if alignment_type == "sim3":
        def warp_sim3(x: np.ndarray, p: np.ndarray):
            warped = float(np.exp(x[0])) * p
            return warped, warped.reshape(3, 1).copy()
        return warp_sim3

    if alignment_type == "sl4":
        from scipy.linalg import expm_frechet
        basis = _sl4_basis_np()
        cache: dict = {}

        def _exp_and_frechet(x: np.ndarray):
            key = x.tobytes()
            if key not in cache:
                A = np.einsum("i,ijk->jk", x, basis)
                dWs = []
                W = None
                for i in range(9):
                    W, dW = expm_frechet(A, basis[i])
                    dWs.append(dW)
                cache.clear()          # only the current linearization point matters
                cache[key] = (W, np.stack(dWs))
            return cache[key]

        def warp_sl4(x: np.ndarray, p: np.ndarray):
            W, dWs = _exp_and_frechet(np.asarray(x, dtype=np.float64).reshape(9))
            ph = np.append(p, 1.0)
            q = W @ ph
            w = max(float(q[3]), 0.25)              # same guard as the GEDF warp
            warped = q[:3] / w
            # d dehomog / d q_tilde = (1/w) [I3 | -warped]
            ddehom = np.concatenate([np.eye(3), -warped.reshape(3, 1)], axis=1) / w
            dq_dx = dWs @ ph                        # (9, 4)
            return warped, (ddehom @ dq_dx.T)       # (3, 9)
        return warp_sl4

    raise ValueError(f"No alignment warp for type '{alignment_type}'")


def make_aligned_pose_to_point_factor(pose_key, extras_key, landmark_key,
                                      obs_Tc_k_i, noise_model, warp):
    """
    Pose-to-point factor with a per-frame alignment correction on the measured
    camera point: residual = pose.transformTo(l_w) - warp(x)(obs). Keys:
    [pose (Pose3), extras (Vector, E dims), landmark (Point3)].
    """
    obs_Tc_k_i = np.asarray(obs_Tc_k_i, dtype=np.float64).reshape(3,)

    keys = [pose_key, extras_key, landmark_key]

    def error_func(this_factor, values, H):
        pose = values.atPose3(this_factor.keys()[0])
        x = values.atVector(this_factor.keys()[1])
        l_w = values.atPoint3(this_factor.keys()[2])

        warped, dwarp_dx = warp(x, obs_Tc_k_i)

        if H is not None:
            H[0] = np.zeros((3, 6), dtype=np.float64)
            H[2] = np.zeros((3, 3), dtype=np.float64)
            pred = pose.transformTo(l_w, H[0], H[2])
            H[1] = -np.asarray(dwarp_dx, dtype=np.float64)
        else:
            pred = pose.transformTo(l_w)

        return pred - warped

    return gtsam.CustomFactor(noise_model, keys, error_func)


def make_gedf_field_factor(pose_key, points_Tc, field_eval, noise_model):
    """
    ONE batched unary factor tying a pose to a G-EDF distance-field map:
    residual_i = d_hat(T . p_i) for every current-frame keypoint p_i (camera
    frame), an (N,)-dim CustomFactor on the pose alone. Batching matters: one
    factor = one Python callback and ONE batched field query per linearization,
    instead of N of each.

    `field_eval((N,3) world points) -> (r (N,), g (N,3))` must implement the
    field semantics (OOB rows: constant residual, zero gradient; gradient-norm
    clamp) — see Module/Optimization/GTSAM/Graphs.py.

    Jacobian (GTSAM Pose3 tangent [omega, v], q = R p + t, transformFrom
    H_pose = [-R [p]x | R]):  row_i = [ -(g_i^T R) x p_i  |  g_i^T R ].
    """
    points_Tc = np.asarray(points_Tc, dtype=np.float64).reshape(-1, 3)

    def error_func(this_factor, values, H):
        pose: gtsam.Pose3 = values.atPose3(this_factor.keys()[0])
        R = pose.rotation().matrix()                        # (3,3)
        t = np.asarray(pose.translation(), dtype=np.float64).reshape(3)
        q = points_Tc @ R.T + t                             # (N,3) world
        r, g = field_eval(q)                                # (N,), (N,3)

        if H is not None:
            GR = g @ R                                      # (N,3) rows g_i^T R
            J = np.zeros((points_Tc.shape[0], 6), dtype=np.float64)
            # a^T skew(b) = (a x b)^T  =>  -g^T R skew(p) = -(GR x p)
            J[:, :3] = -np.cross(GR, points_Tc)
            J[:, 3:] = GR
            H[0] = J
        return np.asarray(r, dtype=np.float64).reshape(-1)

    return gtsam.CustomFactor(noise_model, [pose_key], error_func)


def pypose_to_pose3(se3: pp.SE3) -> gtsam.Pose3:
    T = se3.matrix().detach().cpu().double().numpy()
    if T.ndim == 3: T = T[0]
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    return gtsam.Pose3(R, t)

def pose3_to_pypose(p: gtsam.Pose3) -> pp.SE3:
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = torch.from_numpy(p.rotation().matrix()).to(torch.float64)
    T[:3, 3] = torch.tensor([p.x(), p.y(), p.z()], dtype=torch.float64)
    T_SE3 = pp.from_matrix(T.unsqueeze(0), pp.SE3_type).to(torch.float32)

    # # Permutation matrix to convert from GTSAM (X-forward, Y-left, Z-up) to MACVO NED (Z-down, X-forward, Y-right)
    # P = torch.tensor([[0, 1, 0, 0],
    #                   [0, 0, 1, 0],
    #                   [1, 0, 0, 0],
    #                   [0, 0, 0, 1]], dtype=torch.float)
    # T_permuted = P @ T @ P.T
    # T_SE3 = pp.from_matrix(T_permuted.unsqueeze(0), pp.SE3_type)
    return T_SE3
