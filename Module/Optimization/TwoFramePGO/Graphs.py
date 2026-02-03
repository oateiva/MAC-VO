import torch
import pypose as pp
import typing as typ
from dataclasses import dataclass

from Module.Map import MatchObs, PointNode
from Utility.Point import pixel2point_NED, point2pixel_NED
from ..PyposeOptimizers import AnalyticModule, FactorGraph
from typing import List,Optional
import rerun as rr
@dataclass
class GraphInput:
    frame_idx         : torch.Tensor
    from_idx          : torch.Tensor
    init_motion       : pp.LieTensor
    baseline          : torch.Tensor
    observations      : MatchObs
    points            : PointNode
    images_intrinsic  : torch.Tensor
    edges_index       : torch.Tensor
    device            : str


@dataclass
class GraphOutput:
    motion   : torch.Tensor
    from_idx : torch.Tensor
    frame_idx: torch.Tensor


############## Optimization Graphs

class ICP_TwoframePGO(FactorGraph):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.device                = graph_data.device
        self.init_motion           = graph_data.init_motion
        self.from_idx              = graph_data.from_idx
        self.frame_idx             = graph_data.frame_idx

        self.pose2opt       = pp.Parameter(pp.SE3(self.init_motion))
        self.edges_index    = graph_data.edges_index

        # ICP-based residual
        self.pts = graph_data.points
        self.obs = graph_data.observations

        self.points_Tc: torch.Tensor
        self.points_Tw: torch.Tensor

        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("points_Tc",
            pixel2point_NED(self.obs.data["pixel2_uv"], self.obs.data["pixel2_d"].squeeze(-1), graph_data.images_intrinsic)
        )
        self.register_buffer("points_Tw", self.pts.data["pos_Tw"])
        self.register_buffer("obs_covTc", self.obs.data["obs2_covTc"])
        self.register_buffer("pts_covTw", self.pts.data["cov_Tw"])


    def forward(self) -> torch.Tensor:
        frame_pose = typ.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        return frame_pose.Act(self.points_Tc) - self.points_Tw

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        frame_pose = typ.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R  = frame_pose.rotation().matrix()
        RT = R.transpose(-2, -1)
        return (R @ self.obs_covTc @ RT) + self.pts_covTw # type: ignore

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx)


class Reproj_TwoFramePGO(FactorGraph):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.from_idx : torch.Tensor = graph_data.from_idx
        self.frame_idx: torch.Tensor = graph_data.frame_idx
        self.init_motion:  pp.LieTensor = graph_data.init_motion

        self.pose2opt       = pp.Parameter(pp.SE3(self.init_motion))
        self.edges_index    = graph_data.edges_index

        self.pts     = graph_data.points
        self.obs     = graph_data.observations

        self.pos_Tc: torch.Tensor
        self.pos_Tw: torch.Tensor
        self.K: torch.Tensor
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("pos_Tw" , self.pts.data["pos_Tw"])
        self.register_buffer("cov_Tw" , self.pts.data["cov_Tw"])
        self.register_buffer("kp2"    , self.obs.data["pixel2_uv"])

        N = self.obs.data["pixel2_uv_cov"].size(0)
        # Build covar matrix for keypoints at t+1
        cov_kp2 = torch.empty((N, 2, 2))
        cov_kp2[:, 0, 0] = self.obs.data["pixel2_uv_cov"][:, 0]
        cov_kp2[:, 1, 1] = self.obs.data["pixel2_uv_cov"][:, 1]
        cov_kp2[:, 0, 1] = self.obs.data["pixel2_uv_cov"][:, 2]
        cov_kp2[:, 1, 0] = self.obs.data["pixel2_uv_cov"][:, 2]
        self.register_buffer("cov_kp2", cov_kp2)

    def forward(self) -> torch.Tensor:
        # Transform map points from world to camera frame at t+1
        self.pos_Tc = self.pose2opt.Inv().Act(self.pos_Tw)
        # Project map points to pixels at t+1
        K = typ.cast(torch.Tensor, self.K)
        kp2_reproj = point2pixel_NED(self.pos_Tc, K)
        # Then calculate reprojection residual
        kp2 = typ.cast(torch.Tensor, self.kp2)
        return kp2_reproj - kp2

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        return typ.cast(torch.Tensor, self.cov_kp2)

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        with torch.no_grad():
            return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx)


class ReprojDisp_TwoFramePGO(Reproj_TwoFramePGO):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)
        self.register_buffer("baseline", graph_data.baseline)
        self.baseline: torch.Tensor
        self.register_buffer("kp2_disparity", graph_data.observations.data["pixel2_disp"])

        cov_kp2 = typ.cast(torch.Tensor, self.cov_kp2)

        # Build covar matrix in 3D
        N = cov_kp2.size(0)
        cov = torch.zeros((N, 3, 3))
        cov[:, :2, :2] = cov_kp2
        # add disparity variance
        cov[:, 2, 2] = graph_data.observations.data["pixel2_disp_cov"].squeeze(-1)
        self.register_buffer("cov", cov)

    def forward(self) -> torch.Tensor:
        # Transform map points from world to camera frame at t+1
        self.pos_Tc = self.pose2opt.Inv() * self.pos_Tw

        ## Reprojection
        # Project map points to pixels at t+1
        K = typ.cast(torch.Tensor, self.K)
        kp2_reproj = point2pixel_NED(self.pos_Tc, K)
        # Then calculate reprojection residual
        kp2 = typ.cast(torch.Tensor, self.kp2)
        reproj_err = kp2_reproj - kp2

        ## Disparity
        bl = typ.cast(torch.Tensor, self.baseline)
        # convert depth (of map points in camera frame) to disparity w. pinhole model
        map_disparity = self.pos_Tc[:, 0:1].reciprocal() * (K[0, 0] * bl)
        kp2_disp = typ.cast(torch.Tensor, self.kp2_disparity)
        disp_err = map_disparity - kp2_disp

        return torch.cat((reproj_err, disp_err), dim=-1)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        return typ.cast(torch.Tensor, self.cov)


class Analytic_ICP_TwoframePGO(ICP_TwoframePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        frame_pose = typ.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R = frame_pose.rotation().matrix()
        p = self.points_Tc
        E = p.shape[0]

        J = torch.zeros((E, 3, 7), device=p.device, dtype=p.dtype)

        I3 = torch.eye(3, device=p.device, dtype=p.dtype).unsqueeze(0)
        J[..., 0:3] = I3
        J[..., 3:6] = -pp.vec2skew(frame_pose.Act(p))

        return J.view(-1, 7)


class Analytic_Reproj_TwoFramePGO(Reproj_TwoFramePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"
        # s = self.K[0, 1] # TODO: add this feature later!

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square
        J_homoKS[:, 0, 1] = fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_square
        J_homoKS[:, 1, 2] = fy / x

        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)  # 7 width because of pypose implementation, last column is useless
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J = (J_homoKS @ J_Tinv_p).view(-1, 7)
        return J


class Analytic_ReprojDisp_TwoFramePGO(ReprojDisp_TwoFramePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        ## Projection jacobian wrt. camera frame
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"
        # s = self.K[0, 1] # TODO: add this feature later!

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square
        J_homoKS[:, 0, 1] = fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_square
        J_homoKS[:, 1, 2] = fy / x

        # Derivaritive of T-1pw wrt pose
        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)  # 7 width because of pypose implementation, last column is useless
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)

        # Combine to final jacobian using chain rule
        J_reproj = (J_homoKS @ J_Tinv_p)

        # Disparity row: d(disparity)/dp = -(b*fx)/(x^2) * d(x)/dp
        # x row because in NED, x is depth/disparity direction
        J_disp = (-(self.baseline * fx) / x_square).view(-1, 1, 1) * J_Tinv_p[:, 0:1, :]

        # Stack to (N*3, 7)
        J = torch.cat((J_reproj, J_disp), dim=1).view(-1, 7)
        return J


class ReprojDepth_TwoFramePGO(Reproj_TwoFramePGO):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

        # ------------------
        # From MatchObs / observations:
        # - depth (m) for frame t+1 at kp2 locations
        # - optional depth variance (m^2)
        # We'll convert to inverse-depth and its variance.
        kp2_depth      = graph_data.observations.data["pixel2_d"]          # (N,1)
        kp2_depth_cov  = graph_data.observations.data["pixel2_d_cov"]      # (N,1) or missing/filled -1

        # Build inverse depth safely
        eps = 1e-8
        d   = torch.clamp(kp2_depth, min=eps) # (N,1) avoid div-by-zero
        idepth = d.reciprocal()                     # 1 / depth
        self.register_buffer("kp2_idepth", idepth)

        # Propagate depth variance to inverse-depth variance if provided
        if (kp2_depth_cov is not None) and (kp2_depth_cov.numel() > 0) and torch.all(kp2_depth_cov >= 0):
            # var(1/x) ≈ var(x) / x^4
            idepth_var = kp2_depth_cov / d.pow(4)
        else:
            sigma0 = 0.02     # ~2 cm base std in meters
            alpha  = 0.02     # ~2% relative error per meter

            sigma_d = torch.sqrt(sigma0**2 + (alpha * d)**2)          # (N,1)
            sigma_rho = sigma_d / (d**2 + eps)                        # std of inverse depth
            idepth_var = sigma_rho.pow(2)

        # clamp to reasonable range
        idepth_var_min = 1e-4   # (1/m)^2  -> std ~ 0.01 1/m
        idepth_var_max = 1.0    # (1/m)^2  -> std ~ 1.0 1/m
        idepth_var = torch.clamp(idepth_var, min=idepth_var_min, max=idepth_var_max)

        # Assemble 3x3 covariance over [u, v, idepth]
        cov_kp2 = self.cov_kp2  # (N, 2, 2) from parent __init__
        N = cov_kp2.size(0)

        cov3 = torch.zeros((N, 3, 3), device=cov_kp2.device, dtype=cov_kp2.dtype)
        cov3[:, :2, :2] = cov_kp2
        cov3[:, 2, 2]   = idepth_var.squeeze(-1)    # (N,)
        self.register_buffer("cov", cov3)

    def forward(self) -> torch.Tensor:
        # Transform points to camera frame at t+1
        self.pos_Tc = self.pose2opt.Inv() * self.pos_Tw

        # Project to pixels
        K = typ.cast(torch.Tensor, self.K)
        reproj_err = point2pixel_NED(self.pos_Tc, K) - typ.cast(torch.Tensor, self.kp2)

        # depth_err = (self.pos_Tc[:, 0:1] - self.kp2_depth)

        # Inverse-depth residual: 1/x_hat - 1/x_obs
        eps = 1e-8
        x = self.pos_Tc[:, 0:1]           # predicted depth (m)
        x_safe = torch.clamp(x, min=eps)
        idepth_hat = x_safe.reciprocal()       # 1 / x_hat

        valid = torch.isfinite(x) & (x > eps)
        idepth_err = idepth_hat - self.kp2_idepth
        idepth_err = torch.where(valid, idepth_err, torch.zeros_like(idepth_err))

        return torch.cat((reproj_err, idepth_err), dim=-1)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        return typ.cast(torch.Tensor, self.cov)


class Analytic_ReprojDepth_TwoFramePGO(ReprojDepth_TwoFramePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        assert self.K[0, 1] == 0, "K[0,1] skew not supported yet."

        eps = 1e-8
        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x = torch.clamp(x, min=eps)
        x_sq = x ** 2

        # 2x3 projection Jacobian in NED (x = depth)
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_sq
        J_homoKS[:, 0, 1] =  fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_sq
        J_homoKS[:, 1, 2] =  fy / x

        # Pose inverse Jacobian wrt Lie params (PyPose's 7-wide layout; last column unused)
        R   = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_Tinv_p[..., :3]  = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)

        # Reprojection rows (2x7 per feature)
        J_reproj = (J_homoKS @ J_Tinv_p)

        # Inverse-depth row: d(1/x)/dp = -(1/x^2) * d(x)/dp  (use x_safe)
        idepth_scale = (-(1.0) / x_sq).view(-1, 1, 1)
        J_idepth = idepth_scale * J_Tinv_p[:, 0:1, :]   # pick x-row

        # Zero Jacobian for invalid features (x not finite or <= eps)
        valid = torch.isfinite(x) & (x > eps)
        valid_ = valid.view(-1, 1, 1)
        J_reproj = torch.where(valid_, J_reproj, torch.zeros_like(J_reproj))
        J_idepth = torch.where(valid_, J_idepth, torch.zeros_like(J_idepth))

        # Stack to (N*3, 7)
        J = torch.cat((J_reproj, J_idepth), dim=1).view(-1, 7)
        # Final cleanup (guards against any remaining nan/inf)
        J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        return J

############### GTSAM ISAM2 Optimization Backend
import numpy as np
import gtsam
import json
import os
# from gtsam_unstable import PoseToPointFactor

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

def optimize_gtsam_lm(context: dict, graph_data: GraphInput):

    # Graph parsing
    idx = graph_data.edges_index.detach().cpu().long().numpy()
    print("Max value in edges_index:", idx.max())

    obs_Tc_1 = pixel2point_NED( # in camera frame at t
        graph_data.observations.data["pixel1_uv"],
        graph_data.observations.data["pixel1_d"].squeeze(-1),
        graph_data.images_intrinsic
    ).detach().cpu().double().numpy()

    obs_Tc_2 = pixel2point_NED( # in camera frame at t+1
        graph_data.observations.data["pixel2_uv"],
        graph_data.observations.data["pixel2_d"].squeeze(-1),
        graph_data.images_intrinsic
    ).detach().cpu().double().numpy()

    pts_Tw = graph_data.points.data["pos_Tw"].detach().cpu().double().numpy()  # (N,3) # in world frame?

    obs1_covTc = graph_data.observations.data["obs1_covTc"].detach().cpu().double().numpy()  # (N,3,3)
    obs2_covTc = graph_data.observations.data["obs2_covTc"].detach().cpu().double().numpy()  # (N,3,3)
    pts_covTw = graph_data.points.data["cov_Tw"].detach().cpu().double().numpy() # (N,3,3)

    # convert values to gtsam coords
    obs_Tc1_gtsam = obs_Tc_1
    obs_Tc2_gtsam = obs_Tc_2
    # obs_Tc1_gtsam = convert_macvo_to_gtsam_coords(obs_Tc_1)
    # obs_Tc2_gtsam = convert_macvo_to_gtsam_coords(obs_Tc_2)

    graph_data_dict = {
        "frame_idx": graph_data.frame_idx.cpu().item(),
        "from_idx": graph_data.from_idx.cpu().item(),
        "init_motion": graph_data.init_motion.cpu().tolist(),
        "pixel1_uv": graph_data.observations.data["pixel1_uv"].cpu().tolist(),
        "pixel2_uv": graph_data.observations.data["pixel2_uv"].cpu().tolist(),
        "pixel1_uv_cov": graph_data.observations.data["pixel1_uv_cov"].cpu().tolist(),
        "pixel2_uv_cov": graph_data.observations.data["pixel2_uv_cov"].cpu().tolist(),
        "obs_Tc_1": obs_Tc_1.tolist(),
        "obs_Tc_2": obs_Tc_2.tolist(),
        "pts_Tw": pts_Tw.tolist(),
        "obs1_covTc": obs1_covTc.tolist(),
        "obs2_covTc": obs2_covTc.tolist(),
        "pts_covTw": pts_covTw.tolist(),
        "images_intrinsic": graph_data.images_intrinsic.cpu().tolist(),
    }

    json_path = os.path.join(os.getcwd(), "graph_data_dump.json")
    with open(json_path, "w") as f:
        json.dump(graph_data_dict, f, indent=2)
    print(f"graph_data saved to {json_path}")

    # Build factor graph
    # Prior: pose 1 at identity
    graph = gtsam.NonlinearFactorGraph()
    pose_1_key = gtsam.symbol('p', 1)
    pose_2_key = gtsam.symbol('p', 2)
    # Initial estimate: pose 1 at identity, pose 2 at init_motion
    # Pose 1
    initial_estimate = gtsam.Values()
    P1 = gtsam.Pose3.Identity()
    P2 = gtsam.Pose3.Identity()
    init_pose = pypose_to_pose3(graph_data.init_motion)
    P1 = init_pose
    P2 = init_pose
    initial_estimate.insert(pose_1_key, P1)
    ini_estimate_noise = gtsam.noiseModel.Constrained.All(6)
    graph.add(
        gtsam.PriorFactorPose3(
            pose_1_key,
            P1,   # fixed at identity
            ini_estimate_noise
        ))
    # Pose 2
    initial_estimate.insert(pose_2_key, P2)

    landmark_keys = []
    for i in range(len(obs_Tc1_gtsam)):
        # Read observation and covariance
        obs_Tc_1_i = obs_Tc1_gtsam[i]
        obs_Tc_2_i = obs_Tc2_gtsam[i]
        cov_Tc_1_i = obs1_covTc[i]
        cov_Tc_2_i = obs2_covTc[i]

        # Create noise model
        noise_model_1 = gtsam.noiseModel.Gaussian.Covariance(cov_Tc_1_i)
        noise_model_2 = gtsam.noiseModel.Gaussian.Covariance(cov_Tc_2_i)
        m_huber = gtsam.noiseModel.mEstimator.Huber.Create(0.1)
        noise_model_1 = gtsam.noiseModel.Robust.Create(
            m_huber,
            noise_model_1
        )
        noise_model_2 = gtsam.noiseModel.Robust.Create(
            m_huber,
            noise_model_2
        )
        noise_model_1 = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        noise_model_2 = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        hubert_noise_1 = gtsam.noiseModel.mEstimator.Huber.Create(0.1)
        noise_model_1 = gtsam.noiseModel.Robust.Create(hubert_noise_1, noise_model_1)
        noise_model_2 = gtsam.noiseModel.Robust.Create(hubert_noise_1, noise_model_2)
        # Create landmark key
        landmark_key = gtsam.symbol('l', i)
        landmark_keys.append(landmark_key)
        # Create factors
        factor1 = make_pose_to_point_factor(pose_1_key, landmark_key, obs_Tc_1_i, noise_model_1)
        factor2 = make_pose_to_point_factor(pose_2_key, landmark_key, obs_Tc_2_i, noise_model_2)
        # factor1 = gtsam_unstable.PoseToPointFactor(pose_1_key, landmark_key, obs_Tc_1_i, noise_model_1)
        # factor2 = gtsam_unstable.PoseToPointFactor(pose_2_key, landmark_key, obs_Tc_2_i, noise_model_2)
        # Add factors to graph
        graph.add(factor1)
        graph.add(factor2)
        # Add initial estimate for landmark
        Pt_landmark = P1.transformFrom(obs_Tc_1_i)
        initial_estimate.insert(landmark_key, Pt_landmark)

    # Optimize the graph
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    # graph.print("Factor Graph:\n")
    # initial_estimate.print("Initial Estimate:\n")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    # print("GTSAM optimization complete. Final Result:\n{}".format(estimate))

    pose_2 = result.atPose3(pose_2_key)
    pose_2_q = pose_2.rotation().toQuaternion()
    pose_2_t = pose_2.translation()
    pose_2 = pose3_to_pypose(pose_2)

    # pose_0_to_1 = graph_data.init_motion
    # pose_0_to_2 = pose_0_to_1 @ pose_1_to_2
    pose_1 = result.atPose3(pose_1_key)
    pose_1_q = pose_1.rotation().toQuaternion()
    pose_1_t = pose_1.translation()
    landmark_positions = [result.atPoint3(landmark_keys[i]) for i in range(len(landmark_keys))]



    # from Utility.Visualize import rr_plt
    # rr.init("caca", spawn=True)
    # i = int(graph_data.from_idx.cpu().item())
    # rr.set_time("frame_idx", sequence=i)
    # rr.log("/world/cam/{}".format(graph_data.from_idx.cpu().item()),
    #         rr.Transform3D(
    #             translation=pose_1_t,
    #             quaternion=[pose_1_q.x(), pose_1_q.y(), pose_1_q.z(), pose_1_q.w()],
    #             axis_length=1.0,
    #         ))
    # K = np.array([[459.2732,   0.0000, 345.8487],
    #                 [  0.0000, 459.2732, 349.7954],
    #                 [  0.0000,   0.0000,   1.0000]])
    # rr.log(
    #     "/world/cam/{}/points_i1".format(graph_data.from_idx.cpu().item()),
    #     rr.Points3D(obs_Tc1_gtsam, colors=[255, 165, 0])
    #     )
    # rr.log(
    #     "/world/cam/{}/points_i2".format(graph_data.frame_idx.cpu().item()),
    #     rr.Points3D(obs_Tc2_gtsam, colors=[255, 0, 0])
    #     )

    # # rr.set_time_sequence("step", graph_data.frame_idx)
    # rr.log("/world/cam/{}".format(graph_data.frame_idx.cpu().item()),
    #         rr.Transform3D(
    #             translation=pose_2_t,
    #             quaternion=[pose_2_q.x(), pose_2_q.y(), pose_2_q.z(), pose_2_q.w()],
    #             axis_length=1.0
    #         ))

    # rr.log(
    #     "/world/optimized/{}/points".format(graph_data.frame_idx.cpu().item()),
    #     rr.Points3D(landmark_positions, colors=[0, 255, 0])
    #     )

    return context, GraphOutput(motion=pose_2, frame_idx=graph_data.frame_idx, from_idx=graph_data.from_idx)
