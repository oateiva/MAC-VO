import torch
import gtsam
import numpy as np
import pypose as pp
import typing as typ
from dataclasses import dataclass

from Module.Map import MatchObs, PointNode
from Utility.PrettyPrint import Logger
from Utility.Point import pixel2point_NED, point2pixel_NED
from Utility.GTSAM_Utils import pypose_to_pose3, pose3_to_pypose, make_pose_to_point_factor
from ..PyposeOptimizers import AnalyticModule, FactorGraph
from collections import defaultdict
from typing import Dict, Tuple, Optional, List
import rerun as rr

@dataclass
class GraphInput:
    frame_idx         : torch.Tensor
    from_idx          : torch.Tensor
    from_pose         : pp.LieTensor
    init_motion       : pp.LieTensor
    baseline          : torch.Tensor
    observations      : MatchObs
    points            : PointNode
    images_intrinsic  : torch.Tensor
    edges_index       : torch.Tensor
    device            : str


@dataclass
class GTSAM_GraphInput:
    previous_graph_data: GraphInput
    current_graph_data : GraphInput
    indexes_prev_curr : typ.List[typ.Tuple[int, int]]


@dataclass
class GraphOutput:
    motion   : torch.Tensor
    from_idx : torch.Tensor
    frame_idx: torch.Tensor

@dataclass
class GTSAM_GraphOutput:
    frame_idexes: List[int]
    pose_estimates: List[torch.Tensor]
    landmark_indexes: Optional[list[int] | None] = None
    map_points: Optional[torch.Tensor] = None
    need_interp: Optional[bool] = None

PosePixelMap = Dict[int, Dict[Tuple[float, float], int]]

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

class GTSAM_Pose2Point(FactorGraph):
    def __init__(self):
        super().__init__()


    def parse_graph_data(self, graph_data: GTSAM_GraphInput):

        # Graph parsing
        idx = graph_data.current_graph_data.edges_index.detach().cpu().long().numpy()
        self.from_idx   = graph_data.current_graph_data.from_idx
        self.frame_idx  = graph_data.current_graph_data.frame_idx

        self.init_pypose = graph_data.current_graph_data.init_motion
        self.init_pose = pypose_to_pose3(graph_data.current_graph_data.init_motion)

        self.P0 = pypose_to_pose3(pp.SE3(graph_data.previous_graph_data.from_pose))

        self.obs_Tc_0 = pixel2point_NED( # in camera frame at t
            graph_data.previous_graph_data.observations.data["pixel1_uv"],
            graph_data.previous_graph_data.observations.data["pixel1_d"].squeeze(-1),
            graph_data.previous_graph_data.images_intrinsic
        ).detach().cpu().double().numpy()

        self.obs_Tc_1 = pixel2point_NED( # in camera frame at t
            graph_data.current_graph_data.observations.data["pixel1_uv"],
            graph_data.current_graph_data.observations.data["pixel1_d"].squeeze(-1),
            graph_data.current_graph_data.images_intrinsic
        ).detach().cpu().double().numpy()

        self.obs_Tc_2 = pixel2point_NED( # in camera frame at t+1
            graph_data.current_graph_data.observations.data["pixel2_uv"],
            graph_data.current_graph_data.observations.data["pixel2_d"].squeeze(-1),
            graph_data.current_graph_data.images_intrinsic
        ).detach().cpu().double().numpy()

        pts_Tw = graph_data.current_graph_data.points.data["pos_Tw"].detach().cpu().double().numpy()  # (N,3) # in world frame?

        self.obs1_covTc = graph_data.current_graph_data.observations.data["obs1_covTc"].detach().cpu().double().numpy()  # (N,3,3)
        self.obs2_covTc = graph_data.current_graph_data.observations.data["obs2_covTc"].detach().cpu().double().numpy()  # (N,3,3)
        self.pts_covTw = graph_data.current_graph_data.points.data["cov_Tw"].detach().cpu().double().numpy() # (N,3,3)

        self.previous_graph_data = graph_data.previous_graph_data
        self.indexes_prev_curr = graph_data.indexes_prev_curr

        # self.log_save_data(graph_data)

    def run_gtsam_optimization(self):

        # Build factor graph
        # Prior: pose 1 at identity

        graph = gtsam.NonlinearFactorGraph()

        pose_1_key = gtsam.symbol('p', int(self.from_idx.cpu().item()))
        pose_2_key = gtsam.symbol('p', int(self.frame_idx.cpu().item()))

        # Initial estimate: pose 1 at identity, pose 2 at init_motion
        initial_estimate = gtsam.Values()
        ini_estimate_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*6, dtype=np.float64))
        # Pose 1
        P1 = self.init_pose
        P2 = self.init_pose

        # Pose 1
        initial_estimate.insert(pose_1_key, P1)
        graph.add(
            gtsam.PriorFactorPose3(
                pose_1_key,
                P1,   # fixed at identity
                ini_estimate_noise
            ))
        # Pose 2
        initial_estimate.insert(pose_2_key, P2)

        p0_index = self.previous_graph_data.from_idx.cpu().item()
        if p0_index >= 0:
            pose_0_key = gtsam.symbol('p', int(p0_index))
            P0 = self.P0
                    # Pose 0
            initial_estimate.insert(pose_0_key, P0)
            graph.add(
                gtsam.PriorFactorPose3(
                    pose_0_key,
                    P0,
                    ini_estimate_noise
                ))

        landmark_keys = []
        landmark_idx = []
        for i in range(len(self.obs_Tc_1)):

            # Create landmark key
            landmark_key = gtsam.symbol('l', i)
            landmark_keys.append(landmark_key)
            landmark_idx.append(i)

            if any(i == idx_pair[1] for idx_pair in self.indexes_prev_curr) and p0_index >= 0:
                pi = [index_prev_curr[0] for index_prev_curr in self.indexes_prev_curr if index_prev_curr[1] == i][0]

                # Add BetweenFactor between pose_0_key and pose_2_key
                # between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5]*6))
                # graph.add(
                #     gtsam.BetweenFactorPose3(
                #         pose_0_key,
                #         pose_2_key,
                #         gtsam.Pose3.Identity(),
                #         between_noise
                #     )
                # )
                obs_Tc_0_i = self.obs_Tc_0[pi]
                cov_Tc_0_i = self.previous_graph_data.observations.data["obs1_covTc"][pi].detach().cpu().numpy()
                noise_model_0 = gtsam.noiseModel.Gaussian.Covariance(cov_Tc_0_i)
                m_huber_0 = gtsam.noiseModel.mEstimator.Huber.Create(1.)
                noise_model_0 = gtsam.noiseModel.Robust.Create(m_huber_0, noise_model_0)
                factor0 = make_pose_to_point_factor(pose_0_key, landmark_key, obs_Tc_0_i, noise_model_0)
                graph.add(factor0)


            # Read observation and covariance
            obs_Tc_1_i = self.obs_Tc_1[i]
            obs_Tc_2_i = self.obs_Tc_2[i]
            cov_Tc_1_i = self.obs1_covTc[i]
            cov_Tc_2_i = self.obs2_covTc[i]

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
            # Create factors
            factor1 = make_pose_to_point_factor(pose_1_key, landmark_key, obs_Tc_1_i, noise_model_1)
            factor2 = make_pose_to_point_factor(pose_2_key, landmark_key, obs_Tc_2_i, noise_model_2)

            # Add factors to graph
            graph.add(factor1)
            graph.add(factor2)

            # Add initial estimate for landmark
            Pt_landmark = P1.transformFrom(obs_Tc_1_i)
            initial_estimate.insert(landmark_key, Pt_landmark)

        # Optimize the graph
        params = gtsam.LevenbergMarquardtParams()
        # params.setVerbosityLM("SUMMARY")
        params.setMaxIterations(30)

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()

        pose_1 = result.atPose3(pose_1_key)
        pose_2 = result.atPose3(pose_2_key)
        landmark_positions = [result.atPoint3(landmark_keys[i]) for i in range(len(landmark_keys))]
        landmark_positions = torch.stack([torch.from_numpy(pos).double() for pos in landmark_positions], dim=0)  # (N,3)

        # self.log_plot_data(pose_1=pose_1, pose_2=pose_2, landmark_positions=landmark_positions)
        need_interp = False
        try:
            pose_1 = pose3_to_pypose(pose_1)
            pose_2 = pose3_to_pypose(pose_2)
        except Exception as e:
            Logger.write("error", f"Error converting optimized poses to PyPose format: {e}")
            pose_1 = self.init_pypose
            pose_2 = self.init_pypose
            need_interp = True

        self.graph_output = GTSAM_GraphOutput(
            frame_idexes=[int(self.from_idx.cpu().item()), int(self.frame_idx.cpu().item())],
            pose_estimates=[pose_1, pose_2],
            landmark_indexes=landmark_idx,
            map_points=landmark_positions,
            need_interp=need_interp
            )

    def write_back(self):
        return self.graph_output

    def covariance_array(self) -> torch.Tensor:
        return torch.from_numpy(self.obs2_covTc)

    def log_save_data(self, graph_data: GTSAM_GraphInput):
        import os
        import json

        frame_idx = graph_data.current_graph_data.frame_idx.cpu().item()

        frame_data = {
            "from_idx": graph_data.current_graph_data.from_idx.cpu().item(),
            "from_pose": graph_data.current_graph_data.from_pose.cpu().tolist(),
            "init_motion": graph_data.current_graph_data.init_motion.cpu().tolist(),
            "pixel1_uv": graph_data.current_graph_data.observations.data["pixel1_uv"].cpu().tolist(),
            "pixel2_uv": graph_data.current_graph_data.observations.data["pixel2_uv"].cpu().tolist(),
            "pixel1_uv_cov": graph_data.current_graph_data.observations.data["pixel1_uv_cov"].cpu().tolist(),
            "pixel2_uv_cov": graph_data.current_graph_data.observations.data["pixel2_uv_cov"].cpu().tolist(),
            "obs_Tc_1": self.obs_Tc_1.tolist(),
            "obs_Tc_2": self.obs_Tc_2.tolist(),
            "pts_Tw": graph_data.current_graph_data.points.data["pos_Tw"].cpu().tolist(),
            "obs1_covTc": self.obs1_covTc.tolist(),
            "obs2_covTc": self.obs2_covTc.tolist(),
            "pts_covTw": self.pts_covTw.tolist(),
            "images_intrinsic": graph_data.current_graph_data.images_intrinsic.cpu().tolist(),
        }

        json_path = os.path.join(os.getcwd(), "graph_data_dump.json")
        tmp_path = json_path + ".tmp"  # kept for structure/compat, but not used for append mode

        # Append-only JSON Lines: one frame per line, constant memory.
        record = {"frame_idx": int(frame_idx), "frame_data": frame_data}

        # Ensure directory exists (cwd should, but safe if you later change path)
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

        with open(json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())  # makes it robust if you crash mid-run



    def log_plot_data(self, pose_1, pose_2, landmark_positions):
        from Utility.Visualize import rr_plt
        rr.init("debug", spawn=True)
        i = int(self.from_idx.cpu().item())
        rr.set_time("frame_idx", sequence=i)
        pose_2_q = pose_2.rotation().toQuaternion()
        pose_2_t = pose_2.translation()
        pose_2 = pose3_to_pypose(pose_2)
        pose_1_q = pose_1.rotation().toQuaternion()
        pose_1_t = pose_1.translation()
        rr.log("/world/cam/{}".format(self.from_idx.cpu().item()),
                rr.Transform3D(
                    translation=pose_1_t,
                    quaternion=[pose_1_q.x(), pose_1_q.y(), pose_1_q.z(), pose_1_q.w()],
                    axis_length=1.0,
                ))
        K = np.array([[459.2732,   0.0000, 345.8487],
                        [  0.0000, 459.2732, 349.7954],
                        [  0.0000,   0.0000,   1.0000]])
        rr.log(
            "/world/cam/{}/points_i1".format(self.from_idx.cpu().item()),
            rr.Points3D(self.obs_Tc_1, colors=[255, 165, 0])
            )
        rr.log(
            "/world/cam/{}/points_i2".format(self.frame_idx.cpu().item()),
            rr.Points3D(self.obs_Tc_2, colors=[255, 0, 0])
            )

        # rr.set_time_sequence("step", graph_data.frame_idx)
        rr.log("/world/cam/{}".format(self.frame_idx.cpu().item()),
                rr.Transform3D(
                    translation=pose_2_t,
                    quaternion=[pose_2_q.x(), pose_2_q.y(), pose_2_q.z(), pose_2_q.w()],
                    axis_length=1.0
                ))

        rr.log(
            "/world/optimized/{}/points".format(self.frame_idx.cpu().item()),
            rr.Points3D(landmark_positions, colors=[0, 255, 0])
            )


class ISAM(FactorGraph):
    def __init__(self):
        super().__init__()
        # self.parse_graph_data(graph_data)
        # self.log_save_data(graph_data)
        self.pose_pixel_landmarks = {}

        # ---- iSAM2 setup ----
        self.isam_params = gtsam.ISAM2Params()
        self.isam_params.setRelinearizeThreshold(0.1)
        self.isam_params.relinearizeSkip = 5
        self.isam_params.enablePartialRelinearizationCheck = True
        self.isam = gtsam.ISAM2(self.isam_params)

        self.gauge_prior_added = False
        self.next_landmark_id = 0


    def parse_graph_data(self, graph_data: GTSAM_GraphInput):

        # Graph parsing
        idx = graph_data.current_graph_data.edges_index.detach().cpu().long().numpy()
        self.from_idx   = graph_data.current_graph_data.from_idx.cpu().item()
        self.frame_idx  = graph_data.current_graph_data.frame_idx.cpu().item()
        self.prev_from_idx = graph_data.previous_graph_data.from_idx.cpu().item()

        self.init_pose = pypose_to_pose3(graph_data.current_graph_data.init_motion)

        self.P0 = pypose_to_pose3(pp.SE3(graph_data.previous_graph_data.from_pose))

        self.pixel0_uv = graph_data.previous_graph_data.observations.data["pixel1_uv"].detach().cpu().double().numpy()
        self.pixel1_uv = graph_data.current_graph_data.observations.data["pixel1_uv"].detach().cpu().double().numpy()
        self.pixel2_uv = graph_data.current_graph_data.observations.data["pixel2_uv"].detach().cpu().double().numpy()

        self.obs_Tc_0 = pixel2point_NED( # in camera frame at t
            graph_data.previous_graph_data.observations.data["pixel1_uv"],
            graph_data.previous_graph_data.observations.data["pixel1_d"].squeeze(-1),
            graph_data.previous_graph_data.images_intrinsic
        ).detach().cpu().double().numpy()

        self.obs_Tc_1 = pixel2point_NED( # in camera frame at t
            graph_data.current_graph_data.observations.data["pixel1_uv"],
            graph_data.current_graph_data.observations.data["pixel1_d"].squeeze(-1),
            graph_data.current_graph_data.images_intrinsic
        ).detach().cpu().double().numpy()

        self.obs_Tc_2 = pixel2point_NED( # in camera frame at t+1
            graph_data.current_graph_data.observations.data["pixel2_uv"],
            graph_data.current_graph_data.observations.data["pixel2_d"].squeeze(-1),
            graph_data.current_graph_data.images_intrinsic
        ).detach().cpu().double().numpy()

        pts_Tw = graph_data.current_graph_data.points.data["pos_Tw"].detach().cpu().double().numpy()  # (N,3) # in world frame?

        self.obs1_covTc = graph_data.current_graph_data.observations.data["obs1_covTc"].detach().cpu().double().numpy()  # (N,3,3)
        self.obs2_covTc = graph_data.current_graph_data.observations.data["obs2_covTc"].detach().cpu().double().numpy()  # (N,3,3)
        self.pts_covTw = graph_data.current_graph_data.points.data["cov_Tw"].detach().cpu().double().numpy() # (N,3,3)

        self.previous_graph_data = graph_data.previous_graph_data

    def run_gtsam_optimization(self):

        # Only add *new* factors and *new* initial guesses each step
        new_factors = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()

        # Current estimate (for "exists" checks)
        est = self.isam.calculateEstimate() #if isam.size() > 0 else gtsam.Values()

        # Robust kernel
        m_huber = gtsam.noiseModel.mEstimator.Huber.Create(0.1)

        # Keys
        if self.prev_from_idx>=0:
            pose_0_key = gtsam.symbol('p', self.prev_from_idx)
            P0 = self.P0
            if not est.exists(pose_0_key):
                new_values.insert(pose_0_key, P0)

        pose_1_key = gtsam.symbol('p', self.from_idx)
        pose_2_key = gtsam.symbol('p', self.frame_idx)


        if est.exists(pose_1_key):
            P1 = est.atPose3(pose_1_key)
        else:
            P1 = self.init_pose
        P2 = self.init_pose

        # Insert pose initials if missing

        if not est.exists(pose_1_key):
            new_values.insert(pose_1_key, P1)
        if not est.exists(pose_2_key):
            new_values.insert(pose_2_key, P2)

        sigmas = np.array([1e-4] * 6, dtype=np.float64)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)

        if not self.gauge_prior_added:
            # Add a prior on the first pose to fix gauge freedom
            new_factors.add(
                gtsam.PriorFactorPose3(
                    pose_1_key,
                    P1,
                    prior_noise
                )
            )
            self.gauge_prior_added = True

        # Add weak factor to regularize the relative pose between pose_1 and pose_2 (encourages small motion, helps convergence)
        # it prevents the new pose from being completely unconstrained when landmarks are messy, and it helps stability.
        between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5,0.5,0.5, 0.5,0.5,0.5]))
        new_factors.add(gtsam.BetweenFactorPose3(pose_1_key, pose_2_key, gtsam.Pose3(), between_noise))


        # Landmarks
        landmark_keys = []
        cov_landmark_1 = []
        cov_landmark_2 = []

        eps = 1e-6

        for i in range(len(self.obs_Tc_1)):
            uv1 = (int(self.pixel1_uv[i][0]), int(self.pixel1_uv[i][1]))
            uv2 = (int(self.pixel2_uv[i][0]), int(self.pixel2_uv[i][1]))

            obs_c1 = self.obs_Tc_1[i]
            obs_c2 = self.obs_Tc_2[i]

            cov_c1 = np.array(self.obs1_covTc[i], dtype=np.float64) + eps * np.eye(3)
            cov_c2 = np.array(self.obs2_covTc[i], dtype=np.float64) + eps * np.eye(3)

            cov_landmark_1.append(cov_c1)
            cov_landmark_2.append(cov_c2)

            nm1 = gtsam.noiseModel.Robust.Create(m_huber, gtsam.noiseModel.Gaussian.Covariance(cov_c1))
            nm2 = gtsam.noiseModel.Robust.Create(m_huber, gtsam.noiseModel.Gaussian.Covariance(cov_c2))

            # Check if landmark already exists in dictionary (and thus, in graph)
            landmark_key = find_nearest_landmark(
                self.pose_pixel_landmarks,
                pose_1_key,
                uv1,
                tol=3.0
            )
            # if it exists, uv2 should have the same key
            if landmark_key is not None:
                self.pose_pixel_landmarks.setdefault(pose_2_key, {}).setdefault(uv2, landmark_key)
                new_factors.add(make_pose_to_point_factor(pose_1_key, landmark_key, obs_c1, nm1))
                new_factors.add(make_pose_to_point_factor(pose_2_key, landmark_key, obs_c2, nm2))

            else:
                landmark_key = gtsam.symbol('l', self.next_landmark_id)
                landmark_keys.append(landmark_key)
                self.pose_pixel_landmarks.setdefault(pose_1_key, {}).setdefault(uv1, landmark_key)
                self.pose_pixel_landmarks.setdefault(pose_2_key, {}).setdefault(uv2, landmark_key)

                # Factors for current frame pair
                new_factors.add(make_pose_to_point_factor(pose_1_key, landmark_key, obs_c1, nm1))
                new_factors.add(make_pose_to_point_factor(pose_2_key, landmark_key, obs_c2, nm2))

                self.next_landmark_id += 1

            # Initial for landmark if missing: use cam1 observation lifted into world with P1
            if not est.exists(landmark_key) and not new_values.exists(landmark_key):
                pw_init = P1.transformFrom(np.asarray(obs_c1, dtype=np.float64).reshape(3,))
                new_values.insert(landmark_key, pw_init)

        try:
            update = self.isam.update(new_factors, new_values)
        except Exception as e:
            Logger.write("error", f"Error during iSAM update: {e}")
        # update = self.isam.update()

        pose_window = self.get_pose_window(self.isam, self.frame_idx, K=50)
        frame_idexes = list(pose_window.keys())
        pose_estimates = [pose_window[idx] for idx in frame_idexes]

        # landmark_positions = [result.atPoint3(landmark_keys[i]) for i in range(len(landmark_keys))]
        landmark_idx, landmark_points = self.get_all_landmarks(self.isam)
        # Convert landmark_points (list of gtsam.Point3) to torch.Tensor (N,3)
        if len(landmark_points) > 0:
            landmark_points = torch.tensor(np.stack([np.asarray(p) for p in landmark_points]), dtype=torch.float32)
        else:
            landmark_points = torch.empty((0, 3), dtype=torch.float32)

        self.graph_output = GTSAM_GraphOutput(
            frame_idexes=frame_idexes,
            pose_estimates=pose_estimates,
            landmark_indexes=landmark_idx,
            map_points=landmark_points,
            )

        # self.log_plot_data(pose_1=pose_1, pose_2=pose_2, landmark_positions=landmark_positions)

        new_factors.resize(0)  # Clear new_factors to save memory; iSAM2 has already incorporated them
        new_values.clear()  # Clear new_values to save memory; iSAM2 has already incorporated them

        # Clear dictionary to speed up future lookups; iSAM2 has already incorporated the landmarks
        if len(self.pose_pixel_landmarks) > 3:
            for k in sorted(self.pose_pixel_landmarks.keys())[:-3]:
                del self.pose_pixel_landmarks[k]


    @staticmethod
    def get_pose_window(isam: gtsam.ISAM2, end_idx: int, K: int):
        """
        Returns dict {pose_index: gtsam.Pose3} for indices in [max(0,end_idx-K+1), end_idx]
        that exist in the current estimate.
        """
        est: gtsam.Values = isam.calculateEstimate()

        start_idx = max(0, end_idx - K + 1)
        poses = {}

        for i in range(start_idx, end_idx + 1):
            key = gtsam.symbol('p', i)
            if est.exists(key):
                poses[i] = pose3_to_pypose(est.atPose3(key))

        return poses

    @staticmethod
    def get_all_landmarks(isam: gtsam.ISAM2):
        """
        Returns:
            indices: list[int]        # sorted landmark indices
            points:  list[gtsam.Point3]
        """
        est = isam.calculateEstimate()

        landmark_keys = []

        # Collect all 'l' symbols
        for key in est.keys():
            sym = gtsam.Symbol(key)
            if sym.chr() == ord('l'):  # landmark symbol
                landmark_keys.append((sym.index(), key))

        # Sort by index
        landmark_keys.sort(key=lambda x: x[0])

        indices = []
        points = []

        for idx, key in landmark_keys:
            indices.append(idx)
            points.append(est.atPoint3(key))

        return indices, points

    def write_back(self):
        return self.graph_output

    def covariance_array(self) -> torch.Tensor:
        return torch.from_numpy(self.obs2_covTc)

    def log_save_data(self, graph_data: GTSAM_GraphInput):
        import os
        import json

        frame_idx = graph_data.current_graph_data.frame_idx.cpu().item()

        frame_data = {
            "from_idx": graph_data.current_graph_data.from_idx.cpu().item(),
            "from_pose": graph_data.current_graph_data.from_pose.cpu().tolist(),
            "init_motion": graph_data.current_graph_data.init_motion.cpu().tolist(),
            "pixel1_uv": graph_data.current_graph_data.observations.data["pixel1_uv"].cpu().tolist(),
            "pixel2_uv": graph_data.current_graph_data.observations.data["pixel2_uv"].cpu().tolist(),
            "pixel1_uv_cov": graph_data.current_graph_data.observations.data["pixel1_uv_cov"].cpu().tolist(),
            "pixel2_uv_cov": graph_data.current_graph_data.observations.data["pixel2_uv_cov"].cpu().tolist(),
            "obs_Tc_1": self.obs_Tc_1.tolist(),
            "obs_Tc_2": self.obs_Tc_2.tolist(),
            "pts_Tw": graph_data.current_graph_data.points.data["pos_Tw"].cpu().tolist(),
            "obs1_covTc": self.obs1_covTc.tolist(),
            "obs2_covTc": self.obs2_covTc.tolist(),
            "pts_covTw": self.pts_covTw.tolist(),
            "images_intrinsic": graph_data.current_graph_data.images_intrinsic.cpu().tolist(),
        }

        json_path = os.path.join(os.getcwd(), "graph_data_dump.json")
        tmp_path = json_path + ".tmp"  # kept for structure/compat, but not used for append mode

        # Append-only JSON Lines: one frame per line, constant memory.
        record = {"frame_idx": int(frame_idx), "frame_data": frame_data}

        # Ensure directory exists (cwd should, but safe if you later change path)
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

        with open(json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())  # makes it robust if you crash mid-run


    def log_plot_data(self, pose_1, pose_2, landmark_positions):
        from Utility.Visualize import rr_plt
        rr.init("debug", spawn=True)
        i = int(self.from_idx.cpu().item())
        rr.set_time("frame_idx", sequence=i)
        pose_2_q = pose_2.rotation().toQuaternion()
        pose_2_t = pose_2.translation()
        pose_2 = pose3_to_pypose(pose_2)
        pose_1_q = pose_1.rotation().toQuaternion()
        pose_1_t = pose_1.translation()
        rr.log("/world/cam/{}".format(self.from_idx.cpu().item()),
                rr.Transform3D(
                    translation=pose_1_t,
                    quaternion=[pose_1_q.x(), pose_1_q.y(), pose_1_q.z(), pose_1_q.w()],
                    axis_length=1.0,
                ))
        K = np.array([[459.2732,   0.0000, 345.8487],
                        [  0.0000, 459.2732, 349.7954],
                        [  0.0000,   0.0000,   1.0000]])
        rr.log(
            "/world/cam/{}/points_i1".format(self.from_idx.cpu().item()),
            rr.Points3D(self.obs_Tc_1, colors=[255, 165, 0])
            )
        rr.log(
            "/world/cam/{}/points_i2".format(self.frame_idx.cpu().item()),
            rr.Points3D(self.obs_Tc_2, colors=[255, 0, 0])
            )

        # rr.set_time_sequence("step", graph_data.frame_idx)
        rr.log("/world/cam/{}".format(self.frame_idx.cpu().item()),
                rr.Transform3D(
                    translation=pose_2_t,
                    quaternion=[pose_2_q.x(), pose_2_q.y(), pose_2_q.z(), pose_2_q.w()],
                    axis_length=1.0
                ))

        rr.log(
            "/world/optimized/{}/points".format(self.frame_idx.cpu().item()),
            rr.Points3D(landmark_positions, colors=[0, 255, 0])
            )

def next_landmark_index(values: gtsam.Values) -> int:
    max_idx = -1
    for k in values.keys():
        s = gtsam.Symbol(k)
        if s.chr() == ord('l'):
            max_idx = max(max_idx, s.index())
    return max_idx + 1

def find_nearest_landmark(
    pose_pixel_landmarks: PosePixelMap,
    pose_key: int,
    uv: Tuple[float, float],
    tol: float = 1.0,
) -> Optional[int]:
    inner = pose_pixel_landmarks.get(pose_key)
    if not inner:
        return None

    best_key = None
    best_dist2 = tol * tol

    u0, v0 = uv
    for (u, v), lk in inner.items():
        d2 = (u - u0)**2 + (v - v0)**2
        if d2 <= best_dist2:
            best_dist2 = d2
            best_key = lk

    return best_key
