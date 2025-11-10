import torch
import pypose as pp
import typing as typ
from dataclasses import dataclass

from Module.Map import MatchObs, PointNode
from Utility.Point import pixel2point_NED, point2pixel_NED
from ..PyposeOptimizers import AnalyticModule, FactorGraph


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
        
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("points_Tc",
            pixel2point_NED(self.obs.data["pixel2_uv"], self.obs.data["pixel2_d"].squeeze(-1), graph_data.images_intrinsic)
        )
        self.points_Tc: torch.Tensor
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
