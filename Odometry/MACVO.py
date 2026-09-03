import torch
import rerun as rr
from Utility.Visualize import rr_plt
import pypose as pp
import typing as T
from dataclasses import dataclass
from types import SimpleNamespace

from rich.columns import Columns
from rich.panel import Panel
from typing import Callable

import Module
from DataLoader import Frame, CameraData
from Module.Map import VisualMap, FrameNode, MatchObs, PointNode
from Module.KeyframeTracker import TrackContext
from Utility.Point import filterPointsInRange, pixel2point_NED, point2pixel_NED
from Utility.PrettyPrint import Logger, GlobalConsole
from Utility.Timer import Timer
from Utility.Visualize import fig_plt
from Utility.Extensions import ConfigTestable
from .Interface import IOdometry

T_SensorFrame = T.TypeVar("T_SensorFrame", bound=Frame)


@dataclass
class _KeyframeState:
    """The reference frame every new frame is additionally matched against.

    `obs` / `point_idx` are the rows (and the points they born) registered when
    this frame was frame0 of a consecutive pair; None until that pair has run, so
    a keyframe adopted at frame k yields its first keyframe rows at frame k+2.
    """
    camera   : CameraData
    frame_idx: int
    depth    : Module.IDepth.Output
    obs      : MatchObs | None = None
    point_idx: torch.Tensor | None = None


class MACVO(IOdometry[T_SensorFrame], ConfigTestable):
    # Type alias of callback hooks for MAC-VO system. Will be called by the system on
    # certain event occurs (optimization finish, for instance.)
    T_SYSHOOK = Callable[["MACVO",], None]

    def __init__(
        self,
        device, num_point, edgewidth, match_cov_default, profile, mapping,
        frontend        : Module.IFrontend,
        motion_model    : Module.IMotionModel[T_SensorFrame],
        kp_selector     : Module.IKeypointSelector,
        map_selector    : Module.IKeypointSelector,
        obs_filter      : Module.IObservationFilter,
        obs_covmodel    : Module.ICovariance2to3,
        post_process    : Module.IMapProcessor,
        kf_selector     : Module.IKeyframeSelector[T_SensorFrame],
        optimizer       : Module.IOptimizer,
        min_num_point   : int = 10,
        keyframe_tracker: Module.IKeyframePolicy | None = None,
        **_excessive_args,
    ) -> None:
        super().__init__(profile=profile)
        if len(_excessive_args) > 0:
            Logger.write("warn", f"Receive excessive arguments for __init__ {_excessive_args}, update/clean up your config!")

        self.graph = VisualMap()
        self.device = device
        self.mapping: bool = mapping
        self.match_cov_default: float = match_cov_default

        # Modules
        self.Frontend = frontend
        self.MotionEstimator = motion_model
        self.KeypointSelector = kp_selector
        self.MappointSelector = map_selector
        self.OutlierFilter = obs_filter
        self.ObsCovModel = obs_covmodel
        self.MapRefiner = post_process
        self.KeyframeSelector = kf_selector
        self.Optimizer = optimizer
        self.KeyframeTracker = keyframe_tracker
        # end

        self.min_num_point = min_num_point
        self.num_point = num_point
        self.edge_width = edgewidth
        self.isinitiated = False

        # Context for tracking
        # [0] - Frame Source Data
        # [1] - Frame index (in visual map)
        # [2] - Frame stereo depth
        self.prev_keyframe: tuple[T_SensorFrame, int, Module.IStereoDepth.Output] | None = None
        # Keyframe-tracker reference frame (None when the tracker is off)
        self.keyframe: _KeyframeState | None = None

        # Hooks
        self.on_optimize_writeback: list[MACVO.T_SYSHOOK] = []

        self.report_config()

    @classmethod
    def from_config(cls, cfg: SimpleNamespace):
        odomcfg = cfg.Odometry
        # Initialize modules for VO
        Frontend            = Module.IFrontend.instantiate(odomcfg.frontend.type, odomcfg.frontend.args)
        MotionEstimator     = Module.IMotionModel[T_SensorFrame].instantiate(odomcfg.motion.type, odomcfg.motion.args)
        KeypointSelector    = Module.IKeypointSelector.instantiate(odomcfg.keypoint.type, odomcfg.keypoint.args)
        MappointSelector    = Module.IKeypointSelector.instantiate(odomcfg.mappoint.type, odomcfg.mappoint.args)
        ObservationFilter   = Module.IObservationFilter.instantiate(odomcfg.outlier.type, odomcfg.outlier.args)
        ObserveCovModel     = Module.ICovariance2to3.instantiate(odomcfg.cov.obs.type, odomcfg.cov.obs.args)
        MapRefiner          = Module.IMapProcessor.instantiate(odomcfg.postprocess.type, odomcfg.postprocess.args)
        KeyframeSelector    = Module.IKeyframeSelector[T_SensorFrame].instantiate(odomcfg.keyframe.type, odomcfg.keyframe.args)
        Optimizer           = Module.IOptimizer.instantiate(odomcfg.optimizer.type, odomcfg.optimizer.args)
        KeyframeTracker     = (Module.IKeyframePolicy.instantiate(odomcfg.keyframe_tracker.type, odomcfg.keyframe_tracker.args)
                               if hasattr(odomcfg, "keyframe_tracker") else None)

        return cls(
            frontend=Frontend,
            motion_model=MotionEstimator,
            kp_selector=KeypointSelector,
            map_selector=MappointSelector,
            obs_filter=ObservationFilter,
            obs_covmodel=ObserveCovModel,
            post_process=MapRefiner,
            kf_selector=KeyframeSelector,
            optimizer=Optimizer,
            keyframe_tracker=KeyframeTracker,
            **vars(odomcfg.args),
        )

    def report_config(self):
        # Cute fine-print boxes
        box1 = Panel.fit(
            "\n".join(
                [
                    f"DepthEstimator cov: {self.Frontend.provide_cov[0]}",
                    f"MatchEstimator cov: {self.Frontend.provide_cov[1]}",
                    f"Observation cov:    {self.ObsCovModel.__class__.__name__}",
                ]
            ),
            title="Odometry Covariance",
            title_align="left",
        )
        box2 = Panel.fit(
            "\n".join(
                [
                    f"Optimizer       -'{self.Optimizer       .__class__.__name__}'",
                    f"Frontend        -'{self.Frontend        .__class__.__name__}'",
                    f"MotionEstimator -'{self.MotionEstimator .__class__.__name__}'",
                    f"KeypointSelector-'{self.KeypointSelector.__class__.__name__}'",
                    f"MappointSelector-'{self.MappointSelector.__class__.__name__}'",
                    f"OutlierFilter   -'{self.OutlierFilter   .__class__.__name__}'",
                    f"MapRefiner      -'{self.MapRefiner      .__class__.__name__}'",
                    f"KeyframeTracker -'{self.KeyframeTracker!r}'",
                ]
            ),
            title="Odometry Modules",
            title_align="left",
        )
        GlobalConsole.print(Columns([box1, box2]))

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        Module.IKeyframeSelector.is_valid_config(config.keyframe)
        Module.IMapProcessor.is_valid_config(config.postprocess)
        Module.IObservationFilter.is_valid_config(config.outlier)
        Module.IMotionModel.is_valid_config(config.motion)
        Module.IKeypointSelector.is_valid_config(config.keypoint)
        Module.ICovariance2to3.is_valid_config(config.cov.obs)
        Module.IFrontend.is_valid_config(config.frontend)
        Module.IOptimizer.is_valid_config(config.optimizer)
        if hasattr(config, "keyframe_tracker"):
            Module.IKeyframePolicy.is_valid_config(config.keyframe_tracker)
            if config.optimizer.type != "ISAM2_Graph":
                raise ValueError("keyframe_tracker needs optimizer.type ISAM2_Graph: it is the only "
                                 "backend that consumes keyframe re-observations (VisualMap.kf_match)")
            if config.keyframe.type != "AllKeyframe":
                raise ValueError("keyframe_tracker needs keyframe.type AllKeyframe: frame subsampling "
                                 "breaks the consecutive frame indexing the keyframe rows rely on")

        args_spec: dict = {
            "device"            : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            "num_point"         : lambda b: isinstance(b, int) and b > 0,
            "edgewidth"         : lambda b: isinstance(b, int) and b > 0,
            "match_cov_default" : lambda b: isinstance(b, (float, int)) and b > 0.0,
            "profile"           : lambda b: isinstance(b, bool),
            "mapping"           : lambda b: isinstance(b, bool),
        }
        if hasattr(config.args, "min_num_point"):
            args_spec["min_num_point"] = lambda b: isinstance(b, int) and b > 0
        cls._enforce_config_spec(config.args, args_spec)

    def initialize(self, frame0: T_SensorFrame):
        depth0          = self.Frontend.estimate_depth(frame0.camera)
        est_pose        = self.MotionEstimator.predict(frame0, None, depth0.depth).unsqueeze(0)

        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame0.camera.T_BS,
            "need_interp" : torch.tensor([0], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame0.camera.frame_ns], dtype=torch.long),
            "K"           : frame0.camera.K,
            "baseline"    : frame0.camera.baseline,
        }))
        self.OutlierFilter.set_meta(frame0.camera)
        self.prev_keyframe = (frame0, int(frame_idx.item()), depth0)
        if self.KeyframeTracker is not None:
            self.keyframe = _KeyframeState(frame0.camera, int(frame_idx.item()), depth0)

    def run_pair(self, frame0: T_SensorFrame, frame1: T_SensorFrame) -> None:
        assert self.prev_keyframe is not None

        # Check if current frame is the keyframe ########################################
        if not self.KeyframeSelector.isKeyframe(frame1):
            self.push_keyframe(frame1, self.graph.frames.data["pose"][self.prev_keyframe[1]].unsqueeze(0), need_interp=True)
            return

        depth0          = self.prev_keyframe[2]

        # Build a sparse metric depth prior from the current map for depth-completion depth
        # models (consumed via CameraData.depth_prior). Landmarks are projected with the
        # previous keyframe pose — which is exactly what the (static) motion model predicts
        # for frame1, since the new frame's own pose is not estimated until after depth.
        # No-op (None) when the map is empty or the depth model ignores depth_prior.
        frame1.camera.depth_prior = self._build_depth_prior(
            self.graph.frames.data["pose"][self.prev_keyframe[1]], frame1.camera
        )
        depth1, match01 = self.Frontend.estimate_pair(frame0.camera, frame1.camera)

        # Receive optimization result from previous step (if exists) ####################
        # NOTE: should always writeback optimized pose to global map before selecting new
        # keypoints (register new 3D point) on that frame.
        self.Optimizer.write_map(self.graph)
        for func in self.on_optimize_writeback: func(self)

        # Motion model provide an initial guess to the pose of frame1 ###################
        # Update motion model (this must be after write_back to get latest result)
        # NOTE: I assume the motion estimator works on stereo camera frame (not body frame)
        self.MotionEstimator.update(pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]]))
        est_pose = self.MotionEstimator.predict(frame1, match01.flow, depth1.depth).unsqueeze(0)

        # Generate Keypoints for frame 0 and 1 ##########################################
        kp0_uv  = self.KeypointSelector.select_point(frame0.camera, self.num_point, depth0, depth1, match01)
        kp1_uv  = kp0_uv + self.Frontend.retrieve_pixels(kp0_uv, match01.flow).T

        inbound_mask= filterPointsInRange(
            kp1_uv,
            (self.edge_width, frame1.camera.width - self.edge_width),
            (self.edge_width, frame1.camera.height - self.edge_width)
        )
        kp0_uv  = kp0_uv[inbound_mask]
        kp1_uv  = kp1_uv[inbound_mask]

        if kp0_uv.size(0) == 0:
            # No keypoint survived selection + inbound filtering (featureless or
            # blurry frame). The covariance projection below cannot take N=0, so
            # coast exactly like the VOLostTrack path: keep the motion-model pose,
            # mark the frame for interpolation, and advance the tracking context
            # so the next pair re-anchors on this frame.
            Logger.write("warn", f"VOLostTrack @ {frame1.frame_idx} - no keypoints survived selection")
            frame_idx = self.push_keyframe(frame1, est_pose, need_interp=True)
            self.prev_keyframe = (frame1, int(frame_idx.item()), depth1)
            return

        # Retrieve depth and depth cov for kp on frame 0 and 1 ##########################
        kp0_d               = self.Frontend.retrieve_pixels(kp0_uv, depth0.depth).squeeze(0)
        kp0_disparity       = self.Frontend.retrieve_pixels(kp0_uv, depth0.disparity)
        kp0_sigma_disparity = self.Frontend.retrieve_pixels(kp0_uv, depth0.disparity_uncertainty)
        kp0_sigma_dd        = self.Frontend.retrieve_pixels(kp0_uv, depth0.cov)
        kp0_sigma_dd        = kp0_sigma_dd.squeeze(0) if kp0_sigma_dd is not None else None

        kp1_d               = self.Frontend.retrieve_pixels(kp1_uv, depth1.depth).squeeze(0)
        kp1_disparity       = self.Frontend.retrieve_pixels(kp1_uv, depth1.disparity)
        kp1_sigma_disparity = self.Frontend.retrieve_pixels(kp1_uv, depth1.disparity_uncertainty)
        kp1_sigma_dd        = self.Frontend.retrieve_pixels(kp1_uv, depth1.cov)
        kp1_sigma_dd        = kp1_sigma_dd.squeeze(0) if kp1_sigma_dd is not None else None


        # Retrieve match cov for kp on frame 0 and 1    #################################
        num_kp = kp0_uv.size(0)

        # kp 0 has a fake sigma uv as it is manually selected pixels. This UV
        # represents the uncertainty introduced by the quantization process when
        # taking photo with discrete pixels.
        kp0_sigma_uv = torch.ones((num_kp, 3), device=self.device) * self.match_cov_default
        kp0_sigma_uv[..., 2] = 0.   # No sigma_uv off-diag term.

        kp1_sigma_uv = self.Frontend.retrieve_pixels(kp0_uv, match01.cov)
        kp1_sigma_uv = kp1_sigma_uv.T if kp1_sigma_uv is not None else None

        # Record color of keypoints (for visualization) #################################
        kp0_uv_cpu = kp0_uv.cpu()
        kp0_color  = frame0.camera.imageL[0:1, ..., kp0_uv_cpu[..., 1], kp0_uv_cpu[..., 0]].squeeze(0).T
        kp0_color  = (kp0_color * 255).to(torch.uint8)

        # Project from 2D -> 3D #########################################################
        pos0_Tc = pixel2point_NED(kp0_uv, kp0_d, frame0.camera.frame_K).cpu()
        pos0_covTc  = self.ObsCovModel.estimate(frame0.camera, kp0_uv, depth0, kp0_sigma_dd, kp0_sigma_uv)
        pos1_covTc  = self.ObsCovModel.estimate(frame1.camera, kp1_uv, depth1, kp1_sigma_dd, kp1_sigma_uv)


        # Run Outlier Filter ############################################################
        match_obs = MatchObs.init({
            "pixel1_uv"      : kp0_uv_cpu,
            "pixel2_uv"      : kp1_uv.cpu(),

            "pixel1_d"       : kp0_d.unsqueeze(-1).cpu(),
            "pixel2_d"       : kp1_d.unsqueeze(-1).cpu(),

            "pixel1_disp"    : torch.empty((num_kp, 1)).fill_(-1) if kp0_disparity is None else kp0_disparity.T.cpu(),
            "pixel2_disp"    : torch.empty((num_kp, 1)).fill_(-1) if kp1_disparity is None else kp1_disparity.T.cpu(),

            "pixel1_disp_cov": torch.empty((num_kp, 1)).fill_(-1) if kp0_sigma_disparity is None else kp0_sigma_disparity.T.cpu(),
            "pixel2_disp_cov": torch.empty((num_kp, 1)).fill_(-1) if kp1_sigma_disparity is None else kp1_sigma_disparity.T.cpu(),

            "pixel1_d_cov"   : torch.empty((num_kp, 1)).fill_(-1) if kp0_sigma_dd is None else kp0_sigma_dd.unsqueeze(-1).cpu(),
            "pixel2_d_cov"   : torch.empty((num_kp, 1)).fill_(-1) if kp1_sigma_dd is None else kp1_sigma_dd.unsqueeze(-1).cpu(),

            "pixel1_uv_cov"  : torch.empty((num_kp, 3)).fill_(-1) if kp0_sigma_uv is None else kp0_sigma_uv,
            "pixel2_uv_cov"  : torch.empty((num_kp, 3)).fill_(-1) if kp1_sigma_uv is None else kp1_sigma_uv,

            "obs1_covTc"     : pos0_covTc,
            "obs2_covTc"     : pos1_covTc,
        })
        assert self.OutlierFilter.verify_shape(match_obs), "The provided MatchFactor does not contain all data for outlier filter."
        mask = self.OutlierFilter.filter(match_obs, torch.device("cpu"))
        match_obs = match_obs[mask]

        # Register the factor graph #####################################################
        prev_pose       = pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]])
        prev_rot        = prev_pose.rotation().matrix().repeat((num_kp, 1, 1)).to(torch.float64)
        num_match_orig  = len(self.graph.match)

        point_idx = self.graph.points.push(PointNode.init({
            "pos_Tw": pp.SE3_type.Act(prev_pose, pos0_Tc)[..., :3],  # NOTE: Refer to https://github.com/pypose/pypose/issues/342
            "cov_Tw": torch.bmm(torch.bmm(prev_rot, pos0_covTc), prev_rot.transpose(1, 2)),
            "color" : kp0_color
        })[mask])
        frame_idx      = self.push_keyframe(frame1, est_pose)
        prev_frame_idx = torch.tensor([self.prev_keyframe[1]], dtype=torch.long)
        match_idx      = self.graph.match.push(match_obs)

        num_match_kp = len(match_obs)
        self.graph.point2match.add(point_idx, match_idx)    # Associate point -> match
        self.graph.match2point.set(match_idx, point_idx)    # Associate match -> point
        self.graph.frame2match.add(prev_frame_idx, torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))   # Associate frame -> match
        self.graph.frame2match.add(frame_idx     , torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))   # Associate frame -> match
        self.graph.match2frame1.set(match_idx    , torch.empty((num_match_kp,), dtype=torch.long).fill_(prev_frame_idx.item()))    # Associate match -> frame1
        self.graph.match2frame2.set(match_idx    , torch.empty((num_match_kp,), dtype=torch.long).fill_(frame_idx.item()     ))    # Associate match -> frame2

        # Keyframe tracking: extra keyframe -> frame1 flow, re-observe the keyframe's points
        lost_track = match_idx.size(0) < self.min_num_point
        if self.KeyframeTracker is not None:
            if lost_track:
                # the pair's job is skipped below, so this frame can never anchor keyframe rows
                self.keyframe = _KeyframeState(frame1.camera, int(frame_idx.item()), depth1)
            else:
                self._track_keyframe(frame1, int(frame_idx.item()), self.prev_keyframe[1], depth1,
                                     match_obs, point_idx, est_pose)

        # Visualization #################################################################
        rr.set_time("frame_idx", sequence=int(frame_idx.cpu().item())-1)
        rr_plt.log_flow(
            "/world/macvo/cam_left/optical_flow",
            match01.flow[0].detach().permute(1, 2, 0))
        rr_plt.log_depth("/world/macvo/cam_left/depth", depth1.depth[0])
        rr_plt.log_covariance("/world/macvo/cam_left/optical_flow_covar", match01.cov)
        rr_plt.log_covariance("/world/macvo/cam_left/depth_covar", depth1.cov)
        fig_plt.plot_imatcher("matching", match01, frame0, frame1)
        fig_plt.plot_istereo ("stereo_d", depth1 , frame1)
        fig_plt.plot_macvo   ("macvo_kp", match_obs, depth1, match01, frame0, frame1)

        # Update the tracking context ###################################################
        self.prev_keyframe = (frame1, int(frame_idx.item()), depth1)

        # Launch Optimization task  #####################################################
        if lost_track:
            # NOTE: if lost track, we do not do mapping since the pose is not reliable anyway.
            Logger.write("warn", f"VOLostTrack @ {frame1.frame_idx} - only get {match_idx.size(0)} observations")
            self.graph.frames.data["need_interp"][frame_idx] = True
            return
        else:
            self.Optimizer.start_optimize(
                self.Optimizer.get_graph_data(self.graph, frame_idx)
            )

        # Add (dense) mapping points to the map #########################################
        if self.mapping:
            map0_uv       = self.MappointSelector.select_point(frame0.camera, 2000, depth0, depth1, match01)
            num_kp        = map0_uv.size(0)
            map0_d        = self.Frontend.retrieve_pixels(map0_uv, depth0.depth).squeeze(0)
            map0_Tc       = pixel2point_NED(map0_uv, map0_d, frame0.camera.frame_K).cpu()

            map0_sigma_dd = self.Frontend.retrieve_pixels(map0_uv, depth0.cov)
            map0_sigma_dd = map0_sigma_dd.squeeze(0) if (map0_sigma_dd is not None) else None
            map0_sigma_uv = torch.ones((num_kp, 3), device=self.device) * self.match_cov_default
            map0_sigma_uv[..., 2] = 0.   # No sigma_uv off-diag term.
            map0_Tc_cov = self.ObsCovModel.estimate(frame0.camera, map0_uv, depth0, map0_sigma_dd, map0_sigma_uv)

            map0_uv_cpu = map0_uv.cpu()
            map0_color  = frame0.camera.imageL[0:1,..., map0_uv_cpu[..., 1], map0_uv_cpu[..., 0]].squeeze(0).T
            map0_color  = (map0_color * 255).to(torch.uint8)

            num_map_orig  = len(self.graph.map_points)
            num_mappoint  = map0_Tc.size(0)
            map_idx = self.graph.map_points.push(PointNode.init({
                "pos_Tw": pp.SE3_type.Act(prev_pose, map0_Tc)[..., :3],
                "cov_Tw": map0_Tc_cov,
                "color" : map0_color,
            }))
            self.graph.frame2map.add(frame_idx, torch.tensor([num_map_orig], dtype=torch.long), torch.tensor([num_mappoint], dtype=torch.long))   # Associate frame -> map

    @torch.no_grad()
    def _build_depth_prior(self, pose: torch.Tensor, target: CameraData) -> torch.Tensor | None:
        """
        Project the current map's landmarks into `target`'s image to produce a sparse metric
        depth map (1x1xHxW, 0 = no prior) for depth-completion depth models.

        `pose` is the camera->world pose used for the projection (the previous keyframe pose).
        Returns None when the map is empty or no landmark falls inside the frame.
        """
        landmark_sets: list[torch.Tensor] = []
        if len(self.graph.points) > 0:
            landmark_sets.append(self.graph.points.data["pos_Tw"].tensor)
        if len(self.graph.map_points) > 0:
            landmark_sets.append(self.graph.map_points.data["pos_Tw"].tensor)
        if len(landmark_sets) == 0:
            return None

        points_w = torch.cat(landmark_sets, dim=0).to(torch.float32)    # (N, 3) world NED
        points_c = pp.SE3(pose).Inv().Act(points_w)                     # (N, 3) camera NED
        depth    = points_c[..., 0]                                     # NED forward = depth
        uv       = point2pixel_NED(points_c, target.frame_K)            # (N, 2) pixel (u, v)

        inbound = filterPointsInRange(uv, (0, target.width - 1), (0, target.height - 1))
        valid   = torch.logical_and(inbound, depth > 0)
        if not bool(valid.any()):
            return None

        u = uv[valid, 0].round().long().clamp(0, target.width  - 1)
        v = uv[valid, 1].round().long().clamp(0, target.height - 1)
        prior = torch.zeros((1, 1, target.height, target.width), dtype=torch.float32)
        prior[0, 0, v, u] = depth[valid].to(torch.float32)
        return prior

    def _relative_translation(self, from_idx: int, to_pose: pp.LieTensor | torch.Tensor) -> float:
        T_from = pp.SE3(self.graph.frames.data["pose"][from_idx].reshape(7))
        T_to   = pp.SE3(torch.as_tensor(to_pose).reshape(7))
        rel = T_from.Inv() @ T_to           # SE3 layout [tx, ty, tz, qx, qy, qz, qw]
        return float(rel[..., :3].norm().item())

    def _track_keyframe(self, frame1: T_SensorFrame, frame_idx: int, prev_idx: int, depth1: Module.IDepth.Output,
                        match_obs: MatchObs, point_idx: torch.Tensor, est_pose: pp.LieTensor | torch.Tensor) -> None:
        """Keyframe step for frame1 (already registered as `frame_idx`).

        frame0 == keyframe: this pair IS the keyframe pair; remember its rows and the
        points they born - later frames re-observe exactly these. Otherwise run the
        keyframe -> frame1 flow, push the surviving re-observations to `kf_match`
        (kfmatch2point = the keyframe row's point, i.e. the same landmark), and let
        the policy decide whether frame1 becomes the keyframe.
        """
        assert self.KeyframeTracker is not None and self.keyframe is not None
        kf = self.keyframe
        frames_since_kf = frame_idx - kf.frame_idx
        translation = self._relative_translation(kf.frame_idx, est_pose)

        if prev_idx == kf.frame_idx:
            kf.obs, kf.point_idx = match_obs, point_idx
            ref = match_obs
        elif kf.obs is not None and kf.point_idx is not None:
            ref = self._observe_keyframe(frame1, frame_idx, depth1, kf.obs, kf.point_idx, kf)
        else:
            return

        n_ref = len(ref)
        ctx = TrackContext(
            frames_since_kf=int(frames_since_kf),
            n_grid=int(len(match_obs) if kf.obs is None else len(kf.obs)),
            n_kf_matches=int(n_ref),
            parallax_px=float((ref.data["pixel2_uv"] - ref.data["pixel1_uv"]).norm(dim=-1).median().item()) if n_ref else float("inf"),
            translation_m=translation,
            median_depth_m=float(ref.data["pixel2_d"].median().item()) if n_ref else 0.,
        )
        if self.KeyframeTracker.should_switch(ctx):
            Logger.write("info", f"Keyframe {kf.frame_idx} -> {frame_idx} [{self.KeyframeTracker!r}] "
                                 f"gap={ctx.frames_since_kf} parallax={ctx.parallax_px:.1f}px "
                                 f"covis={ctx.n_kf_matches}/{ctx.n_grid} t={ctx.translation_m:.3f}")
            self.keyframe = _KeyframeState(frame1.camera, frame_idx, depth1)

    def _observe_keyframe(self, frame1: T_SensorFrame, frame_idx: int, depth1: Module.IDepth.Output,
                          kf_obs: MatchObs, kf_point_idx: torch.Tensor, kf: _KeyframeState) -> MatchObs:
        """Flow the keyframe's registered pixels into frame1 and register the survivors
        in `kf_match`. Returns the pushed rows (possibly empty)."""
        match_kf = self.Frontend.estimate_match(kf.camera, frame1.camera)

        kp_kf = kf_obs.data["pixel1_uv"].to(self.device)
        kp_k  = kp_kf + self.Frontend.retrieve_pixels(kp_kf, match_kf.flow).T
        inbound = filterPointsInRange(
            kp_k,
            (self.edge_width, frame1.camera.width - self.edge_width),
            (self.edge_width, frame1.camera.height - self.edge_width)
        )
        inbound_cpu = inbound.cpu()
        kp_kf, kp_k = kp_kf[inbound], kp_k[inbound]
        kf_obs, kf_point_idx = kf_obs[inbound_cpu], kf_point_idx[inbound_cpu]
        num_kp = kp_k.size(0)

        rr_plt.log_flow("/world/macvo/cam_left/optical_flow_kf", match_kf.flow[0].detach().permute(1, 2, 0))
        if num_kp == 0:
            return kf_obs

        kp_k_d               = self.Frontend.retrieve_pixels(kp_k, depth1.depth).squeeze(0)
        kp_k_disparity       = self.Frontend.retrieve_pixels(kp_k, depth1.disparity)
        kp_k_sigma_disparity = self.Frontend.retrieve_pixels(kp_k, depth1.disparity_uncertainty)
        kp_k_sigma_dd        = self.Frontend.retrieve_pixels(kp_k, depth1.cov)
        kp_k_sigma_dd        = kp_k_sigma_dd.squeeze(0) if kp_k_sigma_dd is not None else None
        kp_k_sigma_uv        = self.Frontend.retrieve_pixels(kp_kf, match_kf.cov)
        kp_k_sigma_uv        = kp_k_sigma_uv.T if kp_k_sigma_uv is not None else None
        pos_k_covTc          = self.ObsCovModel.estimate(frame1.camera, kp_k, depth1, kp_k_sigma_dd, kp_k_sigma_uv)

        new_obs = MatchObs.init({
            "pixel1_uv"      : kf_obs.data["pixel1_uv"],
            "pixel2_uv"      : kp_k.cpu(),
            "pixel1_d"       : kf_obs.data["pixel1_d"],
            "pixel2_d"       : kp_k_d.unsqueeze(-1).cpu(),
            "pixel1_disp"    : kf_obs.data["pixel1_disp"],
            "pixel2_disp"    : torch.empty((num_kp, 1)).fill_(-1) if kp_k_disparity is None else kp_k_disparity.T.cpu(),
            "pixel1_disp_cov": kf_obs.data["pixel1_disp_cov"],
            "pixel2_disp_cov": torch.empty((num_kp, 1)).fill_(-1) if kp_k_sigma_disparity is None else kp_k_sigma_disparity.T.cpu(),
            "pixel1_d_cov"   : kf_obs.data["pixel1_d_cov"],
            "pixel2_d_cov"   : torch.empty((num_kp, 1)).fill_(-1) if kp_k_sigma_dd is None else kp_k_sigma_dd.unsqueeze(-1).cpu(),
            "pixel1_uv_cov"  : kf_obs.data["pixel1_uv_cov"],
            "pixel2_uv_cov"  : torch.empty((num_kp, 3)).fill_(-1) if kp_k_sigma_uv is None else kp_k_sigma_uv.cpu(),
            "obs1_covTc"     : kf_obs.data["obs1_covTc"],
            "obs2_covTc"     : pos_k_covTc,
        })
        mask = self.OutlierFilter.filter(new_obs, torch.device("cpu"))
        new_obs, kf_point_idx = new_obs[mask], kf_point_idx[mask]

        num_orig  = len(self.graph.kf_match)
        num_new   = len(new_obs)
        kfm_idx   = self.graph.kf_match.push(new_obs)
        self.graph.kfmatch2point .set(kfm_idx, kf_point_idx)
        self.graph.kfmatch2frame1.set(kfm_idx, torch.full((num_new,), kf.frame_idx, dtype=torch.long))
        self.graph.kfmatch2frame2.set(kfm_idx, torch.full((num_new,), frame_idx,    dtype=torch.long))
        self.graph.frame2kfmatch.add(torch.tensor([frame_idx], dtype=torch.long),
                                     torch.tensor([num_orig], dtype=torch.long),
                                     torch.tensor([num_new], dtype=torch.long))
        return new_obs

    def push_keyframe(self, frame: T_SensorFrame, est_pose: pp.LieTensor | torch.Tensor, need_interp: bool=False) -> torch.Tensor:
        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame.camera.T_BS,
            "need_interp" : torch.tensor([need_interp], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame.camera.frame_ns], dtype=torch.long),
            "K"           : frame.camera.K,
            "baseline"    : frame.camera.baseline,
        }))
        return frame_idx

    @Timer.cpu_timeit("Odom_Runtime")
    @Timer.gpu_timeit("Odom_Runtime")
    def run(self, frame: T_SensorFrame) -> None:
        """
        The main process that continuously running to manage different modules in MAC-VO.
        The multi-threading part will be managed in this function.
        Args:
            frame (T_SensorFrame): The current stereo frame to be processed.
        Returns:
            None
        """

        if not self.isinitiated:
            self.initialize(frame)
            self.isinitiated = True
            return

        assert self.prev_keyframe is not None
        self.run_pair(self.prev_keyframe[0], frame)

    def get_map(self) -> VisualMap:
        return self.graph

    def terminate(self) -> None:
        super().terminate()
        if self.prev_keyframe is not None:
            self.Optimizer.write_map(self.graph)
        self.Optimizer.finalize(self.graph)     # e.g. ISAM2_Graph's offline batch-LM polish (final_lm)
        self.Optimizer.terminate()
        self.MapRefiner.elaborate_map(self.graph.frames)

    def register_on_optimize_finish(self, func: T_SYSHOOK):
        """
        Install a callback hook when optimization result is written back to the map
        """
        self.on_optimize_writeback.append(func)
