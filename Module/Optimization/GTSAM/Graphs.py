import torch
import gtsam
import numpy as np
import pypose as pp
import typing as typ
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import rerun as rr

from Utility.PrettyPrint import Logger
from Utility.Point import pixel2point_NED
from Utility.GTSAM_Utils import (
    pypose_to_pose3, pose3_to_pypose, make_pose_to_point_factor,
    make_aligned_pose_to_point_factor, make_alignment_warp,
)
from .Augmentations import BuildContext, GraphAugmentation
from ..PyposeOptimizers import FactorGraph
from ..TwoFramePGO.Graphs import GraphInput


@dataclass
class GTSAM_GraphInput:
    previous_graph_data: GraphInput
    current_graph_data : GraphInput
    indexes_prev_curr : typ.List[typ.Tuple[int, int]]
    # Set by get_graph_data (parent side) when a Rerun G-EDF map snapshot is
    # wanted for this frame (graph_type "pose2point+gedf" only).
    want_map_snapshot : bool = False


@dataclass
class GTSAM_GraphOutput:
    frame_idexes: List[int]
    pose_estimates: List[torch.Tensor]
    # Optimized landmark positions, to be written into VisualMap.points (the
    # sparse VO landmarks — NOT map_points, the dense mapping cloud, which no
    # graph here re-estimates). `landmark_indexes` are therefore GLOBAL indices
    # into VisualMap.points, carried in from `GraphInput.points.index`; a
    # graph's own landmark numbering (gtsam symbol index) is a different index
    # space and must never leak out of the graph.
    landmark_indexes: Optional[list[int] | None] = None
    landmark_pos_Tw: Optional[torch.Tensor] = None   # (N, 3) world frame
    # Post-solve landmark visuals (export_landmark_marginals=True only):
    # posterior marginal covariances — what the solve actually concluded about
    # each landmark, unlike the map's creation-time cov_Tw which nothing
    # re-estimates — and the CURRENT frame's measured cloud placed at the
    # optimized pose, so the depth correction (prior displacement including the
    # scale s absorbed) is directly visible against the optimized landmarks.
    landmark_cov_Tw: Optional[torch.Tensor] = None       # (N, 3, 3) world frame
    landmark_meas_pos_Tw: Optional[torch.Tensor] = None  # (N, 3) world frame
    need_interp: Optional[bool] = None
    # Alignment diagnostics ("estimate + report", mirrors GEDF_GraphOutput):
    # the per-frame warp parameters estimated for the CURRENT frame's
    # observations. Poses in `pose_estimates` are always pure SE(3).
    alignment_type: str = "se3"
    alignment_state: Optional[torch.Tensor] = None   # (1,) log_s | (9,) sl4, CPU f32
    scale: Optional[float] = None                    # exp(log_s) / exp(4*x5)
    # G-EDF map snapshot for Rerun (graph_type "pose2point+gedf" only; same
    # payloads and semantics as GEDF_GraphOutput, CPU tensors, pickle-cheap).
    gedf_points: Optional[torch.Tensor] = None       # (M, 3) near-surface sample
    gedf_dist: Optional[torch.Tensor] = None         # (M,)
    gedf_gauss_means: Optional[torch.Tensor] = None  # (N, 3)
    gedf_gauss_sigmas: Optional[torch.Tensor] = None # (N, 3)
    gedf_gauss_weights: Optional[torch.Tensor] = None  # (N,)
    gedf_gauss_mae: Optional[torch.Tensor] = None    # (N,)
    gedf_cube_centers: Optional[torch.Tensor] = None # (C, 3)
    gedf_cube_valid: Optional[torch.Tensor] = None   # (C,) bool
    gedf_cube_mae: Optional[torch.Tensor] = None     # (C,)
    gedf_cube_size: Optional[float] = None
    # Merged per-solve diagnostics from the graph's augmentations (see
    # Augmentations.py) — plain scalars only: this dataclass crosses an
    # mp.Queue in parallel mode. None when the graph runs without
    # augmentations; the parent-side CSV/Rerun emission reads it by key.
    aug_diag: Optional[dict] = None

PosePixelMap = Dict[int, Dict[Tuple[float, float], int]]


class GTSAM_Pose2Point(FactorGraph):
    # extras dimensionality per alignment type
    _ALIGN_DIMS = {"se3": 0, "sim3": 1, "sl4": 9}

    def __init__(self, huber_delta: float = 0.1, huber_delta_prev: float = 1.0,
                 prior_sigma: float = 1e-4, max_iterations: int = 20,
                 alignment_type: str = "se3", alignment_prior_weight: float = 100.0,
                 augmentations: "typ.Sequence[GraphAugmentation]" = (),
                 export_landmark_marginals: bool = False):
        super().__init__()
        # Optimizer hyperparameters (config: Odometry.optimizer.args; defaults
        # reproduce the historical hardcoded values).
        self.huber_delta = huber_delta            # robust kernel on pose->point factors
        self.huber_delta_prev = huber_delta_prev  # robust kernel on prev-frame reobservation factor
        self.prior_sigma = prior_sigma            # gauge prior noise on anchor poses
        self.max_iterations = max_iterations      # LM iteration cap
        # Alignment axis (mirrors the GEDF backend): a per-frame warp variable
        # applied to the CURRENT frame's camera observations, correcting
        # monocular depth-scale (sim3) or projective (sl4) bias. Only the SE(3)
        # poses are written back; the warp is reported for diagnostics.
        # Previous-frame observations stay un-warped so they anchor the
        # landmark scale.
        assert alignment_type in self._ALIGN_DIMS, f"Unknown alignment '{alignment_type}'"
        self.alignment_type = alignment_type
        self.alignment_prior_weight = float(alignment_prior_weight)
        self._warp = make_alignment_warp(alignment_type) if alignment_type != "se3" else None
        # Everything that is NOT the pose2point graph itself (the G-EDF field
        # factor, the correlated log-depth prior, per-solve diagnostics) is an
        # injected GraphAugmentation — see Augmentations.py for the contract.
        # Empty tuple = the bare pose2point graph, byte-identical to the
        # pre-augmentation implementation (regression gate T5). List order is
        # factor insertion order.
        self.augmentations = tuple(augmentations)
        # Ship posterior landmark marginals + the measured cloud at the
        # optimized pose (config: viz_landmark_marginals) — display-only
        # payload, never written to the map.
        self.export_landmark_marginals = export_landmark_marginals


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

        # Global indices into VisualMap.points, row-aligned with the observation
        # arrays above (points arrive as get_match2point(observations), so row i
        # of `points` is the landmark that observation i sees). This is what makes
        # the optimized landmarks writable back into the map — see GTSAM_GraphOutput.
        self.point_global_idx = graph_data.current_graph_data.points.index.detach().cpu().long()
        assert self.point_global_idx.size(0) == self.obs_Tc_1.shape[0], \
            "points and observations must be row-aligned to write landmarks back"

        self.obs1_covTc = graph_data.current_graph_data.observations.data["obs1_covTc"].detach().cpu().double().numpy()  # (N,3,3)
        self.obs2_covTc = graph_data.current_graph_data.observations.data["obs2_covTc"].detach().cpu().double().numpy()  # (N,3,3)
        self.pts_covTw = graph_data.current_graph_data.points.data["cov_Tw"].detach().cpu().double().numpy() # (N,3,3)

        self.previous_graph_data = graph_data.previous_graph_data
        self.indexes_prev_curr = graph_data.indexes_prev_curr

        for aug in self.augmentations:
            aug.on_parse(self, graph_data)

        # self.log_save_data(graph_data)

    def run_gtsam_optimization(self):

        # Build factor graph
        # Prior: pose 1 at identity

        graph = gtsam.NonlinearFactorGraph()

        pose_1_key = gtsam.symbol('p', int(self.from_idx.cpu().item()))
        pose_2_key = gtsam.symbol('p', int(self.frame_idx.cpu().item()))

        # Initial estimate: pose 1 at identity, pose 2 at init_motion
        initial_estimate = gtsam.Values()
        ini_estimate_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.prior_sigma]*6, dtype=np.float64))
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
        pose_0_key = None
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

        # Alignment extras for the CURRENT frame's observations (see __init__).
        extras_key = None
        if self.alignment_type != "se3":
            E = self._ALIGN_DIMS[self.alignment_type]
            extras_key = gtsam.symbol('a', int(self.frame_idx.cpu().item()))
            initial_estimate.insert(extras_key, np.zeros(E, dtype=np.float64))
            align_sigma = 1.0 / np.sqrt(self.alignment_prior_weight)
            graph.add(gtsam.PriorFactorVector(
                extras_key, np.zeros(E, dtype=np.float64),
                gtsam.noiseModel.Diagonal.Sigmas(np.full(E, align_sigma, dtype=np.float64))))

        # First-wins map: current-frame landmark index -> previous-frame obs index
        curr_to_prev: Dict[int, int] = {}
        for p, c in self.indexes_prev_curr:
            curr_to_prev.setdefault(c, p)

        # The graph's own factor handles, in insertion order — exposed to the
        # augmentations through BuildContext (e.g. for cost-split diagnostics).
        point_factors: list = []

        landmark_keys = []
        for i in range(len(self.obs_Tc_1)):

            # Create landmark key. The symbol index is local to this solve (one
            # landmark per current-frame observation); the map-side identity of
            # landmark i is self.point_global_idx[i].
            landmark_key = gtsam.symbol('l', i)
            landmark_keys.append(landmark_key)

            if i in curr_to_prev and p0_index >= 0:
                pi = curr_to_prev[i]

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
                m_huber_0 = gtsam.noiseModel.mEstimator.Huber.Create(self.huber_delta_prev)
                noise_model_0 = gtsam.noiseModel.Robust.Create(m_huber_0, noise_model_0)
                factor0 = make_pose_to_point_factor(pose_0_key, landmark_key, obs_Tc_0_i, noise_model_0)
                graph.add(factor0)
                point_factors.append(factor0)


            # Read observation and covariance
            obs_Tc_1_i = self.obs_Tc_1[i]
            obs_Tc_2_i = self.obs_Tc_2[i]
            cov_Tc_1_i = self.obs1_covTc[i]
            cov_Tc_2_i = self.obs2_covTc[i]

            # Create noise model
            noise_model_1 = gtsam.noiseModel.Gaussian.Covariance(cov_Tc_1_i)
            noise_model_2 = gtsam.noiseModel.Gaussian.Covariance(cov_Tc_2_i)
            m_huber = gtsam.noiseModel.mEstimator.Huber.Create(self.huber_delta)
            noise_model_1 = gtsam.noiseModel.Robust.Create(
                m_huber,
                noise_model_1
            )
            noise_model_2 = gtsam.noiseModel.Robust.Create(
                m_huber,
                noise_model_2
            )
            # Create factors. Frame-1 observations anchor the landmarks
            # un-warped; the current frame's factor carries the alignment warp.
            factor1 = make_pose_to_point_factor(pose_1_key, landmark_key, obs_Tc_1_i, noise_model_1)
            if extras_key is not None:
                factor2 = make_aligned_pose_to_point_factor(
                    pose_2_key, extras_key, landmark_key, obs_Tc_2_i, noise_model_2, self._warp)
            else:
                factor2 = make_pose_to_point_factor(pose_2_key, landmark_key, obs_Tc_2_i, noise_model_2)

            # Add factors to graph
            graph.add(factor1)
            graph.add(factor2)
            point_factors.append(factor1)
            point_factors.append(factor2)

            # Add initial estimate for landmark
            Pt_landmark = P1.transformFrom(obs_Tc_1_i)
            initial_estimate.insert(landmark_key, Pt_landmark)

        # Non-pose2point factors (correlated depth prior, G-EDF field, ...) are
        # added by the injected augmentations, in list order, at this exact
        # point of the build — see Augmentations.py for the contract.
        build_ctx = BuildContext(
            pose_1_key=pose_1_key, pose_2_key=pose_2_key, pose_0_key=pose_0_key,
            landmark_keys=landmark_keys, point_factors=point_factors)
        for aug in self.augmentations:
            aug.on_build(self, graph, initial_estimate, build_ctx)

        # Optimize the graph
        params = gtsam.LevenbergMarquardtParams()
        # params.setVerbosityLM("SUMMARY")
        params.setMaxIterations(self.max_iterations)

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()

        pose_1 = result.atPose3(pose_1_key)
        pose_2 = result.atPose3(pose_2_key)
        landmark_positions = [result.atPoint3(landmark_keys[i]) for i in range(len(landmark_keys))]
        landmark_positions = torch.stack([torch.from_numpy(pos).double() for pos in landmark_positions], dim=0)  # (N,3)

        # Augmentation diagnostics at the optimized state (child side in
        # parallel mode — plain scalars only). A diagnostics failure must never
        # take down the solve.
        aug_diag: dict = {}
        for aug in self.augmentations:
            try:
                aug_diag.update(aug.on_solved(
                    self, graph, result, pose_1, pose_2, landmark_positions.numpy()))
            except Exception as e:
                Logger.write("warn", f"{type(aug).__name__}.on_solved failed ({e})")

        # Display-only payload: posterior landmark marginals (what the solve
        # concluded, vs the map's never-re-estimated creation-time cov_Tw) and
        # the measured cloud at the optimized pose (the "before" the depth
        # correction is visible against). Best-effort — never fails the solve.
        landmark_cov_Tw = None
        landmark_meas_pos_Tw = None
        if self.export_landmark_marginals and len(landmark_keys) > 0:
            R2 = pose_2.rotation().matrix()
            t2 = np.asarray(pose_2.translation(), dtype=np.float64).reshape(3)
            landmark_meas_pos_Tw = torch.from_numpy(self.obs_Tc_2 @ R2.T + t2)
            try:
                marginals = gtsam.Marginals(graph, result)
                landmark_cov_Tw = torch.from_numpy(np.stack(
                    [marginals.marginalCovariance(k) for k in landmark_keys]))
            except Exception as e:
                Logger.write("warn", f"landmark marginals unavailable ({e})")

        # self.log_plot_data(pose_1=pose_1, pose_2=pose_2, landmark_positions=landmark_positions)
        need_interp = False
        landmark_idx: Optional[list[int]] = self.point_global_idx.tolist()
        try:
            pose_1 = pose3_to_pypose(pose_1)
            pose_2 = pose3_to_pypose(pose_2)
        except Exception as e:
            Logger.write("error", f"Error converting optimized poses to PyPose format: {e}")
            pose_1 = self.init_pypose
            pose_2 = self.init_pypose
            need_interp = True
            # The poses fell back to the initial guess, so the landmarks this
            # solve produced are not consistent with what lands in the map.
            landmark_idx, landmark_positions = None, None
            landmark_cov_Tw, landmark_meas_pos_Tw = None, None

        align_state: Optional[torch.Tensor] = None
        align_scale: Optional[float] = None
        if extras_key is not None:
            x = np.asarray(result.atVector(extras_key), dtype=np.float64)
            align_state = torch.from_numpy(x).float()
            align_scale = float(np.exp(x[0])) if self.alignment_type == "sim3" \
                else float(np.exp(4.0 * x[5]))

        self.graph_output = GTSAM_GraphOutput(
            frame_idexes=[int(self.from_idx.cpu().item()), int(self.frame_idx.cpu().item())],
            pose_estimates=[pose_1, pose_2],
            landmark_indexes=landmark_idx,
            landmark_pos_Tw=landmark_positions,
            landmark_cov_Tw=landmark_cov_Tw,
            landmark_meas_pos_Tw=landmark_meas_pos_Tw,
            need_interp=need_interp,
            alignment_type=self.alignment_type,
            alignment_state=align_state,
            scale=align_scale,
            aug_diag=aug_diag or None,
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

        # Global indices into VisualMap.points, row-aligned with the observation
        # arrays above (see GTSAM_Pose2Point.parse_graph_data).
        self.point_global_idx = graph_data.current_graph_data.points.index.detach().cpu().long()
        assert self.point_global_idx.size(0) == self.obs_Tc_1.shape[0], \
            "points and observations must be row-aligned to write landmarks back"

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
        # gtsam landmark key observed by row i of this call's observations, so the
        # solve's landmarks can be mapped back onto self.point_global_idx. iSAM2
        # merges tracks across frames by pixel proximity, so its own symbol
        # numbering bears no relation to the map's point indices, and several rows
        # (here or in an earlier call) can share one landmark.
        frame_landmark_keys: list[int] = []

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

            frame_landmark_keys.append(landmark_key)

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

        # Write back only the landmarks THIS call observed, keyed by their map
        # index. iSAM2 keeps refining older landmarks too, but they are not
        # reachable from here without a persistent key -> map-index table that
        # would grow with every observation in the sequence; the whole-graph
        # dump (get_all_landmarks) is in iSAM2's own index space and must not be
        # handed to the map. A failed update leaves new keys absent - skip those.
        est_after: gtsam.Values = self.isam.calculateEstimate()
        landmark_idx: list[int] = []
        landmark_points_np: list[np.ndarray] = []
        for global_idx, landmark_key in zip(self.point_global_idx.tolist(), frame_landmark_keys):
            if not est_after.exists(landmark_key): continue
            landmark_idx.append(global_idx)
            landmark_points_np.append(np.asarray(est_after.atPoint3(landmark_key), dtype=np.float64))

        if len(landmark_points_np) > 0:
            landmark_points = torch.tensor(np.stack(landmark_points_np), dtype=torch.float32)
        else:
            landmark_points = torch.empty((0, 3), dtype=torch.float32)

        self.graph_output = GTSAM_GraphOutput(
            frame_idexes=frame_idexes,
            pose_estimates=pose_estimates,
            landmark_indexes=landmark_idx,
            landmark_pos_Tw=landmark_points,
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
