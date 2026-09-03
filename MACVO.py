import argparse
import time
import torch
import rerun as rr
import numpy as np
import pypose as pp
from pathlib import Path
from typing import List, TYPE_CHECKING

from DataLoader import SequenceBase, Frame, smart_transform
from Evaluation.EvalSeq import EvaluateSequences
from Odometry.MACVO import MACVO

from Utility.Config import load_config, asNamespace
from Utility.PrettyPrint import print_as_table, ColoredTqdm, Logger
from Utility.Sandbox import Sandbox
from Utility.Visualize import fig_plt, rr_plt
from Utility.Visualize.Rerun_Visualize import (
    compute_track_ages, age_colors, update_landmark_obs, filter_persistent_landmarks,
    update_landmark_colors, landmark_color_array,
)
from Utility.Timer import Timer
from Utility.read_eiva_vslam_outputs import load_vslam_track

# ISAM2-only debug visualization (keyframe links + track-age plots) needs the
# live tracker; gtsam (and therefore ISAM2_Graph) may not be installed, so
# guard the import the same way Module/Optimization/__init__.py does.
try:
    from Module.Optimization import ISAM2_Graph
    import gtsam
except ImportError:
    ISAM2_Graph = None  # type: ignore[assignment]
    gtsam = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from Module.Optimization.GTSAM.ISAM2Optimizer import ISAM2FlowTracker


def _isam2_tracker(system: MACVO) -> "ISAM2FlowTracker | None":
    """Return the live ISAM2FlowTracker backing `system.Optimizer`, or None
    when the backend isn't ISAM2 (or gtsam isn't installed) - callers then
    skip all ISAM2-only debug visualization at zero cost."""
    if ISAM2_Graph is None or not isinstance(system.Optimizer, ISAM2_Graph):
        return None
    return system.Optimizer._tracker()


# Feature C: iSAM2-optimized landmark cloud (`/world/isam2/*`). The tracker
# keeps no table for dead landmarks, so viz maintains its own persistent
# lm_key -> max n_obs dict across frames, plus a first-seen RGB per landmark
# (from `/world/vo_tracking`'s image-color rows) so the cloud looks like the
# same scene rather than being colored by observation count.
_lm_n_obs: dict[int, int] = {}
_lm_color: dict[int, tuple[int, int, int]] = {}
_LM_CLOUD_MIN_OBS = 3


def _isam2_landmark_snapshot(
    tracker: "ISAM2FlowTracker", lm_n_obs: dict[int, int], n_frames: int, min_obs: int = _LM_CLOUD_MIN_OBS,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray]:
    """
    One `calculateEstimate()` pull -> (lm_pos (M,3) f32, lm_counts (M,) i64,
    kept_keys (list[int], len M), pose_xyz (n_frames,3) f32, traj (K,3) f32).
    `lm_pos`/`lm_counts`/`kept_keys` cover persistent landmarks
    (`filter_persistent_landmarks`) that still `exists()` in the estimate
    (marginalized-away keys under `marg_lag > 0` are silently dropped, not
    errored); `kept_keys` is the landmark key aligned row-for-row with
    `lm_pos`/`lm_counts`, e.g. for looking up per-landmark colors via
    `landmark_color_array`. `pose_xyz` is a DENSE per-graph-frame-index array
    (row `k` = the pose at `tracker.pose_keys`'s frame index `k`, NaN where
    that pose key doesn't exist in the estimate or is >= n_frames) - this is
    what callers pass to `kf_link_segments` so a kf_match edge's frame/kf
    indices index straight into it, with `kf_link_segments` itself dropping
    any edge touching a NaN row. `traj` is `pose_xyz` with the NaN rows
    dropped (same frame order as before). Both `pose_xyz` and `traj` are in
    the GRAPH gauge, NOT the CHAIN gauge that `/world/est` uses under
    `readout: chain` - see the gauge caveat where this is called.
    """
    assert gtsam is not None
    est = tracker.isam.calculateEstimate()

    keys, counts = filter_persistent_landmarks(lm_n_obs, min_obs)
    lm_pos: list[np.ndarray] = []
    lm_counts: list[int] = []
    kept_keys: list[int] = []
    for key, count in zip(keys, counts):
        if not est.exists(key):
            continue
        lm_pos.append(np.asarray(est.atPoint3(key), dtype=np.float32))
        lm_counts.append(int(count))
        kept_keys.append(key)

    pose_xyz = np.full((n_frames, 3), np.nan, dtype=np.float32)
    for k in sorted(tracker.pose_keys):
        if k >= n_frames:
            continue
        p_k = gtsam.symbol("p", k)  #type: ignore[reportAttributeAccessIssue]
        if not est.exists(p_k):
            continue
        pose_xyz[k] = np.asarray(est.atPose3(p_k).translation(), dtype=np.float32)

    lm_pos_arr = np.array(lm_pos, dtype=np.float32).reshape(-1, 3)
    lm_counts_arr = np.array(lm_counts, dtype=np.int64)
    traj_arr = pose_xyz[np.isfinite(pose_xyz).all(axis=1)]
    return lm_pos_arr, lm_counts_arr, kept_keys, pose_xyz, traj_arr


def VisualizeRerunCallback(
    frame: Frame, system: MACVO, pb: ColoredTqdm, gt: List[torch.Tensor] | None = None,
    lm_cloud_every: int = 1,
):
    rr.set_time("frame_idx", sequence=frame.frame_idx)

    # Non-key frame does not need visualization
    if system.graph.frames.data["need_interp"][-1]: return

    # ISAM2 tracker backing Features A/B/C; None when rerun is off or the
    # backend isn't ISAM2, in which case those features fall back to
    # backend-independent (chain-gauge) behavior or are skipped entirely.
    tracker = _isam2_tracker(system) if rr_plt.default_mode == "rerun" else None

    # Feature A: accumulated keyframe -> frame re-observation lines. When the
    # backend is ISAM2, this is instead logged in the GRAPH gauge inside
    # Feature C's throttled `calculateEstimate()` pull below (positions from
    # `pose_xyz`), so the fan sits on top of `/world/isam2/traj` instead of
    # floating off it under `readout: chain`; here (no ISAM2 tracker) the
    # CHAIN gauge is the only one available, so log it directly.
    if tracker is None and len(system.graph.kf_match) > 0:
        rr_plt.log_kf_links(
            "/world/kf_links",
            system.graph.kfmatch2frame1.mapping.tensor,
            system.graph.kfmatch2frame2.mapping.tensor,
            system.graph.frames.data["pose"].tensor[:, :3],
        )

    if frame.frame_idx > 0:
        rr_plt.log_trajectory("/world/est", pp.SE3(system.graph.frames.data["pose"].tensor))

    # if gt is not None:
    #     gt = torch.stack(gt)
    #     gt = pp.SE3(gt)
    #     rr_plt.log_trajectory("/world/gt" , gt)

    if gt is not None:
        gt = pp.SE3(torch.stack(gt))  # (N,7) typically

        T_rot = gt[0].Inv()

        R = pp.euler2SO3(torch.tensor([-np.pi/2, 0., 0.], device=gt.device, dtype=gt.dtype))

        # Build SE3 with zero translation + rotation R
        batch = gt.shape[:-1]  # e.g. (N,)
        t0 = torch.zeros(*batch, 3, device=gt.device, dtype=gt.dtype)
        q  = R.tensor().expand(*batch, 4)  # quaternion part from SO3
        T_rot = pp.SE3(torch.cat([t0, q], dim=-1))

        gt = T_rot @ gt   # left-multiply: rotate in /world
        rr_plt.log_trajectory("/world/gt", gt)

    rr_plt.log_camera("/world/macvo/cam_left", pp.SE3(system.graph.frames.data["pose"][-1]), system.graph.frames.data["K"][-1])
    rr_plt.log_image ("/world/macvo/cam_left/rgb", frame.camera.imageL[0].permute(1, 2, 0))
    match_obs = system.graph.get_frame2match(system.graph.frames[-1:])

    # Feature B: landmark persistence (age-colored keypoints + track-count/age
    # plots). Only costs anything when rerun is on AND the backend is ISAM2.
    ages: np.ndarray | None = None
    kpt_colors: np.ndarray | None = None
    if tracker is not None:
        ages = compute_track_ages(
            match_obs.data["pixel2_uv"], {k: t.n_obs for k, t in tracker.tracks.items()}
        )
        kpt_colors = age_colors(ages)
    rr_plt.log_keypoints("/world/macvo/cam_left/kpts", match_obs, colors=kpt_colors)

    map_points = system.graph.get_frame2map(system.graph.frames[-1:])
    # rr_plt.log_points("/world/point_cloud_incremental", map_points.data["pos_Tw"].detach(), map_points.data["color"].detach(), map_points.data["cov_Tw"].detach(), "sphere")

    # map_points = system.graph.get_frame2map(system.graph.frames[:])
    # rr_plt.log_points("/world/point_cloud_all", map_points.data["pos_Tw"].detach(), map_points.data["color"].detach(), map_points.data["cov_Tw"].detach(), "sphere")

    vo_points  = system.graph.get_match2point(system.graph.get_frame2match(system.graph.frames[-1:]))
    rr_plt.log_points("/world/vo_tracking", vo_points.data["pos_Tw"].detach(), vo_points.data["color"].detach(), vo_points.data["cov_Tw"].detach(), "sphere")

    if tracker is not None and kpt_colors is not None:
        # 3D twin of the age-colored keypoint overlay; rows align with
        # match_obs via get_match2point.
        rr_plt.log_points(
            "/world/vo_tracking_age",
            vo_points.data["pos_Tw"].detach(),
            torch.from_numpy(kpt_colors),
            None,
            "none",
        )

    if tracker is not None and tracker.stats and tracker.stats[-1]["frame"] == frame.frame_idx:
        stat = tracker.stats[-1]
        rr_plt.log_scalar("/plots/tracks/live_count", stat["n_tracks"], name="live_count")
        rr_plt.log_scalar("/plots/tracks/kf_obs", stat["n_kf_obs"], name="kf_obs")
        if ages is not None and ages.size > 0:
            rr_plt.log_scalar("/plots/tracks/age_mean", float(ages.mean()), name="age_mean")
            rr_plt.log_scalar("/plots/tracks/age_max", float(ages.max()), name="age_max")

    # Feature C: iSAM2-optimized landmark cloud + graph-gauge trajectory.
    # GAUGE CAVEAT: under `readout: chain`, `/world/est`/`vo_tracking` are in
    # the CHAIN gauge while this estimate (landmarks + pose keys) is in the
    # GRAPH gauge; they can diverge after low-support stretches. Judge cloud
    # consistency against `/world/isam2/traj`, NEVER against `/world/est`.
    if tracker is not None and lm_cloud_every > 0 and tracker.stats and tracker.stats[-1]["frame"] == frame.frame_idx:
        # Update the persistent n_obs table every stepped frame so counts stay
        # complete even when the (expensive) estimate pull below is throttled.
        update_landmark_obs(_lm_n_obs, ((t.lm_key, t.n_obs) for t in tracker.tracks.values()))
        # Same cadence: stash each newly-seen landmark's first RGB (from the
        # image-colored /world/vo_tracking rows) so the cloud below matches
        # that look instead of being colored by observation count.
        update_landmark_colors(
            _lm_color, match_obs.data["pixel2_uv"], vo_points.data["color"],
            {k: t.lm_key for k, t in tracker.tracks.items()},
        )
        if frame.frame_idx % lm_cloud_every == 0:
            lm_pos, lm_counts, kept_keys, pose_xyz, traj = _isam2_landmark_snapshot(
                tracker, _lm_n_obs, len(system.graph.frames), _LM_CLOUD_MIN_OBS
            )
            if lm_pos.shape[0] > 0:
                # Two sibling entities from the same snapshot so either can be
                # toggled independently in the viewer: image-RGB (matches the
                # /world/vo_tracking look) vs. the original plasma-by-
                # observation-count coloring.
                rr_plt.log_points(
                    "/world/isam2/landmarks",
                    torch.from_numpy(lm_pos),
                    torch.from_numpy(landmark_color_array(kept_keys, _lm_color)),
                    None,
                    "none",
                )
                rr_plt.log_points(
                    "/world/isam2/landmarks_obs",
                    torch.from_numpy(lm_pos),
                    torch.from_numpy(age_colors(lm_counts)),
                    None,
                    "none",
                )
            rr_plt.log_path("/world/isam2/traj", traj, color=(255, 160, 40))
            # Feature A (ISAM2 case): same GRAPH-gauge positions as the traj
            # above (one `calculateEstimate()` pull, no second per-frame
            # pull), so the fan sits on top of `/world/isam2/traj` instead of
            # the chain-gauge trajectory. `kf_link_segments` drops any edge
            # touching a NaN `pose_xyz` row (pose key not yet/anymore in the
            # estimate).
            if len(system.graph.kf_match) > 0:
                rr_plt.log_kf_links(
                    "/world/kf_links",
                    system.graph.kfmatch2frame1.mapping.tensor,
                    system.graph.kfmatch2frame2.mapping.tensor,
                    torch.from_numpy(pose_xyz),
                )


def VisualizeVRAMUsage(frame: Frame, system: MACVO, pb: ColoredTqdm, iter_ms: float | None = None) -> float | None:
    if torch.cuda.is_available():
        reserved_gb  = torch.cuda.memory_reserved(0)  / 1e9
        allocated_gb = torch.cuda.memory_allocated(0) / 1e9
        vram_gb  = allocated_gb
        vram_str = f"{allocated_gb:.2f}/{reserved_gb:.2f} GB (alloc/reserved)"
    else:
        vram_gb = None
        vram_str = "N/A"

    fps_str = f", FPS={1000.0 / iter_ms:.1f}" if (iter_ms is not None and iter_ms > 0) else ""
    pb.set_description(desc=f"{system.graph}, VRAM={vram_str}{fps_str}")
    return vram_gb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom", type=str, default = "Config/Experiment/MACVO/MACVO.yaml")
    parser.add_argument("--data", type=str, default = "Config/Sequence/TartanAir_abandonfac_001.yaml")
    parser.add_argument(
        "--seq_to",
        type=int,
        default=None,
        help="Crop sequence to frame# when ran. Set to -1 (default) if wish to run whole sequence",
    )
    parser.add_argument(
        "--seq_from",
        type=int,
        default=0,
        help="Crop sequence from frame# when ran. Set to 0 (default) if wish to start from first frame",
    )
    parser.add_argument(
        "--resultRoot",
        type=str,
        default="./Results",
        help="Directory to store trajectory and files generated by the script."
    )
    parser.add_argument(
        "--useRR",
        action="store_true",
        help="Activate RerunVisualizer to generate <config.Project>.rrd file for visualization.",
    )
    parser.add_argument(
        "--saveplt",
        action="store_true",
        help="Activate PLTVisualizer to generate <frame_idx>.jpg file in space folder for covariance visualization.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload entire trajectory into RAM to reduce data fetching overhead during runtime."
    )
    parser.add_argument(
        "--autoremove",
        action="store_true",
        help="Cleanup result sandbox after script finishs / crashed. Helpful during testing & debugging."
    )
    parser.add_argument(
        "--noeval",
        action="store_true",
        help="Evaluate sequence after running odometry."
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Record timing for system (active Utility.Timer for global time recording)"
    )
    parser.add_argument(
        "--vlsam",
        type=str,
        default=None,
        help="Path to VSLAM track txt file for visualization in Rerun."
    )
    parser.add_argument(
        "--n_to_align",
        type=int,
        default=-1,
        help="Number of poses used for trajectory alignment during evaluation (-1 = all)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed torch/numpy/random for reproducible runs (e.g. keypoint selection uses randperm)."
    )
    parser.add_argument(
        "--lm_cloud_every",
        type=int,
        default=1,
        help="Pull the iSAM2 landmark-cloud estimate (--useRR, ISAM2 backend only) every N stepped "
             "frames; 0 disables it. Under marg_lag==0 (unbounded graph), calculateEstimate() is a "
             "fresh full back-substitution whose cost grows with the run - raise N on long sequences. "
             "Free (cached) under marg_lag>0.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Metadata setup & visualizer setup
    cfg, cfg_dict = load_config(Path(args.odom))
    odomcfg, odomcfg_dict = cfg.Odometry, cfg_dict["Odometry"]
    datacfg, datacfg_dict = load_config(Path(args.data))
    project_name = odomcfg.name + "@" + datacfg.name

    exp_space = Sandbox.create(Path(args.resultRoot), project_name)
    if args.autoremove: exp_space.set_autoremove()
    exp_space.config = {
        "Project": project_name,
        "Odometry": odomcfg_dict,
        "Data": {"args": datacfg_dict, "end_idx": args.seq_to, "start_idx": args.seq_from},
    }

    # Setup logging and visualization. The .rrd file sink must be attached at
    # init time (a trailing rr.save after streaming yields an empty file).
    if args.useRR:
        rr_plt.default_mode = "rerun"
        rr_plt.init_connect(project_name, save_rrd=str(exp_space.path(f"{project_name}.rrd")))

    if hasattr(datacfg.args, "vslam") and datacfg.args.vslam:
        vslam_track = load_vslam_track(datacfg.args.vslam, entity_name=project_name)

    Timer.setup(active=args.timing)
    fig_plt.default_mode = "image" if args.saveplt else "none"

    _perf_last_t: list[float] = [time.monotonic()]
    _perf_iter_ms: list[float] = []
    _perf_vram_gb: list[float] = []

    def onFrameFinished(frame: Frame, system: MACVO, pb: ColoredTqdm, gt: List[torch.Tensor] | None = None):
        now = time.monotonic()
        iter_ms = (now - _perf_last_t[0]) * 1000.0
        _perf_last_t[0] = now
        if frame.frame_idx > 0:
            _perf_iter_ms.append(iter_ms)

        VisualizeRerunCallback(frame, system, pb, gt, lm_cloud_every=args.lm_cloud_every)
        vram_gb = VisualizeVRAMUsage(frame, system, pb, iter_ms if frame.frame_idx > 0 else None)
        if vram_gb is not None:
            _perf_vram_gb.append(vram_gb)

    # Initialize data source
    sequence = smart_transform(
        SequenceBase[Frame].instantiate(datacfg.type, datacfg.args).clip(args.seq_from, args.seq_to),
        cfg.Preprocess
    )

    if args.preload:
        sequence = sequence.preload()

    system = MACVO[Frame].from_config(asNamespace(exp_space.config))
    system.receive_frames(sequence, exp_space, on_frame_finished=onFrameFinished)

    if _perf_iter_ms:
        np.save(exp_space.path("iter_time_ms.npy"), np.array(_perf_iter_ms))
    if _perf_vram_gb:
        np.save(exp_space.path("vram_usage_gb.npy"), np.array(_perf_vram_gb))
    optimizer_stats = system.Optimizer.frame_stats()
    if optimizer_stats:
        np.savez(exp_space.path("optimizer_stats.npz"), **optimizer_stats)

    rr_plt.log_trajectory("/world/est"  , torch.tensor(np.load(exp_space.path("poses.npy"))[:, 1:]))
    try:
        rr_plt.log_points    ("/world/point_cloud",
                                system.get_map().map_points.data["pos_Tw"].tensor,
                                system.get_map().map_points.data["color"].tensor,
                                system.get_map().map_points.data["cov_Tw"].tensor,
                                "color")
    except RuntimeError:
        Logger.write("warn", "Unable to log full point cloud - is mapping mode on?")

    Timer.report()
    Timer.save_elapsed(exp_space.path("elapsed_time.json"))

    if not args.noeval:
        header, result = EvaluateSequences([str(exp_space.folder)], align=True, correct_scale=True ,align_origin=False, n_to_align=args.n_to_align)
        print_as_table(header, result)
