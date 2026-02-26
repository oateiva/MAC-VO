import argparse
from pathlib import Path
import torch
import rerun as rr
import pypose as pp
import numpy as np
from Utility.Visualize import fig_plt, rr_plt
from DataLoader import SequenceBase, Frame, smart_transform
from Evaluation.EvalSeq import EvaluateSequences
from Odometry.BaselineTartanVO import TartanVO
from Utility.Config import build_dynamic_config, load_config
from Utility.PrettyPrint import ColoredTqdm, ColoredTqdm, Logger, print_as_table
from Utility.Sandbox import Sandbox

def VisualizeRerunCallback(frame: Frame, system: TartanVO, pb: ColoredTqdm, gt: list[torch.Tensor] | None = None):
    rr.set_time("frame_idx", sequence=frame.frame_idx)


    if frame.frame_idx > 0:
        rr_plt.log_trajectory("/world/est", pp.SE3(system.gmap.frames.data["pose"].tensor))

    if gt is not None:
        gt = torch.stack(gt)
        gt = pp.SE3(gt)
        rr_plt.log_trajectory("/world/gt" , gt)

    if gt is not None:
        T0 = gt[0]  # reference frame
        gt = T0.Inv() @ gt  # transform to reference frame
        rr_plt.log_trajectory("/world/gt", gt)

    rr_plt.log_camera("/world/tartanvo/cam_left", pp.SE3(system.gmap.frames.data["pose"][-1]), system.gmap.frames.data["K"][-1])
    rr_plt.log_image ("/world/tartanvo/cam_left/rgb", frame.camera.imageL[0].permute(1, 2, 0))
    match_obs = system.gmap.get_frame2match(system.gmap.frames[-1:])
    rr_plt.log_keypoints("/world/tartanvo/cam_left/kpts", match_obs)
    \
    map_points = system.gmap.get_frame2map(system.gmap.frames[:])
    rr_plt.log_points("/world/point_cloud_all", map_points.data["pos_Tw"].detach(), map_points.data["color"].detach(), map_points.data["cov_Tw"].detach(), "sphere")

    vo_points  = system.gmap.get_match2point(system.gmap.get_frame2match(system.gmap.frames[-1:]))
    rr_plt.log_points("/world/vo_tracking", vo_points.data["pos_Tw"].detach(), vo_points.data["color"].detach(), vo_points.data["cov_Tw"].detach(), "sphere")


def onFrameFinished(frame: Frame, system: TartanVO, pb: ColoredTqdm, gt: list[torch.Tensor] | None = None):
    """
    Callback executed after each frame is processed.
    Handles visualization and VRAM usage logging.
    """
    VisualizeRerunCallback(frame, system, pb, gt)
    # VisualizeVRAMUsage(frame, system, pb)

def execute_experiment(name, cfg, cfg_dict, root_box: Sandbox) -> str:
    # Execute an experiment, and return the directory of result sandbox
    exp_space = root_box.new_child(name)
    exp_space.config = cfg_dict

    sequence = smart_transform(
        SequenceBase[Frame].instantiate(cfg.Data.type, cfg.Data.args),
        cfg.Preprocess
    )#.preload()
    system = TartanVO.from_config(cfg, sequence)
    system.receive_frames(sequence, exp_space, on_frame_finished=onFrameFinished)

    return str(exp_space.folder)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    args.add_argument(
        "--resultRoot", type=str, default="Results"
    )
    args.add_argument("--useRR", action="store_true", help="Enable Rerun visualization")
    args = args.parse_args()

    Logger.write("info", f"Using configuration from: {args.config}")
    cfg, cfg_dict = load_config(Path(args.config))
    odometry_cfg = cfg_dict["Odometry"]
    data_cfgs = cfg_dict["Datas"]

    run_configs = [
        {
            "Project": odometry_cfg["name"] + "@" + data_cfg["name"],
            "Data": data_cfg,
            "Odometry": odometry_cfg,
            "Preprocess": cfg_dict["Preprocess"]
        }
        for data_cfg in data_cfgs
    ]

    root_box = Sandbox.create(
        Path(args.resultRoot), Path(args.config).name.split(".")[0]
    )
    spaces = []

    for run_cfg_template in run_configs:
        if args.useRR:
                    rr_plt.default_mode = "rerun"
                    rr_plt.init_connect(run_cfg_template["Project"])
        cfg, cfg_dict = build_dynamic_config(run_cfg_template)
        Logger.write("info", cfg_dict)
        spaces.append(execute_experiment(cfg.Project, cfg, cfg_dict, root_box))

    Logger.write(
        "info",
        "Finished experiment group, the results are stored in"
        + "\n"
        + " ".join(spaces),
    )
    with root_box.open("runs.txt", "w") as f:
        f.write("\n".join(spaces))

    eval_header, eval_results = EvaluateSequences(spaces, correct_scale=True)
    print_as_table(eval_header, eval_results)
