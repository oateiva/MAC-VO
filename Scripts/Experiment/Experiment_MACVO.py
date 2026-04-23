# Experiment runner for MACVO system
# This script loads configuration(s), runs experiments, logs results, and evaluates sequences.

import argparse
from pathlib import Path

import torch

# Import custom modules for data loading, evaluation, odometry, config, logging, and visualization
from DataLoader import SequenceBase, Frame, smart_transform
from Evaluation.EvalSeq import EvaluateSequences
from Odometry.MACVO import MACVO
from Utility.Config import build_dynamic_config, load_config
from Utility.PrettyPrint import ColoredTqdm, Logger, print_as_table
from Utility.Sandbox import Sandbox

# Visualization tools
import rerun as rr
from typing import List
from Utility.Visualize import fig_plt, rr_plt
from MACVO import VisualizeRerunCallback, VisualizeVRAMUsage
from Utility.read_eiva_vslam_outputs import load_vslam_track, load_vslam_pointcloud

def onFrameFinished(frame: Frame, system: MACVO, pb: ColoredTqdm, gt: List[torch.Tensor] | None = None):
    """
    Callback executed after each frame is processed.
    Handles visualization and VRAM usage logging.
    """
    VisualizeRerunCallback(frame, system, pb, gt)
    VisualizeVRAMUsage(frame, system, pb)


def execute_experiment(name, cfg, cfg_dict, root_box: Sandbox) -> str:
    """
    Execute a single experiment run.
    Args:
        name (str): Name of the experiment/project.
        cfg: Configuration object for the run.
        cfg_dict: Dictionary version of the config.
        root_box (Sandbox): Root sandbox for results.
    Returns:
        str: Path to the result folder for this experiment.
    """
    exp_space = root_box.new_child(name)
    exp_space.config = cfg_dict



    # Set seq_from and seq_to based on presence in cfg.Data.args
    seq_from = getattr(cfg.Data.args, 'seq_from', 0)
    seq_to = getattr(cfg.Data.args, 'seq_to', -1)

    # Instantiate and preprocess the sequence
    sequence = smart_transform(
        SequenceBase[Frame].instantiate(cfg.Data.type, cfg.Data.args).clip(seq_from, seq_to),
        cfg.Preprocess
    )
    # .preload() can be enabled for large RAM systems

    # Initialize MACVO system and process frames
    system = MACVO[Frame].from_config(cfg)
    system.receive_frames(sequence, exp_space, on_frame_finished=onFrameFinished)

    del system
    torch.cuda.empty_cache()

    return str(exp_space.folder)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MACVO experiments with specified configs.")
    parser.add_argument("--config", type=str, nargs='+', required=True, help="Path(s) to config file(s)")
    parser.add_argument("--resultRoot", type=str, default="Results", help="Root directory for results")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of runs per config")
    parser.add_argument("--useRR", action="store_true", help="Enable Rerun visualization")

    args = parser.parse_args()

    config_list = args.config
    spaces = []  # List to store result directories

    # Iterate over each config file
    for config in config_list:
        Logger.write("info", f"Using configuration from: {config}")

        # Load config and extract odometry/data settings
        cfg, cfg_dict = load_config(Path(config))
        odometry_cfg = cfg_dict["Odometry"]
        data_cfgs = cfg_dict["Datas"]

        # Build run configs for each data sequence
        run_configs = [
            {
                "Project": odometry_cfg["name"] + "@" + data_cfg["name"],
                "Data": data_cfg,
                "Odometry": odometry_cfg,
                "Preprocess": cfg_dict["Preprocess"]
            }
            for data_cfg in data_cfgs
        ]

        # Create sandbox for experiment results
        root_box = Sandbox.create(
            Path(args.resultRoot), Path(config).name.split(".")[0]
        )

        # Run experiments for each config and repetition
        for run_cfg_template in run_configs:
            for run_idx in range(args.n_runs):
                Logger.write(
                    "info",
                    f"Starting experiment: {run_cfg_template['Project']} (Run {run_idx + 1}/{args.n_runs})",
                )
                # Setup visualization if requested
                if args.useRR:
                    rr_plt.default_mode = "rerun"
                    rr_plt.init_connect(run_cfg_template["Project"])
                    vslam_path = run_cfg_template["Data"]["args"].get("vslam")
                    if vslam_path:
                        load_vslam_track(vslam_path, entity_name=run_cfg_template["Project"])
                    vslampc_path = run_cfg_template["Data"]["args"].get("vslampc")
                    if vslampc_path:
                        load_vslam_pointcloud(vslampc_path, entity_name=run_cfg_template["Project"])
                # Build dynamic config for this run
                cfg, cfg_dict = build_dynamic_config(run_cfg_template)
                Logger.write("info", cfg_dict)
                # Execute experiment and collect result folder
                spaces.append(execute_experiment(cfg.Project, cfg, cfg_dict, root_box))
                if args.useRR:
                    rr.save(str(Path(spaces[-1]) / f"{run_cfg_template['Project']}.rrd"))

    # Log summary and save run directories
    Logger.write(
        "info",
        "Finished experiment group, the results are stored in\n" + " ".join(spaces),
    )
    with root_box.open("runs.txt", "w") as f:
        f.write("\n".join(spaces))

    # Evaluate all experiment results and print summary table
    # eval_header, eval_results = EvaluateSequences(spaces, correct_scale=False)
    # print_as_table(eval_header, eval_results)
