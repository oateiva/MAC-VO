import argparse
from pathlib import Path

from DataLoader import SequenceBase, Frame, smart_transform
from Evaluation.EvalSeq import EvaluateSequences
from Odometry.MACVO import MACVO
from Utility.Config import build_dynamic_config, load_config
from Utility.PrettyPrint import ColoredTqdm, Logger, print_as_table
from Utility.Sandbox import Sandbox

import rerun as rr
from Utility.Visualize import fig_plt, rr_plt
from MACVO import VisualizeRerunCallback, VisualizeVRAMUsage

def onFrameFinished(frame: Frame, system: MACVO, pb: ColoredTqdm):
    VisualizeRerunCallback(frame, system, pb)
    VisualizeVRAMUsage(frame, system, pb)


def execute_experiment(name, cfg, cfg_dict, root_box: Sandbox) -> str:
    # Execute an experiment, and return the directory of result sandbox
    exp_space = root_box.new_child(name)
    exp_space.config = cfg_dict

    sequence = smart_transform(
        SequenceBase[Frame].instantiate(cfg.Data.type, cfg.Data.args),
        cfg.Preprocess
    )#.preload() # Preload can be enabled here if needed. And if you have a big ass RAM.
    system = MACVO[Frame].from_config(cfg)
    system.receive_frames(sequence, exp_space, on_frame_finished=onFrameFinished)

    return str(exp_space.folder)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, nargs='+', required=True)
    args.add_argument(
        "--resultRoot", type=str, default="Results"
    )
    args.add_argument("--n-runs", type=int, default=1)
    args.add_argument("--useRR", action="store_true")

    args = args.parse_args()


    config_list = args.config

    spaces = []

    for config in config_list:
        Logger.write("info", f"Using configuration from: {config}")

        cfg, cfg_dict = load_config(Path(config))
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
            Path(args.resultRoot), Path(config).name.split(".")[0]
        )

        for run_cfg_template in run_configs:
            for run_idx in range(args.n_runs):
                Logger.write(
                    "info",
                    f"Starting experiment: {run_cfg_template['Project']} (Run {run_idx + 1}/{args.n_runs})",
                )
                # Setup logging and visualization
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

    eval_header, eval_results = EvaluateSequences(spaces, correct_scale=False)
    print_as_table(eval_header, eval_results)
