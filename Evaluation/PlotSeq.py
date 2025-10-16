import argparse
import numpy as np
from pathlib import Path
from typing import Literal

from Utility.Plot import getColor, AnalyzeRotation, AnalyzeTranslation, \
      PlotTrajectory, AnalyzeRTE_cdf, AnalyzeROE_cdf
from Utility.PrettyPrint import ColoredTqdm, Logger
from Utility.Sandbox import Sandbox
from Utility.Trajectory import Trajectory

import matplotlib.pyplot as plt

NEED_ALIGN_SCALE: dict[str, Literal["Dynamic"] | float] = {
    "dpvo"         : "Dynamic",
    "droid"        : "Dynamic",
    "tartanvo_mono": "Dynamic",
    "mast3r"       : "Dynamic",
    "macvo"        : "Dynamic",
    "eiva"         : "Dynamic",
}

def test_plot_trajectory(data, title="Trajectory", xlabel="X", ylabel="Y"):
    poses = data.poses
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], marker='o')
    ax.set_xlabel('tx')
    ax.set_ylabel('ty')
    plt.title(title)
    plt.show(block=False)


def plot_separately(
        spaces: list[str],
        correct_scale: bool = False,
        align_origin: bool = True,
        align: bool = False
        ):
    for spaceid in ColoredTqdm(spaces, desc="Plotting"):
        exp_space = Sandbox.load(spaceid)
        config = exp_space.config

        try:
            gt_traj, est_traj = Trajectory.from_sandbox(exp_space, align_time="est->gt")
            est_traj.plot_kwargs |= {"color": getColor("-", 4, 0)}

            Logger.write("warn", f"Est: {est_traj.data.poses[0:5]}")
            # test_plot_trajectory(est_traj.data, title=f"Estimated Trajectory: {est_traj.name}")
            # test_plot_trajectory(gt_traj.data, title=f"Ground Truth Trajectory: {gt_traj.name}")
                        # save trajectory file
            # Save estimated trajectory data to a txt file
            output_path = Path(spaces[0], "EstimatedTrajectory.txt")
            # Concatenate timestamps and poses for each row and save to txt
            data_to_save = np.hstack((est_traj.data.time.reshape(-1, 1), est_traj.data.poses))
            np.savetxt(output_path, data_to_save, fmt="%.6f")

            if gt_traj is None:
                Logger.write("warn", f"Unable to plot error analysis for {est_traj} since could not load GT trajectory.")
                return

            gt_traj.plot_kwargs  |= {"linewidth": 3, "linestyle": ":"}

            for key, scale in NEED_ALIGN_SCALE.items():
                if key not in est_traj.name.lower(): continue

                Logger.write("info", f"{est_traj} --[align_scale={scale}]-> {gt_traj}")
                if scale == "Dynamic" and correct_scale and not align:
                    est_traj.data = est_traj.data.align_scale(gt_traj.data, correct_scale=correct_scale, align=align)
                elif scale == "Dynamic" and align:
                    est_traj.data = est_traj.data.evo_align(gt_traj.data, correct_scale=correct_scale)
                else: est_traj.data = est_traj.data.scale(scale)
                break

            if align_origin:
                est_traj.data = est_traj.data.align_origin(gt_traj.data)
            name = config.Project if hasattr(config, "Project") else est_traj.name
            AnalyzeTranslation(
                [(gt_traj.apply(lambda traj: traj.as_motion), est_traj.apply(lambda traj: traj.as_motion))],
                Path("Results", f"{name}_TranslationErr.png")
            )
            AnalyzeRotation(
                [(gt_traj.apply(lambda traj: traj.as_motion), est_traj.apply(lambda traj: traj.as_motion))],
                Path("Results", f"{name}_RotationErr.png")
            )
            PlotTrajectory([gt_traj, est_traj], Path(spaces[0], f"{name}_Trajectory.png"))

            # save trajectory file
            # Save estimated trajectory data to a txt file
            output_path = Path(spaces[0], f"{name}_EstimatedTrajectory_Aligned.txt")
            # Concatenate timestamps and poses for each row and save to txt
            data_to_save = np.hstack((est_traj.data.time.reshape(-1, 1), est_traj.data.poses))
            np.savetxt(output_path, data_to_save, fmt="%.6f")

            # Save ground truth trajectory data to a txt file
            output_path = Path(spaces[0], f"{name}_GroundTruthTrajectory.txt")
            # Concatenate timestamps and poses for each row and save to txt
            data_to_save = np.hstack((gt_traj.data.time.reshape(-1, 1), gt_traj.data.poses))
            np.savetxt(output_path, data_to_save, fmt="%.6f")

        except Exception as e:
            Logger.show_exception()


def plot_jointly(
        spaces: list[str],
        correct_scale: bool = False,
        align_origin: bool = True,
        align: bool = False
        ):
    trajs = [Trajectory.from_sandbox(Sandbox.load(space)) for space in spaces]

    for idx, (gt_traj, est_traj) in enumerate(trajs):
        est_traj.plot_kwargs |= {"color": getColor("-", (idx * 2) % 7, 0)}
        gt_traj.plot_kwargs  |= {"linewidth": 4, "linestyle": "--"}

        for key, scale in NEED_ALIGN_SCALE.items():
            if key not in est_traj.name.lower(): continue

            Logger.write("info", f"{est_traj} --[align_scale={scale}]-> {gt_traj}")
            if scale == "Dynamic" and correct_scale and not align:
                est_traj.data = est_traj.data.align_scale(gt_traj.data)
            elif scale == "Dynamic" and align:
                est_traj.data = est_traj.data.evo_align(gt_traj.data, correct_scale=correct_scale)
            else: est_traj.data = est_traj.data.scale(scale)
            break
        if align_origin:
            est_traj.data = est_traj.data.align_origin(gt_traj.data)

    gt_traj, est_trajs = trajs[0][0], [est for _, est in trajs]
    for space in spaces:
        PlotTrajectory([gt_traj] + est_trajs, Path(space, f"{gt_traj.name}_Compare.png"))

    trajs_motion = [
        (gt_traj.apply(lambda x: x.as_motion), est_traj.apply(lambda x: x.as_motion))
        for (gt_traj, _), est_traj in zip(trajs, est_trajs)
    ]

    for space in spaces:

        AnalyzeTranslation(
            trajs_motion,
            Path(space, f"Combined_trel.png")
        )
        AnalyzeRotation(
            trajs_motion,
            Path(space, f"Combined_rrel.png")
        )
        AnalyzeRTE_cdf(
            trajs_motion,
            None,
            Path(space, f"Combined_RTEcdf.png")
        )
        AnalyzeROE_cdf(
            trajs_motion,
            None,
            Path(space, f"Combined_ROEcdf.png")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", type=str, nargs="+", default=[])
    parser.add_argument("--correctScale", action="store_true")
    parser.add_argument("--alignOrigin", action="store_true", help="Align origin of estimated trajectory to ground truth.")
    parser.add_argument("--align", action="store_true", help="Align estimated trajectory to ground truth.")
    parser.add_argument("--recursive", action="store_true", help="Find and evaluate on leaf sandboxes only.")
    args = parser.parse_args()

    if args.recursive:
        spaces = []
        for space in args.spaces:
            spaces.extend([str(child.folder.absolute()) for child in Sandbox.load(space).get_leaves()])
        Logger.write("info", f"Found {len(spaces)} spaces to plot.")
    else:
        spaces = args.spaces
    plot_separately(spaces, args.correctScale, args.alignOrigin, args.align)
    plot_jointly(spaces, args.correctScale, args.alignOrigin, args.align)
