"""
Hyperparameter sweep driver for the MAC-VO optimizer backends.

Runs MACVO.py as a subprocess once per parameter combination: the base odometry
config is loaded (with !include resolution), patched with dotted-key overrides,
and written to <out>/configs/<run>.yaml. Each run's sandbox lands in
<out>/runs/<run>/, and the resulting trajectory is evaluated Sim3
scale-corrected + aligned against ground truth (the correct treatment for
monocular VO). Metrics are appended to <out>/results.csv after every run, so
the sweep is crash-safe and resumable (finished run names are skipped).

Runs spec (JSON):
    [ {"name": "huber_0.5", "overrides": {"Odometry.optimizer.args.huber_delta": 0.5}}, ... ]
Override value null deletes the key (used e.g. to drop device_depth).

Example:
    python Scripts/Experiment/SweepOptimizer.py \
        --base Config/Experiment/MACVO/MACVO_MonoDAv2.yaml \
        --data Config/Sequence/EIVA_plane_nose_mono.yaml \
        --runs sweep_runs.json --out Results/sweeps/dav2_gtsam \
        --seq_from 40 --seq_to 200 --gpu 1
"""
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CSV_FIELDS = [
    "name", "status", "wall_s",
    "ate_mean", "ate_std", "ate_rmse",
    "rte_mean", "rte_std", "rte_rmse",
    "roe_mean", "roe_std", "roe_rmse",
    "rpe_mean", "rpe_std", "rpe_rmse",
    "overrides",
]


def set_dotted(cfg: dict, dotted_key: str, value) -> None:
    """Set (or, for value None, delete) a dotted key inside a nested dict."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    if value is None:
        node.pop(parts[-1], None)
    else:
        node[parts[-1]] = value


def patch_config(base_cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    # Sweep-wide invariants:
    #   - single visible GPU per run -> a second depth device cannot exist
    #   - sequential optimizer: 'parallel: true' drops results on a 1s timeout,
    #     which makes runs nondeterministic and hides slow-parameter effects
    set_dotted(cfg, "Odometry.frontend.args.device_depth", None)
    set_dotted(cfg, "Odometry.optimizer.args.parallel", False)
    set_dotted(cfg, "Odometry.optimizer.args.parallel_timeout_s", None)
    for key, value in overrides.items():
        set_dotted(cfg, key, value)
    return cfg


def finished_runs(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return {row["name"] for row in csv.DictReader(f) if row["status"] == "ok"}


def append_row(csv_path: Path, row: dict) -> None:
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_sandbox(result_root: Path) -> dict:
    """Evaluate the single sandbox under result_root: Sim3 scaled + aligned."""
    from Evaluation.EvalSeq import EvaluateSequences  # heavy import, deferred

    # Sandboxes are nested: <result_root>/<odom>@<data>/<timestamp>/poses.npy
    candidates = sorted({p.parent for p in result_root.rglob("poses.npy")})
    assert len(candidates) == 1, f"Expected exactly one sandbox in {result_root}, found {candidates}"
    _, results = EvaluateSequences(
        [str(candidates[0])], correct_scale=True, align=True, align_origin=False,
    )
    row = results[0]  # [name, ate μ/σ/rmse, rte μ/σ/rmse, roe μ/σ/rmse, rpe μ/σ/rmse]
    if row[1] is None:
        raise RuntimeError("Evaluation failed (no trajectory?)")
    keys = ["ate_mean", "ate_std", "ate_rmse", "rte_mean", "rte_std", "rte_rmse",
            "roe_mean", "roe_std", "roe_rmse", "rpe_mean", "rpe_std", "rpe_rmse"]
    return dict(zip(keys, [float(v) for v in row[1:13]]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=str, required=True, help="Base odometry config")
    parser.add_argument("--data", type=str, required=True, help="Sequence config")
    parser.add_argument("--runs", type=str, required=True, help="JSON list of {name, overrides}")
    parser.add_argument("--out", type=str, required=True, help="Sweep output directory")
    parser.add_argument("--seq_from", type=int, default=40)
    parser.add_argument("--seq_to", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout_s", type=int, default=3600, help="Per-run timeout")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSONVALUE",
                        help="Global dotted-key override applied to every run "
                             "(before the per-run overrides), e.g. "
                             "--set Odometry.frontend.args.parallel=false")
    args = parser.parse_args()

    from Utility.Config import load_config  # after sys.path fix

    _, base_cfg = load_config(Path(args.base))
    global_overrides: dict = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        global_overrides[key] = json.loads(raw)
    with open(args.runs, "r", encoding="utf-8") as f:
        runs: list[dict] = json.load(f)

    out = Path(args.out)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"

    done = finished_runs(csv_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for spec in runs:
        name, overrides = spec["name"], spec.get("overrides", {})
        if name in done:
            print(f"[sweep] {name}: already done, skipping", flush=True)
            continue

        cfg = patch_config(base_cfg, global_overrides | overrides)
        cfg_path = out / "configs" / f"{name}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        result_root = out / "runs" / name
        if result_root.exists():  # leftovers from a crashed attempt
            import shutil
            shutil.rmtree(result_root)

        cmd = [
            sys.executable, str(REPO / "MACVO.py"),
            "--odom", str(cfg_path), "--data", args.data,
            "--seq_from", str(args.seq_from), "--seq_to", str(args.seq_to),
            "--resultRoot", str(result_root),
            "--seed", str(args.seed), "--noeval",
        ]
        log_path = out / "logs" / f"{name}.log"
        print(f"[sweep] {name}: starting ({overrides})", flush=True)
        t0 = time.monotonic()
        row = {k: "" for k in CSV_FIELDS}
        row |= {"name": name, "overrides": json.dumps(overrides)}
        try:
            with open(log_path, "w", encoding="utf-8") as log_f:
                subprocess.run(cmd, cwd=str(REPO), env=env, stdout=log_f,
                               stderr=subprocess.STDOUT, timeout=args.timeout_s, check=True)
            row |= evaluate_sandbox(result_root)
            row["status"] = "ok"
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
        except Exception as e:  # never kill the sweep on a single bad run
            row["status"] = f"failed:{type(e).__name__}"
        row["wall_s"] = f"{time.monotonic() - t0:.1f}"
        append_row(csv_path, row)
        print(f"[sweep] {name}: {row['status']} ATE_rmse={row.get('ate_rmse', '')} "
              f"({row['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
