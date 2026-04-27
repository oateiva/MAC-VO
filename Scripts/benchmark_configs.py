#!/usr/bin/env python3
"""
Benchmark: compare 1GPU vs 2GPU, parallel vs no-parallel.
Runs each MACVO.py experiment as a subprocess, then collects ATE, FPS, and VRAM.
"""
import subprocess
import sys
import glob
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Evaluation.EvalSeq import EvaluateSequences

CONFIGS = [
    ("1GPU  Parallel", "Config/Experiment/MACVO/_bench_1gpu_para.yaml",    "bench-1gpu-para"),
    ("2GPU  No-Para",  "Config/Experiment/MACVO/_bench_2gpu_nopara.yaml",  "bench-2gpu-nopara"),
    ("2GPU  Parallel", "Config/Experiment/MACVO/_bench_2gpu_para.yaml",    "bench-2gpu-para"),
]

RESULTS_ROOT = "./Results/benchmark"
DATA_CONFIG  = "Config/Sequence/EIVA_Dataset/plane_nose.yaml"

summary: list[tuple] = []

for label, odom_config, odom_name in CONFIGS:
    print(f"\n{'='*70}", flush=True)
    print(f"  RUNNING: {label}", flush=True)
    print(f"{'='*70}", flush=True)

    cmd = [
        sys.executable, "MACVO.py",
        "--useRR",
        "--data",        DATA_CONFIG,
        "--odom",        odom_config,
        "--n_to_align",  "20",
        "--resultRoot",  RESULTS_ROOT,
        "--noeval",
    ]

    proc = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace")
    print(f"\n  subprocess exit code: {proc.returncode}", flush=True)

    # Locate the most recent sandbox for this config
    search = str(Path(RESULTS_ROOT) / f"{odom_name}@plane_nose" / "*")
    dirs = sorted(glob.glob(search))
    if not dirs:
        print(f"  ERROR: no result dir found at {search}", flush=True)
        summary.append((label, None, None, None, None, None))
        continue
    result_dir = Path(dirs[-1])
    print(f"  Result dir: {result_dir}", flush=True)

    # Per-frame metrics
    iter_ms_path = result_dir / "iter_time_ms.npy"
    vram_gb_path = result_dir / "vram_usage_gb.npy"
    iter_ms = np.load(str(iter_ms_path)) if iter_ms_path.exists() else None
    vram_gb = np.load(str(vram_gb_path)) if vram_gb_path.exists() else None

    median_fps = float(1000.0 / np.median(iter_ms)) if iter_ms is not None and len(iter_ms) > 0 else None
    peak_vram  = float(np.max(vram_gb))              if vram_gb is not None and len(vram_gb) > 0 else None
    avg_vram   = float(np.mean(vram_gb))             if vram_gb is not None and len(vram_gb) > 0 else None

    # ATE evaluation
    try:
        _, rows = EvaluateSequences(
            [str(result_dir)],
            align=True, correct_scale=True, align_origin=False, n_to_align=20,
        )
        ate_mean = rows[0][1] if rows and rows[0][1] is not None else None
        ate_rmse = rows[0][3] if rows and rows[0][3] is not None else None
    except Exception as exc:
        print(f"  Eval error: {exc}", flush=True)
        ate_mean, ate_rmse = None, None

    print(f"\n  Median FPS  : {median_fps:.2f}" if median_fps else "  Median FPS : N/A")
    print(f"  Peak VRAM   : {peak_vram:.3f} GB" if peak_vram else "  Peak VRAM  : N/A")
    print(f"  Avg VRAM    : {avg_vram:.3f} GB"  if avg_vram  else "  Avg VRAM   : N/A")
    print(f"  mu_ATE       : {ate_mean:.4f} m"   if ate_mean  else "  mu_ATE      : N/A")
    print(f"  RMSE_ATE    : {ate_rmse:.4f} m"   if ate_rmse  else "  RMSE_ATE   : N/A")

    summary.append((label, median_fps, peak_vram, avg_vram, ate_mean, ate_rmse))

# Final table
print("\n\n" + "="*82)
print("FINAL COMPARISON")
print("="*82)
hdr = f"{'Config':<20} | {'FPS':>7} | {'PeakVRAM(GB)':>13} | {'AvgVRAM(GB)':>12} | {'mu_ATE(m)':>10} | {'RMSE_ATE':>10}"
sep = "-" * len(hdr)
print(hdr)
print(sep)
for row in summary:
    label, fps, pvram, avram, ate, rmse = row
    print(
        f"{label:<20} | "
        f"{f'{fps:.2f}':>7} | "
        f"{f'{pvram:.3f}':>13} | "
        f"{f'{avram:.3f}':>12} | "
        f"{f'{ate:.4f}':>10} | "
        f"{f'{rmse:.4f}':>10}"
        if all(v is not None for v in [fps, pvram, avram, ate, rmse])
        else f"{label:<20} | {'N/A':>7} | {'N/A':>13} | {'N/A':>12} | {'N/A':>10} | {'N/A':>10}"
    )
print(sep)
