# Progress Report — Optimizer hyperparameter sweep (GTSAM & G-EDF backends)

| | |
|---|---|
| **Date** | 2026-07-06 → 07-07 |
| **Branch** | `feature/claude-refactor` |
| **Task** | Tune the optimizer blocks of the DAv2 / DAv3 monocular setups: `GTSAM_Graph` (pose2point) and `GEDF_PGO` (`gedf+icp`) |
| **Status** | ✅ Complete — ~50 runs, optima applied to configs (`4a743ad`) |

Sequence `plane_nose[40:200]` (160 frames). Metric: **ATE RMSE (m), Sim(3) scale-corrected +
aligned** (`EvaluateSequences(align=True, correct_scale=True)`). Driver:
[`Scripts/Experiment/SweepOptimizer.py`](../Scripts/Experiment/SweepOptimizer.py) — seeded,
deterministic (identical configs reproduce bit-identical metrics), optimizer `parallel: false`
(the 1 s parallel timeout drops results nondeterministically), GPU 1 only. Every candidate
optimum re-run under a second seed. Raw data: `Results/sweeps/{dav2,dav3}_{gtsam,gedf}[_seed1]/`.

## Verdict

| Setup | Baseline (s0 / s1) | Tuned (s0 / s1) | Optimum |
|---|--:|--:|---|
| DAv2 + GTSAM | 0.563 / 0.566 | **0.512 / 0.496** (−9 / −12 %) | `huber_delta 0.05, huber_delta_prev 3.0, match_atol 2.0, prior_sigma 1e-3, max_iterations 20` |
| DAv2 + G-EDF (`gedf+icp`) | 1.247 / 1.125 | **0.910 / 1.025** (−27 / −9 %) | `field.sigma 0.20` (rest default) |
| DAv3 + GTSAM | 1.702 | — | defaults optimal; DAv2 combo transfers **negatively** (1.757) |
| DAv3 + G-EDF (`gedf+icp`) | 1.442 / 1.478 | — | defaults; `sigma 0.10` won seed 0 (1.312) but lost seed 1 (1.571) |

GTSAM hyperparameters were hardcoded before this work; they are now config keys with
defaults equal to the old values (`Module/Optimization/GTSAM/`).

## Findings

- **GTSAM gains only appear in combination.** Each single change gave ≤ 0.8 %; the four-parameter
  combo gave 9 %. The optimum sits next to cliffs: `match_atol 3.0` (+21 %) and `prior_sigma 0.01`
  (+20 %) are sharply worse.
- **Nothing transfers between depth models.** Every DAv2-tuned GTSAM component is neutral-to-harmful
  under the DAv3 frontend; the G-EDF sigma optimum differs too. Depth-covariance scales change what
  the optimizer wants — re-tune per frontend.
- **G-EDF's error landscape is noisy** (seed swing ±0.12 ATE ≈ 10 %); effects below ~15 % need
  multi-seed averaging. GTSAM seed noise is ~1 %.
- **Model ranking (this segment):** DAv2+GTSAM 0.51 < DAv2+GEDF 0.91 < DAv3+GEDF 1.44 < DAv3+GTSAM 1.70.

## Engineering found & fixed along the way

- **Parallel-frontend VRAM leak** (`dbb6782`): a fresh CUDA stream per frame stranded ~11 GB of
  inference transients per frame in per-stream allocator pools; Windows/WDDM spilled to shared
  memory and slowed frames ~30×. Streams are now created once per frontend.
- **DAv3+GTSAM is optimizer-bound**: at `num_point 1000`, ~92 % of frame time (~20 s) is the GTSAM
  LM solve calling Python `CustomFactor` callbacks (py-spy). Association code was vectorized
  (bit-identical output, `3ab895e`), but the solve itself needs a native C++ factor or fewer
  keypoints for a real speedup.
- `.rrd` recordings must attach the file sink at init; a trailing `rr.save` records nothing
  (fixed earlier, `06b1455`).

## Reproduce

```bash
python Scripts/Experiment/SweepOptimizer.py \
    --base Config/Experiment/MACVO/MACVO_MonoDAv2.yaml \
    --data Config/Sequence/EIVA_plane_nose_mono.yaml \
    --runs <runs.json> --out Results/sweeps/<name> \
    --seq_from 40 --seq_to 200 --gpu 1     # add --seed 1 for confirmation runs
```

Full narrative + tables: `Results/sweeps/REPORT.md`.
