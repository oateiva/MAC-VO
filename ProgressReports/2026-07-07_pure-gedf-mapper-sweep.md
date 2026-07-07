# Progress Report — Pure G-EDF backend: mapper hyperparameter sweep

| | |
|---|---|
| **Date** | 2026-07-07 |
| **Branch** | `feature/claude-refactor` |
| **Task** | Tune all G-EDF mapper/field hyperparameters for a backend relying **purely** on field registration (`graph_type: gedf`, no ICP factor) — DAv2, DAv3, and a new stereo Fast setup |
| **Status** | ✅ Complete — ~90 runs, configs + report committed (`2de7c7f`) |

Sequence `plane_nose[80:160]` (80 frames). Metric: ATE RMSE (m), Sim(3) scale-corrected + aligned.
Swept: `num_gaussians`, `cube_size` (+`margin`), `sample_points`, `budget_cubes_per_frame`,
`mae_target`, `cov_trace_gate`, `min_gaussians`, `field.sigma`. Protocol as the 07-06 sweep
(seeded/deterministic, sequential optimizer, stage 1 one-at-a-time → stage 2 combos → seed-1
confirmation). New configs: [`MACVO_GEDF_Fast.yaml`](../Config/Experiment/MACVO/MACVO_GEDF_Fast.yaml)
(from `MACVO_Fast`, fp16/bf16 CUDAGraph stereo frontend, `graph_type: gedf`) and
[`EIVA_plane_nose_stereo.yaml`](../Config/Sequence/EIVA_plane_nose_stereo.yaml) (stereo EIVA needs
`window_length: 1`). Raw data: `Results/sweeps/gedf_pure_{dav2,dav3,stereo}[_seed1]/`.
Interactive report: <https://claude.ai/code/artifact/ebe6d045-7357-469a-a4bb-e5d5c7b563c4>.

## Verdict — cross-seed validated

| Setup | Baseline (s0 / s1) | Tuned (s0 / s1) | Optimum |
|---|--:|--:|---|
| DAv2 pure-GEDF | 1.108 / 0.968 | **0.671 / 0.698** (−34 %) | `budget_cubes_per_frame: 16` (rest default) |
| DAv3 pure-GEDF | 2.247 / 2.170 | **1.259 / 2.101** (−24 % mean; seed-1 marginal) | `num_gaussians: 32` (rest default) |
| Stereo Fast pure-GEDF | 1.757 / 1.372 | — | defaults; nothing replicated across seeds |

Stereo's apparent stage-1 winner (`cov_trace_gate: 0`, 1.757 → 1.005 on seed 0) **flipped** on
seed 1 (1.372 → 2.216) — single-seed luck, not signal.

## Findings

- **Tune pure G-EDF one knob at a time.** Every stage-2 combination underperformed its best single
  ingredient, in all three setups — the exact opposite of the GTSAM backend, where gains only
  appeared in combination.
- **`cov_trace_gate` is covariance-scale-critical.** Stereo-scale gates (0.0675 / 0.27 m²) reject
  every monocular point → map never reaches `min_gaussians` → field factor inert → static,
  degenerate trajectory (Umeyama alignment fails). DAv3's covariances pass at 0.27; DAv2's only at
  ≫ 1, where the gate no longer rejects anything.
- **`num_gaussians` is the most consistent stage-1 lever** — monotonic 4 → 32 in all three setups,
  nothing extra at 64 — but only DAv3's gain survived the seed test.
- **Hard constraints:** `cube_size` stays 1.0 (0.5 and 2.0 degrade or degenerate);
  `min_gaussians ≤ 100` (400 leaves pure registration mapless for most of an 80-frame clip).
- **Pure-G-EDF seed noise is large** (up to ±40 % on stereo — vs ±10 % for `gedf+icp`): the map is
  fitted on randomly subsampled points, and without the ICP factor nothing damps that variance.
  Single-seed wins routinely flip; validate everything on a second seed.
- **The ICP factor is essential.** Even tuned, pure G-EDF (0.67 on DAv2) does not reach the fused
  backend's accuracy on this sequence.

## Reproduce

```bash
python Scripts/Experiment/SweepOptimizer.py \
    --base Config/Experiment/MACVO/MACVO_GEDF_Fast.yaml \
    --data Config/Sequence/EIVA_plane_nose_stereo.yaml \
    --runs <runs.json> --out Results/sweeps/<name> \
    --seq_from 80 --seq_to 160 --gpu 1
# mono setups: --base MACVO_GEDF_DAv{2,3}.yaml --data EIVA_plane_nose_mono.yaml \
#              --set 'Odometry.optimizer.args.graph_type="gedf"'
```

Full narrative: `Results/sweeps/REPORT_pure_gedf.md`.
