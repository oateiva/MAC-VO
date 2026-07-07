# Progress Report — GEDF alignment-manifold benchmark

| | |
|---|---|
| **Date** | 2026-07-07 |
| **Branch** | `feature/claude-refactor` |
| **Task** | Evaluate `GEDF_PGO` accuracy under the SE(3) / Sim(3) / SL(4) alignment manifolds |
| **Status** | ✅ Complete — 6/6 runs, metrics + Rerun recordings saved |

How the per-frame **alignment manifold** (`optimizer.args.alignment.type`, see
[`Module/Optimization/README.md`](../Module/Optimization/README.md#alignment-axis-monocular-depth-bias-correction)
and [`Module/Optimization/GEDF/Alignment.py`](../Module/Optimization/GEDF/Alignment.py)) affects
trajectory accuracy for two monocular depth frontends on `plane_nose[80:160]` (80 frames).

Only the `GEDF_PGO` configs are swept — the manifold is a GEDF feature; the `GTSAM_Graph`
configs (`MACVO_MonoDAv2`, `MACVO_MonoDAv3`) are SE(3)-only and are out of scope here.

## Verdict

**Best configuration: `MACVO_GEDF_DAv2` + `sim3` — ATE RMSE 0.251 m.**

The alignment manifold *helps* the DAv2 frontend (Sim(3) beats SE(3) by 34 %) but *hurts* the
already scale-consistent DA3NESTED-giant frontend. SL(4)'s projective freedom overfits in both.
DAv2-vitl outperforms DAv3-giant across every manifold on this segment.

## Accuracy matrix

Monocular evaluation: the estimated trajectory is aligned to ground truth with **Sim(3)**
(rotation + translation + scale) before error computation (`EvaluateSequences(..., align=True,
correct_scale=True)`). Lower is better; **bold** = best manifold per frontend.

| Frontend | Manifold | ATE RMSE (m) | ATE mean (m) | RTE RMSE (m) | ROE RMSE (°) | RPE RMSE (m) | wall (s) |
|---|---|--:|--:|--:|--:|--:|--:|
| GEDF-DAv2 | se3  | 0.379 | 0.330 | 0.203 | 5.80 | 0.248 | 337 |
| GEDF-DAv2 | **sim3** | **0.251** | 0.212 | 0.196 | 5.98 | 0.246 | 347 |
| GEDF-DAv2 | sl4  | 0.410 | 0.359 | 0.199 | 5.86 | 0.246 | 626 |
| GEDF-DAv3 | **se3**  | **0.618** | 0.527 | 0.204 | 5.79 | 0.249 | 610 |
| GEDF-DAv3 | sim3 | 0.757 | 0.620 | 0.202 | 5.88 | 0.248 | 733 |
| GEDF-DAv3 | sl4  | 0.786 | 0.673 | 0.280 | 5.62 | 0.312 | 965 |

_ATE / RTE / RPE in metres · ROE in degrees · wall = single-GPU (DAv2) / dual-GPU (DAv3) run time._

## Alignment scale channel

The per-frame depth-scale correction the optimizer estimated and logged live to
`/world/gedf_alignment/scale` (identity = 1.0). This is the diagnostic behind the accuracy:
a bounded, stable correction helps; a diverging one hurts. `se3` has no scale channel (identity warp).

| Run | scale min | scale median | scale max |
|---|--:|--:|--:|
| DAv2 · sim3 | 0.610 | 0.989 | 1.106 |
| DAv2 · sl4  | 0.883 | 1.032 | **2.881** |
| DAv3 · sim3 | 0.294 | 0.957 | 1.283 |
| DAv3 · sl4  | 0.429 | 1.110 | **7.976** |

## Findings

- **↓ Sim(3) is the win for DAv2.** One log-scale DoF absorbs DAv2-vitl's per-frame depth-scale
  drift, cutting ATE RMSE 0.379 → 0.251 m (−34 %); its scale stays tight (0.61–1.11).
- **↑ The manifold hurts DAv3.** DA3NESTED-giant already produces scale-consistent depth, so the
  extra warp DoF only inject noise: SE(3) 0.618 → Sim(3) 0.757 → SL(4) 0.786. Plain SE(3) is best.
- **✕ SL(4) overfits, everywhere.** The 9 projective DoF let the warp diverge (max 2.9× and 8.0×),
  giving the worst ATE in both frontends while running 2–3× slower (up to 965 s).
- **≈ Local odometry is untouched.** RTE (~0.20 m) and ROE (~5.8°) are essentially flat across all
  six runs — the manifold moves global scale / drift (ATE), not frame-to-frame motion.

## Method & caveats

**Setup**
- Backend `GEDF_PGO`, `graph_type: gedf+icp`, online map.
- Controlled variable: all six runs use `autodiff: true` and `parallel: false`, so the manifold is
  the only difference — deterministic, and no optimizer-timeout artifacts.
- Frontends: DAv2 = DepthAnythingV2-vitl (dataset `window_length: 1`); DAv3 = DA3NESTED-giant-large
  (`window_length: 2`, depth offloaded to `cuda:1`).

**Read with care**
- These `se3` numbers use `autodiff: true`. The shipped `MACVO_GEDF_DAv3.yaml` default is
  `autodiff: false` (analytic, SE(3)-only) and may differ slightly.
- DAv3 needs the depth model on a second GPU — `frontend.args.device_depth: cuda:1` is now set in
  `MACVO_GEDF_DAv3.yaml` to avoid a single-GPU OOM (DA3NESTED-giant + FlowFormer + the online map
  do not fit on one 24 GB card).
- 80-frame segment, single seed. Directional, not a paper-grade average over seeds / sequences.

## Reproduce

Per run (manifold ∈ {se3, sim3, sl4}; `sim3`/`sl4` require `autodiff: true`):

```yaml
# Config/Experiment/MACVO/MACVO_GEDF_DAv{2,3}.yaml → optimizer.args
optimizer:
  type: GEDF_PGO
  args:
    autodiff: true          # required for sim3 / sl4
    parallel: false         # deterministic (no timeout fallback)
    alignment:
      type: sim3            # se3 | sim3 | sl4
      prior_weight: 100.0
```

Run on `plane_nose` frames 80–160, then evaluate with Sim(3) alignment:

```bash
python MACVO.py --odom Config/Experiment/MACVO/MACVO_GEDF_DAv2.yaml \
                --data Config/Sequence/EIVA_Dataset/plane_nose.yaml \
                --seq_from 80 --seq_to 160 --useRR
# EvaluateSequences(..., align=True, correct_scale=True)  → monocular Sim(3) ATE
```

> **Data path note:** `Config/Sequence/EIVA_Dataset/plane_nose.yaml` currently has a stale `root`
> (`D:\Datasets\EIVA\plane_nose`); the data actually lives at `D:\Datasets\EIVA\vobster_quay\plane_nose`.
> Fix the `root` before running the command above.

Each run's Rerun recording carries `/world/est`, `/world/gt`, `/world/gedf_map`, depth / flow /
keypoints, and — for `sim3`/`sl4` — the live `/world/gedf_alignment/scale` plot.
