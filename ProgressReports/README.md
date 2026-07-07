# Progress Reports — combined conclusions

Living summary of the studies in this folder. Updated 2026-07-07.

| Report | Scope | Segment |
|---|---|---|
| [2026-07-06 · Optimizer hyperparameter sweep](2026-07-06_optimizer-hyperparameter-sweep.md) | GTSAM (pose2point) & G-EDF (`gedf+icp`) optimizer blocks, DAv2/DAv3 mono | `plane_nose[40:200]` |
| [2026-07-07 · Pure G-EDF mapper sweep](2026-07-07_pure-gedf-mapper-sweep.md) | All G-EDF mapper/field knobs, `graph_type: gedf` (no ICP), DAv2/DAv3/stereo-Fast | `plane_nose[80:160]` |
| [2026-07-07 · G-EDF alignment-manifold benchmark](2026-07-07_gedf-alignment-manifold-benchmark.md) | SE(3)/Sim(3)/SL(4) per-frame alignment in `GEDF_PGO` (`gedf+icp`), DAv2/DAv3 mono | `plane_nose[80:160]` |

All three evaluate monocular-style: trajectories Sim(3) scale-corrected + aligned to GT before
ATE/RTE/ROE (`EvaluateSequences(align=True, correct_scale=True)`).

## Headline: best known configurations

On `plane_nose[80:160]`, ranked by ATE RMSE:

| Rank | Configuration | ATE RMSE (m) | Source |
|--:|---|--:|---|
| 1 | **G-EDF `gedf+icp` + DAv2 + Sim(3) alignment** | **0.251** | alignment benchmark |
| 2 | G-EDF `gedf+icp` + DAv2 (SE(3), autodiff) | 0.379 | alignment benchmark |
| 3 | G-EDF `gedf+icp` + DAv3 (SE(3), autodiff) | 0.618 | alignment benchmark |
| 4 | pure G-EDF + DAv2, tuned (`budget 16`) | 0.671 / 0.698 (two seeds) | pure-GEDF sweep |
| 5 | pure G-EDF + DAv2, defaults | 1.108 / 0.968 | pure-GEDF sweep |
| — | pure G-EDF + stereo Fast / DAv3 | 1.26 – 2.25 | pure-GEDF sweep |

On `plane_nose[40:200]` (not directly comparable — 2× longer, includes a different trajectory
portion): tuned DAv2+GTSAM reaches 0.512/0.496 and tuned DAv2+G-EDF (`gedf+icp`, `sigma 0.20`)
0.910/1.025. A head-to-head of tuned GTSAM vs `gedf+icp`+Sim(3) on the *same* segment has not
been run yet (see open questions).

## Conclusions that hold across all reports

1. **DAv2-vitl beats DA3NESTED-giant on plane_nose, in every backend and every manifold** —
   GTSAM, fused G-EDF, pure G-EDF, and all three alignment types. DAv3 is also 2–10× slower on a
   single GPU (giant-model VRAM pressure; GTSAM CustomFactor bottleneck at `num_point 1000`).
2. **The depth model's covariance/scale character decides what the optimizer wants; nothing
   transfers between frontends.** The tuned DAv2 GTSAM combo *hurts* DAv3; G-EDF sigma optima
   differ per frontend; `cov_trace_gate` values that work for stereo reject 100 % of mono points
   (degenerate static trajectories); Sim(3) alignment helps exactly the frontend whose depth scale
   drifts (DAv2, −34 %) and hurts the scale-consistent one (DAv3). **Re-tune per depth model, and
   scale covariance-gates to the frontend's covariance magnitude.**
3. **The ICP factor is load-bearing in G-EDF.** Pure field registration, even after tuning all
   mapper knobs, stays ≥ 0.67 — roughly 2.7× worse than the fused backend with alignment on the
   same segment. The field factor's value is anchoring drift *on top of* ICP, not replacing it.
4. **Interaction structure differs by backend — tune accordingly.** GTSAM improvements only
   materialize in combination (singles ≤ 0.8 %, combo 9 %) and sit next to cliffs; pure-G-EDF
   combinations consistently *underperform* their best single ingredient. Sweep GTSAM as combos,
   G-EDF one knob at a time.
5. **Extra degrees of freedom must earn their keep.** One well-chosen DoF (Sim(3) log-scale)
   gives −34 %; nine (SL(4)) overfit and diverge (per-frame scale up to 8×) while running 2–3×
   slower. Same theme in the sweeps: `num_gaussians` 8→32 helps, 64 adds nothing;
   `budget_cubes_per_frame` 8→16 helps DAv2, 32 gives it back.
6. **Determinism + multi-seed validation is mandatory for G-EDF conclusions.** Runs are
   bit-reproducible under a fixed seed (`MACVO.py --seed`), but the map is fitted on randomly
   subsampled points: seed-to-seed ATE swings are ±10 % for `gedf+icp` and up to ±40 % for pure
   G-EDF. Two single-seed "wins" (DAv2 `sigma 0.10`, stereo `cov_trace_gate 0`) flipped sign on
   the second seed. GTSAM noise is ~1 %. The alignment benchmark is single-seed and its DAv2
   margin (−34 %) exceeds `gedf+icp` noise (±10 %), but a seed confirmation would make it solid.
7. **Warm-up matters for online maps on short clips.** `min_gaussians` 400+ leaves the field
   factor inert for most of an 80-frame segment (pure G-EDF degenerates outright); keep ≤ 100.
   `cube_size` 1.0 is a hard sweet spot — halving or doubling degrades or degenerates.

## Current recommended settings

- **DAv2 mono, best accuracy:** `MACVO_GEDF_DAv2.yaml` (`gedf+icp`) + `alignment: {type: sim3,
  prior_weight: 100}`, `autodiff: true`, `parallel: false`.
- **DAv2 mono, GTSAM:** `MACVO_MonoDAv2.yaml` ships the tuned combo (`huber 0.05, hprev 3.0,
  atol 2.0, psigma 1e-3`).
- **DAv3 mono:** keep optimizer defaults everywhere; prefer SE(3) (no alignment warp); needs
  `device_depth: cuda:1` for G-EDF (single 24 GB card OOMs).
- **Pure G-EDF** (when running field-only): DAv2 `budget_cubes_per_frame 16`; DAv3
  `num_gaussians 32`; stereo Fast (`MACVO_GEDF_Fast.yaml`) defaults.

## Open questions

- Head-to-head on one segment & multiple seeds: tuned DAv2+GTSAM vs `gedf+icp`+Sim(3) — the two
  family champions have never met under identical conditions.
- Seed-confirm the alignment benchmark (single-seed today); ±10 % `gedf+icp` noise vs −34 % margin
  suggests it will hold.
- Do the pure-G-EDF mapper optima (`budget 16`, `ngauss 32`) help the *fused* backend too?
  Untested under `gedf+icp`.
- Longer/other sequences: every number above is one 80–160-frame plane_nose segment; nothing here
  is a paper-grade multi-sequence average.
- DAv3+GTSAM runtime (~20 s/frame at `num_point 1000`): needs a native C++ pose→point factor or
  fewer keypoints before DAv3 sweeps get cheap.
