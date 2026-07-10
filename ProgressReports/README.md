# Progress Reports — combined conclusions

Living summary of the studies in this folder. Updated 2026-07-07.

| Report | Scope | Segment |
|---|---|---|
| [2026-07-06 · Optimizer hyperparameter sweep](2026-07-06_optimizer-hyperparameter-sweep.md) | GTSAM (pose2point) & G-EDF (`gedf+icp`) optimizer blocks, DAv2/DAv3 mono | `plane_nose[40:200]` |
| [2026-07-07 · Pure G-EDF mapper sweep](2026-07-07_pure-gedf-mapper-sweep.md) | All G-EDF mapper/field knobs, `graph_type: gedf` (no ICP), DAv2/DAv3/stereo-Fast | `plane_nose[80:160]` |
| [2026-07-07 · G-EDF alignment-manifold benchmark](2026-07-07_gedf-alignment-manifold-benchmark.md) | SE(3)/Sim(3)/SL(4) per-frame alignment in `GEDF_PGO` (`gedf+icp`), DAv2/DAv3 mono | `plane_nose[80:160]` |
| [2026-07-07 · GTSAM alignment-manifold sweep](2026-07-07_gtsam-alignment-sweep.md) | Sim(3)/SL(4) alignment × prior_weight in `GTSAM_Graph` (pose2point), DAv2/DAv3 mono | `plane_nose[80:160]` |

All four evaluate monocular-style: trajectories Sim(3) scale-corrected + aligned to GT before
ATE/RTE/ROE (`EvaluateSequences(align=True, correct_scale=True)`).

## Headline: best known configurations

On `plane_nose[80:160]`, ranked by ATE RMSE:

| Rank | Configuration | ATE RMSE (m) | Source |
|--:|---|--:|---|
| 1 | **tuned DAv2+GTSAM (pose2point, SE(3))** | **0.233 / 0.229** (two seeds) | GTSAM alignment sweep |
| 2 | G-EDF `gedf+icp` + DAv2 + Sim(3) alignment | 0.251 | alignment benchmark |
| 3 | G-EDF `gedf+icp` + DAv2 (SE(3), autodiff) | 0.379 | alignment benchmark |
| 4 | G-EDF `gedf+icp` + DAv3 (SE(3), autodiff) | 0.618 | alignment benchmark |
| 5 | DAv3+GTSAM (defaults, SE(3)) | 0.689 | GTSAM alignment sweep |
| 6 | pure G-EDF + DAv2, tuned (`budget 16`) | 0.671 / 0.698 (two seeds) | pure-GEDF sweep |
| — | pure G-EDF + stereo Fast / DAv3 | 1.26 – 2.25 | pure-GEDF sweep |

The former open question — tuned GTSAM vs `gedf+icp`+Sim(3) head-to-head — is now answered:
**tuned DAv2+GTSAM wins on this segment (0.233/0.229 vs 0.251)**, seed-confirmed. On
`plane_nose[40:200]` (not directly comparable — 2× longer, different trajectory portion):
tuned DAv2+GTSAM 0.512/0.496, tuned DAv2+G-EDF (`gedf+icp`, `sigma 0.20`) 0.910/1.025.

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
5. **Extra degrees of freedom must earn their keep — and whether they can depends on the
   backend's structure.** In G-EDF (fixed map during the solve) one well-chosen DoF (Sim(3)
   log-scale) gives −34 %; in GTSAM (landmarks re-estimated jointly per two-frame problem) the
   same DoF is redundant — sim3 is neutral-to-worse on both frontends because the scale error is
   already absorbed into the landmarks. Nine DoF (SL(4)) overfit everywhere: they diverge in
   G-EDF (per-frame scale up to 8×), hurt DAv3+GTSAM outright (+41 %), and need
   `prior_weight ≥ 1000` on DAv2+GTSAM just to avoid catastrophe. Same theme in the sweeps:
   `num_gaussians` 8→32 helps, 64 adds nothing; `budget_cubes_per_frame` 8→16 helps DAv2,
   32 gives it back.
6. **Determinism + multi-seed validation is mandatory — now also for alignment conclusions.**
   Runs are bit-reproducible under a fixed seed (`MACVO.py --seed`), but seed-to-seed ATE swings
   are ±10 % for `gedf+icp`, up to ±40 % for pure G-EDF, and ~1 % for plain GTSAM. Three
   single-seed "wins" have now flipped sign on the second seed: DAv2 `sigma 0.10`, stereo
   `cov_trace_gate 0`, and DAv2+GTSAM `sl4 w1000` (0.208 → 0.243, vs se3 0.233 → 0.229).
   The alignment DoF itself inflates GTSAM's seed sensitivity to ±8 %. The G-EDF alignment
   benchmark remains single-seed; its DAv2 margin (−34 %) exceeds the ±10 % noise band but a
   seed confirmation would make it solid.
7. **Warm-up matters for online maps on short clips.** `min_gaussians` 400+ leaves the field
   factor inert for most of an 80-frame segment (pure G-EDF degenerates outright); keep ≤ 100.
   `cube_size` 1.0 is a hard sweet spot — halving or doubling degrades or degenerates.

## Current recommended settings

- **DAv2 mono, best accuracy:** `MACVO_MonoDAv2.yaml` (GTSAM pose2point, SE(3)) with the shipped
  tuned combo (`huber 0.05, hprev 3.0, atol 2.0, psigma 1e-3`) — 0.233/0.229 on the reference
  segment, seed-confirmed. **Do not enable `alignment:` on GTSAM** (sim3 neutral-to-worse, sl4
  seed-unstable at best).
- **DAv2 mono, G-EDF:** `MACVO_GEDF_DAv2.yaml` (`gedf+icp`) + `alignment: {type: sim3,
  prior_weight: 100}`, `autodiff: true`, `parallel: false` — close second (0.251), and the
  configuration of choice when the online distance-field map itself is wanted.
- **DAv3 mono:** keep optimizer defaults everywhere; prefer SE(3) (no alignment warp, in both
  backends); needs `device_depth: cuda:1` for G-EDF (single 24 GB card OOMs).
- **Pure G-EDF** (when running field-only): DAv2 `budget_cubes_per_frame 16`; DAv3
  `num_gaussians 32`; stereo Fast (`MACVO_GEDF_Fast.yaml`) defaults.

## Open questions

- ~~Head-to-head on one segment: tuned DAv2+GTSAM vs `gedf+icp`+Sim(3)~~ — **answered** by the
  GTSAM alignment sweep: tuned DAv2+GTSAM wins (0.233/0.229 vs 0.251, seed-confirmed).
- Seed-confirm the G-EDF alignment benchmark (single-seed today); ±10 % `gedf+icp` noise vs
  −34 % margin suggests it will hold — and the GTSAM sl4 flip is a fresh warning.
- Do the pure-G-EDF mapper optima (`budget 16`, `ngauss 32`) help the *fused* backend too?
  Untested under `gedf+icp`.
- Longer/other sequences: every number above is one 80–160-frame plane_nose segment; nothing here
  is a paper-grade multi-sequence average.
- DAv3+GTSAM runtime (~20 s/frame at `num_point 1000`): needs a native C++ pose→point factor or
  fewer keypoints before DAv3 sweeps get cheap.
