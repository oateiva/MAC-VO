# Correlated log-depth prior, Phase 1: the prior owns the depth measurement

**Date:** 2026-08-05 · **Branch:** `feature/claude-refactor` · **Env:** `AITraining12`
**Builds on:** `ProgressReports/2026-08-04_gp-depth-prior-phase0.md` (Phase-0 structural
null: the prior's `yhat` duplicated the `sigma_dd` already in `Sigma_p`).

Phase 1 removes the double count: with `gp_own_depth: true`, the
`CorrelatedDepthPrior` augmentation replaces the point factors' covariances with a
**depth-free** construction and becomes the sole per-point depth measurement model.
Result: the mechanism that was structurally dead in Phase 0 now works — the synthetic
smooth-error component halves, and both real frontends improve on t_rel (DAv3 −11.7 %,
DepthCov −7.4 % vs prior-off), with the prior carrying a healthy 23–33 % of total cost.

## 1. What changed (all inside the augmentation layer; `Graphs.py` untouched)

- **Depth-free point covariance** (`DepthPrior.depth_free_covariance`):
  `Sigma_df = L + (slide_rel·d)^2 · r rᵀ` — pixel-noise lateral block plus a large
  ray-aligned slide. NOT zeroed `sigma_dd` (the observation `d·ray` embeds the measured
  depth; zero along-ray variance would pin it as exact), and NOT a huge `sigma_dd` inside
  `Covariance_2to3_full` (its second-order `sigma_uu·sigma_dd` term blows up the lateral
  directions). The slide term doubles as the rank floor. Applied in `on_parse` to the
  graph's numpy copies only — map, frontend, and the k−2 re-observation factor are
  untouched (k−2 keeps `sigma_dd`; the prior doesn't cover that frame).
- **Heteroscedastic nugget** (`gp_prior_nugget: measured`): per-point
  `Var(log d) = pixel*_d_cov / d²` (for DepthCov literally its GP's log-depth variance),
  floored at `sigma_n²`; `-1` sentinels and missing fields fall back to the floor.
  `matern32_kernel` accepts scalar or per-point nugget (scalar path bit-identical).
- Config: `gp_own_depth` (default false), `gp_slide_sigma_rel` (10.0),
  `gp_prior_nugget` (`fixed`). Rejected: `own_depth` without the prior, or with
  `gp_prior_frames != [prev, curr]` (a depth-freed frame without prior coverage would
  lose its depth measurement outright).

## 2. Mechanism test (the one Phase 0 could not pass)

Same T3 synthetic (planar scene, smooth tilt on frame B + 2 % iid noise), landmark
log-depth error:

| formulation | RMS | smooth (tilt) | rough |
|---|---|---|---|
| Phase 0, prior off | 0.0146 | 0.0524 | 0.1044 |
| Phase 0, prior on | 0.0152 | 0.0524 | 0.1095 |
| **Phase 1, own-depth** | **0.0150** | **0.0268** | 0.1166 |

The smooth component — the spatially-correlated error a depth network actually makes —
**halves**; Phase 0 could not move it at all. The rough component pays a small price
(the prior's nugget now carries the per-point measurement), netting ~even on this scene
where the synthetic geometry is near-perfect. Asserted in
`test_phase1_mechanism_own_depth_beats_phase0`.

## 3. Real-data results (plane_nose[80:160], seed 0)

| arm | t_rel RMSE (m/fr) | r_rel (deg/fr) | prior share | path/GT |
|---|---|---|---|---|
| DAv3 off | 0.3668 | 6.283 | — | 2.138 |
| DAv3 Phase-0 on | 0.3722 | 6.348 | 13.7 % | 2.183 |
| **DAv3 Phase-1 own** | **0.3240 (−11.7 %)** | **6.234** | 32.7 % | **1.847** |
| DepthCov off | 0.2036 | 6.534 | — | 0.809 |
| DepthCov Phase-0 on | 0.1925 | 6.657 | 16.6 % | 0.693 |
| **DepthCov Phase-1 own** | **0.1885 (−7.4 %)** | 6.548 | 23.1 % | 0.662 |

Stratified t_rel (low/high parallax): DAv3 0.310/0.415 (off) → **0.284/0.359** (own);
DepthCov 0.162/0.237 (off) → **0.152/0.218** (own) — improvement in BOTH strata, not
just the low-parallax tail.

Diagnostics: prior cost share rose from ~14 % (Phase 0) to 23–33 % — the prior now
carries the depth information, as designed. `s_curr` becomes strongly active: median
−0.12 (DAv3) and −0.27 (DepthCov) per solve — with geometry freed from the depth
measurement, `s` cleanly measures how hot each network's depth field runs against the
rigid structure. On DAv3 this pulled the trajectory scale TOWARD ground truth
(path/GT 2.138 → 1.847; the `f/300` metric heuristic runs ~2× hot on EIVA optics).
`z_min` clamps appeared (352 and 416 over 79 solves, ~1.5 % of point-visits) —
landmarks wander more during LM iterations with free along-ray directions; benign at
this rate but worth watching. Huber rejections dropped (292→245 DAv3), consistent with
the kernel now gating on flow consistency only.

## 4. Caveats

- **DepthCov trajectory-scale shrinkage worsened** (path/GT 0.809 → 0.662): the prior
  improves relative motion, but the map→`depth_prior` feedback loop (anchors from an
  ever-shrinking map) still contracts absolute scale, and freeing the point factors
  removes some resistance. This is the pre-existing structural collapse mode
  ([[depthcov-scale-collapse]]), now cleanly measured by `s` (−0.27/solve) rather than
  fixed. Any fix belongs to the anchor path, not this prior.
- `plane_nose` is a frontal planar target — good for exercising the mechanism, weak for
  generalization (same caveat as Phase 0). Next data point: a sequence with depth
  discontinuities, where the stationary Matern kernel should start to hurt and the
  learned (DepthCov nonstationary) kernel becomes the Phase-2 candidate.
- The k−2 re-observation factor still carries `sigma_dd` (partial double count on the
  re-observed subset) — documented, not fixed.

## 5. Gates

- Phase-0 path unchanged: GP-on (`own_depth` off) run bit-exact vs the 08-05 gate run
  (max abs diff 0.0). Feature-off path untouched by construction (no code outside the
  augmentation's `if`).
- Full suite: 68 passed + T3 xfail (unchanged, still documents the Phase-0 regime).

## 6. Recommendation

Phase 1 validates the redesign: with the double count removed, the correlated prior is
a net win on both frontends. Worth pursuing next, in order: (a) a `gp_own_depth` sweep
of `sigma_f`/`ell` now that the prior share is load-bearing; (b) a discontinuity-rich
sequence to bound the stationary kernel; (c) Phase 2 — the DepthCov nonstationary
kernel as a `KERNEL_BUILDERS` entry (requires exposing the kernel through the DepthCov
wrapper, see Phase-0 report precondition 3).
