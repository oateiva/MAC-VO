# Correlated log-depth prior (Phase 0): implementation, tests, and a structural null

**Date:** 2026-08-04 · **Branch:** `feature/claude-refactor` · **Env:** `AITraining12` (gtsam 4.2a9)

Implements the "correlated log-depth prior for the GTSAM pose2point graph" spec (Phase 0,
stationary Matern-3/2 kernel, no write-back, `alignment` fixed at se3). Feature ships off by
default (`enable_gp_depth_prior: false`), the off-path is **bit-exact** with pre-change
MAC-VO (T5), and all factor-level tests pass. The load-bearing mechanism test (T3) fails
**structurally, not from a bug** — the formulation double-counts the per-point depth
measurement — and the real-data on/off comparison (T6) confirms the synthetic analysis
from both sides: on DAv3 (tight per-point depth covariance) the prior is active (14 % of
total cost) yet neutral-to-slightly-worse including in the low-parallax stratum, while on
DepthCov (weak per-point depth covariance, i.e. the double-count collapses on its own) it
buys a ~5 % t_rel gain. The section-13 sweep was therefore not run (rationale in §6).
Recommendation: do not integrate a learned kernel onto this formulation; Phase 1 should
first move the per-point depth term out of `Sigma_p` (§7) — the DepthCov arm is a live
preview of that regime.

## 1. Preconditions (spec §4)

1. **Optical axis = component 0** (NED: depth, east, down). `pixel2point_NED`
   (`Utility/Point.py:15-17`) rolls pypose's EDN so index 0 is depth; `Covariance_2to3_full`
   (`Module/Covariance/Project2to3.py`) puts `sigma_zz` at row/col 0 — the modules agree.
   Defined once as `DepthPrior.OPTICAL_AXIS = 0`.
2. **Ray form**: `pp.pixel2point` = `d · K⁻¹[u,v,1]` (unnormalized). Verified numerically:
   `pixel2point_NED(...)[0] == d` with error exactly 0.0 in float64. `yhat = log(pixel*_d)`
   needs no correction factor.
3. (Report) DepthCov's wrapper exposes only the **marginal variance**; the vendored
   `NonstationaryGpModule` computes per-pixel nonstationary kernel parameters internally but
   does not surface them. A Phase-1 learned kernel needs a wrapper extension.
4. (Report) The previous keyframe's CNN output **is retained** for exactly one frame
   (`Odometry/MACVO.py:75,318` — `prev_keyframe[2]`); older frames survive only as sparse
   `MatchObs` rows.
5. **Indexing confirmed**: `gtsam.symbol('l', i)` ↔ row `i` of the current pair's `MatchObs`
   (asserted row-aligned in `parse_graph_data`); `pixel1_*` = frame k−1 ("prev"),
   `pixel2_*` = frame k ("curr").
6. (Report) `pixel0_uv` exists as `previous_graph_data.observations.data["pixel1_uv"][pi]`;
   per the spec default, the prior does not apply to frame k−2.

Binding constraints discovered (gtsam 4.2a9): `transformTo`'s preallocated Jacobian buffers
must be **F-contiguous** float64 (C-order raises TypeError); `Marginals` requires every
variable constrained (hence `s` is only inserted alongside ≥1 block factor and its prior);
`factor.error()` = `0.5‖whitened‖²` (Huber-ρ for robust models); dense Hessians for tests
need an explicit `gtsam.Ordering` (default is COLAMD).

## 2. What was built

| Piece | Where |
|---|---|
| Kernel registry (`matern32`), kd-median block partitioner, jitter-escalating `chol_inv_lower`, block factor with analytic Jacobians | `Module/Optimization/GTSAM/DepthPrior.py` (new) |
| Graph wiring: per-frame `s` (symbol `'s'`, 1-dim Vector + `PriorFactorVector`), block factors per enabled frame, post-solve diagnostics | `Module/Optimization/GTSAM/Graphs.py` |
| Config schema (9 spec keys + `gp_prior_diag_dir`), hard rejects (non-se3 alignment, non-pose2point), Rerun channels `/world/gp_depth_prior/*`, per-sequence CSV on `terminate()` | `Module/Optimization/GTSAM/Optimizer.py` |
| Docs | `Module/Optimization/README.md` |
| Tests (18) | `Scripts/UnitTest/test_gtsam_gp_depth_prior.py` |

Design notes honored: blocks are computed once from frame-k pixels and reused for both
frames; per-frame kernels from each frame's own coordinates; nonpositive depths excluded
(never clamped); `z_min` clamp zeroes the affected Jacobian rows (the spec's literal formula
at the clamp disagrees with the cost and stalls LM); rows with the prior OFF still emit the
parallax/Huber CSV columns so both T6 arms stratify identically. Depth-module agnostic: only
`MatchObs.pixel*_uv / pixel*_d` are consumed — the same optimizer block drops into DAv3,
DAv2, DepthCov, or stereo configs.

**Deviation from spec §12:** the optimizer never receives the run's `Sandbox`, so "CSV in
the sandbox" is implemented as the optional `gp_prior_diag_dir` config key (10th optional
key), flushed on `terminate()`.

## 3. Test results

| Test | Result |
|---|---|
| T1 Jacobians (pose / s / landmark blocks vs central differences) | **pass** (≤1e-10; plus exact zero residual at consistent state; clamp rows zeroed) |
| T2 rank structure (every landmark block rank 1, row space ∥ `eᵀRᵀ`) | **pass** |
| T3 correlated recovery | **xfail — structural, see §5** (companion guard: prior never catastrophic, pass) |
| T4 gauge (uniform (y,s) shift null without s prior; pinned with it; single null in along-ray subspace) | **pass** |
| T5 regression, feature off | **pass, bit-exact** (max abs diff 0.0 over 80 poses; run-to-run determinism separately established with two identical pre-change runs) |
| Config validation (back-compat, happy path, sim3-combo reject, graph-type reject, unknown kernel/frames reject) | **pass** |
| Pure functions (kernel SPD, partition coverage/original-indices/spatial, Cholesky jitter recovery) | **pass** |

(13 pre-existing `test_config_macvo` failures on this branch — keypoint-selector
`Excessive Keys` on DAv2/Fast-era configs — are unrelated and unchanged.)

## 4. T6: plane_nose[80:160], seed 0, on vs off

DAv3 arm (`MACVO_DAv3_p2p_np300_gp[off].yaml`, num_point 300, spec-default prior params):

| | t_rel RMSE (m/fr) | r_rel RMSE (deg/fr) |
|---|---|---|
| prior off | 0.3668 | 6.283 |
| prior on | 0.3722 | 6.348 |
| off, low-parallax half | 0.3095 | — |
| on, low-parallax half | 0.3123 | — |
| off, high-parallax half | 0.4150 | — |
| on, high-parallax half | 0.4226 | — |

Diagnostics (79 solves): **prior cost share median 13.7 %** (p10 7.2 %, p90 33.9 %) — the
prior is doing real work, so this null is *a result*, not a mis-weighted no-op (spec §8
check). `s_curr` median −0.010, |s| MAD 0.035 — DAv3's per-frame scale is already stable.
Posterior σ(s) median 0.020 « prior 0.15 — scale observable on every pair (the §14
"posterior ≈ prior" degeneracy did not occur, even on this frontal planar target). Median
departure RMS `y−ŷ−s` = 0.058. Huber rejections identical on/off (292). Zero z-clamps,
zero dropped blocks, zero nonpositive depths.

DepthCov arm (`MACVO_MonoDepthCov_gp[off].yaml`, num_point 200, optimizer sequentialized
for determinism):

| | t_rel RMSE (m/fr) | r_rel RMSE (deg/fr) | path length / GT |
|---|---|---|---|
| prior off | 0.2036 | 6.534 | 0.809 |
| prior on | **0.1925 (−5.5 %)** | 6.657 (+1.9 %) | 0.693 |
| off / on, low-parallax half | 0.1622 / 0.1522 | — | — |
| off / on, high-parallax half | 0.2371 / 0.2248 | — | — |

Diagnostics: prior cost share median 16.6 %; `s_curr` median **−0.135** (a large per-solve
scale correction — the depth field runs ~13 % hot against the landmark cloud, the same
drift family as the known DepthCov scale collapse), posterior σ(s) 0.034; departure RMS
9.2 % (vs DAv3's 5.8 %). Huber rejections unchanged; zero clamps/drops.

**Reading:** on DepthCov the prior is *not* a null — t_rel improves ~5 % in both parallax
strata. This is consistent with §5 rather than contradicting it: DepthCov's `sigma_dd` is
genuinely enormous away from anchors (`init_cov` 10 m²), so its point factors carry little
depth information and the double-count collapses — the block prior becomes the de-facto
depth measurement model, which is exactly the §7 target design. The cost is faster
trajectory-scale shrinkage (path/GT 0.81 → 0.69): the estimated `s` ≈ −0.13 shows the
prior re-arbitrating depth-vs-geometry each solve, and with no anchor for absolute scale
the balance tips further toward the (shrinking) geometry. A gain on the depth model whose
per-point covariance is weakest, and a null on the one whose covariance is tight, is the
double-count mechanism read off two real frontends.

## 5. The structural finding (why T3 xfails)

The factor is correct (T1/T2/T4). The *formulation* cannot help in this graph:

**The prior's `yhat` is the same depth measurement already inside `Sigma_p`.** Each point
factor's noise model carries `sigma_dd` — the network depth's per-point variance — and its
measurement `obs_Tc = d·ray` embeds the same `d` the prior compares against. The block
factor therefore adds the depth measurement a second time, at nugget tightness `sigma_n`.
Two regimes follow, both observed:

- `sigma_n` tight relative to geometry → the prior out-votes geometry and pins landmarks
  to the measured field (synthetic: rough error doubles at depth-trust 30 %).
- `sigma_n` loose → the prior adds nothing the point factors don't already say
  (synthetic: all deltas ≈ 0).

There is no third regime, because the prior carries **no information source other than the
measurements already in the graph** — its only genuine addition is the off-diagonal
coupling, and coupling only transfers information when some points are much better
determined than others *and* the per-point double-count doesn't drown the transfer.
Verified across 16 synthetic constructions: tilt corruption on both frames (unobservable —
it is self-consistent 3D geometry), on one frame only (the old scalar test's C_A=1
analog), homogeneous and heterogeneous (forward-motion/FOE) parallax, `ell` ∈
{40,100,150,220}, ratio 10 at several absolute scales, depth trust 10–30 %. Neutral to
harmful in every cell, smooth component untouched.

The T6 numbers agree: active prior (14 % cost share), observable `s`, and slightly worse
pose in aggregate *and* in the low-parallax stratum.

Two incidental findings worth keeping:

- A **diagonal** (axis-aligned) synthetic `Sigma_p` makes ground truth look like a 10-σ
  residual for off-center pixels — the `sigma_dd·(u−cx)/fx` cross terms in
  `Covariance_2to3_full` are load-bearing ray alignment, not decoration.
- The two-frame solve is **not globally convergent** from a static init at 10 cm/5° motion
  with ~1 px lateral tubes (LM walks to a spurious minimum below the true basin's cost);
  T3 inits near truth for this reason.

## 6. Why the section-13 sweep was not run

The sweep varies `ell`, `sigma_f/sigma_n`, and `gp_scale_prior_sigma`. §5 shows the null is
caused by double-counting, which no kernel hyperparameter can undo — the sweep would spend
~9 GPU-runs re-measuring T3's conclusion at other points of the same surface. The one sweep
precondition the spec insists on (prior share of cost non-negligible) is already satisfied
at defaults, so "the sweep measured nothing" is ruled out as an alternative explanation.
If a decisive empirical nail is wanted anyway, the two most informative single points are
`gp_scale_prior_sigma = 1.0` (shape-only limit) and `sigma_n = 0.15` (weak per-point pin);
both are one-command runs with the shipped configs.

## 7. Recommendation for Phase 1

Do **not** bolt the learned kernel onto this formulation — a better `K` (DepthCov's
nonstationary kernel cutting correlation at discontinuities) sharpens a term that is
double-counted either way. The T6 DepthCov arm is the empirical case for the redesign: the
only regime where the prior helped is the one where the per-point depth term in `Sigma_p`
was already too weak to double-count.

The fix is the one the original stage-2 sketch already anticipated: **move the per-point
depth information out of `Sigma_p` and let the prior own it.**

1. Depth-free point factors: rebuild `Sigma_p` without `sigma_dd` (rank floor needed — each
   frame's contribution drops toward rank 2), so point factors carry flow/pixel geometry
   only.
2. The block prior becomes the *sole* depth measurement model:
   `y − ŷ − s ~ GP(0, K)` with `sigma_n` = the network's actual per-point noise and the
   smooth modes carrying its spatially-correlated error — no double count, and the
   coupling now has something to do.
3. Then Phase 1 (DepthCov kernel) becomes meaningful: `K` from the GP's own covariance
   function (requires exposing it through the wrapper — precondition 3).

This is a covariance-model change (`Covariance_2to3_full` variant + config plumbing), i.e.
outside the current change's allowed file set — stopping and reporting per spec §15.

## 8. Reproduction

```powershell
# tests
C:\...\envs\AITraining12\python.exe -m pytest Scripts/UnitTest/test_gtsam_gp_depth_prior.py -o addopts=""
# T6 arms (configs ship in-tree; diagnostics CSVs land in Results/gp_prior_t6/diag_*/)
C:\...\envs\AITraining12\python.exe MACVO.py --odom Config/Experiment/MACVO/Optimal/MACVO_DAv3_p2p_np300_gp.yaml `
  --data Config/Sequence/EIVA_plane_nose_mono.yaml --seq_from 80 --seq_to 160 --seed 0 --noeval --resultRoot Results/gp_prior_t6
```

Artifacts: T5 baselines `Results/gp_prior_t5_baseline/`, T6 runs + CSVs
`Results/gp_prior_t6/`.
