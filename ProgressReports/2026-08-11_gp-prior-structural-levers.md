# Structural levers: far-range bias gate + constant-velocity motion prior

**Date:** 2026-08-11 · **Setup:** DAv3 np300 gpown (ell100/sf0.15), plane_nose[80:160],
seeds 0+1 · **Follows:** `2026-08-11_gp-prior-sweep.md`; both levers imported from
`learningUAVO/gtsam_backend/FINDINGS.md` ("bias can only be gated, not down-weighted";
soft cv prior −35 % APE there).

## What was built

- **Far-range gate** in `CovAwareSelector_NoDepth` (`Module/KeypointSelector.py`):
  optional `max_depth` (absolute, VO units) and `max_depth_rel` (× median candidate
  depth), same hasattr-gated pattern as `depth_cov_rel`. Rationale in-code: beyond some
  range this footage's depth AND flow are systematically biased (open water) — bias
  cannot be down-weighted by covariance, only gated.
- **`MotionPrior` augmentation** (`Module/Optimization/GTSAM/Augmentations.py`): one soft
  `BetweenFactorPose3` pose_1→pose_2, measurement = the previous pair's optimized
  relative motion, σ_rot = 2·σ_t, skipped on the first pair; enabled by
  `motion_prior_sigma`. Deliberately a soft factor, not an extrapolated init (UAVO:
  coasting is 4–8× worse than freezing).

## Individual arms (seed 0)

| arm | ATE (sim3) | t_rel | align scale |
|---|---|---|---|
| base (gpown winner) | 0.691 | 0.311 | 0.538 |
| gate abs 6.0 | **0.638** | 0.322 | 0.515 |
| gate abs 8.0 | 0.716 | 0.326 | 0.508 |
| gate rel 1.5 | 0.735 | 0.312 | 0.540 |
| mp 0.0225 | 0.690 | **0.303** | 0.541 |
| mp 0.05 / 0.1 | 0.692 | 0.309 / 0.310 | 0.536 |

Thresholds were set from the measured candidate-depth distribution (median 4.6, p90 7.9,
p99 11.5 VO-units): abs 6 trims the far ~25 %. The UAVO absolute value (3.5) sits below
our median — thresholds do not transfer across scale conventions, the *principle* does.
The median-relative variant loses: a per-frame moving threshold adds selection jitter.
The motion prior ALONE is neutral on ATE (its −35 % did not transfer; MAC-VO's graph
already carries inter-pair information through the k−2 re-observation factors and
shared-landmark coupling that their bare chain lacked).

## The combo, seed-controlled (the adopted result)

| | seed 0 | seed 1 |
|---|---|---|
| OFF (no prior) | 0.728 / 0.367 | 0.729 / 0.367 |
| base gpown | 0.691 / 0.311 | 0.731 / 0.317 |
| gate abs6 | 0.638 / 0.322 | 0.708 / 0.315 |
| **combo abs6 + mp 0.0225** | **0.555 / 0.300** | **0.657 / 0.305** |

(cells: ATE / t_rel)

The ordering combo < gate < base holds at BOTH seeds — unlike the prior-only ATE gain,
this survives the noise floor. Vs the prior-off baseline: **ATE −24 % / −10 %, t_rel
−18 % / −17 %**, and the best trajectory scale of the campaign (align-scale 0.58 vs 0.44
off). The two levers synergize: gating removes the biased far points but thins each
pair's geometry; the cv factor supplies exactly the stabilization the thinner geometry
lost — separately ~neutral (mp) or modest (gate), together the largest improvement since
Phase 1 itself.

Adopted into `MACVO_DAv3_p2p_np300_gpown.yaml`: `max_depth: 6.0` (keypoint block),
`motion_prior_sigma: 0.0225` (optimizer block).

## Caveats

- abs-6.0 is tuned to plane_nose's depth distribution under DAv3's ~1.9× scale
  inflation; other sequences/frontends need their own threshold (or a robust re-derivation
  from the candidate distribution — the rel variant as implemented is NOT it).
- Single sequence, two seeds. The next data point should be a second sequence before
  promoting these into non-experiment configs.
- All numbers vs a reference with a measured noise floor (~2.5 cm RMS, ~9°/pair
  orientation — see the UAVO ledger); aligned-ATE deltas of this size are meaningful,
  the absolute values are not.
