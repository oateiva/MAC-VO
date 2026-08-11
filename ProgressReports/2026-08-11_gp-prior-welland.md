# Cross-dataset test: the GP-prior stack on Welland

**Date:** 2026-08-11 · **Setup:** DAv3 np300, `Welland_test_first_dive[750:850]` (steadiest
100-frame window by GT motion profile; 4.4 cm/frame), seeds 0+1 · **Follows:**
`2026-08-11_gp-prior-structural-levers.md`. New data config:
`Config/Sequence/EIVA_welland_mono.yaml`.

Welland is the first structurally different scene for the stack: canal survey, nearer
and more structured than plane_nose (candidate depths median 2.8 / p90 5.2 / p99 11.2 VO
units — a fat biased far tail), different camera calibration (different DAv3 metric
factor).

## Results (ATE sim3-aligned / t_rel, seed 0; seed 1 where run)

| arm | ATE | t_rel | align scale |
|---|---|---|---|
| OFF | 0.0767 (s1: 0.0771) | 0.0817 (s1: 0.0816) | 0.572 |
| prior only (adopted plane_nose tuning, no gate/mp) | **0.1486 (+94 %)** | 0.0754 | 0.641 |
| prior only, ell 20 | 0.1662 | 0.0755 | 0.644 |
| **combo: prior + gate 3.5 + mp 0.0225** | **0.0566 (s1: 0.0591)** | 0.0798 | 0.590 |
| combo with ell 20 | 0.0580 | 0.0791 | 0.595 |

## Findings

1. **The stack generalizes — with its protocol, not its constants.** Combo beats OFF by
   −26 % / −23 % (seeds 0/1), consistent with plane_nose's −24 % / −10 %. The gate
   threshold had to be re-derived from the local candidate-depth distribution (3.5 ≈ p78
   here vs 6.0 on plane_nose); the derivation, not the number, is what transfers.
2. **Ungated, the prior is actively harmful on far-tail-heavy scenes (+94 % ATE).** The
   prior's coupling is the mechanism: it propagates the biased far-tail depths into the
   clean near cloud — precisely the "slide together" behaviour, applied to points that
   should have been excluded. On plane_nose (milder tail) this surfaced as a modest gate
   improvement; on Welland it's a prerequisite. Promoted to a hard usage note in
   `Module/Optimization/README.md`: **do not enable the prior without a far-range gate on
   scenes with unbounded range.**
3. **No kernel-scale sensitivity** (ell 20 ≈ ell 100 everywhere): the stationary kernel's
   correlation reach is not the binding constraint on this window, so this test provides
   NO go-signal for Phase 2 (the learned nonstationary kernel). A genuinely
   discontinuity-dominated window (occlusion boundaries at close range) is still needed to
   probe that; this one's damage was tail bias, not edge smoothing.
4. Per-step t_rel is near the GT noise floor on this easy window (all arms within 8 %),
   so ATE carries the verdict here.

## Caveats

- One 100-frame window of one dive; the [750:850] window was selected for steady motion,
  not for difficulty. The full-dive run (4719 frames) is the natural next escalation.
- The gate threshold protocol is currently manual (inspect the candidate-depth
  distribution, cut ~p75–p80). If this stack goes beyond experiment configs, an
  auto-derived robust threshold belongs in the selector (the naive median-relative
  variant is NOT it — it lost on both datasets due to per-frame threshold jitter).
