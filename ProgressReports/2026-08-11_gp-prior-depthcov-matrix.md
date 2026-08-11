# DepthCov arm: completing the GP-prior ablation matrix

**Date:** 2026-08-11 · **Setup:** DepthCov np200, plane_nose[80:160], seeds 0+1 ·
**Completes:** the frontend column started in `2026-08-05_gp-depth-prior-phase1.md`;
structural levers from `2026-08-11_gp-prior-structural-levers.md`.

| arm (seed 0 / seed 1) | ATE (sim3) | t_rel | path/GT |
|---|---|---|---|
| off | 1.533 / 1.363 | 0.204 / 0.206 | 0.809 / 0.864 |
| P1 own-depth (scale prior .15) | 1.485 | 0.189 | 0.662 |
| base2 = adopted DAv3 tuning (scale prior 2.0) | 1.324 | 0.166 | **0.481** |
| combo (gate 2.7 + mp) | **collapsed** (path 0.000, both seeds) | — | — |
| combo2 (mp, no gate, scale .15) | 1.535 / 1.389 | 0.190 / 0.190 | 0.675 / 0.754 |

## Findings

1. **The DAv3 combo recipe does not transfer.** The verdict arm (combo2) is
   indistinguishable from Phase-1 own-depth on every metric; the motion prior adds
   nothing here beyond the t_rel −7 % the prior already gave. DepthCov's ATE seed spread
   (±0.09–0.17 — an order larger than DAv3's) swallows all aligned-shape differences.
2. **The gate is a hard hazard on map-anchored frontends: a bootstrap interlock.** Gate
   threshold 2.7 < DepthCov's `init_depth` 3.0 fallback ⇒ with an empty map every
   candidate carries the fallback depth, the gate rejects all of them, no landmarks are
   created, so the map never grows anchors and the depth never leaves the fallback —
   a deterministic deadlock (pose spread exactly 0.0, need_interp 75/80, both seeds).
   The gate is also pointless here even above the interlock threshold: DepthCov's
   anchor-driven selection (`depth_cov_rel`) already compresses the candidate
   distribution (p99/median 1.8 vs 4.0 on DAv3-Welland) — nothing to cut.
3. **Loosening the scale prior — free on DAv3 — is a shrinkage amplifier on DepthCov**
   (path/GT 0.662 at σ_s=0.15 → 0.481 at 2.0): the tight `s` prior was load-bearing
   resistance against the map→anchor feedback loop. base2's "best ATE" (1.324) is the
   forgiveness artifact in its purest form: the fastest-shrinking trajectory scores the
   best similarity-aligned shape.
4. **What actually binds DepthCov is unchanged**: the anchor-feedback scale loop
   ([[depthcov-scale-collapse]]). Every parametric lever tested either feeds it or is
   neutral to it — consistent with the UAVO forgiveness principle that only an
   in-estimator persistent scale state can address it.

## Adopted per-frontend recipes (final)

| | DAv3 | DepthCov |
|---|---|---|
| prior (own-depth) | ✓ ell100 / σf .15 / slide 10 | ✓ ell100 / σf .15 / slide 10, **nugget measured** |
| scale prior σ_s | 2.0 (loose; free) | **0.15 (tight; load-bearing)** |
| far-range gate | ✓ re-derive ~p75–80 per dataset | **✗ never** (interlock + nothing to cut) |
| cv motion prior | ✓ 0.0225 (synergizes with gate) | ✗ (neutral) |
| net vs off | ATE −24/−10 % (plane_nose), −26/−23 % (Welland) | t_rel −7 %; ATE within seed noise; scale loop open |

`MACVO_MonoDepthCov_gpown.yaml` stays the recommended DepthCov config (unchanged).
