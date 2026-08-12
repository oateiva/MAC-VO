# Nugget semantics fix: the kernel is added to the network's variance (UAVO parity)

**Date:** 2026-08-12 · **Fixes:** `gp_prior_nugget` defaulted to `fixed`, and every DAv3
arm in the 08-11 campaigns ran that way — discarding DAv3's calibrated per-pixel
variance and replacing it with a flat sigma_n floor. UAVO's `depth_prior.py` has no such
mode: the Matern smooth part is always ADDED to the network's measured log-depth
variance, `nugget^2 = max(Var(d)/d^2, sigma_n^2)`, with the floor standing alone only
where the network reports nothing.

**Change:** `measured` is now the default (mechanics were already identical — same
formula, floor, and sentinel fallback); adopted DAv3 configs updated. `fixed` remains
as the explicit escape hatch for networks whose stored variance is fake and must not be
consumed (DAv2: `0.1·d` — positive, so not auto-detectable). Guard test:
`test_nugget_defaults_to_measured`.

## Measured vs fixed, adopted combo configs (ATE sim3 / t_rel, seeds 0/1)

| | fixed (08-11 numbers) | measured |
|---|---|---|
| plane_nose combo | 0.555 / 0.657 · 0.300 / 0.305 | 0.656 / 0.730 · **0.296 / 0.301** |
| Welland combo | 0.0566 / 0.0591 · 0.0798 / 0.0802 | **0.0547 / 0.0575** · 0.0797 / 0.0799 |

Reading: Welland improves across the board (small, both seeds). plane_nose loses ATE
(+18 %/+11 %, both seeds) and lands almost exactly on the gate-only arm (0.638) —
because DAv3's calibrated variance at plane_nose depths is LOOSER than the fixed 0.05
floor, so the honest nugget weakens the prior. The fixed-nugget ATE "win" was therefore
partly an artifact of over-trusting depth beyond its calibrated reliability — it paid on
a frontal plane whose depth shape happened to be good, i.e. exactly the kind of gain that
does not generalize (and Welland agrees).

## Follow-up

`sigma_f` (and the sigma_n floor) were tuned under the fixed nugget; the effective
sigma_f/nugget ratio has now shifted. A short re-sweep of `sigma_f` under measured
nugget on both datasets is the natural next step if plane_nose ATE matters; the 08-11
sweep table suggests trying sigma_f 0.2–0.3 first.
