# GP depth prior (own-depth) hyperparameter sweep

**Date:** 2026-08-11 · **Setup:** DAv3 np300, plane_nose[80:160], seed 0 (+seed-1 controls)
· **Builds on:** `2026-08-05_gp-depth-prior-phase1.md`; evaluation lessons adopted from
`learningUAVO/gtsam_backend/FINDINGS.md` (similarity-aligned ATE alongside raw t_rel;
seed controls before believing small margins).

## Wave 1: ell × sigma_f (3×3, slide 10, scale prior 2.0)

| | ATE (sim3-aligned) | t_rel (m/fr) | prior share | align scale |
|---|---|---|---|---|
| ell20, sf .08/.15/.30 | .771/.715/.779 | .332/.316/.300 | ~.30 | .52–.56 |
| ell40, sf .08/.15/.30 | .740/.707/.754 | .331/.313/.298 | ~.31 | .52–.56 |
| ell100, sf .08/**.15**/.30 | .717/**.691**/.750 | .328/.311/.295 | ~.31 | .52–.57 |
| OFF | .728 | .367 | — | .44 |

`sigma_f` is the active axis and trades per-step accuracy against global shape exactly as
the UAVO ledger predicted: t_rel improves monotonically with sigma_f while aligned ATE
peaks at 0.15 and degrades at 0.30. `ell` mildly prefers long (100). Winner: **ell 100,
sigma_f 0.15** (adopted into `MACVO_DAv3_p2p_np300_gpown.yaml`).

## Wave 2: slide_rel × scale_prior at the winner, + the UAVO transplant

| cell | ATE | t_rel | align scale | s_curr med | share |
|---|---|---|---|---|---|
| winner (slide 10, scale 2) | .691 | .311 | .538 | −0.14 | .31 |
| scale 4 | .688 | .310 | .541 | −0.15 | .31 |
| slide 1 | .844 | .304 | .545 | −0.19 | .29 |
| slide 1 + scale 4 | .851 | .303 | .546 | −0.20 | .29 |
| **slide 3** | **1.489 (collapsed)** | .199* | 1.210 | **−2.24** | .18 |
| UAVO-tuned transplant (ell60, sf.25, sl1, sc4) | .718 | .312 | .512 | −0.13 | .26 |

- **slide_rel 10 confirmed; slide 1 hurts; slide 3 is pathological.** At slide 1 the
  point factors re-carry meaningful depth (partial return of the Phase-0 double count →
  ATE +22 %). At slide 3 the point factors' along-ray weight is *comparable* to the
  prior's and the two enter open competition: s runs away (median −2.24 ≈ claiming the
  depth field is 9× too large), the trajectory collapses to half GT length, and the
  seemingly great t_rel* is an artifact of tiny steps. The stable regimes are the two
  ends; the middle is a fight zone. Do not ship intermediate slide values.
- **scale_prior 2 vs 4: indifferent** (posterior σ(s) ≈ 0.03–0.05 — the prior hasn't
  been binding since ~1.0).
- **The UAVO optimum does not transfer** (0.718 vs 0.691): tuning is backend-specific
  (free landmarks + Huber + cross-frame factor0 vs their chain), as suspected.

## Seed controls (the honest headline)

| | seed 0 | seed 1 |
|---|---|---|
| winner ATE / t_rel / scale | .691 / .311 / .538 | .731 / .317 / .523 |
| OFF ATE / t_rel / scale | .728 / .367 / .439 | .729 / .367 / .438 |

Aligned-ATE seed spread on the prior arm is ±6 % — **the ATE gain vs OFF is within seed
noise**. What survives the controls, consistently across both seeds:

- **t_rel −15 %** (0.31 vs 0.37) — real per-step improvement;
- **absolute scale** substantially closer to truth (align-scale 0.52–0.54 vs 0.44, i.e.
  trajectory inflation 1.9× vs 2.3×) — real;
- aligned ATE — a wash, because similarity alignment forgives precisely the scale error
  the prior fixes (the forgiveness principle from the UAVO ledger, observed here in a
  second backend).

## Adopted + next

Adopted: `ell=100, sigma_f=0.15, slide_rel=10, scale_prior=2` (scale 4 equivalent).
Claim to carry forward: the prior buys per-step accuracy and metric scale, not
aligned-ATE shape on this sequence. Next levers, per the UAVO ledger, are structural
rather than parametric: a max-depth bias gate on keypoint selection and a
constant-velocity motion-prior augmentation.
