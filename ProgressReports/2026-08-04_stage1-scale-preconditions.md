# Stage 1 (per-frame log-depth scale variables): precondition report — STOPPED

**Date:** 2026-08-04
**Branch:** `feature/claude-refactor`
**Status:** Implementation **not started**, per the spec's own instruction in section 1:
*"If any is false, stop and report rather than adapting the spec silently, because the
derivations below depend on them."* Preconditions **2, 3, and 4 are false**. Precondition 3
is the one the spec singles out as design-breaking: the 3D points are free variables in the
graph, not fixed measurements.

This document is deliverable 3 of section 11, minus the T5 metrics table and the `s`-plot
(both require an implementation). It answers all seven section-1 preconditions, explains why
the design must change, and records the hazards and decisions the revised spec should absorb.

---

## 1. Section-1 precondition answers

### 1.1 Optimizer is GTSAM — TRUE (Python bindings)

gtsam **4.2a9**, Python bindings, imported guardedly in `Module/Optimization/__init__.py`
(absent from `requirements.txt`; present only in the `AITraining12` conda env,
`C:\Users\oat\AppData\Local\anaconda3\envs\AITraining12\python.exe`). The two-frame
point-to-point backend is `GTSAM_Pose2Point` (`Module/Optimization/GTSAM/Graphs.py:91`),
selected in YAML by `optimizer.type: GTSAM_Graph` + `args.graph_type: pose2point`. Solver is
batch Levenberg–Marquardt, `max_iterations` default 20, rebuilt from scratch every frame
(`Graphs.py:322-328` region). A separate PyPose backend (`TwoFrame_PGO`) exists but is not
the target of the shipped mono configs.

### 1.2 Relative pose is a single Pose3 — FALSE

The graph holds **absolute** `Pose3` variables, not a relative one
(`Graphs.py:180-214`):

- `pose_1` = frame k−1: inserted and immediately pinned by a `PriorFactorPose3` with
  σ = `prior_sigma` = 1e-4 on all six DoF — effectively fixed.
- `pose_2` = frame k: free. This is "the motion".
- `pose_0` = frame k−2: inserted and pinned the same way, when cross-frame re-observations
  exist.

The relative pose the spec reasons about is implicit in `pose_1⁻¹ · pose_2`. A caveat for any
redesign: `pose_1` is initialized *and priored* at `init_motion` (`Graphs.py:189-199`), which
equals the previous frame's pose only under `StaticMotionModel` — the anchor is wrong for any
predictive motion model.

### 1.3 3D points are fixed measurements, not variables — FALSE ← stop condition

One **free `Point3` landmark variable per current-frame observation**
(`gtsam.symbol('l', i)`, `Graphs.py:238`), initialized at `P1.transformFrom(obs_Tc_1_i)`
(`Graphs.py:295-296`). The camera-frame back-projections `obs_Tc_0/1/2` (computed from stored
`pixel*_uv` + `pixel*_d` via `pixel2point_NED`, `Graphs.py:138-154`) are the *measurements*
captured in each factor's closure — but they constrain free landmarks, they are not
differenced against each other.

The spec's own instruction for this case applies verbatim: the per-frame scale is partially
redundant with free point positions, and the design must change. Section 2 below details the
redundancy, which is not hypothetical — it has already been measured in this codebase.

### 1.4 Residual is `p_prev − T·p_curr` with combined Σ per point — FALSE

The residual is **`pose.transformTo(landmark_w) − obs_Tc`** per (pose, landmark) pair,
implemented as `gtsam.CustomFactor` with GTSAM-provided analytic Jacobians
(`Utility/GTSAM_Utils.py:22-43`). Each landmark gets 2–3 such factors: an un-warped one on
`pose_1`, one on `pose_2`, and optionally one on `pose_0` for re-observed points
(`Graphs.py:241-292`).

Consequently there is **no combined `Σ_A + R_AB Σ_B R_ABᵀ` noise model**: each factor
carries its *single* observation's 3×3 covariance
(`gtsam.noiseModel.Gaussian.Covariance(cov_Tc_k_i)`, `Graphs.py:263-280`). The
Σ-combination the spec describes exists in this repo only in the PyPose ICP backend
(`Module/Optimization/TwoFramePGO/Graphs.py`, `covariance_array`). The spec's section-4
"dependence on rotation" paragraph therefore has no counterpart in the GTSAM path.

### 1.5 Robust kernel — YES: Huber, two thresholds

Every point factor's noise model is wrapped in
`noiseModel.Robust.Create(mEstimator.Huber.Create(δ), ·)`:

| Factor | Threshold | Default |
|---|---|---|
| `pose_1` / `pose_2` ↔ landmark | `huber_delta` | 0.1 |
| `pose_0` ↔ landmark (re-observation) | `huber_delta_prev` | 1.0 |

(`Graphs.py:254-280`; config keys in `Module/Optimization/GTSAM/Optimizer.py:173-179`.)
Section 6 of the spec (kernel vs. scale competition) is therefore live and must survive into
the revision.

### 1.6 σ_d source in the monocular fork — network confidence head, metric units

For the default mono model (Depth Anything V3,
`Module/Network/Depth/DepthAnythingV3/api.py:401-433`):

```
var_raw     = 1 / (depth_conf + 1e-8) · q90 · q_calibration      (dimensionless, raw units)
d_metric    = d_raw · s,   var_metric = var_raw · s²,   s = f_mean / 300
```

**Metric: yes, by construction** — one factor `s` drives both depth and variance, an
invariant documented in-source (`api.py:413-420`) and pinned by
`Scripts/UnitTest/test_depth_metric_units.py`. It flows per-keypoint into
`Covariance_2to3_full` as `sigma_zz` (`Module/Covariance/Project2to3.py:413-425`) and is
stored per observation as `MatchObs.pixel*_d_cov` alongside the full 3×3 `obs*_covTc`.

Caveats the revised spec should note:

- **DAv2**: its "covariance" is `0.1 · depth` — not a variance, not m², flagged TODO in
  source. Any scale experiment validated on the DAv2 config measures noise.
- **DepthCov** (new): emits `Var(d) = d² · Var(log d)` (delta method, metric) — already the
  exact Jacobian of the log-depth parameterization; but see hazard 3.2.

Nothing was changed, per the spec.

### 1.7 Regression command and output location (for eventual T4/T5)

```powershell
C:\Users\oat\AppData\Local\anaconda3\envs\AITraining12\python.exe MACVO.py `
  --odom Config/Experiment/MACVO/Optimal/MACVO_DAv3_p2p.yaml `
  --data Config/Sequence/EIVA_plane_nose_mono.yaml `
  --seq_from 80 --seq_to 160 --seed 0 --resultRoot Results/<tag>
```

- Trajectory: `<resultRoot>/<odom.name>@<data.name>/<MM_DD_HHMMSS>/poses.npy`, shape (N, 8):
  `time_ns, tx, ty, tz, qx, qy, qz, qw` (written by `Odometry/Interface.py` via
  `Utility/Sandbox.py`).
- Data verified present at `D:\Datasets\EIVA\vobster_quay\plane_nose` (566 frames). The
  TartanAir sample data referenced by `Config/Sequence/TartanAir_example.yaml` is **not** on
  this machine.
- Runtime: ~20 s/frame at `num_point: 1000` — the per-point Python `CustomFactor` callbacks
  are ≈92 % of frame time (documented in the `MACVO_DAv3_p2p.yaml` header). Reduce
  `num_point` while iterating. This also bounds the spec's section-7 performance concern:
  the callback overhead is already far beyond the spec's 20 % threshold *before* adding
  scale keys.
- Unit tests in this env need `-o addopts=""` (jaxtyping 0.3.3 breaks pytest collection via
  the `pyproject.toml` addopts).

---

## 2. Why the design must change

### 2.1 The redundancy is structural

With free landmarks, frame k−1's **un-warped** factors are what anchor the scale of the
landmark cloud; the landmarks are re-estimated from scratch every pair. A global depth-scale
error on frame k's observations can be absorbed by the joint (landmarks, `pose_2`
translation) adjustment before a scale variable sees it. The gauge analysis of spec §5 (one
null direction over `(ω, v, s_A, s_B)`) does not transfer: the Hessian carries 3N landmark
blocks, so **test T3 as specified is unimplementable against this graph**, and T1/T2's
factor construction does not match the implemented residual.

### 2.2 The feature ~90 % exists, as `alignment: sim3`

`GTSAM_Pose2Point` already supports a per-frame log-scale variable
(`Graphs.py:216-225, 283-292`; warp + analytic Jacobian in
`Utility/GTSAM_Utils.py:59-75, make_aligned_pose_to_point_factor`):

- key `gtsam.symbol('a', frame_idx)`, a 1-dim `Vector` (the repo idiom for scalars —
  `insertDouble` is used nowhere);
- residual `pose.transformTo(l_w) − exp(x₀)·obs_Tc`, i.e. exactly a log-depth scale on the
  measured camera point (`obs = d·ray`, so `exp(x₀)·obs ≡ exp(x₀)·d`);
- zero-mean Gaussian prior, σ = 1/√`prior_weight` (default 0.1 — the spec's σ_s=0.15 is the
  same mechanism, different number);
- applied to the **current frame only**; previous-frame factors deliberately un-warped as
  the scale anchor;
- **estimate-and-report only**: the solved scale is logged
  (`/world/gtsam_alignment/scale`) but never written to poses, depths, or the map.

### 2.3 Empirical prior art: the estimation half has already been tried and measured

`ProgressReports/2026-07-07_gtsam-alignment-sweep.md`:

> **≈ sim3 is neutral-to-slightly-worse on GTSAM (both frontends).** … the pose2point graph
> re-estimates landmarks jointly per two-frame problem, so a per-frame depth-scale error is
> largely absorbed into the landmark positions already; the explicit scale DoF has nothing
> left to fix and only adds variance. … **Keep SE(3)**.

This is the spec's §10 failure mode "posterior sigma ≈ prior sigma / scale absorbs nothing",
observed in advance. What the spec adds that the sweep did **not** test — and where a revised
Stage 1 could still earn its keep:

1. **exp(2s)-consistent noise.** The existing sim3 factor leaves the noise model built from
   *un-warped* covariances — a real inconsistency the spec's outer loop fixes.
2. **Baking `s` into stored depths + carry-forward** of the posterior sigma. sim3 discards
   its estimate each solve; the spec's filter-along-the-sequence behaviour is genuinely new.
3. **Robust-kernel first-pass interaction** (spec §6) — never explored here.

---

## 3. Hazards a revised spec must address (decisions already taken are marked)

### 3.1 Scale-collapse feedback loop — DECISION: write-back must be guarded

Feeding a raw per-solve scale forward has a **documented death spiral** in this codebase:
one diverged solve shrinks the next insertion about the camera center, the shrunken targets
pull the next estimate lower, and within ~10 frames the map collapses (scale → 1e-3, frozen
trajectory) — observed on `plane_nose[128:140]`. The standing guard is an accept-gate
`[0.5, 2.0]` + log-space EMA (α = 0.3) (`Module/Optimization/GEDF/Optimizer.py:60-88`),
regression-tested in `Scripts/UnitTest/test_gedf_registration.py` ("plane_nose scale death
spiral").

**Decision (owner, 2026-08-04): the revised Stage 1 must reuse this guard pattern for the
depth/point write-back — not the spec's ungated `d ← exp(s*)·d` bake.** The spec's outer
loop tightens the feedback relative to GEDF's (covariances rescale too, changing next-solve
weights), so the ungated version is strictly more dangerous than the case that already
collapsed.

### 3.2 DepthCov frontend closes the loop tighter still

DepthCov's absolute scale comes entirely from sparse anchors projected out of the running
map (`Odometry/MACVO.py:358` `_build_depth_prior` → `CameraData.depth_prior` →
`DepthCov._extract_anchors`, which takes `log d` of map landmarks). Rescaling stored
`pos_Tw` therefore rescales the *next frame's predicted depth* — a closed positive-feedback
loop over `log d` with no external anchor. DAv3 has no such path (scale fixed by
`f_mean/300`). The revised spec must either declare the DepthCov config unsupported for
scale write-back or add an explicit guard + regression test.

### 3.3 "Rescale stored depths" must be a sparse rescale, about the camera center

Dense depth maps live only for the current + previous keyframe
(`Odometry/MACVO.py:75, 173, 318`); older frames survive only as sparse rows. The complete
mutable set per frame is: `MatchObs.pixel1_d/pixel2_d`, `pixel1_d_cov/pixel2_d_cov`,
`obs1_covTc/obs2_covTc`, and `PointNode.pos_Tw/cov_Tw`. Two consistency rules:

- frame B's depth is consumed by **both** pairs (A,B) and (B,C) — a per-frame `exp(s_B)`
  must touch both pairs' rows;
- the `pos_Tw` rescale is a uniform scaling **about the camera center**, not the world
  origin, with `cov_Tw` scaled by `s²`. Working precedent:
  `Module/Optimization/GEDF/Optimizer.py:321-343`.

### 3.4 Covariance homogeneity holds exactly — the outer loop's step 1 is cheap

Substituting `d → eˢd`, `σ_dd → e²ˢσ_dd` scales every entry of `Covariance_2to3_full`
(`Project2to3.py:413-425`) by exactly `e²ˢ` (every term is degree-2 in (d, σ_d)). So
"rebuild Σ_p from updated depths" reduces to a scalar multiply of the stored `obs*_covTc` —
no re-projection needed. The exactness breaks only at the clamps (`min_depth_cov`,
`min_flow_cov`) and, in the mono DAv3 path, nowhere else.

### 3.5 Mechanical constraints for the implementer

- **Config:** `_enforce_config_spec` rejects unknown keys
  (`Utility/Extensions/Testable.py:37-41`); every new option must be added to
  `GTSAM_Graph.is_valid_config` (`GTSAM/Optimizer.py:120-197`), following the existing
  `hasattr`-gated optional pattern. The optimizer schema is documented in
  `Module/Optimization/README.md`, not `Config/ConfigSpec.md`.
- **Symbols:** `'p'` (poses), `'l'` (landmarks), `'a'` (alignment) are taken; `'s'` is free.
  Scalars go in as 1-dim `Vector` + `PriorFactorVector` (repo idiom).
- **Parallel mode:** graph inputs/outputs are pickled copies; anything destined for the map
  must ride in `GTSAM_GraphOutput` and be applied in `write_graph_data` on the parent —
  in-place mutation inside `_optimize` is silently lost.
- **T4 feasibility:** bit-exact regression with the feature off is achievable — the sim3
  machinery shows the pattern (default `se3` leaves the original factor path untouched).
- **Performance:** every added key on the per-point `CustomFactor` raises the
  already-dominant Python callback cost; the spec's stacked-3N fallback breaks per-point
  Huber (as the spec itself notes) and the kernel is always on here.

---

## 4. Recommendation to the spec author

Reframe Stage 1 as an **extension of the existing alignment axis**, not a new factor family.
Two viable shapes:

1. **Keep the landmark graph** (small diff, preserves architecture): generalize `alignment`
   to a per-frame log-depth scale on *both* frames' observations — `s_A` on the `pose_1`
   factors (prior carried forward from the previous solve's posterior, clamped per spec §5),
   `s_B` on the `pose_2` factors (fresh prior σ_s) — with the exp(2s) outer loop operating
   on the stored sparse rows (3.3/3.4), a guarded write-back (3.1), and the §6 kernel
   schedule. Note the sweep result (2.3) predicts the pure-estimation signal is weak on this
   graph; the write-back + covariance consistency is the part that has not been tried.
2. **Match the spec's math literally**: add a new `graph_type` whose factors bake points as
   fixed measurements against a single relative pose (`p_prev − T·p_curr`, combined
   `Σ_A + R Σ_B Rᵀ` noise). This restores the spec's Jacobians, gauge analysis, and T1–T3
   verbatim, at the cost of a second graph implementation diverging from the existing one —
   and it discards the landmark re-estimation that currently absorbs structured error.

Whichever is chosen: T3's Hessian analysis and T1/T2's construction must be rewritten for
that graph; T4 stays as specified; the diagnostics of spec §9 map naturally onto the
existing Rerun scalar channels (`/world/gtsam_alignment/scale`) plus a per-sequence
`np.save`/CSV in the sandbox.
