# Extending Optimizer in MAC-VO

## The `IOptimizer` Interface

`IOptimizer` is the interface for the optimizer used in MAC-VO. It is the most complex interface in this project since it allows running any optimizer in sequential/parallel mode according to the config.

The Optimizer runs in two modes but the user only need to implement a single set of interface, which contains *four methods* and *three data (message) types*

Data:
* `init_context` - initialize the "context" of the optimizer. Essentially, context is like the `self` in Python but is represented as a separate instance since `self` cannot be sent directly to the child process.
* `get_graph_data` - given the map constructed by odometry and some frames to optimize on, this method extracts all information required to build the optimization problem.
* `_optimize` - Given context and argument, construct the optimization problem, solve it, and return the updated context and result.
* `write_graph_data` - Given the result returned from `_optimize`, update the map (write the result back to the map)

Data (Message) Type:
* `T_Context` - an arbitrary class that stores the optimizer state accumulated/modified across frames.
* `T_GraphInput` - a **subclass of `ITransferable`** since this message may be communicated across processes. Contains the inputs required to construct the optimization problem.
* `T_GraphOutput` - a **subclass of `ITransferable`** since this message may be communicated across processes. Contains results (of interest) for the optimization problem.

These message classes are necessary due to the multi-thread module of the optimizer.
Detailed specification of methods to be implemented is provided below:

```python
class IOptimizer(ABC, Generic[T_GraphInput, T_Context, T_GraphOutput], SubclassRegistry):
    """
    Interface for optimization module. When config.parallel set to `true`, will spawn a child process
    to run optimization loop in "background".

    `IOptimizer.optimize(global_map: TensorMap, frames: BatchFrames) -> None`

    * In sequential mode, will run optimization loop in blocking mannor and retun when optimization is finished.

    * In parallel mode, will send optimization job to child process and return immediately (non-blocking).

    `IOptimizer.write_back(global_map: TensorMap) -> None`

    * In sequential mode, will write back optimization result to global_map immediately and return.

    * In parallel mode, will wait for child process to finish optimization job and write back result to global_map. (blocking)

    `IOptimizer.terminate() -> None`

    Force terminate child process if in parallel mode. no-op if in sequential mode.
    """
    ### Internal interface to be implemented
    @abstractmethod
    def _get_graph_args(self, global_map: TensorMap, frames: BatchFrame) -> T_GraphInput:
        """
        Given current global map and frames of interest (actual meaning depends on the implementation),
        return T_GraphArgs that will be used by optimizer to construct optimization problem.
        """
        ...

    @staticmethod
    @abstractmethod
    def init_context(config) -> T_Context:
        """
        Given config, initialize a *mutable* context object that is preserved between optimizations.

        Can also be used to avoid repetitive initialization of some objects (e.g. optimizer, robust kernel).
        """
        ...

    @staticmethod
    @abstractmethod
    def _optimize(context: T_Context, graph_args: T_GraphInput) -> tuple[T_Context, T_GraphOutput]:
        """
        Given context and argument, construct the optimization problem, solve it and return the
        updated context and result.
        """
        ...

    @staticmethod
    @abstractmethod
    def _write_map(result: T_GraphOutput | None, global_map: TensorMap) -> None:
        """
        Given the result, write back the result to global_map.
        """
        ...
```

Below we demonstrate how the internal interfaces mentioned above are orchestrated in sequential and parallel optimization mode.

**Parallel Mode**

![ParallelMode](https://github.com/user-attachments/assets/98348cb8-7a22-44f5-b160-4568fe196f50)

**Sequential Mode**

![SequentialMode](https://github.com/user-attachments/assets/b297a5db-f348-46b0-8213-fd60b5c4a006)

---

# Optimizer backends & shipped setups

Four backends implement `IOptimizer`. All are selected purely through the
`Odometry.optimizer` config block (`type:` + `args:`), all support the
sequential / parallel-worker execution modes described above (except
`ISAM2_Graph`, sequential-only), and all write a 7-float SE(3) pose into
`frames.data["pose"]`.

| Backend (`type:`) | Library | Graph types | Alignment axis | Extra state written back | Notes |
|---|---|---|---|---|---|
| `TwoFrame_PGO` | PyPose (LM, autodiff or analytic Jacobians) | `icp`, `reproj`, `disp` | se3 only | pose only | The MAC-VO default; frame-to-frame, covariance-weighted |
| `GTSAM_Graph` | GTSAM (`LevenbergMarquardtOptimizer`, `ISAM2`) | `pose2point`, `isam` | se3 / sim3 / sl4 (`pose2point` only) | pose **and landmark positions** (into `points`) | Optional dependency (guarded import); multi-frame / incremental |
| `ISAM2_Graph` | GTSAM (`ISAM2`, persistent) | `pose2point`, `pose2point_native`, `bearingrange` factors | se3 only | pose only | One graph over the whole sequence; flow-tracked landmarks via `TrackingCovAwareSelector`; optional GNC-GM; sequential-only |
| `GEDF_PGO` | PyPose + custom G-EDF mapper | `gedf`, `gedf+icp` | se3 / sim3 / sl4 (autodiff mode) | pose only (+ diagnostics) | Builds a distance-field map online and registers scan-to-map; see [`GEDF/README.md`](GEDF/README.md) |

## `TwoFrame_PGO` (PyPose two-frame pose graph)

Optimizes the new frame's absolute pose against the previous keyframe using
covariance-weighted residuals; rebuilt per frame pair.

```yaml
optimizer:
  type: TwoFrame_PGO        # variants: Local_TwoFrame_PGO (solves in the previous-
  args:                     # keyframe frame for numerical stability), Empty_TwoFrame_PGO (no-op)
    device: cpu
    vectorize: true
    parallel: true          # non-blocking worker, pipelined with the frontend
    graph_type: icp         # icp | reproj | disp
    autodiff: true          # false = hand-derived analytic Jacobians (LM_analytic)
```

* `icp` — residual `T·p_cam − p_world`, weighted by `R Σ_obs Rᵀ + Σ_landmark`.
* `reproj` — reprojection-pixel residual weighted by keypoint uv covariance.
* `disp` — reprojection + disparity row (stereo).

## `GTSAM_Graph`

Factor graph over `gtsam.Pose3` poses and landmarks; the only backend that also
writes optimized **landmark positions** back into the map. Registered only when
`gtsam` is installed.

**Installing gtsam (validated version: 4.3a2).** On Linux/Docker:
`pip install gtsam==4.3a2` (pre-release wheels for Python 3.11–3.14). PyPI ships
**no Windows wheels**, so on Windows build the wheel from source with
`Scripts/build_gtsam_windows.ps1` (VS 2022 + CMake + Ninja; ~10 min build), e.g.

```powershell
powershell -File Scripts/build_gtsam_windows.ps1 -Tag 4.3a2 `
    -GtsamClone C:\Users\oat\Documents\Github\gtsam `
    -PythonExe C:\Users\oat\.conda\envs\AITraining12\python.exe `
    -PatchFile Scripts/patches/gtsam-posetopoint-wrapper.patch
```

`-PatchFile` adds the Python wrapper for `gtsam_unstable::PoseToPointFactor`
(not wrapped upstream) that the `pose2point_native` iSAM2 factor type needs;
omit it for a stock build.

Note `gtsam` exposes no `__version__` attribute — check with
`importlib.metadata.version("gtsam")`. For local pyright runs, pin the
interpreter (`pyright --pythonpath <env>\python.exe ...`) or it will resolve
`gtsam` from whatever `python` is first on PATH.

**Architecture:** `GTSAM_Pose2Point` builds exactly the pose-to-point two-frame
graph and nothing else. Every optional extension — the G-EDF field factor
(`GEDFField`), the correlated log-depth prior (`CorrelatedDepthPrior`), per-solve
diagnostics (`SolveDiagnostics`) — implements the `GraphAugmentation` protocol
(`GTSAM/Augmentations.py`) and is injected by `init_context` at three fixed
lifecycle hooks: `on_parse` (stash inputs), `on_build` (add factors/variables
pre-LM; list order = factor insertion order), `on_solved` (plain-scalar
diagnostics merged into `GTSAM_GraphOutput.aug_diag`). With no augmentations the
built graph is byte-identical to the bare pose2point path (regression-gated).
New extensions add a class + an `init_context` line — the graph itself stays
closed.

The landmarks land in `VisualMap.points` — the sparse VO landmarks, at the global
indices `GraphInput.points.index` carried into the solve. They do *not* touch
`map_points` (the dense mapping cloud), which no graph here re-estimates, and a
graph's own landmark numbering (a gtsam symbol index, which `isam` reuses across
frames as it merges tracks) never leaves the graph. `pose2point` writes back one
landmark per current-frame observation; `isam` writes back the landmarks the
current call observed, so refinements iSAM2 makes to older landmarks outside that
set stay inside the solver.

With `--useRR` each refined batch is logged to **`/world/vo_tracking_opt`** at the
same `frame_idx` where `MACVO.VisualizeRerunCallback` logged the same batch's
first estimate to `/world/vo_tracking` — so the two entities overlay directly, and
widening the visible time range accumulates the optimized cloud. Under
`pose2point` each landmark is refined exactly once, so the accumulation *is* the
final map; under `isam` a landmark can be re-refined on later frames, and only its
newest batch reflects that.

How far each batch moved is logged alongside as
**`/world/gtsam_landmark_disp/median`** and **`/p95`** (‖refined − first estimate‖
in world units). The median is the typical depth correction the pose-to-point
factors apply — governed by the `min_depth_cov` / `min_flow_cov` floors in
`Odometry.cov.obs`, which set how tightly the solve trusts each observation — and
p95 is the tail the `huber_delta` kernels did not clamp.

```yaml
optimizer:
  type: GTSAM_Graph
  args:
    device: cpu
    vectorize: true
    parallel: false
    autodiff: true          # required by the config spec (not used for graph selection)
    graph_type: pose2point  # pose2point | isam | pose2point+gedf
    # Optional pose2point hyperparameters (defaults shown):
    # huber_delta: 0.1
    # huber_delta_prev: 1.0
    # prior_sigma: 1.0e-4
    # max_iterations: 20
    # match_atol: <pixel tolerance for cross-frame landmark association>
    # Optional alignment axis (pose2point only; isam and pose2point+gedf stay SE3):
    # alignment:
    #   type: sim3          # se3 (default) | sim3 | sl4
    #   prior_weight: 100.0
    # Required for pose2point+gedf — the G-EDF map/field/viz sub-configs
    # (same keys as GEDF_PGO minus insert_dense; see Optimal/MACVO_Fast_GEDF.yaml):
    # gedf: { map: {...}, field: {...}, viz: {...} }
    # Optional correlated log-depth prior (pose2point only; mutually exclusive
    # with a non-se3 alignment — the two occupy the same global-scale direction
    # and the combination is REJECTED at config validation). Defaults shown:
    # enable_gp_depth_prior: false
    # gp_prior_frames: ["prev", "curr"]   # which frame's factors get the prior
    # gp_prior_block_size: 16             # spatial block size (kd median splits)
    # gp_prior_kernel: matern32           # DepthPrior.KERNEL_BUILDERS entry
    # gp_prior_length_scale_px: 40.0      # correlation reach, pixels
    # gp_prior_sigma_f: 0.15              # collective departure, log-depth units
    # gp_prior_sigma_n: 0.05              # per-point departure, log-depth units
    # gp_scale_prior_sigma: 0.15          # prior on the per-frame scale s
    # gp_prior_z_min: 0.05                # depth clamp inside the factor, metres
    # gp_prior_diag_dir: <dir>            # flush the per-sequence diagnostics CSV here
```

* `pose2point` — batch LM over two frames with pose-to-point custom factors and
  cross-frame landmark association by pixel proximity.
* `isam` — incremental `ISAM2` with a sliding pose window, between-factors, and
  pixel-keyed landmark reuse.
* `pose2point+gedf` — `pose2point` plus **one batched G-EDF field factor** on
  the current pose (`make_gedf_field_factor`): residual `d̂(T·pᵢ)` per current
  keypoint against the online (or prebuilt) G-EDF map, inside the same joint
  solve that re-estimates the landmarks — "GTSAM's ICP" fused with whole-map
  registration. The map is fed and refit per call exactly like `GEDF_PGO`, from
  `points` as it stands at `get_graph_data` time — so a call inserts the
  landmarks the *previous* call refined, never its own; the field factor is inert
  until the map is ready and acts on the raw camera points (hence SE(3)-only).
  Map snapshots log to the same `/world/gedf_map` Rerun entities.

### Correlated log-depth prior (`enable_gp_depth_prior`, Phase 0)

`Sigma_p` is strongly elongated along the viewing ray, so each `pose2point`
landmark slides along its own ray nearly for free — one unconstrained depth
degree of freedom per point, which is why a per-frame scale variable (sim3) had
nothing to do (2026-07-07 sweep). This prior couples those freedoms instead: per
spatially-local block of landmarks (`Module/Optimization/GTSAM/DepthPrior.py`),
a factor whitens the landmarks' log-depth departure from the *measured* depths
with the inverse Cholesky factor of a Matern-3/2 kernel over pixel coordinates,
and one per-frame scalar `s` (symbol `'s'`, weak zero prior) subtracts global
scale so only the *shape* of the depth field is asserted. Nothing is written
back — stored depths, covariances and the dense maps are read-only here.

Frontend-agnostic by construction: the prior reads only `MatchObs.pixel*_uv` /
`pixel*_d`, so the same optimizer block drops into any experiment config (DAv3,
DAv2, DepthCov, stereo) regardless of `monodepth.type`. Note DAv2's *covariance*
is `0.1·depth` (not a variance), which degrades the point factors themselves —
prefer DAv3 or DepthCov configs for experiments.

**Phase 1 — the prior owns depth (`gp_own_depth: true`).** Phase 0 measured a
structural null: the prior's target `log(pixel*_d)` duplicates the depth
measurement already weighted by `sigma_dd` inside `Sigma_p`, so the prior could
only re-weight, never add. With `gp_own_depth` the augmentation replaces the
point factors' covariances (numpy copies only — the map is untouched) with a
depth-free construction: pixel-noise lateral block plus a large ray-aligned
slide term `(gp_slide_sigma_rel * d)^2 · r rᵀ` that makes the along-ray
direction nearly uninformative (NOT zero-variance — the observation `d·ray`
embeds the measured depth, and zeroing would pin it as exact). The prior is
then the sole per-point depth model. Its block covariance is the kernel's
smooth part ADDED to the network's own per-point log-depth variance
(`gp_prior_nugget: measured`, the default): nugget² = max(`pixel*_d_cov`/d²,
σ_n²) — for DAv3 that is the calibrated confidence-head variance, for DepthCov
exactly its GP's `Var(log d)`; the σ_n floor stands alone wherever the network
reports nothing. `gp_prior_nugget: fixed` is the explicit escape hatch for
networks whose stored variance is fake and must not be consumed (DAv2:
`0.1·d` — positive, so it cannot be auto-detected as invalid).
Requires `gp_prior_frames: [prev, curr]` (rejected otherwise — a depth-freed
frame without prior coverage would lose its depth measurement outright); the
frame k−2 re-observation factor keeps `sigma_dd` (the prior does not cover
k−2). Note the Huber kernel on depth-free factors gates on lateral/flow
consistency only. Extra keys: `gp_own_depth: false`, `gp_slide_sigma_rel: 10.0`,
`gp_prior_nugget: fixed|measured`.

**Usage warning (measured on Welland, 2026-08-11): do not enable the prior without a
far-range gate on scenes with unbounded range.** The prior's coupling propagates biased
far-tail depths into the clean near cloud — ungated it cost +94 % ATE on a canal survey
while the gated combo beat prior-off by 26 %. Set the keypoint selector's `max_depth`
from the scene's candidate-depth distribution (cut ~p75–p80; thresholds do NOT transfer
across datasets/scale conventions), and pair with `motion_prior_sigma` — the gate thins
each pair's geometry and the soft cv factor supplies the stabilization back.

**Except on map-anchored frontends (DepthCov): never gate.** A `max_depth` below the
frontend's bootstrap fallback depth (`init_depth`) is a deterministic deadlock — the
empty-map fallback depth fails the gate, so no landmarks are ever created and the map
never grows anchors (measured: frozen trajectory, both seeds). And above the interlock
threshold the gate has nothing to cut (anchor-driven selection already compresses the
range). DepthCov also needs the TIGHT scale prior (`gp_scale_prior_sigma: 0.15`) — the
loose value that is free on DAv3 amplifies its anchor-feedback scale shrinkage
(path/GT 0.66 → 0.48). Per-frontend recipes:
`ProgressReports/2026-08-11_gp-prior-depthcov-matrix.md`.

Diagnostics per solve (also with the prior off, for on/off stratification):
Rerun scalars under `/world/gp_depth_prior/*` (per-frame `s`, RMS of
`y − ŷ − s`, cost shares, median parallax angle, Huber rejections) and a
per-sequence CSV flushed to `gp_prior_diag_dir` on `terminate()`. Check the
cost split before interpreting any sweep: a negligible prior share means the
sweep measured nothing.

## `ISAM2_Graph` (persistent-graph iSAM2 with flow-tracked landmarks)

One `gtsam.ISAM2` factor graph over the whole sequence — a pose key per frame,
landmark keys that live as long as their track — updated incrementally per
frame pair. Port of learningUAVO's `gtsam_backend/isam2_tracker.py` (see that
repo's `FINDINGS.md` for every hyperparameter's provenance); the shipped
configs under `Config/Experiment/MACVO/ISAM2/` reproduce the three winning
arms of its plane_nose_full online sweep.

**Requires `keypoint: TrackingCovAwareSelector`.** Landmark identity is
positional: the stateful selector carries each keypoint along the optical flow
(mirroring `run_pair`'s `kp1 = kp0 + flow(kp0)` bit-for-bit), and the backend
chains rows into tracks by exact integer pixel association
(`pixel1_uv == round(previous pixel2_uv)`). A carried row's pixel1 observation
is skipped (it re-quantizes the previous pair's pixel2 obs); an unmatched row
births a landmark whose pixel1 fields are its first observation. Flow variance
accumulates along tracks; per-row 3×3 covariances are recomposed in-solver from
the MatchObs scalars. Pair it with `outlier: IdentityFilter` — a row dropped
between selector and map splits its track — and `keyframe: AllKeyframe`.

```yaml
optimizer:
  type: ISAM2_Graph
  args:
    device: cpu
    parallel: false           # enforced: stateful persistent graph
    factor_type: bearingrange # bearingrange (C++ relin, cheap) | pose2point (exact model, Python
                              # callback) | pose2point_native (exact model AND C++ relin — needs a
                              # gtsam wheel built with Scripts/patches/gtsam-posetopoint-wrapper.patch,
                              # which wraps gtsam_unstable::PoseToPointFactor for Python)
    kernel: huber             # huber|cauchy|geman|tukey|welsch|none; ignored when GNC on
    kernel_delta: 0.1
    relin_threshold: 0.05     # LOOSER than the gtsam default is better online
    relin_skip: 1
    extra_updates: 3
    warmup_frames: 10         # early frames get warmup_extra additional update rounds
    warmup_extra: 5
    depth_var_scale: 2.5
    accumulate_fvar: true
    motion_init: cv           # static | cv (T_prev @ T_rel_prev)
    motion_prior_sigma: 0.3   # soft BetweenFactorPose3; 0 = off; rot sigma = 2x
    coast_sigma: 0.1          # weak Between on low-support / skipped frames
    min_support: 6
    readout: chain            # chain (composed current-belief rel motions) | online
    min_flow_cov: 0.25
    min_depth_cov: 0.01
    match_cov_default: 0.25
    # GNC-GM instead of the kernel (runs damped under Dogleg):
    # gnc_rounds: 5
    # gnc_c: 0.4              # residual scale in the depth map's own units — re-sweep per depth source
    # gnc_mu_rate: 5.0
```

The written-back pose is the **chain readout** by default — current-belief
relative motions composed into a frozen chain, immune to the gauge slide of
weakly-bridged young segments that scrambles raw online poses. Landmarks are
*not* written back (the chain gauge and the live graph's gauge diverge after
low-support stretches). The solver runs QR factorization unconditionally:
Huber-downweighted cliques throw `IndeterminantLinearSystemException` under
Cholesky. Frames the odometry skips (`VOLostTrack`) are coasted in with a weak
between-factor at the previous relative motion.

### Keyframe re-observations (`Odometry.keyframe_tracker`)

Optional, `ISAM2_Graph` only. A keyframe *policy* (`Module/KeyframeTracker.py`,
port of learningUAVO's `keyframe_selection.py`: `EveryN`, `Parallax`,
`Covisibility`, `BaselineRatio`, `AnyOf`) holds one reference frame. Every frame
k the odometry runs one extra flow inference keyframe -> k
(`IFrontend.estimate_match`), carries the keyframe's registered keypoints (the
pixel1 rows of pair kf -> kf+1) into frame k and stores the survivors in
`VisualMap.kf_match` (`kfmatch2point` = the point those rows born). The backend
associates each row to the landmark that integer pixel resolved to when pair
(kf, kf+1) was stepped and adds a pose-to-point factor `p_k -> l` on that SAME
key — an extra, non-consecutive connection whose variance is quantization + one
flow step instead of the accumulated chain variance. The chain observation of the
same landmark is kept; `kf_cov_scale` inflates the keyframe factors (both share
the frame-k depth sample). Under `marg_lag` the keyframe pose and its
re-observed landmarks are re-stamped every frame; already-marginalized landmarks
are skipped. `frame_stats()` gains `n_kf_obs` and `kf_idx`. Requires
`keyframe: AllKeyframe`. Config: `Config/Experiment/MACVO/ISAM2/MACVO_ISAM2_p2p_kf.yaml`.

## `GEDF_PGO` (scan-to-map against an online distance-field map)

Builds a G-EDF map from MAC-VO's own landmarks during the run (or loads a
prebuilt GDF1 `.bin`) and registers each frame's keypoints against it — a
drift-resisting map factor fused with the ICP factor. Full documentation,
config reference, math, and results: [`GEDF/README.md`](GEDF/README.md).

```yaml
optimizer:
  type: GEDF_PGO
  args:
    device: cpu
    vectorize: true
    parallel: true
    autodiff: false           # analytic (se3 alignment only); true enables sim3/sl4
    graph_type: gedf+icp      # gedf (field-only) | gedf+icp (joint hybrid, recommended)
                              # | icp->gedf (sequential: ICP init, whole-map field final)
    map:    { source: online, ... }      # online mapping or prebuilt .bin
    field:  { weighting: mahalanobis, sigma: 0.30, ... }
    solver: { coarse/fine two-stage robust LM ... }
    viz:    { every: 10, ... }           # live Rerun map view (--useRR)
    # alignment: { type: sim3, prior_weight: 100.0 }   # optional, monocular (see below)
```

### Alignment axis (monocular depth-bias correction)

`GEDF_PGO` (any graph type, requires `autodiff: true` for non-se3) and
`GTSAM_Graph` (`pose2point` only, via custom Sim3/SL4-warped pose-to-point
factors with analytic Jacobians) support a per-frame **alignment manifold**
estimated jointly with the pose (`alignment.type`):

| `alignment.type` | Extra DoF | Absorbs | Reported as |
|---|---|---|---|
| `se3` (default) | — | — | — |
| `sim3` | 1 (log-scale) | monocular depth-scale bias | `scale` on the graph output; `/world/gedf_alignment/scale` / `/world/gtsam_alignment/scale` in Rerun |
| `sl4` (experimental) | 9 (projective) | affine/projective depth bias | `alignment_state` (9,) |

Semantics are *estimate + report*: the warp corrects the residuals during the
solve, only the SE(3) component reaches the map. In GTSAM the warp applies to
the CURRENT frame's observations only (previous-frame factors anchor landmark
scale) and is reported at `/world/gtsam_alignment/scale`. TwoFramePGO and the
GTSAM `isam` graph remain SE(3)-only.

**`GEDF_PGO` + `sim3` additionally feeds the scale forward at map insertion**:
the landmarks inserted at the start of each `_optimize` call come from the
*previous* frame's depth, so a scale correction is applied to them (uniform
scaling about the previous camera center; covariances pick up s²) before they
reach the online map and the ICP rows. Without this the map accumulates
geometry at raw (drifting) monocular depth scale while the poses follow the
corrected scale — visible as double surfaces / map-vs-estimate misalignment on
long sequences. `sl4` is *not* fed forward (its warp is not a uniform scale);
`se3` has no scale channel.

The applied correction is a **gated, damped state**, not the raw per-solve
estimate (`_update_align_scale_state` in `GEDF/Optimizer.py`): estimates
outside `[0.5, 2.0]` are rejected (with a warning) and accepted ones advance a
log-space EMA (α = 0.3). Feeding raw estimates forward is a positive-feedback
loop — one diverged solve or a genuine depth-scale transient shrinks the next
insertion, the shrunken ICP targets pull the next estimate lower, and within
~10 frames the map and warp collapse to a point (observed on
`plane_nose[128:150]`: scale 1.0 → 1e-3, frozen trajectory). The Rerun channel
`/world/gedf_alignment/scale` still shows the raw per-solve estimate.

## Shipped experiment configs (`Config/Experiment/MACVO/`)

| Config | Optimizer setup | Frontend | Intended use |
|---|---|---|---|
| `MACVO_Performant.yaml` | `TwoFrame_PGO` icp, autodiff, parallel | stereo FlowFormer | Stereo default (best accuracy) |
| `MACVO_Fast.yaml` | `TwoFrame_PGO` icp, autodiff, parallel | stereo, mixed precision | ~2x speed, ~5% RTE/ROE cost |
| `MACVO_gtsam.yaml` | `GTSAM_Graph` pose2point, sequential | stereo FlowFormer | GTSAM baseline / landmark refinement; commented `alignment:` example |
| `MACVO_MonoDAv3.yaml` | `GTSAM_Graph` pose2point, sequential | mono DepthAnythingV3 | Monocular GTSAM baseline; uncomment `alignment: sim3` for per-frame scale correction |
| `MACVO_GEDF.yaml` | `GEDF_PGO` gedf+icp, analytic, parallel, online map | stereo FlowFormer | Stereo + scan-to-map drift resistance |
| `MACVO_GEDF_DAv3.yaml` | `GEDF_PGO` gedf+icp, analytic, parallel, online map, relative depth-cov selectors | mono DepthAnythingV3 | Monocular + scan-to-map; uncomment `alignment: sim3` (+ `autodiff: true`) for per-frame scale correction |

Choosing:
* **Stereo, short sequences** — `MACVO_Performant` (frame-to-frame ICP is hard to beat when depth is metric and drift is small).
* **Stereo, long sequences / revisits** — `MACVO_GEDF` (the map factor anchors the trajectory to previously seen structure).
* **Monocular** — `MACVO_GEDF_DAv3` (scan-to-map) or `MACVO_MonoDAv3` (GTSAM); on either, enable `alignment: sim3` when the depth model's scale wanders per frame (GEDF additionally needs `autodiff: true`; GTSAM's factors are analytic and need no mode switch).
* **Landmark refinement / incremental smoothing** — `GTSAM_Graph` (`isam` for a sliding window).
