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

Three backends implement `IOptimizer`. All are selected purely through the
`Odometry.optimizer` config block (`type:` + `args:`), all support the
sequential / parallel-worker execution modes described above, and all write a
7-float SE(3) pose into `frames.data["pose"]`.

| Backend (`type:`) | Library | Graph types | Extra state written back | Notes |
|---|---|---|---|---|
| `TwoFrame_PGO` | PyPose (LM, autodiff or analytic Jacobians) | `icp`, `reproj`, `disp` | pose only | The MAC-VO default; frame-to-frame, covariance-weighted |
| `GTSAM_Graph` | GTSAM (`LevenbergMarquardtOptimizer`, `ISAM2`) | `pose2point`, `isam` | pose **and landmark positions** | Optional dependency (guarded import); multi-frame / incremental |
| `GEDF_PGO` | PyPose + custom G-EDF mapper | `gedf`, `gedf+icp` | pose only (+ diagnostics) | Builds a distance-field map online and registers scan-to-map; see [`GEDF/README.md`](GEDF/README.md) |

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

```yaml
optimizer:
  type: GTSAM_Graph
  args:
    device: cpu
    vectorize: true
    parallel: false
    autodiff: true          # required by the config spec (not used for graph selection)
    graph_type: pose2point  # pose2point | isam
    # Optional pose2point hyperparameters (defaults shown):
    # huber_delta: 0.1
    # huber_delta_prev: 1.0
    # prior_sigma: 1.0e-4
    # max_iterations: 20
    # match_atol: <pixel tolerance for cross-frame landmark association>
```

* `pose2point` — batch LM over two frames with pose-to-point custom factors and
  cross-frame landmark association by pixel proximity.
* `isam` — incremental `ISAM2` with a sliding pose window, between-factors, and
  pixel-keyed landmark reuse.

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
    graph_type: gedf+icp      # gedf (field-only) | gedf+icp (hybrid, recommended)
    map:    { source: online, ... }      # online mapping or prebuilt .bin
    field:  { weighting: mahalanobis, sigma: 0.30, ... }
    solver: { coarse/fine two-stage robust LM ... }
    viz:    { every: 10, ... }           # live Rerun map view (--useRR)
    # alignment: { type: sim3, prior_weight: 100.0 }   # optional, monocular (see below)
```

### Alignment axis (monocular depth-bias correction)

`GEDF_PGO` additionally supports a per-frame **alignment manifold** estimated
jointly with the pose (`alignment.type`, requires `autodiff: true` for non-se3):

| `alignment.type` | Extra DoF | Absorbs | Reported as |
|---|---|---|---|
| `se3` (default) | — | — | — |
| `sim3` | 1 (log-scale) | monocular depth-scale bias | `scale`, `/world/gedf_alignment/scale` in Rerun |
| `sl4` (experimental) | 9 (projective) | affine/projective depth bias | `alignment_state` (9,) |

Semantics are *estimate + report*: the warp corrects the residuals during the
solve, only the SE(3) component reaches the map. GTSAM and TwoFramePGO remain
SE(3)-only.

## Shipped experiment configs (`Config/Experiment/MACVO/`)

| Config | Optimizer setup | Frontend | Intended use |
|---|---|---|---|
| `MACVO_Performant.yaml` | `TwoFrame_PGO` icp, autodiff, parallel | stereo FlowFormer | Stereo default (best accuracy) |
| `MACVO_Fast.yaml` | `TwoFrame_PGO` icp, autodiff, parallel | stereo, mixed precision | ~2x speed, ~5% RTE/ROE cost |
| `MACVO_gtsam.yaml` | `GTSAM_Graph` pose2point, sequential | stereo FlowFormer | GTSAM baseline / landmark refinement |
| `MACVO_MonoDAv3.yaml` | `GTSAM_Graph` pose2point, sequential | mono DepthAnythingV3 | Monocular baseline |
| `MACVO_GEDF.yaml` | `GEDF_PGO` gedf+icp, analytic, parallel, online map | stereo FlowFormer | Stereo + scan-to-map drift resistance |
| `MACVO_GEDF_DAv3.yaml` | `GEDF_PGO` gedf+icp, analytic, parallel, online map, relative depth-cov selectors | mono DepthAnythingV3 | Monocular + scan-to-map; uncomment `alignment: sim3` (+ `autodiff: true`) for per-frame scale correction |

Choosing:
* **Stereo, short sequences** — `MACVO_Performant` (frame-to-frame ICP is hard to beat when depth is metric and drift is small).
* **Stereo, long sequences / revisits** — `MACVO_GEDF` (the map factor anchors the trajectory to previously seen structure).
* **Monocular** — `MACVO_GEDF_DAv3`; enable `alignment: sim3` when the depth model's scale wanders per frame.
* **Landmark refinement / incremental smoothing** — `GTSAM_Graph` (`isam` for a sliding window).
