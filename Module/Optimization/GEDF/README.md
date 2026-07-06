# G-EDF Scan-to-Map Optimization Backend (`GEDF_PGO`)

A pose-optimization backend that builds a **G-EDF map** (Gaussian Euclidean Distance
Field) online from MAC-VO's own landmarks and registers each new frame's keypoints
against it — point-to-distance-field residuals in the style of G-EDF-Loc's LiDAR
localization, fused with MAC-VO's covariance-weighted ICP factor. The map factor
resists drift: it anchors every new pose to the structure observed in *all* earlier
frames, not just the previous one.

Selectable from YAML like any other backend:

```yaml
optimizer:
  type: GEDF_PGO
  args: { ... }        # full reference below
```

References:
* **G-EDF** (robotics-upo, IROS 2026): block-sparse Gaussian distance fields —
  the field model, per-cube GMM fitting, and the GDF1 binary map format.
* **G-EDF-Loc**: 6-DoF localization against a G-EDF map with Ceres — the
  registration residual, analytic Jacobian, out-of-map and two-stage-solve
  semantics ported here.

Everything is pure Python/PyTorch — no C++ builds. The field math is verified at
`atol 1e-9` against G-EDF's reference NumPy evaluator (`scripts/gdf1_field.py`)
and against real maps produced by the C++ trainer.

---

## 1. The field model

Space is partitioned into a sparse, world-anchored grid of cubes (default 1 m³,
key = `floor(x / cube_size)`). Each cube holds a small Gaussian mixture that
approximates the local **unsigned Euclidean distance field**:

```
d̂(x) = Σ_k  w_k · exp( -0.5 · Σ_a (x_a - μ_ka)² / λ_ka )
```

* diagonal scales with the GDF1 **root parameterization**: the stored "sigma" is
  `p`, and `λ = p⁴` (effective std `σ_eff = p²`) — this keeps scales positive
  during fitting without a manifold;
* weights may be **negative** (the mixture is a function approximator, not a
  density): negative Gaussians carve the near-surface valley, positive ones
  raise the field away from it;
* neighboring cubes overlap by `margin` (0.25 m) and are blended with a
  Smoothstep weight `t²(3-2t)` for C¹ continuity across cube faces;
* empty / out-of-map regions return the sentinel `oob_value = 20.0` with zero
  gradient.

`d̂(x) ≈ distance to the mapped surface`, so a point known to lie **on** a
surface should evaluate to ≈ 0 — that is the registration residual.

## 2. Architecture

```
Module/Optimization/GEDF/
  Config.py     GEDFConfig - all mapper parameters (defaults mirror G-EDF's config.yaml)
  Mapper.py     GEDFMapper - the map: storage, query, incremental lifecycle, sampling
  Fitting.py    batched per-cube GMM fitting (EDT targets, NMS init, batched LM)
  Export.py     GDF1 .bin reader/writer (byte-compatible with the C++ tools)
  Graphs.py     factor graphs (pure registration + hybrid ICP) and message types
  Optimizer.py  GEDF_PGO(IOptimizer) - plugs everything into the MAC-VO backend API
```

### `GEDFMapper` (Mapper.py)

Storage is a padded structure-of-arrays over cube "slots" (`[C, K, ...]`
tensors; padding entries carry `w = 0, 1/λ = 0` so they contribute exactly zero
without masking). A packed-int64-key dictionary serves inserts; a lazily sorted
key tensor + `searchsorted` serves fully batched queries.

* `query(points) -> dist` — blended field value; **differentiable w.r.t. the
  query points** (used by the autodiff factor graphs) and device-transparent.
* `query_with_grad(points) -> (dist, grad)` — value plus the **exact analytic
  spatial gradient**, including the blend-weight quotient-rule term (the C++
  treats blend weights as constants; here analytic == autograd, which the tests
  assert).
* `insert(points, cov=None)` — bin into cubes, dedup on a 2.5 cm voxel, cap at
  512 points/cube, mark touched cubes (and halo-affected neighbors) dirty.
  Optional absolute covariance gate (`cov_trace_gate`); selecting *reliable*
  points is otherwise the job of the keypoint / mappoint selectors upstream.
* `refit(camera_pos, budget)` — fit up to `budget_cubes_per_frame` dirty cubes,
  highest priority first (cold cubes and cubes near the camera win). Failed
  fits keep the last valid parameters.
* `sample_surface(resolution, iso, max_points)` — near-surface point cloud of
  the field (grid-sample valid cubes, keep `d̂ < iso`); shared by the Rerun
  live visualization and `Scripts/VisualizeGEDF.py`.
* `from_gdf1(path)` / `export_gdf1(path)` — load a pre-built map (frozen) /
  write the current map, byte-compatible with G-EDF's visualization tools
  (`visualize_slice.py`, `visualize_3d.py`, `gaussian_to_ply`).

### Fitting (Fitting.py)

Direct port of the G-EDF trainer, batched over cubes:

1. **Targets**: PURE-mode EDT = distance to the nearest cloud point, computed
   *exactly at the sample locations* with a masked `cdist` (equivalent to the
   reference's per-voxel kd-tree, cheaper for sparse VO clouds). Samples are
   half near-surface (jittered around cloud points), half uniform over the cube
   ± margin. SIGNED mode is out of scope: VO landmarks carry no free-space
   evidence.
2. **Initialization** (cold cubes): NMS over local extrema of a coarse distance
   grid — positive Gaussians at maxima, negative at minima (the surface),
   random fill as fallback; `p = √0.08 (σ_eff = 8 cm)`, `w = ±0.1`.
3. **Solver**: hand-rolled batched Levenberg-Marquardt with the analytic
   Jacobian of `DynamicGMMCostFunction` (`∂r/∂w = e`, `∂r/∂μ = w·e·v/q²`,
   `∂r/∂p = 2w·e·v²p/q³`, `q = p² + ε`), float64 fitting / float32 storage,
   per-cube damping and accept/reject; 30 iterations cold, 10 warm-started.
4. **Validation**: per-cube MAE against the sample targets; a fit is usable
   when finite and `MAE ≤ mae_threshold_max`. The map's running mean MAE is
   exposed as `mapper.sigma` and feeds the residual weighting.

### Factor graphs (Graphs.py)

Both run in the **world frame only** (the map is world-anchored — never wrap
this backend in a `Local_`-style frame transform).

* `GEDF_Registration` — one scalar row per keypoint:
  `r_i = d̂(T · p_i)` with `p_i` back-projected from `pixel2_uv`/`pixel2_d`.
  Out-of-map rows (`d̂ ≥ 19`) are clamped to a constant `oob_residual = 5.0`
  with zero Jacobian, exactly as in G-EDF-Loc.
* `GEDF_ICP` — the hybrid: 4 rows per keypoint = the standard covariance-
  weighted ICP 3-vector (`T·p_c - p_w`) plus the field row. While the map is
  not ready the field row is inert (zero residual/Jacobian, unit variance), so
  the graph degrades to pure ICP with no shape changes mid-run.
* `Analytic_*` variants supply hand-derived Jacobians in PyPose's 7-wide SE3
  layout: field row `J = [ gᵀ | -gᵀ[q]ₓ | 0 ]` with `q = T·p`, `g = ∇d̂(q)`
  (per-point gradient norm clamped to `max_grad_norm` as solver-side
  robustification). Verified against autograd by `AnalyticModule.verify_jacobian`.

Residual weighting (`field.weighting`):
* `mahalanobis` (default) — `σ_f,i² = gᵢᵀ (R·Σ_obs,i·Rᵀ) gᵢ + max(map MAE, field.sigma)²`,
  putting the scalar field residual on the same m² footing as the ICP rows;
* `fixed` — constant `field.sigma²`.

### `GEDF_PGO` (Optimizer.py)

Implements the four `IOptimizer` methods. Per `_optimize` call (runs in the
spawned worker when `parallel: true`):

1. **insert** the frame pair's new landmarks (safe pre-solve: they are anchored
   at the *previous, already-optimized* keyframe pose, and each landmark set is
   created exactly once — no double insertion);
2. **refit** a budgeted number of dirty cubes (camera-proximity prioritized);
3. cold-start guard (`graph_type: gedf` with a not-ready map returns the
   initial motion unchanged; the hybrid proceeds as pure ICP);
4. **two-stage robust LM** over the selected graph — coarse (Huber δ=3.0) then
   fine (δ=0.5), mirroring G-EDF-Loc's Cauchy coarse→fine schedule — with
   block-diagonal inverse-covariance weights.

The map lives in the optimizer **context**, i.e. entirely inside the worker
process in parallel mode. Only the pose (and, when requested, a bounded map
snapshot for visualization) crosses the result queue.

## 3. Configuration reference

```yaml
optimizer:
  type: GEDF_PGO
  args:
    device: cpu             # solver device (the mapper has its own below)
    vectorize: true
    parallel: true          # spawn worker process (pipelined with the frontend)
    autodiff: false         # false = analytic Jacobians (LM_analytic), recommended
    graph_type: gedf+icp    # "gedf" = pure field registration | "gedf+icp" = hybrid

    map:
      source: online        # "online" = build during the run | "prebuilt" = load .bin
      path: ""              # GDF1 .bin path (only for source: prebuilt; "" = unset -
                            # note the YAML loader maps `null` to an empty namespace)
      insert_keypoints: true    # feed the sparse VO landmarks into the map
      insert_dense: false       # additionally feed the dense mapping points
      min_gaussians: 800        # field factor stays inert until the map holds this many
      online:                   # GEDFConfig - mapper parameters (all optional)
        device: cuda
        cube_size: 1.0          # m
        num_gaussians: 8        # K per cube (fixed; raise for high-curvature scenes)
        sample_points: 1000     # training samples per cube fit
        budget_cubes_per_frame: 8
        mae_target: 0.03        # m
        cov_trace_gate: 0.0675  # m^2 absolute insert gate; 0 disables (use 0 for mono)
        # ... every GEDFConfig field is accepted here; see Config.py for the
        # full list incl. LM iteration counts, refit triggers, dedup voxel,
        # and the CPU-budget knobs documented in its docstring.

    field:
      weighting: mahalanobis    # or "fixed"
      sigma: 0.30               # m - field-error floor. IMPORTANT: this is the main
                                # knob balancing map factor vs ICP. 0.30 for young
                                # online maps; 0.10 is fine for good prebuilt maps.
      oob_value_threshold: 19.0
      oob_residual: 5.0
      max_grad_norm: 10.0

    solver:
      coarse_kernel_delta: 3.0
      coarse_steps: 10
      fine_kernel_delta: 0.5
      fine_steps: 10

    viz:                        # Rerun map visualization (active only with --useRR)
      every: 10                 # log a snapshot every N keyframes; 0 = disabled
      iso: 0.10                 # keep sampled points with field value < iso (m)
      resolution: 0.10          # sampling grid (m)
      max_points: 100000
```

Ready-made configs:
* `Config/Experiment/MACVO/MACVO_GEDF.yaml` — stereo (Performant base).
* `Config/Experiment/MACVO/MACVO_GEDF_DAv3.yaml` — monocular DepthAnythingV3
  base. Mono notes: the absolute insert gate is disabled (mono covariances are
  large in absolute terms); reliable-pixel selection is done upstream by the
  selectors via their `depth_cov_rel` (median-relative depth-cov filter) keys.

## 4. Visualization

**Live** (same Rerun recording as the rest of MAC-VO): run with `--useRR`; the
map appears as a near-surface point cloud at `/world/gedf_map`, refreshed every
`viz.every` keyframes *once the map is ready* (with `min_gaussians: 800` and
`every: 10` the first snapshot lands around keyframe 20). Snapshots are sampled
in the worker and shipped over the result queue; there is zero overhead when
Rerun is off.

**Offline** (any GDF1 map — from the C++ trainer or `export_gdf1`):

```bash
python Scripts/VisualizeGEDF.py --map path/to/map.bin [--iso 0.05] [--resolution 0.05] [--save out.rrd]
```

## 5. Results

30-frame TartanAir `abf001` benchmark (same sequence, same frontend):

| Backend | ATE | RTE | ROE | RPE |
|---|---|---|---|---|
| `TwoFrame_PGO` (ICP baseline) | 0.0034 | 0.0032 | 0.0456 | 0.0034 |
| `GEDF_PGO` (`gedf+icp`, online map) | **0.0036** | **0.0029** | **0.0411** | **0.0031** |

Parallel and sequential modes agree to 4 decimal places. Runtime overhead vs
the ICP baseline is ~10-15% per keyframe (GPU mapper); the drift-resisting
benefit of the map factor is expected on longer sequences with revisits, which
a 30-frame drift-free clip cannot show.

Weighting sensitivity: with `field.sigma: 0.10` on the young online map the
field factor dominated ICP and *degraded* ATE ~7x; `0.30` + `min_gaussians:
800` is the validated default. Lower `sigma` only for good prebuilt maps.

## 6. Tests

`Scripts/UnitTest/test_gedf_field.py` (format + query math parity, analytic
gradients vs autograd, OOB, blend continuity), `test_gedf_mapper.py` (EDT
exactness, fit quality on synthetic shapes, incremental lifecycle, export
round-trips, surface sampling, CPU/CUDA equivalence), `test_gedf_registration.py`
(Jacobian verification, pose recovery in all four graph/autodiff combinations,
cold start, snapshot plumbing, registry integration). Local env note: run with
`python -m pytest ... -o addopts=""` (the pinned jaxtyping/typeguard combo
breaks collection otherwise).

## 7. Known limitations / future work

* Fixed `K` Gaussians per cube (the C++ adaptive `K ∈ {8,16,32}` loop is not
  ported); raise `num_gaussians` for high-curvature scenes.
* PURE (unsigned) EDT only — no SIGNED mode.
* The online map absorbs early trajectory drift into itself (landmarks are
  anchored at estimated poses); the field factor enforces map-consistency, not
  global ground-truth accuracy.
* No empty-neighbor-cube fitting (would widen the attraction basin under large
  drift), no covariance-weighted training samples, no loop-closure use of the
  field, no background fitting thread.
