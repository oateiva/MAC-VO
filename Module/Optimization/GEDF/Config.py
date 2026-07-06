from dataclasses import dataclass, fields
from types import SimpleNamespace


@dataclass
class GEDFConfig:
    """
    Configuration for the online G-EDF mapper (Gaussian Euclidean Distance Field).

    Field semantics follow the reference implementation in the G-EDF repository
    (robotics-upo). Defaults mirror `G-EDF/config/config.yaml` where applicable.

    CPU-budget knobs: if fitting is too slow without CUDA, reduce
    `budget_cubes_per_frame` (e.g. 4), `sample_points` (e.g. 600),
    `lm_iters_cold` (e.g. 15) and/or set `fit_dtype: float32`.
    """
    device: str = "cuda"
    cube_size: float = 1.0
    margin: float = 0.25              # blending + training margin (= margin_pure)
    halo: float = 0.50                # EDT context reach = margin + edt_extension
    num_gaussians: int = 8            # fixed K per cube
    sample_points: int = 1000         # training samples per cube
    surface_sample_frac: float = 0.5
    surface_jitter: float = 0.12      # m, jitter around source points for near-surface samples
    init_voxel_size: float = 0.10     # NMS init grid resolution
    nms_radius: int = 2               # voxels (suppression + peak radius)
    init_sigma_param: float = 0.2828427  # sqrt(0.08) -> sigma_eff = p^2 = 0.08 m
    init_weight: float = 0.1
    importance_weighting: bool = False   # exp(-5|d|) residual weighting
    # Batched Levenberg-Marquardt
    lm_iters_cold: int = 30
    lm_iters_warm: int = 10
    lm_lambda_init: float = 1e-3
    lm_lambda_up: float = 4.0
    lm_lambda_down: float = 3.0
    lm_step_tol: float = 1e-4
    fit_dtype: str = "float64"
    # Validation
    mae_target: float = 0.03
    mae_threshold_max: float = 0.30
    cold_restart_mae_factor: float = 2.0
    # Incremental lifecycle
    min_points_fit: int = 10
    refit_min_new: int = 20           # refit when >= this many new points since last fit
    refit_growth: float = 0.30        # ... or >= this fraction growth
    ctx_refit_min_new: int = 40       # ... or this many new halo points from neighbor inserts
    budget_cubes_per_frame: int = 8   # max cube fits per refit() call
    max_points_per_cube: int = 512    # cap (random subsample beyond)
    dedup_voxel: float = 0.025        # m, fine-voxel dedup of inserted points
    cov_trace_gate: float = 0.0675    # m^2 (= 3 * 0.15^2); 0 disables gating
    camera_dist_tau: float = 4.0      # m, distance prior scale in refit priority
    oob_value: float = 20.0           # sentinel for empty / out-of-map regions

    @classmethod
    def from_namespace(cls, ns: SimpleNamespace | None) -> "GEDFConfig":
        """Build a config from a (possibly partial) SimpleNamespace; unknown keys raise."""
        cfg = cls()
        if ns is None:
            return cfg
        known = {f.name: f.type for f in fields(cls)}
        for key, value in vars(ns).items():
            if key not in known:
                raise KeyError(f"Unknown G-EDF mapper config key: '{key}'. "
                               f"Valid keys: {sorted(known.keys())}")
            # YAML often parses e.g. `1e-3` as str and small floats as int; coerce numerics.
            default = getattr(cfg, key)
            if isinstance(default, bool):
                assert isinstance(value, bool), f"Config key '{key}' expects bool, got {value!r}"
            elif isinstance(default, float) and isinstance(value, (int, str)):
                value = float(value)
            elif isinstance(default, int) and isinstance(value, float) and value.is_integer():
                value = int(value)
            assert isinstance(value, type(default)), \
                f"Config key '{key}' expects {type(default).__name__}, got {value!r}"
            setattr(cfg, key, value)
        assert 0.0 < cfg.margin < 0.5 * cfg.cube_size, \
            "G-EDF blending margin must satisfy 0 < margin < cube_size / 2"
        return cfg
