from functools import wraps
from Utility.PrettyPrint import Logger
import typing as T
import pypose as pp
import numpy  as np
import torch
from flow_vis import flow_to_color
from Module.Map.Template import MatchObs

try:
    import rerun as rr
except ImportError:
    rr = None


T_Mode    = T.Literal["none", "rerun"]
T_Input   = T.ParamSpec("T_Input")
T_Output  = T.TypeVar("T_Output")


# NOTE: Since rerun does not ensure compatibilty between different versions,
#       We explicitly constrain the version of rerun sdk
if (rr is not None) and (rr.__version__ <= "0.24.0"):
    Logger.write("warn", f"Please re-install rerun_sdk to have version of 0.24.0. Current version is {rr.__version__}")
    rr = None


# --- Pure numeric helpers (no rerun/matplotlib dependency) ------------------
# Kept free of `rr`/matplotlib imports so they are unit-testable without a
# rerun install; matplotlib is imported locally inside `age_colors` only,
# mirroring the `log_points` cov_mode="color" branch below.

_KF_PALETTE = np.array([
    [ 31, 119, 180], [255, 127,  14], [ 44, 160,  44], [214,  39,  40],
    [148, 103, 189], [140,  86,  75], [227, 119, 194], [127, 127, 127],
    [188, 189,  34], [ 23, 190, 207],
], dtype=np.uint8)   # matplotlib tab10, hardcoded to avoid importing matplotlib here


def kf_link_segments(
    kf_ids: torch.Tensor, frame_ids: torch.Tensor, positions: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse (keyframe, frame) re-observation rows into one 3D line segment per
    unique (keyframe, frame) pair, colored per-keyframe via `_KF_PALETTE`
    (color index = rank of the keyframe id among all keyframe ids seen so far,
    mod 10 - stable across a run since keyframe indices only ever grow).

    Malformed ids (< 0, >= len(positions)) and self-pairs (kf_id == frame_id)
    are dropped before dedup. A pair is also dropped if either endpoint
    position is non-finite (NaN/inf) - e.g. a graph-gauge `pose_xyz` row for a
    frame index whose pose key doesn't (yet/anymore) `exists()` in the
    estimate - and a keyframe marker is dropped the same way if its own
    position is non-finite. Empty input -> empty outputs of the right
    shape/dtype so callers never need to special-case the first frames.

    Returns (segments (P,2,3) f32, seg_colors (P,3) u8, kf_pos (K,3) f32, kf_colors (K,3) u8).
    """
    empty_seg = np.zeros((0, 2, 3), dtype=np.float32)
    empty_col = np.zeros((0, 3), dtype=np.uint8)
    empty_pos = np.zeros((0, 3), dtype=np.float32)

    if kf_ids.numel() == 0 or frame_ids.numel() == 0:
        return empty_seg, empty_col, empty_pos, empty_col

    n_pos = positions.shape[0]
    kf_ids    = kf_ids.detach().cpu().long()
    frame_ids = frame_ids.detach().cpu().long()

    valid = (
        (kf_ids >= 0) & (kf_ids < n_pos) &
        (frame_ids >= 0) & (frame_ids < n_pos) &
        (kf_ids != frame_ids)
    )
    kf_ids, frame_ids = kf_ids[valid], frame_ids[valid]
    if kf_ids.numel() == 0:
        return empty_seg, empty_col, empty_pos, empty_col

    pairs = torch.unique(torch.stack([kf_ids, frame_ids], dim=1), dim=0)   # (P, 2)
    uniq_kf, inverse = torch.unique(pairs[:, 0], sorted=True, return_inverse=True)

    positions_np = positions.detach().cpu().numpy()
    pairs_np     = pairs.numpy()
    segments   = positions_np[pairs_np].astype(np.float32)                # (P, 2, 3)
    seg_colors = _KF_PALETTE[(inverse % 10).numpy()]

    seg_finite = np.isfinite(segments).all(axis=(1, 2))
    segments, seg_colors = segments[seg_finite], seg_colors[seg_finite]

    kf_pos    = positions_np[uniq_kf.numpy()].astype(np.float32)
    kf_colors = _KF_PALETTE[(torch.arange(uniq_kf.numel()) % 10).numpy()]

    kf_finite = np.isfinite(kf_pos).all(axis=1)
    kf_pos, kf_colors = kf_pos[kf_finite], kf_colors[kf_finite]

    return segments, seg_colors, kf_pos, kf_colors


def compute_track_ages(pixel2_uv: torch.Tensor, n_obs_of: dict[tuple[int, int], int]) -> np.ndarray:
    """
    Per-row track age (observation count) for `pixel2_uv` rows, looked up by
    the tracker's own key convention: `np.rint`'d integer (u, v) - NOT Python
    `round()`, which disagrees with numpy's round-half-to-even on .5 boundaries.
    Rows with no matching key default to age 1 (unobserved-by-tracker / fresh).
    Empty input -> empty int64 array.
    """
    if pixel2_uv.numel() == 0:
        return np.zeros((0,), dtype=np.int64)
    uv = pixel2_uv.detach().cpu().numpy()
    keys = np.rint(uv).astype(np.int64)
    return np.array([n_obs_of.get((int(u), int(v)), 1) for u, v in keys], dtype=np.int64)


def age_colors(ages: np.ndarray, cap: int = 30) -> np.ndarray:
    """Plasma-colormap track ages into (N,3) uint8; ages are clipped to [1, cap]
    (matplotlib imported locally, same pattern as `log_points` cov_mode="color")."""
    if ages.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    normalized = Normalize(vmin=1, vmax=cap, clip=True)(ages.astype(np.float64))
    colormap = plt.cm.plasma   # type: ignore
    return (colormap(normalized)[..., :3] * 255).astype(np.uint8)


def update_landmark_obs(lm_n_obs: dict[int, int], live: T.Iterable[tuple[int, int]]) -> dict[int, int]:
    """
    Max-merge `(lm_key, n_obs)` pairs from the currently-live tracks into the
    persistent `lm_key -> n_obs` table (mutated in place and returned).
    `n_obs` never decreases for a live landmark within a run, but this is a
    max-merge (not overwrite) defensively; landmarks that drop out of the live
    set (dead/pruned) simply stop being updated and keep their last count.
    """
    for lm_key, n_obs in live:
        prev = lm_n_obs.get(lm_key)
        if prev is None or n_obs > prev:
            lm_n_obs[lm_key] = n_obs
    return lm_n_obs


def update_landmark_colors(
    lm_color: dict[int, tuple[int, int, int]],
    pixel2_uv: torch.Tensor,
    colors: torch.Tensor,
    lm_key_of: dict[tuple[int, int], int],
) -> dict[int, tuple[int, int, int]]:
    """
    Set-if-absent: for each `pixel2_uv` row, join to a live track via the
    tracker's own `np.rint`'d integer (u, v) key convention (see
    `compute_track_ages`), resolve that track to its `lm_key`, and - only if
    the landmark has no stored color yet - stash the row's RGB from `colors`
    (row-aligned with `pixel2_uv`). A landmark keeps its first-seen color for
    the rest of the run (stable across frames), so later sightings never
    overwrite it. Rows with no matching track key are skipped. Mutates and
    returns `lm_color`; empty input is a no-op.
    """
    if pixel2_uv.numel() == 0 or colors.numel() == 0:
        return lm_color
    uv = pixel2_uv.detach().cpu().numpy()
    rgb = colors.detach().cpu().numpy()
    keys = np.rint(uv).astype(np.int64)
    for (u, v), color in zip(keys, rgb):
        lm_key = lm_key_of.get((int(u), int(v)))
        if lm_key is None or lm_key in lm_color:
            continue
        lm_color[lm_key] = (int(color[0]), int(color[1]), int(color[2]))
    return lm_color


def landmark_color_array(
    keys: list[int] | np.ndarray,
    lm_color: dict[int, tuple[int, int, int]],
    fallback: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """(M,3) uint8 array of `lm_color[key]` in `keys` order, `fallback` for
    keys with no stored color yet. Empty-safe: `[]` -> `(0, 3)` uint8 array."""
    if len(keys) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.array([lm_color.get(int(k), fallback) for k in keys], dtype=np.uint8)


def filter_persistent_landmarks(lm_n_obs: dict[int, int], min_obs: int = 3) -> tuple[list[int], np.ndarray]:
    """
    Keys with `n_obs >= min_obs`, sorted ascending (a stable order since
    landmark ids only ever grow), paired with an aligned `(M,)` int64 array of
    their observation counts. Empty-safe: `({}, ...)` -> `([], np.zeros((0,), int64))`.
    """
    keys = sorted(k for k, n in lm_n_obs.items() if n >= min_obs)
    counts = np.array([lm_n_obs[k] for k in keys], dtype=np.int64)
    return keys, counts


class Rerun_Visualizer:
    func_mode: T.ClassVar[dict[str, T_Mode | T.Literal["default"]]] = dict()
    default_mode: T.ClassVar[T_Mode] = "none"
    _series_styled: T.ClassVar[set[str]] = set()

    @staticmethod
    def init_connect(application_id: str, save_rrd: str | None = None):
        """
        Spawn a live viewer; when `save_rrd` is given, ALSO record the stream to
        that file from the very first log call. NOTE: a trailing `rr.save(...)`
        after streaming does NOT work - the file sink only records data logged
        after it is attached, so late saves produce empty recordings.
        """
        assert rr is not None, "Can't initialize rerun since rerun is not installed or have incorrect version."
        rr.init(application_id)
        if save_rrd is None:
            rr.spawn()
        else:
            rr.spawn(connect=False)
            rr.set_sinks(rr.GrpcSink(), rr.FileSink(save_rrd))
        rr.log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)

    @staticmethod
    def init_save(application_id: str, save_rrd: str):
        """Record to an .rrd file only (no viewer)."""
        assert rr is not None, "Can't initialize rerun since rerun is not installed or have incorrect version."
        rr.init(application_id)
        rr.save(save_rrd)
        rr.log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)

    @staticmethod
    def set_fn_mode(func: T.Callable[T.Concatenate[str, T_Input], None], mode: T_Mode | T.Literal["default"]):
        Rerun_Visualizer.func_mode[func.__name__] = mode

    @staticmethod
    def get_fn_mode(func: T.Callable[T.Concatenate[str, T_Input], None]) -> T_Mode:
        assert func.__name__ in Rerun_Visualizer.func_mode
        func_mode = Rerun_Visualizer.func_mode[func.__name__]
        if func_mode == "default": return Rerun_Visualizer.default_mode
        return func_mode

    @staticmethod
    def register(func: T.Callable[T.Concatenate[str, T_Input], None]) -> T.Callable[T.Concatenate[str, T_Input], None]:
        @wraps(func)
        def implement(rerun_path: str, *args: T_Input.args, **kwargs: T_Input.kwargs) -> None:
            if func.__name__ not in Rerun_Visualizer.func_mode:
                Rerun_Visualizer.func_mode[func.__name__] = "default"

            func_mode = Rerun_Visualizer.get_fn_mode(func)
            match func_mode:
                case "none": return None
                case "rerun": func(rerun_path, *args, **kwargs)
        return implement

    @register
    @staticmethod
    def log_trajectory(rerun_path: str, trajectory: pp.LieTensor | torch.Tensor, **kwargs):
        assert rr is not None
        if not isinstance(trajectory, pp.LieTensor):
            trajectory = pp.SE3(trajectory)

        position = trajectory.translation().detach().cpu().numpy()
        from_pos = position[:-1]
        to_pos = position[1:]
        rr.log(rerun_path, rr.LineStrips3D(np.stack([from_pos, to_pos], axis=1), **kwargs))

    @register
    @staticmethod
    def log_path(rerun_path: str, positions: np.ndarray, color: tuple[int, int, int] | None = None, radius: float = 0.01):
        """
        Log a polyline through `positions` (N,3) as consecutive-segment
        `rr.LineStrips3D` - the plain-numpy counterpart to `log_trajectory`
        (which takes a `pp.LieTensor`/`torch.Tensor` pose sequence). Useful
        for trajectories reconstructed from a raw position array (e.g. a
        gtsam `Values` estimate) rather than a pose tensor. No-ops on < 2
        points (a single point has no segment to draw).
        """
        assert rr is not None
        if positions.shape[0] < 2:
            return
        from_pos = positions[:-1]
        to_pos = positions[1:]
        rr.log(
            rerun_path,
            rr.LineStrips3D(
                np.stack([from_pos, to_pos], axis=1),
                colors=[color] if color is not None else None,
                radii=[radius],
            ),
        )

    @register
    @staticmethod
    def log_camera(rerun_path: str, pose: pp.LieTensor | torch.Tensor, K: torch.Tensor, **kwargs):
        assert rr is not None
        cx = K[0][2].item()
        cy = K[1][2].item()

        if not isinstance(pose, pp.LieTensor):
            pose = pp.SE3(pose)
        frame_position = pose.translation().detach().cpu().numpy()
        frame_rotation = pose.rotation().detach().cpu().numpy()

        rr.log(
            "/".join(rerun_path.split("/")[:-1]),
            rr.Transform3D(
                translation=frame_position,
                rotation=rr.datatypes.Quaternion(xyzw=frame_rotation),
            ),
        )
        rr.log(
            rerun_path,
            rr.Pinhole(
                resolution=[cx * 2, cy * 2],
                image_from_camera=K.detach().cpu().numpy(),
                camera_xyz=rr.ViewCoordinates.FRD,
                image_plane_distance=0.25
            ),
        )

    @register
    @staticmethod
    def log_points(rerun_path: str, position: torch.Tensor, color: torch.Tensor | None, cov_Tw: torch.Tensor | None, cov_mode: T.Literal["none", "axis", "sphere", "color"]="sphere"):
        assert rr is not None
        rr.log(
            rerun_path,
            rr.Points3D(positions=position, colors=color.detach().cpu().numpy() if (color is not None) else None)
        )

        match cov_Tw, cov_mode:
            case None, _: return
            case _, "none": return
            case _, "axis":
                eigen_val, eigen_vec = torch.linalg.eig(cov_Tw)
                eigen_val, eigen_vec = eigen_val.real, eigen_vec.real

                delta = position.repeat(1, 3, 1).reshape(-1, 3)
                eigen_vec_Tw = eigen_vec.transpose(-1, -2).reshape(-1, 3)
                eigen_val = eigen_val.unsqueeze(-1).repeat(1, 1, 3).reshape(-1, 3)
                eigen_vec_Tw = eigen_vec_Tw * eigen_val.sqrt()
                eigen_vec_Tw_a = delta + .1 * eigen_vec_Tw
                eigen_vec_Tw_b = delta - .1 * eigen_vec_Tw
                rr.log(
                    rerun_path + "/cov",
                    rr.LineStrips3D(
                        torch.stack([eigen_vec_Tw_a, eigen_vec_Tw_b], dim=1).numpy(),
                        radii=[0.003],
                        colors=color.unsqueeze(0).repeat(3, 1, 1).reshape(-1, 3) if (color is not None) else None
                    ),
                )
            case _, "sphere":
                radii  = (cov_Tw.det().sqrt() * 1e2).clamp(min=0.03, max=0.5)
                rr.log(
                    rerun_path + "/cov",
                    rr.Points3D(positions=position, colors=color.detach().cpu().numpy() if (color is not None) else None,
                                radii=radii)
                )
            case _, "color":
                import matplotlib.pyplot as plt
                from matplotlib.colors import Normalize
                cov_value = cov_Tw.det()
                cov_det_normalized = Normalize(vmin=0, vmax=cov_value.quantile(0.99).item())(cov_value)
                colormap = plt.cm.plasma    #type: ignore
                c = colormap(cov_det_normalized)[..., :3]
                rr.log(rerun_path + "/cov", rr.Points3D(position, colors=c))

    @register
    @staticmethod
    def log_kf_links(rerun_path: str, kf_ids: torch.Tensor, frame_ids: torch.Tensor, positions: torch.Tensor):
        """
        Log accumulated keyframe -> frame re-observation lines (see
        `kf_link_segments`) at `rerun_path`, plus keyframe marker points at
        `{rerun_path}/keyframes`. Rebuilt from the full kf_match edge tables
        each call, so endpoints track updated pose estimates; accumulation
        over the run is free since kf_match is append-only.
        """
        assert rr is not None
        segments, seg_colors, kf_pos, kf_colors = kf_link_segments(kf_ids, frame_ids, positions)
        rr.log(rerun_path, rr.LineStrips3D(segments, colors=seg_colors, radii=[0.002]))
        rr.log(rerun_path + "/keyframes", rr.Points3D(kf_pos, colors=kf_colors, radii=[0.03]))

    @register
    @staticmethod
    def log_gedf_map(rerun_path: str, points: torch.Tensor, dist: torch.Tensor | None,
                     radius: float = 0.02):
        """
        Log a G-EDF near-surface sample cloud (see GEDFMapper.sample_surface):
        points (M,3) colored by their field value (viridis, near-surface = dark).
        """
        assert rr is not None
        if points.numel() == 0:
            return
        colors = None
        if dist is not None and dist.numel() > 0:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            d = dist.detach().cpu().numpy()
            normalized = Normalize(vmin=0.0, vmax=max(float(d.max()), 1e-6))(d)
            colormap = plt.cm.viridis   # type: ignore
            colors = colormap(normalized)[..., :3]
        rr.log(rerun_path, rr.Points3D(points.detach().cpu().numpy(),
                                       colors=colors, radii=[radius]))

    @register
    @staticmethod
    def log_gedf_gaussians(rerun_path: str, means: torch.Tensor, sigmas: torch.Tensor,
                           weights: torch.Tensor, cube_mae: torch.Tensor,
                           n_sigma: float = 1.0):
        """
        Log G-EDF GMM components (see GEDFMapper.gaussians) as axis-aligned
        wireframe ellipsoids with half-sizes = n_sigma * sigmas.

        Confidence encoding:
        - hue  = per-cube fit MAE via cividis (dark = low MAE = trustworthy;
          deliberately not viridis, which log_gedf_map uses for distance),
        - alpha = normalized |weight| (faint = low-amplitude component),
        - sign  = entity split: positive components at `{path}/pos`, negative
          ("carving") components at `{path}/neg` in fixed magenta — separate
          entities so each population can be toggled in the viewer.
        Empty populations are logged as empty so stale instances clear.

        Ellipsoid axes are the world NED axes (the field model is diagonal, no
        rotation); size is each component's fitted spatial support — the scale
        of the field feature it encodes, NOT uncertainty, importance, or the
        cube grid. See Module/Optimization/GEDF/README.md §4 ("How to read the
        ellipsoids").
        """
        assert rr is not None
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        mu = means.detach().cpu().numpy()
        half = n_sigma * sigmas.detach().cpu().numpy()
        w = weights.detach().cpu().numpy()
        mae = cube_mae.detach().cpu().numpy()

        if mu.shape[0] == 0:
            for sub in ("/pos", "/neg"):
                rr.log(rerun_path + sub, rr.Ellipsoids3D(half_sizes=np.zeros((0, 3))))
            return

        w_abs = np.abs(w)
        alpha = 0.3 + 0.7 * np.clip(w_abs / max(float(np.quantile(w_abs, 0.95)), 1e-9), 0., 1.)
        norm = Normalize(vmin=0.0, vmax=max(float(np.quantile(mae, 0.95)), 1e-6), clip=True)
        rgba = plt.cm.cividis(norm(mae))    # type: ignore
        rgba[..., 3] = alpha
        neg = w < 0
        rgba[neg, :3] = np.array([200, 30, 200]) / 255.0
        rgba_u8 = (rgba * 255).round().astype(np.uint8)

        for sub, mask in (("/pos", ~neg), ("/neg", neg)):
            rr.log(rerun_path + sub, rr.Ellipsoids3D(
                centers=mu[mask], half_sizes=half[mask], colors=rgba_u8[mask],
                fill_mode=rr.components.FillMode.MajorWireframe))

    @register
    @staticmethod
    def log_gedf_cubes(rerun_path: str, centers: torch.Tensor, valid: torch.Tensor,
                       mae: torch.Tensor, cube_size: float):
        """
        Log the G-EDF sparse cube grid (see GEDFMapper.cubes) as wireframe
        boxes of edge `cube_size`. Fitted (valid) cubes are colored by their
        fit MAE on cividis — the same scale the ellipsoid layer uses — and
        not-yet-fitted cubes are faint gray, so coverage and fit quality are
        visible at a glance.
        """
        assert rr is not None
        c = centers.detach().cpu().numpy()
        if c.shape[0] == 0:
            rr.log(rerun_path, rr.Boxes3D(half_sizes=np.zeros((0, 3))))
            return
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        v = valid.detach().cpu().numpy().astype(bool)
        m = mae.detach().cpu().numpy()
        rgba = np.full((c.shape[0], 4), [0.5, 0.5, 0.5, 0.25])   # pending: faint gray
        if v.any():
            norm = Normalize(vmin=0.0, vmax=max(float(np.quantile(m[v], 0.95)), 1e-6),
                             clip=True)
            rgba[v] = plt.cm.cividis(norm(m[v]))    # type: ignore
            rgba[v, 3] = 0.8
        rgba_u8 = (rgba * 255).round().astype(np.uint8)
        rr.log(rerun_path, rr.Boxes3D(
            centers=c, half_sizes=np.full((c.shape[0], 3), cube_size / 2.0),
            colors=rgba_u8))

    @register
    @staticmethod
    def log_image(rerun_path: str, image: torch.Tensor | np.ndarray):
        assert rr is not None
        if isinstance(image, torch.Tensor): np_image = image.cpu().numpy()
        else: np_image = image

        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)
        rr.log(rerun_path, rr.Image(np_image).compress())

    @register
    @staticmethod
    def log_flow(rerun_path: str, flow: torch.Tensor | np.ndarray):
        assert rr is not None
        if isinstance(flow, torch.Tensor): np_flow = flow.cpu().numpy()
        else: np_flow = flow
        # Convert flow to color for visualization
        # We use the standard flow visualization method where hue represents
        # the flow direction,
        # and saturation represents the flow magnitude.
        # The color wheel corresponds to the one in
        # http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
        flow_img = flow_to_color(np_flow, convert_to_bgr=False)
        rr.log(rerun_path, rr.Image(flow_img).compress())

    @register
    @staticmethod
    def log_covariance(rerun_path: str, covariance: torch.Tensor):
        assert rr is not None
        # If the flow covariance has two channels (e.g., shape [H, W, 2]), visualize each channel separately
        if covariance.ndim == 4 and covariance.shape[1] > 1:
            # To visualize the covariance of the optical flow, we compute the determinant of the covariance matrix
            # and take the logarithm to enhance visibility. The determinant gives a measure of the uncertainty
            # associated with the flow vectors.
            # We then use a colormap to represent the uncertainty visually.
            flow_cov_det = (covariance[:, 0] * covariance[:, 1] - covariance[:, 2].square())[0].log10()
            rr.log(rerun_path, rr.DepthImage(flow_cov_det, colormap=4))
        if covariance.ndim == 4 and covariance.shape[1] == 1:
            rr.log(rerun_path, rr.DepthImage(covariance, colormap=4))

    @register
    @staticmethod
    def log_depth(rerun_path: str, depth: torch.Tensor):
        assert rr is not None
        rr.log(rerun_path, rr.DepthImage(depth, colormap=4))

    @register
    @staticmethod
    def log_keypoints(rerun_path: str, match_obs: MatchObs, colors: np.ndarray | None = None):
        assert rr is not None
        # Generate the radii based on the uncertainty (covariance) of the keypoints
        # The radii size is inversely proportional to the uncertainty
        # radii = 1 / (cov_u**2 + cov_v**2), normalized to [0, 4]
        radii = 1 / \
            (match_obs.data["pixel2_uv_cov"][:, 0]**2+match_obs.data["pixel2_uv_cov"][:, 1]**2)
        if radii.numel() > 0:
            radii = 4 * radii / radii.max(dim=0).values
        rr.log(
            rerun_path,
            rr.Points2D(
                match_obs.data["pixel2_uv"],
                colors=colors if colors is not None else match_obs.data["pixel2_uv_cov"]*255,
                radii=radii
            )
        )

    @register
    @staticmethod
    def log_scalar(rerun_path: str, value: float, name: str | None = None, color: tuple[int, int, int] | None = None):
        """
        Log one scalar sample at `rerun_path`. On the first call for a given
        path, also logs a static `rr.SeriesLines` styling entry (legend name /
        color) so per-frame `rr.Scalars` calls need not repeat it.
        """
        assert rr is not None
        if rerun_path not in Rerun_Visualizer._series_styled:
            Rerun_Visualizer._series_styled.add(rerun_path)
            rr.log(
                rerun_path,
                rr.SeriesLines(
                    names=[name] if name is not None else None,
                    colors=[color] if color is not None else None,
                ),
                static=True,
            )
        rr.log(rerun_path, rr.Scalars(float(value)))
