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

class Rerun_Visualizer:
    func_mode: T.ClassVar[dict[str, T_Mode | T.Literal["default"]]] = dict()
    default_mode: T.ClassVar[T_Mode] = "none"

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
    def log_keypoints(rerun_path: str, match_obs: MatchObs):
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
                colors=match_obs.data["pixel2_uv_cov"]*255,
                radii=radii
            )
        )
