#Evaluate the model on the validation set
import argparse
import torch
import typing as T
from Utility.Visualize import fig_plt, rr_plt
import pypose as pp
import random
import numpy as np
from pathlib import Path

from Module.Network.FlowFormerCov import build_flowformer
from Module.Network.FlowFormer.configs.submission import get_cfg
from Utility.Config import load_config
from Utility.PrettyPrint import ColoredTqdm
from typing import Dict
from DataLoader import SequenceBase, Frame, smart_transform
T_SensorFrame = T.TypeVar("T_SensorFrame", bound=Frame)
from Utility.Utils import reflect_torch_dtype
from Module.Network.ModelSelector import build_depth_model
from Module.Network.Depth.base import DepthModelProtocol
from DataLoader import CameraData

def set_determinism(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_nc(nc: torch.Tensor, max_keep: int, rng: torch.Generator) -> torch.Tensor:
    # nc: 1D tensor on CPU
    nc = nc[torch.isfinite(nc)]
    if nc.numel() <= max_keep:
        return nc
    idx = torch.randint(0, nc.numel(), (max_keep,), generator=rng)
    return nc[idx]


class ConformalStatistics:
    def __init__(self, args):

        cfg, self.cfg_dict = load_config(Path(args.config))

        self.data_cfgs = self.cfg_dict["Data"]

        self.isinitialized = False
        self.prev_keyframe = None
        self.initialize_models(cfg.Frontend.args)

        # Calibration results
        self.flow_q90: float | None = None
        self.depth_q90: float | None = None

        rr_plt.default_mode = "rerun"
        rr_plt.init_connect("ConformalStatistics")

    def initialize_models(self, cfg):
        # Flow model
        flowformer_config = get_cfg()
        flow_model = build_flowformer(
            flowformer_config,
            reflect_torch_dtype(cfg.enc_dtype),
            reflect_torch_dtype(cfg.dec_dtype)
        )
        flow_model.eval().to(cfg.device)
        ckpt = torch.load(cfg.flow.weight, map_location=cfg.device, weights_only=True)
        flow_model.load_ddp_state_dict(ckpt)

        # Monodepth model
        monodepth_name = cfg.monodepth.type
        monodepth_args = vars(cfg.monodepth.args)
        monodepth_model = build_depth_model(monodepth_name, **monodepth_args)
        monodepth_model.deepodo_initialize(cfg.monodepth.args)
        monodepth_model.eval().to(cfg.device)

        self.model: Dict[str, torch.nn.Module | DepthModelProtocol] = {
            "flow_model": flow_model,
            "monodepth_model": monodepth_model,
        }

    def initialize(self, frame0: Frame):
        self.prev_keyframe = frame0
        self.isinitialized = True

    # ---------- Depth ----------
    def estimate_depth(self, frame: CameraData):
        """
        Return depth prediction and DEPTH VARIANCE (not stddev).
        depth_output.cov is assumed to be "depth confidence/precision-like".
        If your depth_output already returns variance, adjust accordingly.
        """
        eps = 1e-8
        depth_output = self.model["monodepth_model"].deepodo_inference(frame)

        depth = depth_output.depth

        # depth_output.cov is already a variance
        var_depth = depth_output.cov

        sky_mask = depth_output.mask

        return depth, var_depth, sky_mask

    # ---------- Flow ----------
    def estimate_flow(self, frame1: CameraData, frame2: CameraData):
        """
        Return flow prediction and FLOW VARIANCE per component (2 channels, px^2).
        Your flow model already returns exp(cov_pre*2) -> variance, per channel.
        """
        img1 = frame1.imageL.cuda()[0:1, :, :, :]
        img2 = frame2.imageL.cuda()[0:1, :, :, :]

        flow, var_uv = self.model["flow_model"].inference(img1, img2)
        # var_uv is expected shape: (1, 2, H, W) or (2, H, W)

        return flow, var_uv

    def estimate_pair(self, frame1: CameraData, frame2: CameraData):
        depth, var_depth, sky_mask = self.estimate_depth(frame2)
        flow, var_uv = self.estimate_flow(frame1, frame2)
        return depth, var_depth, sky_mask, flow, var_uv

    # ---------- Errors ----------
    def flow_error_sq(self, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        """
        Returns per-pixel squared magnitude error: ||e||^2 = du^2 + dv^2
        Shape: (N_pixels,)
        """
        # pred_flow and gt_flow should be (1,2,H,W)
        e = pred_flow - gt_flow
        e2 = (e[:, 0] ** 2) + (e[:, 1] ** 2)
        return e2.flatten()

    def depth_error_sq(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor, sky_mask: torch.Tensor, var_depth: torch.Tensor) -> torch.Tensor:
        """
        Returns per-pixel squared error: (d - d_gt)^2, only for valid (non-sky) pixels.
        Shape: (N_valid_pixels,)
        """
        # Set pred_depth and gt_depth to inf where sky_mask is True
        mask = sky_mask.bool()[:, 0, :, :].unsqueeze(0)  # shape (1,1,H,W)
        valid_mask = ~sky_mask[:, 0, :, :].unsqueeze(0)

        ratio = (gt_depth[valid_mask] / (pred_depth[valid_mask] + 1e-8)).clamp(0.1, 10.0)
        s = ratio.median()
        pred_depth = s * pred_depth
        var_depth = s**2 * var_depth

        pred_depth_inf = pred_depth.clone()
        gt_depth_inf = gt_depth.clone()
        pred_depth_inf[mask] = float('inf')
        gt_depth_inf[mask] = float('inf')

        e2 = (pred_depth_inf - gt_depth_inf) ** 2
        # Only keep non-sky pixels (sky_mask == False)

        e2_valid = e2[valid_mask]
        # print(f"[Depth Prediction]   min: {pred_depth_inf.min().item():.4f}, median: {pred_depth_inf.median().item():.4f}, max: {pred_depth_inf.max().item():.4f}")
        # print(f"[Depth Ground Truth] min: {gt_depth_inf.min().item():.4f}, median: {gt_depth_inf.median().item():.4f}, max: {gt_depth_inf.max().item():.4f}")
        # print(f"[Abs Depth Error]    median: {e2_valid.median().item():.4f}, mean: {e2_valid.mean().item():.4f}, max: {e2_valid.max().item():.4f}")

        # rerun logging for debugging
        K = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
        rr_plt.log_camera("/world/cam", pp.SE3([0,0,0,0,0,0,0]), K)
        rr_plt.log_depth("/world/cam/depth_prediction", pred_depth_inf)
        rr_plt.log_depth("/world/cam/depth_gt", gt_depth_inf)
        rr_plt.log_depth("/world/cam/depth_error", e2)
        return e2, valid_mask, var_depth

    # ---------- Predicted variances to scalar variances ----------
    def flow_equivalent_variance(self, var_uv: torch.Tensor) -> torch.Tensor:
        """
        Convert 2-channel (sigma_uu, sigma_vv) into a scalar equivalent variance.
        Use avg: (sigma_uu + sigma_vv)/2
        Shape: (N_pixels,)
        """
        eps = 1e-12
        if var_uv.dim() == 3:
            # (2,H,W) -> add batch
            var_uv = var_uv.unsqueeze(0)

        var_u = var_uv[:, 0]
        var_v = var_uv[:, 1]
        var_eq = 0.5 * (var_u + var_v)
        return (var_eq + eps).flatten()

    # ---------- Conformal nonconformity ----------
    def nonconformity_score(self, error_sq: torch.Tensor, var_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Dimensionless score: a = e^2 / var
        """
        error_sq = error_sq[mask] if mask is not None else error_sq
        var_pred = var_pred[mask] if mask is not None else var_pred
        assert error_sq.shape == var_pred.shape, f"Shape mismatch: error_sq {error_sq.shape}, var_pred {var_pred.shape}"

        nc = torch.zeros_like(error_sq)
        valid_mask = torch.isfinite(error_sq) & torch.isfinite(var_pred) & (var_pred > 1e-12)
        nc[valid_mask] = error_sq[valid_mask] / var_pred[valid_mask]

        # print(f"[Nonconformity Score] median: {nc.median().item():.4f}, mean: {nc.mean().item():.4f}, max: {nc.max().item():.4f}")

        return nc[valid_mask].flatten()

    def nonconformity_quantile(self, nc_scores: torch.Tensor, q: float = 0.90, clip_q: float = 0.99):
        nc = nc_scores[torch.isfinite(nc_scores)]
        hi = torch.quantile(nc, clip_q)
        nc = torch.clamp(nc, max=hi)
        return torch.quantile(nc, q)

    def plot_nonconformity_boxplot(self, nc_scores: torch.Tensor, title: str):
        import matplotlib.pyplot as plt
        nc_scores = nc_scores.cpu().numpy()
        plt.figure()
        plt.boxplot(nc_scores)
        plt.title(title)
        plt.show(block=False)

    # ---------- Main evaluation ----------
    def evaluate(self, max_pairs: int) -> Dict[str, float]:

        # Build run configs for each data sequence
        run_configs = [
            {
                "Data": data_cfg,
                "Preprocess": self.cfg_dict["Preprocess"]
            }
            for data_cfg in self.data_cfgs
        ]

        for run_cfg in run_configs:

        # Initialize data source
            sequence = smart_transform(
                SequenceBase[Frame].instantiate(run_cfg["Data"]["type"], run_cfg["Data"]["args"]).clip(0, -1),
                run_cfg["Preprocess"]
            )

            pb = ColoredTqdm(sequence)

            rng = torch.Generator().manual_seed(0)

            all_flow_nc = []
            all_depth_nc = []

            with torch.no_grad():
                n_pairs = 0
                PAIR_KEEP_PROB = 0.1
                MAX_KEEP_PER_PAIR_FLOW  = 50_000
                MAX_KEEP_PER_PAIR_DEPTH = 50_000
                for frame in pb:
                    if not self.isinitialized:
                        self.initialize(frame)
                        continue
                    if n_pairs >= max_pairs:
                        break


                    assert self.prev_keyframe is not None
                    frame1, frame2 = self.prev_keyframe, frame

                    if torch.rand(1, generator=rng).item() > PAIR_KEEP_PROB:
                        self.prev_keyframe = frame2
                        continue

                    gt_flow = frame1.mono.gt_flow.cuda()     # flow from frame1->frame2
                    gt_depth = frame2.mono.gt_depth.cuda()   # depth at frame2

                    depth, var_depth, sky_mask, flow, var_uv = self.estimate_pair(frame1.mono, frame2.mono)

                    # Compute per-pixel squared errors
                    flow_e2 = self.flow_error_sq(flow, gt_flow)
                    depth_e2, depth_error_mask, var_depth = self.depth_error_sq(depth, gt_depth, sky_mask, var_depth)

                    # Predicted variances (per pixel)
                    flow_var = self.flow_equivalent_variance(var_uv)
                    depth_var = var_depth

                    # Nonconformity scores
                    flow_nc = self.nonconformity_score(flow_e2, flow_var, None)
                    depth_nc = self.nonconformity_score(depth_e2, depth_var, depth_error_mask)

                    # move to CPU and subsample to keep memory bounded + quantile stable
                    flow_nc_cpu  = sample_nc(flow_nc.detach().cpu(),  MAX_KEEP_PER_PAIR_FLOW,  rng)
                    depth_nc_cpu = sample_nc(depth_nc.detach().cpu(), MAX_KEEP_PER_PAIR_DEPTH, rng)

                    all_flow_nc.append(flow_nc_cpu)
                    all_depth_nc.append(depth_nc_cpu)

                    self.prev_keyframe = frame2
                    n_pairs += 1

        flow_q90 = self.nonconformity_quantile(torch.cat(all_flow_nc), q=0.90).item()
        depth_q90 = self.nonconformity_quantile(torch.cat(all_depth_nc), q=0.90).item()
        depth_q95 = self.nonconformity_quantile(torch.cat(all_depth_nc), q=0.95).item()
        depth_q80 = self.nonconformity_quantile(torch.cat(all_depth_nc), q=0.80).item()
        depth_q85 = self.nonconformity_quantile(torch.cat(all_depth_nc), q=0.85).item()
        depth_q75 = self.nonconformity_quantile(torch.cat(all_depth_nc), q=0.75).item()
        depth_q50 = self.nonconformity_quantile(torch.cat(all_depth_nc), q=0.50).item()
        print(f"Estimated depth nonconformity quantiles: q50={depth_q50:.4f}, q75={depth_q75:.4f}, q80={depth_q80:.4f}, q85={depth_q85:.4f}, q90={depth_q90:.4f}, q95={depth_q95:.4f}")
        all_depth_nc = torch.cat(all_depth_nc)
        print(f"[Depth NC] median: {all_depth_nc.median().item():.4f}, mean: {all_depth_nc.mean().item():.4f}, max: {all_depth_nc.max().item():.4f}")
        self.plot_nonconformity_boxplot(all_depth_nc, title="Depth Nonconformity Scores")
        self.plot_nonconformity_boxplot(torch.cat(all_flow_nc), title="Flow Nonconformity Scores")
        self.flow_q90 = flow_q90
        self.depth_q90 = depth_q90

        return {"flow_q90": flow_q90, "depth_q90": depth_q90}



if __name__ == "__main__":
    # set_determinism(0)
    parser = argparse.ArgumentParser(description="Evaluate the model on TartanAirV2")
    parser.add_argument("--ckpt", default = "Model/15001_FlowFormerCov.pth", type=str, help="Path to the model weight")
    parser.add_argument("--config", type=str, default="Config/Conformal/TartanAir.yaml")
    parser.add_argument("--length", type=int, default=2000, help="Length of the evaluation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--wandb", action="store_true", help="Use wandb to log the results")
    # New
    parser.add_argument("--data", type=str, default = "Config/Sequence/TartanAir_Sample_dataset/TartanAir_abf001.yaml")
    parser.add_argument(
        "--seq_to",
        type=int,
        default=None,
        help="Crop sequence to frame# when ran. Set to -1 (default) if wish to run whole sequence",
    )
    parser.add_argument(
        "--seq_from",
        type=int,
        default=0,
        help="Crop sequence from frame# when ran. Set to 0 (default) if wish to start from first frame",
    )
    args = parser.parse_args()


    cf = ConformalStatistics(args)
    stats = cf.evaluate(max_pairs=args.length)
    print(stats)
    print(f"Estimated flow nonconformity quantile (q={0.90}): {cf.flow_q90:.4f}")
    print(f"Estimated depth nonconformity quantile (q={0.90}): {cf.depth_q90:.4f}")
