"""
precompute_md2e.py — run the MDE repo's LR2Depth / MD2E model offline.

Runs in the isolated `D:/envs/mde` env (torch 1.13 + cu117 + mmcv-full 1.7.1).
The user-supplied weights under Z:/Research/weights/depth_estimation/mde are the
LR2Depth "one-path" family (1P-L-K / 1P-M-K / LRDepth-S-K / LRDepth-XL-N); each
maps to a `configs/LRDepth/1P_*.py`. RGB-only, metric depth (mmseg-toolbox style).
No confidence output -> CachedDepth uses the heuristic cov.

  <env python> Scripts/Depth/precompute_md2e.py \
      --frames D:/macvo_depth/frames --out D:/macvo_depth/cache/md2e \
      --config D:/macvo_depth/repos/MDE/configs/LRDepth/1P_L_kitti.py \
      --ckpt Z:/Research/weights/depth_estimation/mde/1P-L-K.pth --limit 200
"""
import sys, argparse
from pathlib import Path

import numpy as np
import cv2
import torch

REPO = "D:/macvo_depth/repos/MDE"
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--res", type=int, default=704, help="square inference size (multiple of 32)")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import mmcv
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from depth.models import build_depther

    cfg = Config.fromfile(args.config)
    # We load the full checkpoint below, so don't fetch the Swin backbone init.
    if hasattr(cfg.model, "pretrained"):
        cfg.model["pretrained"] = None
    if "backbone" in cfg.model and "pretrained" in cfg.model.backbone:
        cfg.model.backbone.pretrained = None

    # Build + load manually (these checkpoints lack the mmseg 'meta' wrapper that
    # init_depther assumes for CLASSES/PALETTE, which inference doesn't need).
    model = build_depther(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model.cfg = cfg
    model = model.to(args.device).eval()
    print("[md2e] model ready", flush=True)

    import numpy as _np
    MEAN = _np.array([123.675, 116.28, 103.53], dtype=_np.float32)
    STD = _np.array([58.395, 57.12, 57.375], dtype=_np.float32)

    def infer(bgr):
        H0, W0 = bgr.shape[:2]
        r = cv2.resize(bgr, (args.res, args.res))                # WxH square, /32
        norm = mmcv.imnormalize(r, MEAN, STD, to_rgb=True)       # BGR->RGB + ImageNet norm
        t = torch.from_numpy(norm.transpose(2, 0, 1))[None].float().to(args.device)
        meta = dict(filename=None, ori_filename=None, ori_shape=(H0, W0, 3),
                    img_shape=(args.res, args.res, 3), pad_shape=(args.res, args.res, 3),
                    scale_factor=_np.array([1., 1., 1., 1.], dtype=_np.float32),
                    flip=False, flip_direction=None,
                    img_norm_cfg=dict(mean=MEAN, std=STD, to_rgb=True), cam_intrinsic=None)
        with torch.no_grad():
            out = model(return_loss=False, rescale=True, img=[t], img_metas=[[meta]])
        return _np.squeeze(_np.asarray(out[0] if isinstance(out, (list, tuple)) else out))

    frames_root, out_dir = Path(args.frames), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = args.sequences or sorted(d.name for d in frames_root.iterdir() if d.is_dir())
    first = True
    for seq in seqs:
        pngs = sorted((frames_root / seq).glob("*.png"))
        if args.limit:
            pngs = pngs[:args.limit]
        print(f"=== {seq}: {len(pngs)} frames", flush=True)
        for i, png in enumerate(pngs):
            out = out_dir / f"{png.stem}.npz"
            if out.exists() and not args.overwrite:
                continue
            img = cv2.imread(str(png))                      # BGR HxWx3
            depth = infer(img).astype(np.float32)           # HxW at ori resolution
            if first:
                print(f"[md2e] depth {depth.shape} range [{np.nanmin(depth):.3f},{np.nanmax(depth):.3f}]", flush=True)
                first = False
            np.savez_compressed(out, depth=depth)
            if i % 100 == 0:
                print(f"  {seq} {i}/{len(pngs)}", flush=True)
    print("=== MD2E PRECOMPUTE DONE ===", flush=True)


if __name__ == "__main__":
    main()
