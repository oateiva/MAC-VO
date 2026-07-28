"""
precompute_ppd.py — run Pixel-Perfect Depth (gangweix/pixel-perfect-depth) offline.

Runs in the isolated `D:/envs/ppd` env. Uses the DA2 semantics variant (relative /
affine-invariant depth); the MAC-VO trajectory is scale-corrected at evaluation time
(eiva_trajectory_evaluator uses correct_scale=true), so relative depth is fine.

PPD has no confidence output -> CachedDepth falls back to the DAv2 heuristic cov.

  <env python> Scripts/Depth/precompute_ppd.py \
      --frames D:/macvo_depth/frames --out D:/macvo_depth/cache/ppd \
      --ppd D:/macvo_depth/weights/ppd/ppd.pth \
      --da2 C:/Users/oat/workspace/MAC-VO/Model/depth_anything_v2_vitl.pth \
      --sampling_steps 4
"""
import sys, argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

REPO = "D:/macvo_depth/repos/pixel-perfect-depth"
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ppd", required=True, help="ppd.pth")
    ap.add_argument("--da2", required=True, help="depth_anything_v2_vitl.pth backbone")
    ap.add_argument("--sampling_steps", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only first N frames per sequence (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from ppd.models.ppd import PixelPerfectDepth
    model = PixelPerfectDepth(semantics_model="DA2", semantics_pth=args.da2,
                              sampling_steps=args.sampling_steps)
    model.load_state_dict(torch.load(args.ppd, map_location="cpu"), strict=False)
    model = model.to(args.device).eval()
    print("[ppd] model ready", flush=True)

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
            image = cv2.imread(str(png))          # BGR HxWx3, matches run.py
            H, W = image.shape[:2]
            with torch.inference_mode():
                depth, _ = model.infer_image(image)
                depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            depth = depth.squeeze().float().cpu().numpy().astype(np.float32)
            if first:
                print(f"[ppd] depth {depth.shape} range [{depth.min():.3f},{depth.max():.3f}]", flush=True)
                first = False
            np.savez_compressed(out, depth=depth)
            if i % 100 == 0:
                print(f"  {seq} {i}/{len(pngs)}", flush=True)
    print("=== PPD PRECOMPUTE DONE ===", flush=True)


if __name__ == "__main__":
    main()
