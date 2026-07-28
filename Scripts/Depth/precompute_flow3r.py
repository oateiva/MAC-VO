"""
precompute_flow3r.py — run Flow3r (CVPR 2026, Kidrauh/flow3r) offline.

Runs in the isolated `D:/envs/flow3r` env (torch 2.5.1). Flow3r is a multi-view
visual-geometry transformer; here it is run per frame with N=1 (documented
compromise — single-frame loses its multi-view scale consistency, but the MAC-VO
trajectory is scale-corrected at evaluation). Depth is the camera-frame z of the
predicted local pointmap; a per-pixel confidence map is emitted (edge pixels zeroed,
matching gradio_app.py) and used as CachedDepth's variance source.

Output is scale-invariant (relative) -> eiva_trajectory_evaluator correct_scale=true.

  <env python> Scripts/Depth/precompute_flow3r.py \
      --frames D:/macvo_depth/frames --out D:/macvo_depth/cache/flow3r \
      --ckpt D:/macvo_depth/weights/flow3r/flow3r.bin --res 518 --limit 200
"""
import sys, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = "D:/macvo_depth/repos/flow3r"
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Flow3r's attention hits torch SDPA; flash/mem-efficient kernels aren't
    # available for this dtype/config here ("No available kernel"). Force the math
    # backend (always available) and run fp32.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    from flow3r.models.flow3r import Flow3r
    model = Flow3r()
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model = model.to(args.device).eval()
    print("[flow3r] model ready", flush=True)

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
            img = Image.open(png).convert("RGB")
            t = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0   # 3xHxW
            t = F.interpolate(t[None], size=(args.res, args.res), mode="bilinear", align_corners=False)  # 1x3xr xr
            imgs = t.to(args.device)                                                  # (N=1,3,r,r)
            with torch.inference_mode():
                pred = model(imgs[None])                                              # (B=1,N=1,...) fp32
            local = pred["local_points"].float()                                      # B,N,H,W,3
            depth = local[0, 0, ..., 2].cpu().numpy().astype(np.float32)              # HxW
            conf = torch.sigmoid(pred["conf"].float())[0, 0].squeeze().cpu().numpy().astype(np.float32)
            if first:
                print(f"[flow3r] depth {depth.shape} range [{depth.min():.3f},{depth.max():.3f}] "
                      f"conf [{conf.min():.3f},{conf.max():.3f}]", flush=True)
                first = False
            np.savez_compressed(out, depth=depth, conf=conf)
            if i % 100 == 0:
                print(f"  {seq} {i}/{len(pngs)}", flush=True)
    print("=== FLOW3R PRECOMPUTE DONE ===", flush=True)


if __name__ == "__main__":
    main()
