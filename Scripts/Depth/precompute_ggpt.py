"""
precompute_ggpt.py — GGPT (ChenYutongTHU/GGPT) single-image depth backbone, offline.

GGPT proper is a multi-view SfM + PointTransformerV3 refinement pipeline (needs
>=2 views + RoMa + spconv/pointcept custom CUDA — not a monocular depth net and
not Windows-feasible). Its single-image geometry comes from its *feedforward
backbone*, the default `vggt-depth` = stock VGGT-1B (`facebook/VGGT-1B`). That is
what we integrate here: single-image depth + confidence from VGGT-1B (distinct
from DGGT, which uses a Waymo-finetuned VGGT).

Runs in the `D:/envs/dggt` env (torch 2.4.1, VGGT-compatible). Relative/up-to-scale
depth -> eiva_trajectory_evaluator correct_scale=true. Confidence emitted but
CachedDepth uses the bounded heuristic cov (required by the p2p backend).

  <env python> Scripts/Depth/precompute_ggpt.py --frames D:/macvo_depth/frames \
      --out D:/macvo_depth/cache/ggpt --res 518 --sequences EIVA_plane_nose
"""
import sys, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GGPT = "D:/macvo_depth/repos/GGPT"
sys.path.insert(0, GGPT)
sys.path.insert(0, GGPT + "/vggt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(args.device).eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print("[ggpt] VGGT-1B backbone ready", flush=True)

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
            t = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0     # 3xHxW
            t = F.interpolate(t[None], size=(args.res, args.res), mode="bilinear", align_corners=False)  # 1x3xr xr
            imgs = t.to(args.device)                                                    # (S=1,3,r,r)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                out_d = model(imgs)                                                     # dict
            depth = out_d["depth"][0, 0].squeeze().float().cpu().numpy()               # HxW
            conf = out_d["depth_conf"][0, 0].squeeze().float().cpu().numpy()           # HxW
            if first:
                print(f"[ggpt] depth {depth.shape} range [{np.nanmin(depth):.3f},{np.nanmax(depth):.3f}] "
                      f"conf [{np.nanmin(conf):.3f},{np.nanmax(conf):.3f}]", flush=True)
                first = False
            np.savez_compressed(out, depth=depth.astype(np.float32), conf=conf.astype(np.float32))
            if i % 100 == 0:
                print(f"  {seq} {i}/{len(pngs)}", flush=True)
    print("=== GGPT PRECOMPUTE DONE ===", flush=True)


if __name__ == "__main__":
    main()
