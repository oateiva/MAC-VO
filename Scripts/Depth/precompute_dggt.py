"""
precompute_dggt.py — run DGGT (xiaomi-research/dggt) offline over dumped frames.

Runs in the isolated `D:/envs/dggt` env (torch 2.4.1). Reads the RGB frames dumped
by Scripts/Depth/dump_frames.py and writes, per frame, a metric depth + confidence
map that Module/Network/Depth/Cached/api.py (CachedDepth) serves back to MAC-VO.

DGGT emits a direct per-pixel metric depth (predictions["depth"]) and a confidence
(predictions["depth_conf"]). It is a single forward pass; we run it per frame with
S=1 (VGGT aggregator handles arbitrary sequence length).

  <env python> Scripts/Depth/precompute_dggt.py \
      --frames D:/macvo_depth/frames --out D:/macvo_depth/cache/dggt \
      --ckpt D:/macvo_depth/weights/dggt/model_latest_waymo.pt --res 518
"""
import sys, types, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# DGGT's dggt.models.sky / fusion / utils.gs import gsplat + open3d at module top,
# used only for rendering — never in the depth forward. Stub them with MagicMocks so
# importing VGGT needs neither a gsplat CUDA build nor open3d.
from unittest import mock as _mock
for _m in ("gsplat", "gsplat.rendering", "gsplat.cuda", "open3d", "open3d.geometry",
           "open3d.utility", "open3d.visualization"):
    sys.modules.setdefault(_m, _mock.MagicMock())

REPO = "D:/macvo_depth/repos/dggt"
sys.path.insert(0, REPO)


def frame_key(ns) -> str:
    return str(int(round(float(ns))))


def load_model(ckpt: str, device: str):
    from dggt.models.vggt import VGGT
    model = VGGT()
    sd = torch.load(ckpt, map_location="cpu")
    for k in ("model", "state_dict", "ema"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[dggt] loaded ckpt: {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model.to(device).eval()


@torch.inference_mode()
def infer_one(model, png: Path, res: int, device: str):
    img = Image.open(png).convert("RGB")
    t = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0    # 3xHxW in [0,1]
    t = F.interpolate(t[None], size=(res, res), mode="bilinear", align_corners=False)  # 1x3xres xres
    imgs = t.to(device)                                                       # [S=1,3,res,res]
    with torch.cuda.amp.autocast(dtype=torch.float32):
        pred = model(imgs)
    depth = pred["depth"][0, 0].squeeze().float().cpu().numpy()               # HxW
    conf = pred["depth_conf"][0, 0].squeeze().float().cpu().numpy()           # HxW
    return depth.astype(np.float32), conf.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only first N frames per sequence (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)
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
            depth, conf = infer_one(model, png, args.res, args.device)
            if first:
                print(f"[dggt] depth {depth.shape} range [{depth.min():.3f},{depth.max():.3f}] "
                      f"conf [{conf.min():.3f},{conf.max():.3f}]", flush=True)
                first = False
            np.savez_compressed(out, depth=depth, conf=conf)
            if i % 100 == 0:
                print(f"  {seq} {i}/{len(pngs)}", flush=True)
    print("=== DGGT PRECOMPUTE DONE ===", flush=True)


if __name__ == "__main__":
    main()
