"""
precompute_gemdepth.py — run GemDepth (Yuecheng919/GemDepth) offline.

Runs in the isolated `D:/envs/gemdepth` env (torch 2.3.1). GemDepth is a VIDEO
depth transformer: it consumes a whole clip and returns temporally-aligned
relative depth per frame (arbitrary scale). We feed each EIVA sequence as one
video and cache per-frame depth keyed by frame_ns. Uses attn_implementation
"pytorch_naive" to avoid the flash-attention build. No confidence output ->
CachedDepth uses the bounded heuristic cov (required for the p2p backend).

Output is relative -> eiva_trajectory_evaluator correct_scale=true.

  <env python> Scripts/Depth/precompute_gemdepth.py \
      --frames D:/macvo_depth/frames --out D:/macvo_depth/cache/gemdepth \
      --ckpt D:/macvo_depth/weights/gemdepth/gemdepth.pth --sequences EIVA_plane_nose
"""
import sys, argparse
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# GemDepth's attention calls torch SDPA; no flash/mem-efficient kernel is available
# here and even MATH errors ("No available kernel"). Replace SDPA with a pure-matmul
# implementation BEFORE importing the model, so its `from torch.nn.functional import
# scaled_dot_product_attention` binds this version.
def _manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    scale = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    attn = (query @ key.transpose(-2, -1)) * scale
    if is_causal:
        L, S = query.size(-2), key.size(-2)
        cm = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
        attn = attn.masked_fill(~cm, float("-inf"))
    if attn_mask is not None:
        attn = attn.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else attn + attn_mask
    attn = attn.softmax(dim=-1)
    return attn @ value

torch.nn.functional.scaled_dot_product_attention = _manual_sdpa
torch.scaled_dot_product_attention = _manual_sdpa

REPO = "D:/macvo_depth/repos/GemDepth"
sys.path.insert(0, REPO)

MODEL_CONFIGS = {"vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--invert", action="store_true", help="store 1/value (if the net emits disparity)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # GemDepth's VGGT layers call torch SDPA directly; flash/mem-efficient kernels
    # aren't available here ("No available kernel"). Force the math backend.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    from model.gemdepth import GemDepth
    model = GemDepth(**MODEL_CONFIGS["vitl"], attn_implementation="pytorch_naive")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model = model.to(args.device).eval()
    print("[gemdepth] model ready", flush=True)

    frames_root, out_dir = Path(args.frames), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = args.sequences or sorted(d.name for d in frames_root.iterdir() if d.is_dir())
    for seq in seqs:
        pngs = sorted((frames_root / seq).glob("*.png"))
        if args.limit:
            pngs = pngs[:args.limit]
        if not pngs:
            continue
        print(f"=== {seq}: {len(pngs)} frames (video)", flush=True)
        video = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in pngs])   # (N,H,W,3) RGB uint8
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.MATH]):                                          # flash/efficient unavailable here
            depths, _ = model.infer_video_depth(video, target_fps=30, input_size=args.input_size,
                                                device=args.device, fp32=True)        # (N,H,W) relative
        print(f"[gemdepth] {seq} depth {depths.shape} range [{np.nanmin(depths):.3f},{np.nanmax(depths):.3f}]", flush=True)
        for p, d in zip(pngs, depths):
            out = out_dir / f"{p.stem}.npz"
            if out.exists() and not args.overwrite:
                continue
            d = d.astype(np.float32)
            if args.invert:
                d = 1.0 / np.clip(d, 1e-6, None)
            np.savez_compressed(out, depth=d)
    print("=== GEMDEPTH PRECOMPUTE DONE ===", flush=True)


if __name__ == "__main__":
    main()
