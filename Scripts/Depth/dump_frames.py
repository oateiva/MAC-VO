"""
dump_frames.py — export the exact preprocessed frames MAC-VO feeds its frontend.

Runs in MAC-VO's env (AITraining12). For each sequence in a config's ``Datas`` it
instantiates the dataloader + Preprocess pipeline identically to
``Scripts/Experiment/Experiment_MACVO.py`` and writes, per frame:

  <out>/<sequence_name>/<frame_ns>.png    RGB uint8, exactly as seen by the matcher
  <out>/<sequence_name>/meta.json         { K, height, width, frames: [frame_ns...] }

External depth networks (run offline in their own envs) consume these PNGs and
write depth caches keyed by the same ``frame_ns``; ``CachedDepth`` then serves them
back to MAC-VO. This guarantees pixel + timestamp alignment without importing the
MAC-VO dataloader into the exotic net environments.

Usage:
  python -m Scripts.Depth.dump_frames --config Config/Experiment/MACVO/Custom/MACVO_dggt_GEDF.yaml \
      --out D:/macvo_depth/frames
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from DataLoader import SequenceBase, Frame, smart_transform
from Utility.Config import load_config


def frame_key(frame_ns) -> str:
    """Canonical, notation-stable cache key from a (possibly float) ns timestamp.
    MUST match Module/Network/Depth/Cached/api.py and every precompute_*.py."""
    return str(int(round(float(frame_ns))))


def chw_to_uint8(img) -> np.ndarray:
    """Any leading dims x C x H x W float[0,1] RGB tensor -> HxWx3 uint8 RGB."""
    t = img.detach().float().cpu()
    t = t.reshape(-1, t.shape[-3], t.shape[-2], t.shape[-1])[0]   # -> C x H x W
    t = t.clamp(0, 1).mul(255).round().byte()                    # C x H x W uint8
    return t.permute(1, 2, 0).numpy()                            # H x W x 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="A MAC-VO config with Datas + Preprocess")
    ap.add_argument("--out", required=True, help="Output root for dumped frames")
    ap.add_argument("--seq_from", type=int, default=0)
    ap.add_argument("--seq_to", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg, _ = load_config(Path(args.config))
    out_root = Path(args.out)

    for data_cfg in cfg.Datas:
        name = data_cfg.name
        seq_from = getattr(data_cfg.args, "seq_from", args.seq_from)
        seq_to = getattr(data_cfg.args, "seq_to", args.seq_to)
        sequence = smart_transform(
            SequenceBase[Frame].instantiate(data_cfg.type, data_cfg.args).clip(seq_from, seq_to),
            cfg.Preprocess,
        )
        seq_dir = out_root / name
        seq_dir.mkdir(parents=True, exist_ok=True)
        frames: list[int] = []
        K = None
        H = W = None
        print(f"=== {name}: {len(sequence)} frames -> {seq_dir}", flush=True)
        for i, frame in enumerate(sequence):
            cam = frame.camera
            key = frame_key(cam.frame_ns)
            if key in frames:
                raise RuntimeError(f"Duplicate frame key {key} in {name}; keying assumption broken")
            img = chw_to_uint8(cam.imageL)
            H, W = img.shape[0], img.shape[1]
            if K is None:
                K = cam.frame_K.detach().cpu().numpy().tolist()
            png = seq_dir / f"{key}.png"
            if args.overwrite or not png.exists():
                Image.fromarray(img).save(png)
            frames.append(key)
            if i % 100 == 0:
                print(f"  {name} {i}/{len(sequence)}", flush=True)
        (seq_dir / "meta.json").write_text(json.dumps(
            {"sequence": name, "K": K, "height": H, "width": W, "frames": frames}, indent=2))
        print(f"    wrote {len(frames)} frames + meta.json", flush=True)


if __name__ == "__main__":
    main()
