"""
Build a GDF1 .bin G-EDF map from a .ply pointcloud, using the mapper
parameters of a MAC-VO config so the offline map matches the online setup
(cube_size, margin, num_gaussians, ... are read from the config's
`map.online` block — GEDF_PGO layout — or `gedf.map.online` — GTSAM hybrid).

    python Scripts/BuildGEDFMap.py --ply path/to/model.ply \
        --config Config/Experiment/MACVO/Optimal/MACVO_Fast_GEDF_Pure.yaml \
        [--out path/to/model.bin] [--budget 512]

The result is loadable with `map.source: prebuilt` + `map.path` (frozen map,
pure localization) and by `Scripts/VisualizeGEDF.py`.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Module.Optimization.GEDF import GEDFConfig, GEDFMapper   # noqa: E402
from Utility.Config import load_config                        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help=".ply pointcloud (x/y/z vertex properties)")
    ap.add_argument("--config", required=True,
                    help="MAC-VO odom config supplying the mapper parameters")
    ap.add_argument("--out", default=None,
                    help="output GDF1 .bin (default: <ply>_cube<size>.bin)")
    ap.add_argument("--budget", type=int, default=512, help="cube fits per refit round")
    ap.add_argument("--chunk", type=int, default=500_000, help="points per insert call")
    args = ap.parse_args()

    cfg, _ = load_config(Path(args.config))
    opt_args = cfg.Odometry.optimizer.args
    online = getattr(getattr(opt_args, "map", None), "online", None)
    if online is None:                       # GTSAM hybrid layout (gedf: block)
        online = opt_args.gedf.map.online
    gcfg = GEDFConfig.from_namespace(online)
    gcfg.cov_trace_gate = 0.0                # VO insert gate; a model cloud has no covariance

    v = PlyData.read(args.ply)["vertex"]
    pts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    print(f"[BuildGEDFMap] {args.ply}: {pts.shape[0]} points, "
          f"bbox min {pts.min(0).round(2)} max {pts.max(0).round(2)}")
    print(f"[BuildGEDFMap] mapper: cube {gcfg.cube_size} m, margin {gcfg.margin}, "
          f"K {gcfg.num_gaussians}, device {gcfg.device}")

    mapper = GEDFMapper(gcfg)
    t = torch.from_numpy(pts)
    for i in range(0, t.shape[0], args.chunk):
        mapper.insert(t[i:i + args.chunk])
    print(f"[BuildGEDFMap] inserted -> {mapper.num_cubes} cubes; fitting "
          f"(budget {args.budget}/round)...")

    rounds = 0
    while True:
        fittable = [s for s in mapper._dirty
                    if mapper._n_points.get(s, 0) >= gcfg.min_points_fit]
        if not fittable:
            break
        stats = mapper.refit(budget=args.budget)
        rounds += 1
        print(f"  round {rounds}: +{stats['refitted']} fitted, "
              f"{stats['dirty_remaining']} dirty left, mean MAE {stats['mean_mae']:.4f} m",
              flush=True)

    print(f"[BuildGEDFMap] done: {mapper.num_valid_cubes}/{mapper.num_cubes} cubes valid, "
          f"{mapper.num_valid_gaussians} gaussians, avg MAE {mapper.sigma:.4f} m")
    out = args.out or f"{Path(args.ply).with_suffix('')}_cube{gcfg.cube_size}.bin"
    mapper.export_gdf1(out)
    print(f"[BuildGEDFMap] wrote {out}")


if __name__ == "__main__":
    main()
