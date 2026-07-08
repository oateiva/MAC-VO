"""
Visualize a G-EDF map (GDF1 .bin) in Rerun as a near-surface point cloud,
optionally overlaid with the GMM components as confidence-encoded ellipsoids
(hue = cube fit MAE, alpha = |weight|, magenta = negative/carving components).

Works on maps produced by the G-EDF C++ trainer (e.g. plane_nose_model.bin) and
on maps exported from a MAC-VO run via GEDFMapper.export_gdf1.

Usage:
    python Scripts/VisualizeGEDF.py --map D:/path/to/map.bin
    python Scripts/VisualizeGEDF.py --map map.bin --gaussians --n_sigma 1.5
    python Scripts/VisualizeGEDF.py --map map.bin --iso 0.05 --resolution 0.05 --save map.rrd
"""
import argparse
from pathlib import Path

import torch
import rerun as rr

from Module.Optimization.GEDF import GEDFConfig, GEDFMapper
from Utility.Visualize import rr_plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=str, required=True, help="Path to a GDF1 .bin map")
    parser.add_argument("--iso", type=float, default=0.05,
                        help="Keep sampled points with field value < iso (m)")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Sampling grid resolution (m)")
    parser.add_argument("--max_points", type=int, default=2_000_000)
    parser.add_argument("--gaussians", action="store_true",
                        help="Also render GMM components as ellipsoids "
                             "(hue = cube MAE, alpha = |weight|, magenta = negative)")
    parser.add_argument("--n_sigma", type=float, default=1.0,
                        help="Ellipsoid half-size = n_sigma * per-axis sigma")
    parser.add_argument("--max_gaussians", type=int, default=200_000,
                        help="Cap on rendered gaussians (top-|weight| kept)")
    parser.add_argument("--save", type=str, default=None,
                        help="Write an .rrd file instead of spawning the viewer")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    map_path = Path(args.map)
    mapper = GEDFMapper.from_gdf1(map_path, GEDFConfig(device=args.device),
                                  dtype=torch.float32)
    print(f"[GEDF] {map_path.name}: {mapper.num_valid_cubes} cubes, "
          f"{mapper.num_valid_gaussians} gaussians, avg MAE {mapper.sigma:.4f} m")

    points, dist = mapper.sample_surface(resolution=args.resolution, iso=args.iso,
                                         max_points=args.max_points)
    print(f"[GEDF] sampled {points.shape[0]} near-surface points "
          f"(< {args.iso} m at {args.resolution} m grid)")

    rr.init(f"GEDF@{map_path.stem}", spawn=args.save is None)
    if args.save is not None:
        rr.save(args.save)
    rr.log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)

    rr_plt.default_mode = "rerun"
    rr_plt.log_gedf_map("/world/gedf_map", points, dist, radius=args.resolution / 3)
    if args.gaussians:
        means, sigmas, weights, mae = mapper.gaussians(max_gaussians=args.max_gaussians)
        print(f"[GEDF] rendering {means.shape[0]} gaussians "
              f"({int((weights < 0).sum())} negative) at {args.n_sigma:.1f} sigma")
        rr_plt.log_gedf_gaussians("/world/gedf_map/gaussians", means, sigmas, weights,
                                  mae, n_sigma=args.n_sigma)
    if args.save is not None:
        print(f"[GEDF] wrote {args.save}")


if __name__ == "__main__":
    main()
