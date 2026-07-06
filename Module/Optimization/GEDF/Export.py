"""
GDF1 binary map format reader / writer.

Byte layout follows the reference implementation shipped with G-EDF
(`scripts/gdf1_loader_reference.py` / `include/binary_exporter.hpp`), little endian:

    Map header (120 B):  <4sIIff3f3ffff64x
        magic "GDF1", version u32, num_cubes u32, avg_mae f32, std_dev f32,
        bounds_min[3] f32, bounds_max[3] f32, cube_size f32,
        empty_search_margin f32, cube_margin f32, 64 B padding
    Per cube (24 B):     <3fffI      origin[3], mae, std_dev, num_gaussians
    Per gaussian (32 B): <I3f3ff     id, mean[3], sigma[3], weight

NOTE on "sigma": the stored value is the ROOT parameter `p` of the solver —
the effective scale is lambda = p^4 (NOT p^2). Readers must apply the fourth
power; writers must store `p` raw.
"""
import struct
from pathlib import Path

import numpy as np

_HEADER_FMT = "<4sIIff3f3ffff64x"
_CUBE_FMT = "<3fffI"
_GAUSS_FMT = "<I3f3ff"

_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_CUBE_SIZE = struct.calcsize(_CUBE_FMT)
_GAUSS_SIZE = struct.calcsize(_GAUSS_FMT)

GDF1_VERSION = 1


def read_gdf1(path: str | Path) -> tuple[dict, list[dict]]:
    """
    Load a GDF1 .bin file.

    Returns (header, cubes) where header is a dict with keys
    `version, num_cubes, avg_mae, std_dev, bounds_min, bounds_max, cube_size,
    empty_margin, margin`, and each cube is a dict with keys
    `origin (3,) f64, mae, std_dev, means (G,3) f64, sigmas (G,3) f64 (root
    parameter p; lambda = p**4), weights (G,) f64`.
    """
    raw = Path(path).read_bytes()
    if len(raw) < _HEADER_SIZE:
        raise ValueError(f"Not a GDF1 file (too short): {path}")
    (magic, version, num_cubes, avg_mae, std_dev,
     min_x, min_y, min_z, max_x, max_y, max_z,
     cube_size, empty_margin, cube_margin) = struct.unpack_from(_HEADER_FMT, raw, 0)
    if magic != b"GDF1":
        raise ValueError(f"Not a GDF1 file (magic={magic!r}): {path}")

    header = dict(
        version=version, num_cubes=num_cubes, avg_mae=avg_mae, std_dev=std_dev,
        bounds_min=(min_x, min_y, min_z), bounds_max=(max_x, max_y, max_z),
        cube_size=cube_size, empty_margin=empty_margin, margin=cube_margin,
    )

    cubes: list[dict] = []
    offset = _HEADER_SIZE
    for i in range(num_cubes):
        if offset + _CUBE_SIZE > len(raw):
            raise ValueError(f"Truncated GDF1 file: cube header {i} out of range")
        ox, oy, oz, mae, c_std, n_gauss = struct.unpack_from(_CUBE_FMT, raw, offset)
        offset += _CUBE_SIZE

        blob_size = n_gauss * _GAUSS_SIZE
        if offset + blob_size > len(raw):
            raise ValueError(f"Truncated GDF1 file: gaussians of cube {i} out of range")
        # Vectorized parse: each 32 B record is (u32, 7 x f32)
        rec = np.frombuffer(raw, dtype=np.float32, count=n_gauss * 8, offset=offset)
        rec = rec.reshape(n_gauss, 8).astype(np.float64)
        offset += blob_size

        cubes.append(dict(
            origin=np.array([ox, oy, oz], dtype=np.float64),
            mae=float(mae), std_dev=float(c_std),
            means=rec[:, 1:4].copy(),
            sigmas=rec[:, 4:7].copy(),
            weights=rec[:, 7].copy(),
        ))
    return header, cubes


def write_gdf1(
    path: str | Path,
    origins: np.ndarray,     # (C, 3) cube corner origins (world frame)
    means: np.ndarray,       # (C, K, 3)
    sigmas: np.ndarray,      # (C, K, 3) root parameter p (lambda = p**4)
    weights: np.ndarray,     # (C, K)
    n_gauss: np.ndarray,     # (C,) number of real (non-padding) gaussians per cube
    mae: np.ndarray,         # (C,)
    std_dev: np.ndarray,     # (C,)
    cube_size: float,
    margin: float,
    empty_margin: float = 0.25,
) -> None:
    """Write cubes to a GDF1 .bin. All cubes passed in are written verbatim."""
    C = origins.shape[0]
    if C == 0:
        raise ValueError("Refusing to write an empty GDF1 map (no valid cubes)")

    bounds_min = origins.min(axis=0)
    bounds_max = origins.max(axis=0) + cube_size
    avg_mae = float(np.mean(mae))
    avg_std = float(np.mean(std_dev))

    parts: list[bytes] = [struct.pack(
        _HEADER_FMT, b"GDF1", GDF1_VERSION, C, avg_mae, avg_std,
        float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2]),
        float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2]),
        float(cube_size), float(empty_margin), float(margin),
    )]
    for c in range(C):
        g = int(n_gauss[c])
        parts.append(struct.pack(
            _CUBE_FMT,
            float(origins[c, 0]), float(origins[c, 1]), float(origins[c, 2]),
            float(mae[c]), float(std_dev[c]), g,
        ))
        for k in range(g):
            parts.append(struct.pack(
                _GAUSS_FMT, k,
                float(means[c, k, 0]), float(means[c, k, 1]), float(means[c, k, 2]),
                float(sigmas[c, k, 0]), float(sigmas[c, k, 1]), float(sigmas[c, k, 2]),
                float(weights[c, k]),
            ))
    Path(path).write_bytes(b"".join(parts))
