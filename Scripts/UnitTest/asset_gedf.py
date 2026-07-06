"""
Synthetic shape samplers with analytic (unsigned) Euclidean distance fields,
shared by the G-EDF mapper tests.
"""
import numpy as np
import torch


def sample_plane(n: int = 4000, extent: float = 3.0, z: float = 1.0,
                 seed: int = 0) -> torch.Tensor:
    """Points on the plane z = const, over [0, extent]^2."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, extent, size=(n, 2))
    pts = np.column_stack([xy, np.full(n, z)])
    return torch.from_numpy(pts)


def plane_edf(points: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """Unsigned distance of query points to the (infinite) plane z = const."""
    return (points[:, 2] - z).abs()


def sample_sphere(n: int = 4000, center: tuple = (1.5, 1.5, 1.5),
                  radius: float = 1.0, seed: int = 0) -> torch.Tensor:
    """Points on a sphere surface."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = np.asarray(center) + radius * v
    return torch.from_numpy(pts)


def sphere_edf(points: torch.Tensor, center: tuple = (1.5, 1.5, 1.5),
               radius: float = 1.0) -> torch.Tensor:
    """Unsigned distance of query points to the sphere surface."""
    c = torch.tensor(center, dtype=points.dtype, device=points.device)
    return ((points - c).norm(dim=-1) - radius).abs()


def sample_two_bumps(n: int = 3000, seed: int = 0) -> torch.Tensor:
    """
    Two parallel plane patches (z = 0.15 and z = 0.85) inside a single unit
    cube, giving a distance field with a clear interior maximum between them.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    lo = np.column_stack([rng.uniform(0, 1, size=(half, 2)), np.full(half, 0.15)])
    hi = np.column_stack([rng.uniform(0, 1, size=(n - half, 2)), np.full(n - half, 0.85)])
    return torch.from_numpy(np.concatenate([lo, hi]))
