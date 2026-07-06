"""
G-EDF map: sparse grid of unit cubes, each holding a small Gaussian mixture that
approximates the local Euclidean Distance Field:

    d_hat(x) = sum_k w_k * exp(-0.5 * sum_a (x_a - mu_ka)^2 / lambda_ka)

with diagonal scales lambda = p^4 (p is the stored root parameter) and possibly
negative weights. Neighboring cubes are blended with a Smoothstep weight for C^1
continuity; empty regions return the sentinel `oob_value` (20.0) with zero gradient.

Query semantics match `G-EDF/scripts/gdf1_field.py` (the exact NumPy reference)
and `G-EDF-Loc/include/gaussian_map/gaussian_map.hpp`, with two deliberate
deviations from the C++:
  * no `dsq > 20` exp() cutoff (a speed hack; the NumPy reference has none), and
  * the analytic gradient includes the blend-weight quotient-rule term, i.e. it is
    the exact gradient of the blended field (the C++ treats blend weights as
    constants, which is only an approximation inside blend zones).
"""
import math
import typing as typ
from pathlib import Path

import numpy as np
import torch

from Utility.PrettyPrint import Logger

from .Config import GEDFConfig
from .Export import read_gdf1, write_gdf1

_KEY_BIAS = 1 << 20     # supports cube indices in [-2^20, 2^20), 21 bits per axis
_BW_EPS = 1e-6          # blend weights below this are treated as zero (reference parity)


def pack_keys(idx: torch.Tensor) -> torch.Tensor:
    """Pack integer cube coordinates (..., 3) into a single int64 key."""
    idx = idx.long()
    return (((idx[..., 0] + _KEY_BIAS) << 42)
            | ((idx[..., 1] + _KEY_BIAS) << 21)
            | (idx[..., 2] + _KEY_BIAS))


@typ.runtime_checkable
class GEDFMapProtocol(typ.Protocol):
    """What the GEDF factor graphs need from a map (online mapper or frozen file)."""
    cube_size: float
    margin: float

    @property
    def is_ready(self) -> bool: ...
    @property
    def sigma(self) -> float: ...
    def query(self, points_world: torch.Tensor) -> torch.Tensor: ...
    def query_with_grad(self, points_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def insert(self, points_world: torch.Tensor, cov_world: torch.Tensor | None = None) -> None: ...


class GEDFMapper:
    """
    Incremental G-EDF map. Storage is a padded structure-of-arrays over cube
    "slots" ([C, K, ...] tensors); padding slots use weight 0 and inv_lambda 0 so
    they contribute exactly zero to both value and gradient without masking.
    """

    def __init__(self, cfg: GEDFConfig, dtype: torch.dtype = torch.float32) -> None:
        self.cfg = cfg
        self.cube_size: float = cfg.cube_size
        self.margin: float = cfg.margin
        self.oob_value: float = cfg.oob_value
        self.frozen: bool = False
        # Minimum number of gaussians in valid cubes before `is_ready`; the
        # consuming optimizer overrides this from its own config.
        self.ready_min_gaussians: int = 1

        self._device = torch.device(cfg.device)
        self._dtype = dtype
        K = cfg.num_gaussians

        dev, dt = self._device, self._dtype
        self._origin_i = torch.empty((0, 3), device=dev, dtype=torch.int64)
        self._means = torch.empty((0, K, 3), device=dev, dtype=dt)
        self._p_sigma = torch.empty((0, K, 3), device=dev, dtype=dt)
        self._inv_lam = torch.empty((0, K, 3), device=dev, dtype=dt)
        self._weights = torch.empty((0, K), device=dev, dtype=dt)
        self._n_gauss = torch.empty((0,), device=dev, dtype=torch.int64)
        self._valid = torch.empty((0,), device=dev, dtype=torch.bool)
        self._mae = torch.empty((0,), device=dev, dtype=dt)
        self._std = torch.empty((0,), device=dev, dtype=dt)

        self._key2slot: dict[int, int] = {}
        self._sorted_keys: torch.Tensor | None = None
        self._sorted_slots: torch.Tensor | None = None

        # Incremental-mapping lifecycle (unused for frozen maps)
        self._points: dict[int, torch.Tensor] = {}      # per-slot deduped point buffer
        self._fine_keys: dict[int, torch.Tensor] = {}   # dedup-voxel keys per slot
        self._dirty: set[int] = set()
        self._n_points: dict[int, int] = {}
        self._n_at_fit: dict[int, int] = {}
        self._ctx_new: dict[int, int] = {}
        self._last_fit_frame: dict[int, int] = {}
        self._frame_counter: int = 0

        # 8 corner combinations of the (base, neighbor) candidate set, bits in {0, 1}
        self._combos = torch.tensor(
            [[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)],
            device=dev, dtype=torch.int64,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def num_cubes(self) -> int:
        return int(self._origin_i.shape[0])

    @property
    def num_valid_cubes(self) -> int:
        return int(self._valid.sum().item())

    @property
    def num_valid_gaussians(self) -> int:
        if self.num_cubes == 0:
            return 0
        return int(self._n_gauss[self._valid].sum().item())

    @property
    def is_ready(self) -> bool:
        return self.num_valid_gaussians >= self.ready_min_gaussians

    @property
    def sigma(self) -> float:
        """Running map error (mean per-cube MAE over valid cubes), in meters."""
        if self.num_valid_cubes == 0:
            return float("inf")
        return float(self._mae[self._valid].mean().item())

    # ------------------------------------------------------------------ #
    # Construction from / export to GDF1
    # ------------------------------------------------------------------ #
    @classmethod
    def from_gdf1(cls, path: str | Path, cfg: GEDFConfig | None = None,
                  dtype: torch.dtype = torch.float64) -> "GEDFMapper":
        """
        Load a pre-built map. Cube origins are snapped to the world-anchored grid
        exactly like G-EDF-Loc's loader: idx = floor((origin + 0.01 * cs) / cs);
        the canonical origin used for blending is idx * cs.
        """
        header, cubes = read_gdf1(path)
        cs = float(header["cube_size"])

        cfg = cfg if cfg is not None else GEDFConfig(device="cpu")
        cfg.cube_size = cs
        cfg.margin = float(header["margin"])
        cfg.num_gaussians = max(1, max((c["means"].shape[0] for c in cubes), default=1))

        mapper = cls(cfg, dtype=dtype)
        mapper.frozen = True

        if len(cubes) == 0:
            return mapper

        origins = np.stack([c["origin"] for c in cubes])                 # (C, 3)
        idx = np.floor((origins + 0.01 * cs) / cs).astype(np.int64)      # (C, 3)
        misaligned = np.abs(idx * cs - origins).max()
        if misaligned > 0.02 * cs:
            Logger.write("warn", f"[GEDF] {path}: cube origins deviate from the world grid "
                                 f"by up to {misaligned:.4f} m; queries use the snapped grid.")

        C, K = len(cubes), cfg.num_gaussians
        means = np.zeros((C, K, 3)); p_sig = np.zeros((C, K, 3))
        w = np.zeros((C, K)); ng = np.zeros((C,), dtype=np.int64)
        mae = np.zeros((C,)); std = np.zeros((C,))
        for i, c in enumerate(cubes):
            g = c["means"].shape[0]
            ng[i], mae[i], std[i] = g, c["mae"], c["std_dev"]
            means[i, :g] = c["means"]
            # Sanitize like the C++ loader: |p| < 1e-4 -> 1e-4, non-finite weight -> 0
            sig = c["sigmas"].copy()
            sig[np.abs(sig) < 1e-4] = 1e-4
            p_sig[i, :g] = sig
            wi = c["weights"].copy()
            wi[~np.isfinite(wi)] = 0.0
            w[i, :g] = wi

        mapper._append_cubes(
            torch.from_numpy(idx),
            torch.from_numpy(means), torch.from_numpy(p_sig), torch.from_numpy(w),
            torch.from_numpy(ng),
            torch.from_numpy(mae), torch.from_numpy(std),
            valid=torch.ones(C, dtype=torch.bool),
        )
        return mapper

    def export_gdf1(self, path: str | Path) -> None:
        """Write all valid cubes to a GDF1 .bin (loadable by the G-EDF viz tools)."""
        valid = self._valid.cpu().numpy()
        if not valid.any():
            raise ValueError("GEDF map has no valid cubes to export")
        sel = np.flatnonzero(valid)
        origins = (self._origin_i.cpu().numpy().astype(np.float64) * self.cube_size)[sel]
        write_gdf1(
            path,
            origins=origins,
            means=self._means.cpu().double().numpy()[sel],
            sigmas=self._p_sigma.cpu().double().numpy()[sel],
            weights=self._weights.cpu().double().numpy()[sel],
            n_gauss=self._n_gauss.cpu().numpy()[sel],
            mae=self._mae.cpu().double().numpy()[sel],
            std_dev=self._std.cpu().double().numpy()[sel],
            cube_size=self.cube_size,
            margin=self.margin,
        )

    # ------------------------------------------------------------------ #
    # Slot storage
    # ------------------------------------------------------------------ #
    def _append_cubes(self, origin_i: torch.Tensor, means: torch.Tensor,
                      p_sigma: torch.Tensor, weights: torch.Tensor,
                      n_gauss: torch.Tensor, mae: torch.Tensor, std: torch.Tensor,
                      valid: torch.Tensor) -> None:
        """Append fully-specified cube rows. Padding slots must carry w=0, p=0."""
        dev, dt = self._device, self._dtype
        p = p_sigma.to(dev, dt)
        lam = p.double() ** 4
        inv_lam = torch.where(lam > 0, 1.0 / lam, torch.zeros_like(lam)).to(dt)

        first_slot = self.num_cubes
        self._origin_i = torch.cat([self._origin_i, origin_i.to(dev, torch.int64)])
        self._means = torch.cat([self._means, means.to(dev, dt)])
        self._p_sigma = torch.cat([self._p_sigma, p])
        self._inv_lam = torch.cat([self._inv_lam, inv_lam])
        self._weights = torch.cat([self._weights, weights.to(dev, dt)])
        self._n_gauss = torch.cat([self._n_gauss, n_gauss.to(dev, torch.int64)])
        self._mae = torch.cat([self._mae, mae.to(dev, dt)])
        self._std = torch.cat([self._std, std.to(dev, dt)])
        self._valid = torch.cat([self._valid, valid.to(dev, torch.bool)])

        keys = pack_keys(origin_i).tolist()
        for i, key in enumerate(keys):
            assert key not in self._key2slot, f"Duplicate cube key {key}"
            self._key2slot[key] = first_slot + i
        self._sorted_keys = None    # invalidate the batched-lookup index

    def _lookup(self, keys: torch.Tensor) -> torch.Tensor:
        """Batched key -> slot lookup; -1 for misses. keys: (...,) int64."""
        if self.num_cubes == 0:
            return torch.full_like(keys, -1)
        if self._sorted_keys is None:
            all_keys = pack_keys(self._origin_i)
            self._sorted_keys, order = torch.sort(all_keys)
            self._sorted_slots = torch.arange(
                self.num_cubes, device=self._device, dtype=torch.int64)[order]
        assert self._sorted_slots is not None
        flat = keys.reshape(-1).contiguous()
        pos = torch.searchsorted(self._sorted_keys, flat).clamp_(max=self.num_cubes - 1)
        hit = self._sorted_keys[pos] == flat
        slots = torch.where(hit, self._sorted_slots[pos], torch.full_like(flat, -1))
        return slots.reshape(keys.shape)

    # ------------------------------------------------------------------ #
    # Query path
    # ------------------------------------------------------------------ #
    def _candidates(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each point, the <= 8 cubes whose margin-expanded box may contain it
        (base cube plus the neighbor across each near-boundary axis).

        Returns (slot (M,8) int64 clamped >= 0, origin (M,8,3) world coords of the
        candidate cubes, valid (M,8) bool). Everything except `valid`/indices is
        differentiable w.r.t. `points` (piecewise; index selection is constant).
        """
        cs, margin = self.cube_size, self.margin
        base = torch.floor(points.detach() / cs).long()                  # (M, 3)
        local = points.detach() - base.to(points.dtype) * cs             # (M, 3)
        step = torch.zeros_like(base)
        step = torch.where(local < margin, step - 1, step)
        step = torch.where(local > cs - margin, step + 1, step)          # (M, 3) in {-1,0,1}

        combos = self._combos                                            # (8, 3) in {0,1}
        cand = base.unsqueeze(1) + combos.unsqueeze(0) * step.unsqueeze(1)  # (M, 8, 3)
        # A combo bit set on an axis whose step is 0 duplicates the base cube.
        dup = ((combos.unsqueeze(0) == 1) & (step.unsqueeze(1) == 0)).any(-1)  # (M, 8)

        slot = self._lookup(pack_keys(cand))                             # (M, 8)
        valid = (~dup) & (slot >= 0)
        slot = slot.clamp(min=0)
        valid = valid & self._valid[slot]
        origin = cand.to(points.dtype) * cs                              # (M, 8, 3)
        return slot, origin, valid

    def query(self, points_world: torch.Tensor) -> torch.Tensor:
        """
        Blended field value at world points (M, 3) -> (M,). Differentiable w.r.t.
        the query points (used by the autodiff factor graphs); out-of-map points
        return `oob_value` with zero gradient.
        """
        assert points_world.ndim == 2 and points_world.shape[-1] == 3
        points = points_world.to(self._device)   # device-transparent (autograd flows through .to)
        if points.shape[0] == 0 or self.num_cubes == 0:
            return torch.full(points_world.shape[:1], self.oob_value,
                              device=points_world.device, dtype=points_world.dtype)

        slot, origin, valid = self._candidates(points)
        bw = self._blend_weights(points, origin, valid)                  # (M, 8)
        f = self._eval_cubes(points, slot)                               # (M, 8)

        S = (bw * f).sum(-1)
        W = bw.sum(-1)
        good = W > _BW_EPS
        W_safe = torch.where(good, W, torch.ones_like(W))
        out = torch.where(good, S / W_safe, torch.full_like(S, self.oob_value))
        return out.to(points_world.device)

    @torch.no_grad()
    def query_with_grad(self, points_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Field value and its exact analytic spatial gradient at world points:
        (M, 3) -> ((M,), (M, 3)). Out-of-map points return (oob_value, 0).
        """
        assert points_world.ndim == 2 and points_world.shape[-1] == 3
        points = points_world.to(self._device)
        M = points.shape[0]
        if M == 0 or self.num_cubes == 0:
            return (torch.full((M,), self.oob_value,
                               device=points_world.device, dtype=points_world.dtype),
                    torch.zeros((M, 3), device=points_world.device, dtype=points_world.dtype))

        slot, origin, valid = self._candidates(points)

        # Blend weights and their gradients w.r.t. the point
        d_lo = points.unsqueeze(1) - origin                              # (M, 8, 3)
        d_hi = (origin + self.cube_size) - points.unsqueeze(1)           # (M, 8, 3)
        face = torch.cat([d_lo, d_hi], dim=-1)                           # (M, 8, 6)
        min_dist, fidx = face.min(dim=-1)                                # (M, 8)
        t = torch.clamp(1.0 + min_dist / self.margin, 0.0, 1.0)
        bw = t * t * (3.0 - 2.0 * t)
        active = valid & (bw > _BW_EPS)
        bw = torch.where(active, bw, torch.zeros_like(bw))

        dt = torch.where(active, 6.0 * t * (1.0 - t) / self.margin, torch.zeros_like(t))
        sign = torch.where(fidx < 3, torch.ones_like(min_dist), -torch.ones_like(min_dist))
        g_bw = torch.zeros_like(origin)                                  # (M, 8, 3)
        g_bw.scatter_(-1, (fidx % 3).unsqueeze(-1), (dt * sign).unsqueeze(-1))

        # Per-candidate field value and gradient
        means = self._means[slot].to(points.dtype)                       # (M, 8, K, 3)
        inv_lam = self._inv_lam[slot].to(points.dtype)
        w = self._weights[slot].to(points.dtype)                         # (M, 8, K)
        d = points.unsqueeze(1).unsqueeze(2) - means                     # (M, 8, K, 3)
        e = w * torch.exp(-0.5 * (d * d * inv_lam).sum(-1))              # (M, 8, K)
        f = e.sum(-1)                                                    # (M, 8)
        g_f = -(e.unsqueeze(-1) * (d * inv_lam)).sum(-2)                 # (M, 8, 3)

        # Quotient rule over the blended sum:  d_hat = S / W
        S = (bw * f).sum(-1)                                             # (M,)
        W = bw.sum(-1)
        g_S = (g_bw * f.unsqueeze(-1) + bw.unsqueeze(-1) * g_f).sum(-2)  # (M, 3)
        g_W = g_bw.sum(-2)                                               # (M, 3)

        good = W > _BW_EPS
        W_safe = torch.where(good, W, torch.ones_like(W))
        dist = torch.where(good, S / W_safe, torch.full_like(S, self.oob_value))
        grad = (g_S - dist.unsqueeze(-1) * g_W) / W_safe.unsqueeze(-1)
        grad = torch.where(good.unsqueeze(-1), grad, torch.zeros_like(grad))
        return dist.to(points_world.device), grad.to(points_world.device)

    def _blend_weights(self, points: torch.Tensor, origin: torch.Tensor,
                       valid: torch.Tensor) -> torch.Tensor:
        """Smoothstep blend weight of each candidate cube at each point (M, 8)."""
        d_lo = points.unsqueeze(1) - origin
        d_hi = (origin + self.cube_size) - points.unsqueeze(1)
        min_dist = torch.minimum(d_lo, d_hi).amin(-1)                    # (M, 8)
        t = torch.clamp(1.0 + min_dist / self.margin, 0.0, 1.0)
        bw = t * t * (3.0 - 2.0 * t)
        return torch.where(valid & (bw > _BW_EPS), bw, torch.zeros_like(bw))

    def _eval_cubes(self, points: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """Raw (un-blended) GMM value of each candidate cube at each point (M, 8)."""
        means = self._means[slot].to(points.dtype)                       # (M, 8, K, 3)
        inv_lam = self._inv_lam[slot].to(points.dtype)
        w = self._weights[slot].to(points.dtype)
        d = points.unsqueeze(1).unsqueeze(2) - means
        dsq = (d * d * inv_lam).sum(-1)                                  # (M, 8, K)
        return (w * torch.exp(-0.5 * dsq)).sum(-1)

    @torch.no_grad()
    def sample_surface(self, resolution: float = 0.10, iso: float = 0.10,
                       max_points: int = 100_000) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the near-surface region of the field: grid-sample every valid cube
        at `resolution` and keep points whose field value is below `iso`.

        Returns (points (M,3), dist (M,)) as float32 CPU tensors, M <= max_points
        (evenly-spaced subsample beyond). Empty map -> empty tensors. Used by the
        Rerun visualization (live snapshots and the offline viewer script).
        """
        empty = (torch.zeros((0, 3), dtype=torch.float32),
                 torch.zeros((0,), dtype=torch.float32))
        if self.num_valid_cubes == 0:
            return empty

        cs = self.cube_size
        G = max(1, int(round(cs / resolution)))
        ax = (torch.arange(G, device=self._device, dtype=self._dtype) + 0.5) * (cs / G)
        gz, gy, gx = torch.meshgrid(ax, ax, ax, indexing="ij")
        template = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)      # (G^3, 3)

        origins = self._origin_i[self._valid].to(self._dtype) * cs      # (V, 3)
        pts_all: list[torch.Tensor] = []
        dist_all: list[torch.Tensor] = []
        chunk = max(1, (2 ** 21) // template.shape[0])                   # ~2M points per query
        for i in range(0, origins.shape[0], chunk):
            pts = (origins[i:i + chunk].unsqueeze(1) + template.unsqueeze(0)).reshape(-1, 3)
            dist = self.query(pts)
            keep = dist < iso
            pts_all.append(pts[keep])
            dist_all.append(dist[keep])

        points = torch.cat(pts_all).float().cpu()
        dist = torch.cat(dist_all).float().cpu()
        if points.shape[0] > max_points:
            sel = torch.linspace(0, points.shape[0] - 1, max_points).long()
            points, dist = points[sel], dist[sel]
        return points, dist

    # ------------------------------------------------------------------ #
    # Incremental mapping
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def insert(self, points_world: torch.Tensor, cov_world: torch.Tensor | None = None) -> None:
        """
        Add world-frame points (N, 3) to the per-cube buffers, deduplicating on a
        fine voxel grid and marking touched (and halo-affected neighbor) cubes
        dirty for the next `refit()` call. Does not fit anything by itself.
        """
        assert not self.frozen, "Cannot insert into a frozen (pre-built) G-EDF map"
        cfg = self.cfg
        self._frame_counter += 1

        pts = points_world.to(self._device, self._dtype).reshape(-1, 3)
        keep = torch.isfinite(pts).all(-1)
        if cov_world is not None and cfg.cov_trace_gate > 0:
            tr = cov_world.to(self._device).diagonal(dim1=-2, dim2=-1).sum(-1)
            keep = keep & (tr.reshape(-1).to(self._dtype) <= cfg.cov_trace_gate)
        pts = pts[keep]
        if pts.shape[0] == 0:
            return

        cube_idx = torch.floor(pts / self.cube_size).long()              # (N, 3)
        keys = pack_keys(cube_idx)
        uniq_keys, inverse = torch.unique(keys, return_inverse=True)

        # Create slots for previously unseen cubes (zero params, invalid).
        new_mask = torch.tensor([int(k.item()) not in self._key2slot for k in uniq_keys],
                                device=self._device, dtype=torch.bool)
        if bool(new_mask.any()):
            new_keys = uniq_keys[new_mask]
            sel = (keys.unsqueeze(0) == new_keys.unsqueeze(1)).float().argmax(dim=1)
            new_origin_i = cube_idx[sel]                                 # (n_new, 3)
            self._new_slots(new_origin_i)

        halo_counts: dict[int, int] = {}
        for u, key_t in enumerate(uniq_keys):
            key = int(key_t.item())
            slot = self._key2slot[key]
            p_new = pts[inverse == u]

            # Fine-voxel dedup against the existing buffer and within the batch
            fkey = pack_keys(torch.floor(p_new / cfg.dedup_voxel).long())
            fkey, first_idx = _unique_first(fkey)
            p_new = p_new[first_idx]
            old_fkeys = self._fine_keys.get(slot)
            if old_fkeys is not None and old_fkeys.numel() > 0:
                fresh = ~torch.isin(fkey, old_fkeys)
                p_new, fkey = p_new[fresh], fkey[fresh]
            if p_new.shape[0] == 0:
                continue

            buf = self._points.get(slot)
            buf = p_new if buf is None else torch.cat([buf, p_new])
            fbuf = fkey if old_fkeys is None else torch.cat([old_fkeys, fkey])
            if buf.shape[0] > cfg.max_points_per_cube:                   # cap: subsample
                # evenly spaced over insertion order: deterministic across devices
                sel = torch.linspace(0, buf.shape[0] - 1, cfg.max_points_per_cube,
                                     device=self._device).long()
                buf, fbuf = buf[sel], fbuf[sel]
            self._points[slot], self._fine_keys[slot] = buf, fbuf
            self._n_points[slot] = int(buf.shape[0])

            # Halo marking: which existing neighbor cubes gained EDT-context points
            origin = self._origin_i[slot].to(self._dtype) * self.cube_size
            local = p_new - origin
            step = torch.zeros_like(p_new, dtype=torch.int64)
            step = torch.where(local < cfg.halo, step - 1, step)
            step = torch.where(local > self.cube_size - cfg.halo, step + 1, step)
            combos = self._combos
            cand = self._origin_i[slot].unsqueeze(0).unsqueeze(0) \
                 + combos.unsqueeze(0) * step.unsqueeze(1)               # (n, 8, 3)
            dup = ((combos.unsqueeze(0) == 1) & (step.unsqueeze(1) == 0)).any(-1)
            nkeys = pack_keys(cand)[~dup]
            nkeys = nkeys[nkeys != key]
            for nk, cnt in zip(*[t.tolist() for t in torch.unique(nkeys, return_counts=True)]):
                halo_counts[int(nk)] = halo_counts.get(int(nk), 0) + int(cnt)

            self._maybe_mark_dirty(slot)

        for nk, cnt in halo_counts.items():
            nslot = self._key2slot.get(nk)
            if nslot is None:
                continue
            self._ctx_new[nslot] = self._ctx_new.get(nslot, 0) + cnt
            if bool(self._valid[nslot]) and self._ctx_new[nslot] >= cfg.ctx_refit_min_new:
                self._dirty.add(nslot)

    @torch.no_grad()
    def refit(self, camera_pos: torch.Tensor | None = None, budget: int | None = None) -> dict:
        """
        Fit up to `budget` dirty cubes (highest priority first). Cubes keep
        serving their last valid parameters while dirty / after failed fits.
        """
        assert not self.frozen, "Cannot refit a frozen (pre-built) G-EDF map"
        from .Fitting import fit_batch    # local import: avoids cycle at module load

        cfg = self.cfg
        budget = cfg.budget_cubes_per_frame if budget is None else budget
        candidates = [s for s in self._dirty
                      if self._n_points.get(s, 0) >= cfg.min_points_fit]
        stats = {"refitted": 0, "cold": 0, "dirty_remaining": len(self._dirty),
                 "mean_mae": self.sigma if self.num_valid_cubes > 0 else float("nan")}
        if not candidates:
            return stats

        cam = None if camera_pos is None else camera_pos.to(self._device, self._dtype).reshape(3)

        def score(slot: int) -> float:
            cold = not bool(self._valid[slot])
            n_new = self._n_points.get(slot, 0) - self._n_at_fit.get(slot, 0)
            recency = self._frame_counter - self._last_fit_frame.get(slot, 0)
            s = 10.0 * cold + n_new + 0.5 * self._ctx_new.get(slot, 0) + 0.1 * recency
            if cam is not None:
                center = (self._origin_i[slot].to(self._dtype) + 0.5) * self.cube_size
                s /= 1.0 + float((center - cam).norm().item()) / cfg.camera_dist_tau
            return s

        candidates.sort(key=score, reverse=True)
        batch = candidates[:budget]

        warm_ok = {s: bool(self._valid[s]) and
                      float(self._mae[s].item()) <= cfg.cold_restart_mae_factor * cfg.mae_target
                   for s in batch}
        cold_slots = [s for s in batch if not warm_ok[s]]
        warm_slots = [s for s in batch if warm_ok[s]]

        for slots, warm in ((cold_slots, False), (warm_slots, True)):
            if not slots:
                continue
            pts_pad, mask, origins = self._gather_local_clouds(slots)
            theta_warm = None
            if warm:
                theta_warm = torch.cat([
                    torch.cat([self._means[s], self._p_sigma[s],
                               self._weights[s].unsqueeze(-1)], dim=-1).unsqueeze(0)
                    for s in slots]).double()
            seed = (sum(pack_keys(self._origin_i[slots]).tolist())
                    + sum(self._n_points.get(s, 0) for s in slots)) & 0x7FFFFFFF
            theta, mae, std, usable = fit_batch(pts_pad, mask, origins, theta_warm, cfg, seed=seed)

            for i, slot in enumerate(slots):
                was_valid = bool(self._valid[slot])
                accept = bool(usable[i]) and (
                    not was_valid or float(mae[i]) <= float(self._mae[slot]) * 1.5)
                if accept:
                    self._means[slot] = theta[i, :, :3].to(self._dtype)
                    p = theta[i, :, 3:6].to(self._dtype)
                    self._p_sigma[slot] = p
                    lam = p.double() ** 4
                    self._inv_lam[slot] = torch.where(
                        lam > 0, 1.0 / lam, torch.zeros_like(lam)).to(self._dtype)
                    self._weights[slot] = theta[i, :, 6].to(self._dtype)
                    self._n_gauss[slot] = cfg.num_gaussians
                    self._mae[slot] = mae[i].to(self._dtype)
                    self._std[slot] = std[i].to(self._dtype)
                    self._valid[slot] = True
                    stats["refitted"] += 1
                    stats["cold"] += int(not was_valid)
                # In both cases the counters reset and the cube leaves the queue;
                # it re-enters on the next insert.
                self._n_at_fit[slot] = self._n_points.get(slot, 0)
                self._ctx_new[slot] = 0
                self._last_fit_frame[slot] = self._frame_counter
                self._dirty.discard(slot)

        stats["dirty_remaining"] = len(self._dirty)
        stats["mean_mae"] = self.sigma if self.num_valid_cubes > 0 else float("nan")
        return stats

    def flush(self, camera_pos: torch.Tensor | None = None, max_rounds: int = 100) -> None:
        """Refit until the dirty queue is drained (tests / offline use)."""
        for _ in range(max_rounds):
            fittable = [s for s in self._dirty
                        if self._n_points.get(s, 0) >= self.cfg.min_points_fit]
            if not fittable:
                break
            self.refit(camera_pos=camera_pos)

    def _maybe_mark_dirty(self, slot: int) -> None:
        cfg = self.cfg
        n = self._n_points.get(slot, 0)
        n_fit = self._n_at_fit.get(slot, 0)
        cold = not bool(self._valid[slot])
        n_new = n - n_fit
        if (cold and n >= cfg.min_points_fit) \
           or n_new >= cfg.refit_min_new \
           or (n_fit > 0 and n_new >= cfg.refit_growth * n_fit) \
           or self._ctx_new.get(slot, 0) >= cfg.ctx_refit_min_new:
            self._dirty.add(slot)

    def _new_slots(self, origin_i: torch.Tensor) -> None:
        """Create empty (invalid) slots for new cube coordinates (n, 3)."""
        n, K = origin_i.shape[0], self.cfg.num_gaussians
        dt = self._dtype
        zeros3 = torch.zeros((n, K, 3), dtype=dt)
        self._append_cubes(
            origin_i, zeros3, zeros3.clone(), torch.zeros((n, K), dtype=dt),
            torch.zeros((n,), dtype=torch.int64),
            torch.zeros((n,), dtype=dt), torch.zeros((n,), dtype=dt),
            valid=torch.zeros((n,), dtype=torch.bool),
        )

    def _gather_local_clouds(self, slots: list[int]
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-cube local cloud = own buffer + neighbor points clipped to the halo
        box [origin - halo, origin + cube + halo], padded across the batch.
        """
        cfg = self.cfg
        clouds: list[torch.Tensor] = []
        origins = self._origin_i[torch.tensor(slots, device=self._device)] \
                      .to(self._dtype) * self.cube_size
        offsets = torch.stack(torch.meshgrid(
            *([torch.tensor([-1, 0, 1], device=self._device)] * 3), indexing="ij"),
            dim=-1).reshape(-1, 3)                                       # (27, 3)
        for i, slot in enumerate(slots):
            parts = []
            own = self._points.get(slot)
            if own is not None:
                parts.append(own)
            nkeys = pack_keys(self._origin_i[slot].unsqueeze(0) + offsets)
            lo = origins[i] - cfg.halo
            hi = origins[i] + self.cube_size + cfg.halo
            for nk in nkeys.tolist():
                nslot = self._key2slot.get(int(nk))
                if nslot is None or nslot == slot:
                    continue
                np_buf = self._points.get(nslot)
                if np_buf is None or np_buf.shape[0] == 0:
                    continue
                inside = ((np_buf >= lo) & (np_buf <= hi)).all(-1)
                if bool(inside.any()):
                    parts.append(np_buf[inside])
            cloud = torch.cat(parts) if parts else torch.zeros((0, 3), device=self._device,
                                                               dtype=self._dtype)
            cap = 2 * cfg.max_points_per_cube
            if cloud.shape[0] > cap:
                sel = torch.linspace(0, cloud.shape[0] - 1, cap, device=self._device).long()
                cloud = cloud[sel]
            clouds.append(cloud)

        P = max(1, max(int(c.shape[0]) for c in clouds))
        B = len(clouds)
        pts_pad = torch.zeros((B, P, 3), device=self._device, dtype=self._dtype)
        mask = torch.zeros((B, P), device=self._device, dtype=torch.bool)
        for i, c in enumerate(clouds):
            pts_pad[i, :c.shape[0]] = c
            mask[i, :c.shape[0]] = True
        return pts_pad, mask, origins


def _unique_first(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unique keys plus the index of the first occurrence of each (order-stable)."""
    uniq, inverse = torch.unique(keys, return_inverse=True)
    order = torch.arange(keys.shape[0], device=keys.device)
    first = torch.full((uniq.shape[0],), keys.shape[0], device=keys.device, dtype=torch.long)
    first.scatter_reduce_(0, inverse, order, reduce="amin")
    return uniq, first
