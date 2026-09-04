"""
Audit (and, on request, repair) sandboxes affected by the NED/EDN ground-truth
axis-convention bug.

Background
----------
Some `DataLoader` adapters used to write ground-truth trajectories
(`ref_poses.npy`) with camera axes in OpenCV/EDN convention
`[right, down, forward]`, while MAC-VO's own estimate (`poses.npy`) is always
written in NED convention `[forward, right, down]`. ATE is unaffected (it is
computed after a global SE(3)/Sim(3) alignment, which absorbs a fixed axis
relabeling of the *whole* trajectory), but ROE / RTE / RPE are computed from
*frame-to-frame relative motion* and are corrupted by this per-pose axis
mismatch. The dataloaders have since been fixed, so newly produced sandboxes
are correct. This script finds and (optionally) repairs sandboxes that were
produced by the old, buggy dataloaders and are already sitting on disk.

The correction, `NED2EDN`, is defined once in `Utility.Point` and imported
here (not redefined): rotation `[[0, 1, 0], [0, 0, 1], [1, 0, 0]]`.

Two-signal design: provenance decides the write, measurement decides "already
fixed"
------------------------------------------------------------------------
A first version of this script decided everything from measurement alone: it
scanned each sandbox's own trajectories and inferred whether the axis bug was
present from the numbers. Running that version across the full `Results/`
tree exposed why that is not sufficient on its own:

  - **False positive.** `Aqualoc`'s loader was deliberately left unfixed (its
    axis convention is unverified), yet some short/noisy Aqualoc runs score
    best under the `NED2EDN.T` probe purely by chance. Measurement alone
    would "fix" ground truth that was never touched by the bug, silently
    diverging it from any freshly-produced Aqualoc run.
  - **False negative.** Many real, affected EIVA sandboxes are short (~80
    frame) hyperparameter-sweep runs where the estimate's own VO error
    (several degrees of relative-orientation error) swamps the few-tenths-
    of-a-degree signal the axis probe relies on. No permutation wins
    cleanly, so a measurement-only classifier reports these as `OK` or
    `AMBIGUOUS` even though they need the fix.

The root cause: **the axis convention is a property of the dataloader that
produced a sandbox, not something that can always be reliably inferred from
a short, noisy trajectory.** That provenance is recorded authoritatively in
each sandbox's `config.yaml`, under the sequence's `type` field (e.g.
`Data.type` or, for a clipped sequence, `Data.args.type`) -- the same `type`
string used to `SubclassRegistry.instantiate()` the loader (see
`Utility/Extensions/SubclassRegistry.py`). This script therefore uses two
independent signals:

  - **Provenance** (`config.yaml`'s recorded sequence `type`) decides whether
    a sandbox is *eligible* to be fixed at all. Only the three loaders that
    were actually patched are eligible -- looked up from their registration,
    not guessed:
      - `DataLoader/Dataset/EIVA.py`: `EIVASequence.name()` -> `"EIVA_NoIMU"`
        (the *other* class in that file, `EIVA_StereoSequenceORM`, carries no
        ground truth at all -- `record_to_frame()` sets `gt_pose=None` -- so
        it is excluded on purpose).
      - `DataLoader/Dataset/SubPipe.py`: `SubPipeSequence.name()` ->
        `"SubPipe"`.
      - `DataLoader/Dataset/EiffelTower.py`: `EiffelTowerSequence.name()` ->
        `"EiffelTower_NoIMU"`.
    Every other recorded type (`Aqualoc_NoIMU`, `TartanAir_NoIMU`, `EuRoC`,
    `EuRoC_NoIMU`, `KITTI`, `VBR_Stereo`, anything else) -- and a sandbox
    whose `config.yaml` is missing or unparseable -- is *never* written,
    regardless of what the measurement says.
  - **Measurement** (the permutation scan below) decides whether an eligible
    sandbox has *already* been fixed, so a second pass is a safe no-op (see
    Idempotence, below), and is always reported for every row for
    transparency, independent of the write decision.

Detection algorithm (the measurement signal)
---------------------------------------------
For a sandbox directory containing both `poses.npy` (estimate) and
`ref_poses.npy` (ground truth), both `(N, 8)` arrays of
`[time_ns, tx, ty, tz, qx, qy, qz, qw]` (pypose XYZW order):

  1. Load both, truncate to `n = min(len(est), len(gt))`.
  2. Build rotation matrices `Re` (estimate) and `Rg` (ground truth) from the
     quaternions.
  3. Enumerate the 24 proper (det = +1) signed permutation matrices `S` --
     the rotation group of the cube / octahedral rotation group, which is
     exactly the set of matrices that can turn one right-handed orthogonal
     axis convention into another. For each `S`, right-multiply the
     ESTIMATE's rotations by `S` and score the *median frame-to-frame
     relative orientation error*, in degrees:

         E = (Rg_i^T Rg_{i+1})^T @ ((Re_i @ S)^T @ (Re_{i+1} @ S))
         error_i = degrees(arccos(clip((trace(E) - 1) / 2, -1, 1)))
         score(S) = median_i(error_i)

     This is a *relative*-motion metric (each estimate accrues its own
     drift relative to GT, so an absolute comparison would not isolate the
     axis-convention effect), which is exactly the failure mode of
     ROE/RTE/RPE that motivates this audit.
  4. `scan_verdict` (measurement only, reported for every row):
       - best-scoring `S` is the identity              -> `OK`
       - best-scoring `S` is `NED2EDN.T` (i.e.
         `[[0, 0, 1], [1, 0, 0], [0, 1, 0]]`), AND it beats the identity's
         score by at least `NEEDS_FIX_MARGIN`
         (identity_score >= NEEDS_FIX_MARGIN * best_score)   -> `NEEDS_FIX`
       - anything else                                       -> `AMBIGUOUS`

  `NEEDS_FIX_MARGIN` is a module-level constant chosen below the smallest
  margin observed in the verified NEEDS_FIX examples used to validate this
  script (~1.20x), so all of them clear it, while still requiring the
  NED2EDN.T reading to be a clear, not coincidental, winner.

Equivalence (already proven, used but not re-derived here): correcting the
ESTIMATE's rotation by `Sᵀ` scores identically to leaving the estimate
alone and instead correcting the GROUND TRUTH by `S`. This script *probes*
the estimate side (cheaper: one array of candidates, and it never risks
touching `poses.npy`), but *applies* the repair to the ground-truth side,
by right-multiplying every GT pose by `NED2EDN`.

Write decision (`action`, combines both signals)
---------------------------------------------------
For each sandbox, once `scan_verdict` (measurement) and `provenance_type`
(from `config.yaml`) are known:

    provenance_type is None                        -> UNKNOWN_PROVENANCE
    provenance_type not in FIXED_LOADER_TYPES       -> NOT_APPLICABLE
    a `ref_poses.orig.npy` backup already exists    -> ALREADY_FIXED
    scan is *reliable* and says identity is best    -> CONFLICT
    otherwise                                       -> FIX

Only rows with `action == "FIX"` are ever written by `--fix`. The first two
branches are the provenance gate described above. The third is the ordinary,
expected post-fix state (see Idempotence). The fourth, `CONFLICT`, is the
guard the false-positive/false-negative gap above calls for: provenance says
this sandbox's loader was patched and therefore *could* need fixing, but the
scan reliably shows its ground truth is already NED-convention-consistent
and there is no backup evidence that this script produced that state. That
combination is unexpected (e.g. a sandbox produced by a fixed loader after
the loader was patched, or one migrated by some other means) and is reported
-- never auto-fixed -- so a human can look at it.

A scan is "reliable" -- i.e. trustworthy enough to justify skipping a
provenance-eligible sandbox -- only when it has both enough frames and a
clear separation between the identity and NED2EDN.T readings:

    reliable = (n >= RELIABLE_MIN_N)
               and (best-scoring S is identity)
               and (ned2edn_t_score >= RELIABLE_MARGIN * identity_score)

`RELIABLE_MIN_N` (200) and `RELIABLE_MARGIN` (module-level constants) were
picked by checking every current EIVA/SubPipe/EiffelTower sandbox in
`Results/`: the ~80-frame sweep runs that must NOT be skipped (their own VO
error, several degrees, swamps the signal) all have `n < RELIABLE_MIN_N`, so
they correctly fall through to `FIX`; no sandbox in the current tree reaches
`n >= RELIABLE_MIN_N` with identity already best, so today's `Results/` tree
produces zero spurious `CONFLICT`s -- that branch exists for future runs
(e.g. a fixed-loader sandbox produced after the fix, sitting without a
backup marker) rather than anything observed today.

Idempotence guarantee (revised)
---------------------------------
Idempotence can no longer be "measurement alone decides, so re-running just
re-measures `OK`" -- that was exactly the false-negative failure mode above
(a genuinely-still-broken short sweep run can measure as spuriously `OK`).
Idempotence is instead guaranteed by construction from two independent,
redundant signals, either of which is sufficient to block a second write:

  1. **Backup marker.** The first successful fix on a sandbox writes
     `ref_poses.orig.npy` (unless `--no-backup`). Its mere presence means
     "this script already wrote this sandbox's `ref_poses.npy`", checked
     before ever touching the file again -- independent of what the
     permutation scan measures.
  2. **Reliable measurement.** Independently, after any correct fix,
     `ref_poses.npy`'s rotations score best at `S = identity` (right-
     multiplying by `NED2EDN` is exactly what made the pre-fix probe score
     best at `NED2EDN.T`). Whenever a sandbox has enough frames for that
     post-fix state to be measured reliably (see `RELIABLE_MIN_N`), it is
     independently caught by the `CONFLICT` branch (which never writes) even
     if the backup marker were somehow missing (e.g. `--no-backup` was used).

So: `--fix` writes only `NEEDS_FIX`-by-provenance sandboxes that clear
*neither* of those two guards, and after a write at least one of the guards
is live for any subsequent run of any size. `poses.npy` is never written by
this script under any circumstance, so it, too, is trivially idempotent
(there is nothing to converge to -- it is just read).

Usage
-----
    python Scripts/AdHoc/audit_gt_convention.py <path>              # report only
    python Scripts/AdHoc/audit_gt_convention.py <path> --fix         # repair FIX-action rows
    python Scripts/AdHoc/audit_gt_convention.py <path> --fix --no-backup

`<path>` may be a single sandbox (a directory directly containing both
`poses.npy` and `ref_poses.npy`) or a root to walk recursively for such
pairs.

`--fix` only ever rewrites `ref_poses.npy`, and only for rows whose `action`
is `FIX`; `poses.npy` is never modified. By default, a `ref_poses.orig.npy`
backup is written next to the file before the first fix (skipped if a backup
already exists there, so re-running `--fix` can never clobber the true
original); `--no-backup` disables this.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import pypose as pp
import yaml

from Utility.Point import NED2EDN

# Best-scoring S must be at least this many times better than identity for a
# NED2EDN.T match to be trusted as NEEDS_FIX rather than AMBIGUOUS. Chosen
# below the smallest margin (~1.20x) observed among the verified real-world
# NEEDS_FIX examples used to validate this script.
NEEDS_FIX_MARGIN = 1.15

# A scan is trusted to declare a provenance-eligible sandbox already-correct
# (and therefore skipped, see CONFLICT below) only with at least this many
# frames of relative motion to measure from, AND a clear separation between
# the identity and NED2EDN.T readings (RELIABLE_MARGIN). Below this frame
# count, real ~80-frame EIVA sweep runs are known to be indistinguishable
# from noise (their own VO error swamps the axis-convention signal), so they
# must NOT be treated as reliably-OK.
RELIABLE_MIN_N = 200
RELIABLE_MARGIN = 1.15

# Floor used when dividing by a near-zero score to avoid a spurious "infinite
# margin" from floating-point noise around a perfect match.
SCORE_EPS = 1e-9

IDENTITY_LABEL = "identity"
NED2EDN_T_LABEL = "NED2EDN.T"

# Sequence `type` names (as registered via SubclassRegistry / used as the
# `type:` field in Config/Sequence/*.yaml) for the loaders that were actually
# patched for the NED/EDN ground-truth bug. Looked up directly from each
# loader's `name()` override, not guessed:
#   DataLoader/Dataset/EIVA.py        : EIVASequence.name()        -> "EIVA_NoIMU"
#   DataLoader/Dataset/SubPipe.py     : SubPipeSequence.name()     -> "SubPipe"
#   DataLoader/Dataset/EiffelTower.py : EiffelTowerSequence.name() -> "EiffelTower_NoIMU"
# EIVA.py's other class, EIVA_StereoSequenceORM, is intentionally excluded:
# it never carries ground truth (record_to_frame() sets gt_pose=None), so it
# cannot be affected by this bug. Every other recorded type (Aqualoc_NoIMU,
# TartanAir_NoIMU/TartanAir, EuRoC/EuRoC_NoIMU, KITTI, VBR_Stereo, ...) is
# deliberately excluded -- some of those loaders' conventions are unverified
# (e.g. Aqualoc) and must never be rewritten by this script.
FIXED_LOADER_TYPES = frozenset({"EIVA_NoIMU", "SubPipe", "EiffelTower_NoIMU"})


def signed_permutation_matrices() -> list[np.ndarray]:
    """The 24 proper (det = +1) signed permutation matrices."""
    matrices = []
    for perm in itertools.permutations(range(3)):
        base = np.zeros((3, 3))
        for row, col in enumerate(perm):
            base[row, col] = 1.0
        for signs in itertools.product((1.0, -1.0), repeat=3):
            candidate = base * np.array(signs)[:, None]
            if abs(np.linalg.det(candidate) - 1.0) < 1e-9:
                matrices.append(candidate)
    assert len(matrices) == 24
    return matrices


def label_for(S: np.ndarray, identity: np.ndarray, ned2edn_t: np.ndarray) -> str:
    if np.allclose(S, identity):
        return IDENTITY_LABEL
    if np.allclose(S, ned2edn_t):
        return NED2EDN_T_LABEL
    perm = tuple(int(np.nonzero(S[row])[0][0]) for row in range(3))
    signs = tuple(int(S[row, perm[row]]) for row in range(3))
    return f"perm{perm}*sign{signs}"


@dataclass
class SandboxResult:
    name: str
    path: Path
    n: int
    identity_score: float
    ned2edn_t_score: float
    best_score: float
    best_label: str
    scan_verdict: str            # "OK" | "NEEDS_FIX" | "AMBIGUOUS"   (measurement only)
    provenance_type: str | None  # sequence `type` from config.yaml, or None if unknown
    backup_exists: bool
    reliable_identity_ok: bool
    action: str                  # "FIX" | "ALREADY_FIXED" | "CONFLICT" | "NOT_APPLICABLE" | "UNKNOWN_PROVENANCE"


def load_rotations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rotation matrices (N,3,3), full (N,8) pose array)."""
    arr = np.load(path)
    se3 = pp.SE3(torch.tensor(arr[:, 1:8], dtype=torch.float64))
    rot = se3.rotation().matrix().numpy()
    return rot, arr


def relative_rotations(R: np.ndarray) -> np.ndarray:
    """R_rel[i] = R[i]^T @ R[i+1], for consecutive frames."""
    return np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])


def score_candidate(Rg_rel: np.ndarray, Re: np.ndarray, S: np.ndarray) -> float:
    Re_corrected = Re @ S[None, :, :]
    Re_rel = relative_rotations(Re_corrected)
    E = np.einsum("nij,njk->nik", Rg_rel.transpose(0, 2, 1), Re_rel)
    trace = np.trace(E, axis1=1, axis2=2)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_angle))
    return float(np.median(angles_deg))


_CANDIDATES = signed_permutation_matrices()
_IDENTITY = np.eye(3)


def _ned2edn_rotation_matrix() -> np.ndarray:
    # NED2EDN is a pp.SE3 (translation-free); its rotation part is what the
    # docstring/spec calls NED2EDN. We need NED2EDN.T for the estimate-side
    # probe (see module docstring's equivalence note).
    return NED2EDN.rotation().matrix().numpy()


def get_sequence_type(sandbox_path: Path) -> str | None:
    """Reads the sequence `type` recorded in a sandbox's config.yaml.

    The saved config nests the sequence spec either directly under `Data`
    (`Data.type`), or, when the sequence was clipped (--seq_from/--seq_to),
    one level deeper (`Data.args.type`) -- confirmed by inspecting every
    config.yaml under Results/, which only ever exhibits these two shapes.
    Returns None if the file is missing, unparseable, or neither shape is
    present (unknown provenance -> never fixed).
    """
    config_path = sandbox_path / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None

    data = cfg.get("Data") if isinstance(cfg, dict) else None
    if not isinstance(data, dict):
        return None

    direct = data.get("type")
    if isinstance(direct, str):
        return direct

    nested = data.get("args")
    if isinstance(nested, dict) and isinstance(nested.get("type"), str):
        return nested["type"]

    return None


def decide_action(provenance_type: str | None, backup_exists: bool, reliable_identity_ok: bool) -> str:
    if provenance_type is None:
        return "UNKNOWN_PROVENANCE"
    if provenance_type not in FIXED_LOADER_TYPES:
        return "NOT_APPLICABLE"
    if backup_exists:
        return "ALREADY_FIXED"
    if reliable_identity_ok:
        return "CONFLICT"
    return "FIX"


def analyze_sandbox(name: str, path: Path) -> SandboxResult:
    est_rot, est_arr = load_rotations(path / "poses.npy")
    gt_rot, gt_arr = load_rotations(path / "ref_poses.npy")
    n = min(est_arr.shape[0], gt_arr.shape[0])
    Re, Rg = est_rot[:n], gt_rot[:n]

    ned2edn_t = _ned2edn_rotation_matrix().T

    Rg_rel = relative_rotations(Rg)
    scores = [score_candidate(Rg_rel, Re, S) for S in _CANDIDATES]
    labels = [label_for(S, _IDENTITY, ned2edn_t) for S in _CANDIDATES]

    identity_idx = labels.index(IDENTITY_LABEL)
    ned2edn_idx = labels.index(NED2EDN_T_LABEL)
    identity_score = scores[identity_idx]
    ned2edn_t_score = scores[ned2edn_idx]
    best_idx = int(np.argmin(scores))
    best_score = scores[best_idx]
    best_label = labels[best_idx]

    if best_label == IDENTITY_LABEL:
        scan_verdict = "OK"
    elif best_label == NED2EDN_T_LABEL and identity_score >= NEEDS_FIX_MARGIN * max(best_score, SCORE_EPS):
        scan_verdict = "NEEDS_FIX"
    else:
        scan_verdict = "AMBIGUOUS"

    reliable_identity_ok = (
        n >= RELIABLE_MIN_N
        and best_label == IDENTITY_LABEL
        and ned2edn_t_score >= RELIABLE_MARGIN * max(identity_score, SCORE_EPS)
    )

    provenance_type = get_sequence_type(path)
    backup_exists = (path / "ref_poses.orig.npy").exists()
    action = decide_action(provenance_type, backup_exists, reliable_identity_ok)

    return SandboxResult(
        name=name, path=path, n=n,
        identity_score=identity_score, ned2edn_t_score=ned2edn_t_score,
        best_score=best_score, best_label=best_label,
        scan_verdict=scan_verdict,
        provenance_type=provenance_type, backup_exists=backup_exists,
        reliable_identity_ok=reliable_identity_ok, action=action,
    )


def find_sandboxes(root: Path) -> list[tuple[str, Path]]:
    """Returns (display_name, path) pairs for every dir under root containing
    both poses.npy and ref_poses.npy. If root itself is such a dir, returns
    just that one."""
    if (root / "poses.npy").exists() and (root / "ref_poses.npy").exists():
        return [(root.name, root)]

    found = []
    for gt_path in sorted(root.rglob("ref_poses.npy")):
        sandbox_dir = gt_path.parent
        if (sandbox_dir / "poses.npy").exists():
            try:
                display = str(sandbox_dir.relative_to(root))
            except ValueError:
                display = str(sandbox_dir)
            found.append((display, sandbox_dir))
    return found


def apply_fix(ref_arr: np.ndarray) -> np.ndarray:
    """Returns a new (N,8) array with the rotation (and translation, per the
    SE3 composition law with a zero-translation correction) corrected by
    NED2EDN, timestamp column and dtype preserved."""
    orig_dtype = ref_arr.dtype
    time_col = ref_arr[:, 0:1]
    se3 = pp.SE3(torch.tensor(ref_arr[:, 1:8], dtype=torch.float64))
    corrected = se3 @ NED2EDN.double()
    corrected_np = corrected.numpy()
    return np.concatenate([time_col.astype(np.float64), corrected_np], axis=1).astype(orig_dtype)


def fix_sandbox(path: Path, backup: bool) -> None:
    ref_path = path / "ref_poses.npy"
    ref_arr = np.load(ref_path)

    if backup:
        backup_path = path / "ref_poses.orig.npy"
        if not backup_path.exists():
            np.save(backup_path, ref_arr)

    fixed = apply_fix(ref_arr)
    assert fixed.shape == ref_arr.shape
    assert fixed.dtype == ref_arr.dtype
    np.save(ref_path, fixed)


def print_table(results: list[SandboxResult]) -> None:
    if not results:
        print("No sandboxes with both poses.npy and ref_poses.npy found.")
        return

    name_w = max(len("sandbox"), max(len(r.name) for r in results))
    prov_w = max(len("provenance"), max(len(r.provenance_type or "?") for r in results))
    header = (
        f"{'sandbox':<{name_w}}  {'n':>6}  {'identity(deg)':>14}  "
        f"{'best(deg)':>10}  {'best-S':<20}  {'scan':<10}  "
        f"{'provenance':<{prov_w}}  {'action':<18}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<{name_w}}  {r.n:>6}  {r.identity_score:>14.4f}  "
            f"{r.best_score:>10.4f}  {r.best_label:<20}  {r.scan_verdict:<10}  "
            f"{(r.provenance_type or '?'):<{prov_w}}  {r.action:<18}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit (and optionally fix) NED/EDN ground-truth axis-convention bugs in MAC-VO sandboxes."
    )
    parser.add_argument("path", type=str, help="A sandbox directory, or a root to walk recursively for sandboxes.")
    parser.add_argument("--fix", action="store_true", help="Rewrite ref_poses.npy for FIX-action sandboxes.")
    parser.add_argument("--no-backup", action="store_true", help="Disable writing ref_poses.orig.npy before --fix.")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"Error: path does not exist: {root}", file=sys.stderr)
        return 1

    sandboxes = find_sandboxes(root)
    if not sandboxes:
        print(f"No sandboxes with both poses.npy and ref_poses.npy found under {root}.")
        return 0

    results: list[SandboxResult] = []
    for name, path in sandboxes:
        try:
            results.append(analyze_sandbox(name, path))
        except Exception as e:
            print(f"Error analyzing {name} ({path}): {e}", file=sys.stderr)

    print_table(results)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.action] = counts.get(r.action, 0) + 1
        if r.action == "FIX" and args.fix:
            try:
                fix_sandbox(r.path, backup=not args.no_backup)
            except Exception as e:
                print(f"Error fixing {r.name} ({r.path}): {e}", file=sys.stderr)

    fix_verb = "fixed" if args.fix else "would fix"
    print(
        f"\nSummary: {counts.get('FIX', 0)} {fix_verb}, "
        f"{counts.get('ALREADY_FIXED', 0)} already fixed, "
        f"{counts.get('CONFLICT', 0)} conflict (unwritten), "
        f"{counts.get('NOT_APPLICABLE', 0)} not applicable (other datasets), "
        f"{counts.get('UNKNOWN_PROVENANCE', 0)} unknown provenance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
