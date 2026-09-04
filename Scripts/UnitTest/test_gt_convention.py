"""
Regression tests for the EDN/NED ground-truth axis-convention fix.

MAC-VO uses NED camera axes [forward, right, down] internally (see
``Utility.Point.pixel2point_NED`` / ``point2pixel_NED``). The EIVA, SubPipe and
EiffelTower ground-truth loaders used to emit poses whose camera/body axes were
still OpenCV/EDN [right, down, forward], with no conversion applied. Because
ATE only compares positions, that bug was invisible there; it inflated the
ROE/RTE/RPE (orientation-sensitive) metrics instead.

The fix (already applied to the working tree):
  - ``Utility/Point.py`` defines canonical ``EDN2NED`` / ``NED2EDN`` SE3
    LieTensors.
  - ``DataLoader/Dataset/EIVA.py::loadEIVAGT`` and
    ``DataLoader/Dataset/SubPipe.py::loadSubPipeGT`` now right-multiply the
    parsed poses by ``NED2EDN`` (the world frame was already NED-like, so a
    pure body-axis rebase is enough).
  - ``DataLoader/Dataset/EiffelTower.py::loadEiffelTowerGT`` sandwiches with
    ``ned_R @ ... @ NED2EDN`` (a full change of basis, because there the
    world frame IS the initial camera frame).
  - ``DataLoader/Dataset/TartanAir.py::loadTartanAirGT`` is deliberately left
    unconverted (TartanAir ships GT that is already NED).

This module is pure/synthetic and CPU-only, and is the first test coverage of
these loaders -- it is meant to be the regression net that stops this bug (or
its EiffelTower/SubPipe siblings) from silently coming back.

Style follows the synthetic-pose tests in ``test_gtsam_alignment.py``.
"""
import numpy as np
import pytest
import torch
import pypose as pp
from scipy.spatial.transform import Rotation as R

from Utility.Point import EDN2NED, NED2EDN


# --------------------------------------------------------------------------- #
# Synthetic trajectory helpers
# --------------------------------------------------------------------------- #
def _make_synthetic_edn_trajectory(n: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """N camera->world poses (T_wc), expressed with EDN [right, down, forward]
    camera/body axes, as (translation_xyz, quaternion_xyzw) pairs. Rotations are
    non-trivial (random axis-angle, 0.3-1.2 rad) so the tests actually exercise
    the axis conversion instead of passing vacuously on an identity rotation.
    """
    rng = np.random.default_rng(42)
    poses = []
    t = np.zeros(3)
    for _ in range(n):
        t = t + rng.normal(size=3) * 0.5 + np.array([1.0, 0.0, 0.0])
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = rng.uniform(0.3, 1.2)
        quat = R.from_rotvec(axis * angle).as_quat()  # [x, y, z, w]
        poses.append((t.copy(), quat))
    return poses


def _write_zephyr_gt_file(path, poses_wc: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Write poses_wc (camera->world, EDN) to the on-disk "Zephyr Voyis" format
    that loadEIVAGT / loadSubPipeGT parse: one line per frame,
    ``filename tx ty tz R00 R01 R02 R10 R11 R12 R20 R21 R22`` (space-separated,
    row-major 3x3, world->camera -- the loader inverts it back to camera->world).
    """
    lines = []
    for i, (t, quat) in enumerate(poses_wc):
        T_wc = pp.SE3(torch.tensor([*t, *quat], dtype=torch.float64))
        T_cw = T_wc.Inv()
        mat = T_cw.matrix().numpy()
        R_cw, t_cw = mat[:3, :3], mat[:3, 3]
        vals = [f"{v:.10f}" for v in t_cw] + [f"{v:.10f}" for v in R_cw.flatten()]
        lines.append(f"frame_{i:04d}.png " + " ".join(vals))
    path.write_text("\n".join(lines) + "\n")


def _roe_deg(Rg: torch.Tensor, Re: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame relative orientation error in degrees, evo-style:
    E = (Rg_i^T Rg_{i+1})^T (Re_i^T Re_{i+1}); angle = arccos((trace(E)-1)/2).
    Rg, Re: (N, 3, 3) rotation matrices of equal length; returns (N-1,) degrees.
    """
    dRg = Rg[:-1].transpose(-1, -2) @ Rg[1:]
    dRe = Re[:-1].transpose(-1, -2) @ Re[1:]
    E = dRg.transpose(-1, -2) @ dRe
    tr = E.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((tr - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos_theta))


def _import_loadEIVAGT():
    # DataLoader/Dataset/EIVA.py imports django at module scope. Skip (rather
    # than error) if django is unavailable in this environment, but prefer to
    # actually run: django is a normal dependency here and importing the real
    # module exercises the real code path instead of a reimplementation.
    pytest.importorskip("django")
    from DataLoader.Dataset.EIVA import loadEIVAGT
    return loadEIVAGT


def _import_loadSubPipeGT():
    pytest.importorskip("django")
    from DataLoader.Dataset.SubPipe import loadSubPipeGT
    return loadSubPipeGT


# --------------------------------------------------------------------------- #
# 1. Pin the conversion constants
# --------------------------------------------------------------------------- #
def test_ned_edn_constants():
    expected_edn2ned = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    expected_ned2edn = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])

    assert torch.allclose(EDN2NED.matrix()[:3, :3], expected_edn2ned, atol=1e-6)
    assert torch.allclose(NED2EDN.matrix()[:3, :3], expected_ned2edn, atol=1e-6)

    roundtrip = (EDN2NED @ NED2EDN).matrix()
    assert torch.allclose(roundtrip, torch.eye(4), atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Right-multiplying by NED2EDN must not touch translation (keeps ATE valid)
# --------------------------------------------------------------------------- #
def test_conversion_preserves_translation():
    rng = np.random.default_rng(0)
    N = 8
    t = torch.tensor(rng.normal(size=(N, 3)), dtype=torch.float32)
    quat = torch.tensor(R.from_rotvec(rng.normal(size=(N, 3))).as_quat(), dtype=torch.float32)
    poses = pp.SE3(torch.cat([t, quat], dim=-1))

    converted = poses @ NED2EDN

    # Exact equality (not approximate): NED2EDN has zero translation, so the
    # composed translation is R_a @ 0 + t_a == t_a bit-for-bit.
    assert torch.equal(converted.tensor()[:, :3], poses.tensor()[:, :3])


# --------------------------------------------------------------------------- #
# 3. loadEIVAGT: EDN-on-disk in, NED-out
# --------------------------------------------------------------------------- #
def test_loadEIVAGT_returns_ned(tmp_path):
    loadEIVAGT = _import_loadEIVAGT()

    poses_wc = _make_synthetic_edn_trajectory(n=5)
    gt_path = tmp_path / "pose_gt.txt"
    _write_zephyr_gt_file(gt_path, poses_wc)

    loaded = loadEIVAGT(gt_path)
    assert loaded.shape[0] == len(poses_wc)

    for i, (t, quat) in enumerate(poses_wc):
        T_wc_edn = pp.SE3(torch.tensor([*t, *quat], dtype=torch.float64))
        expected = pp.SE3(T_wc_edn @ NED2EDN.double())
        got = pp.SE3(loaded[i].double())

        # Compare via the Lie-algebra distance to sidestep quaternion sign
        # ambiguity; tolerance accounts for the float32 round-trip through the
        # on-disk matrix representation.
        err = (expected.Inv() @ got).Log()
        assert torch.allclose(err, torch.zeros(6, dtype=torch.float64), atol=1e-3), \
            f"pose {i}: log-distance {err}"


# --------------------------------------------------------------------------- #
# 4. The load-bearing regression test: perfect self-comparison has ~zero ROE,
#    but re-introducing the EDN/NED mix-up produces a large ROE.
# --------------------------------------------------------------------------- #
def test_perfect_estimate_has_zero_roe(tmp_path):
    loadEIVAGT = _import_loadEIVAGT()

    poses_wc = _make_synthetic_edn_trajectory(n=6)
    gt_path = tmp_path / "pose_gt.txt"
    _write_zephyr_gt_file(gt_path, poses_wc)

    loaded = loadEIVAGT(gt_path)
    Rg = loaded.matrix()[:, :3, :3].double()

    # A "perfect" estimate is just the GT compared to itself -> ROE ~ 0.
    roe_perfect = _roe_deg(Rg, Rg)
    assert torch.all(roe_perfect < 0.5), f"expected ~0 ROE, got {roe_perfect}"

    # Re-introduce the bug: compare the (correctly converted) NED GT against
    # the SAME physical trajectory left in its original EDN body axes -- this
    # is exactly what the old, unconverted loadEIVAGT would have produced.
    Re_edn_bug = torch.stack([
        pp.SE3(torch.tensor([*t, *quat], dtype=torch.float64)).matrix()[:3, :3]
        for t, quat in poses_wc
    ])
    roe_bug = _roe_deg(Rg, Re_edn_bug)
    assert torch.all(roe_bug > 5.0), \
        f"expected the EDN/NED mismatch to blow up ROE (>5 deg), got {roe_bug}"


# --------------------------------------------------------------------------- #
# 5. TartanAir GT must stay unconverted (it is already NED)
# --------------------------------------------------------------------------- #
def test_tartanair_gt_unconverted(tmp_path):
    from DataLoader.Dataset.TartanAir import loadTartanAirGT

    rng = np.random.default_rng(1)
    quats = R.from_rotvec(rng.normal(size=(3, 3)) * 0.5).as_quat()
    data = np.concatenate([rng.normal(size=(3, 3)) + np.array([1.0, 2.0, 3.0]), quats], axis=1)

    path = tmp_path / "pose_left.txt"
    np.savetxt(str(path), data)

    loaded = loadTartanAirGT(path)
    # pp.SE3 casts to float32 internally, so compare against the float32-cast
    # input (this is a precision cast, not an axis conversion) with a tight
    # tolerance -- any axis conversion (e.g. an accidental NED2EDN) would move
    # these values by orders of magnitude more than float32 rounding.
    np.testing.assert_allclose(
        loaded.tensor().numpy(), data.astype(np.float32), rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# 6. SubPipe and EIVA loaders are copy-paste twins -- they must not drift
# --------------------------------------------------------------------------- #
def test_subpipe_matches_eiva(tmp_path):
    loadEIVAGT = _import_loadEIVAGT()
    loadSubPipeGT = _import_loadSubPipeGT()

    poses_wc = _make_synthetic_edn_trajectory(n=4)
    gt_path = tmp_path / "pose_gt.txt"
    _write_zephyr_gt_file(gt_path, poses_wc)

    eiva = loadEIVAGT(gt_path)
    subpipe = loadSubPipeGT(gt_path)
    assert torch.equal(eiva.tensor(), subpipe.tensor())


# --------------------------------------------------------------------------- #
# 7. EiffelTower: pose 0 stays identity after the world-frame sandwich
# --------------------------------------------------------------------------- #
def test_eiffeltower_pose0_identity():
    """loadEiffelTowerGT needs a COLMAP-style images.txt with filenames that
    parse_timestamp_to_ns can parse, plus a django-importable module -- not
    worth fabricating just to exercise a single algebraic property. Instead,
    this pins the property loadEiffelTowerGT's comment relies on directly:
    ``ned_R @ NED2EDN == identity``, where ned_R (defined locally in
    EiffelTower.py as quaternion [0.5, 0.5, 0.5, 0.5]) is the same rotation as
    Utility.Point.EDN2NED. Because the first-pose normalization in
    loadEiffelTowerGT sets pose 0 to identity in the pre-conversion (EDN)
    frame, the subsequent ``poses[k] = ned_R @ poses[k] @ NED2EDN`` sandwich
    must map identity to identity, or the "pose 0 == identity" contract the
    rest of the pipeline relies on would silently break.
    """
    ned_R = pp.SE3(torch.tensor([0., 0., 0., 0.5, 0.5, 0.5, 0.5]))

    assert torch.allclose(ned_R.matrix()[:3, :3], EDN2NED.matrix()[:3, :3], atol=1e-6)

    result = (ned_R @ NED2EDN).matrix()
    assert torch.allclose(result, torch.eye(4), atol=1e-6)
