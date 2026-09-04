"""
Tests for the persistent-graph iSAM2 backend (Module/Optimization/GTSAM/
ISAM2Optimizer.py): synthetic-scene pose recovery with exact integer track
association across jobs, new-landmark minting, the VOLostTrack coast, GNC-GM
outlier rejection, bearing-range/pose-to-point information parity, gnc_weights
invariants, idempotent write-back, and config validation.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

gtsam = pytest.importorskip("gtsam")

import pypose as pp

from Module.Map import VisualMap, FrameNode, MatchObs, PointNode
from Module.Optimization.GTSAM.ISAM2Optimizer import (
    ISAM2FlowTracker, ISAM2_Graph, ISAM2_GraphInput, ISAM2_GraphOutput, ISAM2_KeyframeRows,
    _KERNELS, _matrix_to_se3, _NATIVE_P2P, gnc_weights, make_native_bearing_factor,
    make_native_point_factor, make_native_pose_to_point_factor)
from Utility.GTSAM_Utils import make_pose_to_point_factor

FX = FY = 320.0
CX = CY = 160.0
K_T = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=torch.float32)


def make_cfg(**overrides) -> SimpleNamespace:
    cfg = SimpleNamespace(
        device="cpu", parallel=False, factor_type="pose2point",
        kernel="huber", kernel_delta=0.1,
        relin_threshold=0.001, relin_skip=1, extra_updates=3,
        warmup_frames=10, warmup_extra=5,
        depth_var_scale=1.0, accumulate_fvar=True,
        motion_init="cv", motion_prior_sigma=0.0, coast_sigma=0.1,
        min_support=3, readout="online",
        min_flow_cov=0.25, min_depth_cov=0.01, match_cov_default=0.25,
    )
    cfg.__dict__.update(overrides)
    return cfg


# ---- synthetic NED scene -----------------------------------------------------

def landmarks() -> np.ndarray:
    """12 landmarks on a lattice, well separated in pixels (no rint collisions)."""
    ys = np.array([-0.9, -0.3, 0.3, 0.9])
    zs = np.array([-0.6, 0.0, 0.6])
    yy, zz = np.meshgrid(ys, zs)
    return np.stack([np.full(12, 3.0), yy.ravel(), zz.ravel()], axis=-1)


def pose_gt(k: int) -> np.ndarray:
    c, s = np.cos(0.03 * k), np.sin(0.03 * k)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = [0.3 * k, 0.05 * k, 0.0]
    return T


def observe(lms: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(uv (N,2), depth (N,)) of `lms` in frame k's NED camera."""
    T = pose_gt(k)
    obs = (lms - T[:3, 3]) @ T[:3, :3]          # R^T (l - t) = transformTo
    d = obs[:, 0]
    uv = np.stack([CX + FX * obs[:, 1] / d, CY + FY * obs[:, 2] / d], axis=-1)
    return uv, d


def make_job(k: int, uv1: np.ndarray, d1: np.ndarray, uv2: np.ndarray, d2: np.ndarray,
             from_pose: torch.Tensor | None = None) -> ISAM2_GraphInput:
    n = uv1.shape[0]
    uv_cov = torch.tensor([0.09, 0.09, 0.0]).repeat(n, 1)
    return ISAM2_GraphInput(
        frame_idx=k, from_idx=k - 1,
        from_pose=from_pose if from_pose is not None
        else _matrix_to_se3(pose_gt(k - 1)),
        K=K_T.clone(),
        pixel1_uv=torch.from_numpy(uv1).float(),
        pixel2_uv=torch.from_numpy(uv2).float(),
        pixel1_d=torch.from_numpy(d1).float(),
        pixel2_d=torch.from_numpy(d2).float(),
        pixel1_d_cov=torch.full((n,), 0.01),
        pixel2_d_cov=torch.full((n,), 0.01),
        pixel1_uv_cov=uv_cov.clone(),
        pixel2_uv_cov=uv_cov.clone(),
    )


def make_kf_rows(kf: int, k: int, lms: np.ndarray, uv_noise: float = 0.0,
                 rng: np.random.Generator | None = None) -> ISAM2_KeyframeRows:
    """Keyframe rows kf -> k with the odometry's contract: pixel1 is the ROUNDED
    keyframe projection (the pixel1 of pair kf -> kf+1), pixel2 the flow-carried
    position in frame k."""
    uv1, _ = observe(lms, kf)
    uv2, d2 = observe(lms, k)
    if uv_noise > 0 and rng is not None:
        uv2 = uv2 + rng.normal(scale=uv_noise, size=uv2.shape)
    m = uv1.shape[0]
    return ISAM2_KeyframeRows(
        kf_idx=kf,
        pixel1_uv=torch.from_numpy(np.rint(uv1)).float(),
        pixel2_uv=torch.from_numpy(uv2).float(),
        pixel2_d=torch.from_numpy(d2).float(),
        pixel2_d_cov=torch.full((m,), 0.01),
        pixel2_uv_cov=torch.tensor([0.09, 0.09, 0.0]).repeat(m, 1),
    )


def with_kf(job: ISAM2_GraphInput, kf: int, lms: np.ndarray, **kw) -> ISAM2_GraphInput:
    job.kf = make_kf_rows(kf, job.frame_idx, lms, **kw)
    return job


def chained_jobs(n_pairs: int, lms: np.ndarray) -> list[ISAM2_GraphInput]:
    """Pairs (0,1)..(n-1,n) with the selector's contract: pixel1 of pair k is
    rint(pixel2 of pair k-1); the first pair's pixel1 is the rounded projection."""
    jobs = []
    uv_prev, d_prev = observe(lms, 0)
    uv1 = np.rint(uv_prev)
    d1 = d_prev
    for k in range(1, n_pairs + 1):
        uv2, d2 = observe(lms, k)
        jobs.append(make_job(k, uv1, d1, uv2, d2))
        uv1, d1 = np.rint(uv2), d2
    return jobs


# ---- tests -------------------------------------------------------------------

def test_matrix_se3_roundtrip():
    T = pose_gt(7)
    back = pp.SE3(_matrix_to_se3(T).double()).matrix().numpy().reshape(4, 4)
    assert np.allclose(back, T, atol=1e-6)


def test_pose_recovery_and_track_chaining():
    lms = landmarks()
    tracker = ISAM2FlowTracker(make_cfg())
    for job in chained_jobs(3, lms):
        pose = tracker.step(job)
        T_est = pp.SE3(pose.double()).matrix().numpy().reshape(4, 4)
        err = np.linalg.norm(T_est[:3, 3] - pose_gt(job.frame_idx)[:3, 3])
        assert err < 0.02, f"frame {job.frame_idx}: translation error {err:.4f} m"

    # every landmark chained across all three pairs — no track splits
    assert tracker.next_lm_id == len(lms)
    assert len(tracker.tracks) == len(lms)
    assert all(t.n_obs == 4 for t in tracker.tracks.values())
    assert tracker.pose_keys == {0, 1, 2, 3}


def test_unmatched_row_mints_new_landmark():
    lms = landmarks()
    tracker = ISAM2FlowTracker(make_cfg())
    jobs = chained_jobs(3, lms)
    for job in jobs[:2]:
        tracker.step(job)

    extra = np.array([[3.5, 0.6, -0.45]])
    uv1_e, d1_e = observe(extra, 2)
    uv2_e, d2_e = observe(extra, 3)
    job3 = jobs[2]
    job = make_job(
        3,
        np.concatenate([job3.pixel1_uv.numpy(), np.rint(uv1_e)]),
        np.concatenate([job3.pixel1_d.numpy(), d1_e]),
        np.concatenate([job3.pixel2_uv.numpy(), uv2_e]),
        np.concatenate([job3.pixel2_d.numpy(), d2_e]),
    )
    tracker.step(job)
    assert tracker.next_lm_id == len(lms) + 1
    assert len(tracker.tracks) == len(lms) + 1


def test_lost_track_gap_coasts():
    """Pair (1,2) never reaches the backend (VOLostTrack): pair (2,3) must coast
    p_2 in and still produce a finite, sane pose."""
    lms = landmarks()
    tracker = ISAM2FlowTracker(make_cfg())
    jobs = chained_jobs(3, lms)
    tracker.step(jobs[0])

    uv1, d1 = observe(lms, 2)
    uv2, d2 = observe(lms, 3)
    pose = tracker.step(make_job(3, np.rint(uv1), d1, uv2, d2))

    assert tracker.pose_keys == {0, 1, 2, 3}
    T_est = pp.SE3(pose.double()).matrix().numpy().reshape(4, 4)
    assert np.isfinite(T_est).all()
    assert np.linalg.norm(T_est[:3, 3] - pose_gt(3)[:3, 3]) < 0.1


def test_gnc_rejects_outlier():
    lms = landmarks()
    jobs = chained_jobs(2, lms)
    uv2_bad = jobs[1].pixel2_uv.numpy().copy()
    uv2_bad[4, 0] += 40.0                                  # row 4: the outlier
    corrupted = [jobs[0], make_job(
        2,
        jobs[1].pixel1_uv.numpy(), jobs[1].pixel1_d.numpy(),
        uv2_bad, jobs[1].pixel2_d.numpy(),
    )]

    errors = {}
    for name, cfg in {
        "plain": make_cfg(factor_type="bearingrange", kernel="none"),
        "gnc"  : make_cfg(factor_type="bearingrange", kernel="none",
                          gnc_rounds=4, gnc_c=0.4, gnc_mu_rate=5.0),
    }.items():
        tracker = ISAM2FlowTracker(cfg)
        pose = torch.zeros(7)
        for job in corrupted:
            pose = tracker.step(job)
        T_est = pp.SE3(pose.double()).matrix().numpy().reshape(4, 4)
        assert np.isfinite(T_est).all()
        errors[name] = np.linalg.norm(T_est[:3, 3] - pose_gt(2)[:3, 3])
        if name == "gnc":
            assert tracker.n_gnc_rollback == 0

    assert errors["gnc"] < errors["plain"]


def test_gnc_weights_invariants():
    r2 = np.array([0.0, 1.0, 1e6, 1e12, np.inf])
    for floor in (1e-6, 1e-4, 1e-2):
        w = gnc_weights(r2, mu=2.0, c=0.4, floor=floor)
        assert w.min() >= floor
        assert bool(np.all(np.diff(w) <= 0))               # monotone in r^2
        ref = np.maximum((2.0 * 0.4 ** 2 / (r2 + 2.0 * 0.4 ** 2)) ** 2, floor)
        assert np.array_equal(w, ref)                      # bit-parity with bare GM


def test_bearingrange_carries_same_information():
    """The whitened norms of the native BearingRangeFactor3D and the Python
    pose-to-point factor must tie to first order around the measurement."""
    rng = np.random.default_rng(4)
    pose_key, lm_key = gtsam.symbol("p", 0), gtsam.symbol("l", 0)
    for _ in range(4):
        m = np.array([rng.uniform(1.5, 4.0), rng.uniform(-1, 1), rng.uniform(-1, 1)])
        A = rng.normal(size=(3, 3)) * 0.05
        cov = A @ A.T + 0.01 * np.eye(3)
        f_native = make_native_point_factor(pose_key, lm_key, m, cov, "none", 0.1)
        f_python = make_pose_to_point_factor(
            pose_key, lm_key, m, gtsam.noiseModel.Gaussian.Covariance(cov))
        values = gtsam.Values()
        values.insert(pose_key, gtsam.Pose3())
        values.insert(lm_key, gtsam.Point3(*(m + rng.normal(scale=1e-4, size=3))))
        # error() is 0.5 * ||whitened||^2 — the Mahalanobis halves must match
        assert abs(f_native.error(values) - f_python.error(values)) \
            <= 1e-3 * max(f_python.error(values), 1e-12)


def test_bearing_factor_is_bearing_block_of_bearingrange():
    """The bearing-only factor must be exactly the bearing block of the same
    (bearing, range) covariance a BearingRangeFactor3D on this observation would
    use: zero residual on-ray at any positive scale (bearingrange still sees
    the range mismatch), decoupled-and-equal to bearingrange off-ray under
    isotropic noise (where the range residual vanishes and the cross term is
    identically zero), dim() == 2, and finite under Huber.
    """
    rng = np.random.default_rng(5)
    pose_key, lm_key = gtsam.symbol("p", 0), gtsam.symbol("l", 0)
    m = np.array([2.3, 0.4, -0.6])
    A = rng.normal(size=(3, 3)) * 0.05
    cov = A @ A.T + 0.01 * np.eye(3)
    f_bear = make_native_bearing_factor(pose_key, lm_key, m, cov, "none", 0.1)
    f_br = make_native_point_factor(pose_key, lm_key, m, cov, "none", 0.1)
    assert f_bear.dim() == 2

    for s in (0.5, 2.0):
        values = gtsam.Values()
        values.insert(pose_key, gtsam.Pose3())
        values.insert(lm_key, gtsam.Point3(*(s * m)))
        assert f_bear.error(values) < 1e-9, f"scale {s}: bearing error should vanish on-ray"
        assert f_br.error(values) > 1e-6, f"scale {s}: bearingrange error should NOT vanish"

    # cov_br's bearing/range cross term is 0 for isotropic noise (B^T(I-bb^T)b/r == 0)
    iso_cov = 0.02 * np.eye(3)
    f_bear_iso = make_native_bearing_factor(pose_key, lm_key, m, iso_cov, "none", 0.1)
    f_br_iso = make_native_point_factor(pose_key, lm_key, m, iso_cov, "none", 0.1)
    r = np.linalg.norm(m)
    u = m / r
    off_ray = rng.normal(size=3)
    off_ray -= (off_ray @ u) * u
    p_off = u + 0.3 * off_ray / np.linalg.norm(off_ray)
    p_off = p_off * (r / np.linalg.norm(p_off))          # |p_off| == |m| exactly
    values_off = gtsam.Values()
    values_off.insert(pose_key, gtsam.Pose3())
    values_off.insert(lm_key, gtsam.Point3(*p_off))
    assert f_bear_iso.error(values_off) == pytest.approx(f_br_iso.error(values_off), rel=1e-9, abs=1e-12)
    assert f_bear_iso.error(values_off) > 1e-9           # a nontrivial check, not 0 == 0

    f_bear_huber = make_native_bearing_factor(pose_key, lm_key, m, cov, "huber", 0.1)
    values_pert = gtsam.Values()
    values_pert.insert(pose_key, gtsam.Pose3())
    values_pert.insert(lm_key, gtsam.Point3(*(m + rng.normal(scale=0.05, size=3))))
    assert np.isfinite(f_bear_huber.error(values_pert))


@pytest.mark.skipif(_NATIVE_P2P is None,
                    reason="gtsam wheel lacks the PoseToPointFactor wrapper patch "
                           "(Scripts/patches/gtsam-posetopoint-wrapper.patch)")
def test_native_p2p_matches_custom():
    """The wrapped C++ PoseToPointFactor and the Python pose-to-point
    CustomFactor implement the same residual (transformTo(l_w) - obs, same key
    order): error and full linearization must agree to machine precision."""
    rng = np.random.default_rng(7)
    pose_key, lm_key = gtsam.symbol("p", 0), gtsam.symbol("l", 0)
    for _ in range(4):
        m = np.array([rng.uniform(1.5, 4.0), rng.uniform(-1, 1), rng.uniform(-1, 1)])
        A = rng.normal(size=(3, 3)) * 0.05
        cov = A @ A.T + 0.01 * np.eye(3)
        f_native = make_native_pose_to_point_factor(pose_key, lm_key, m, cov, "none", 0.0)
        f_python = make_pose_to_point_factor(
            pose_key, lm_key, m, gtsam.noiseModel.Gaussian.Covariance(cov))
        values = gtsam.Values()
        values.insert(pose_key, gtsam.Pose3.Expmap(rng.normal(scale=0.1, size=6)))
        values.insert(lm_key, gtsam.Point3(*(m + rng.normal(scale=0.3, size=3))))
        assert f_native.error(values) == pytest.approx(f_python.error(values), rel=1e-9, abs=1e-12)
        A_n, b_n = f_native.linearize(values).jacobian()
        A_p, b_p = f_python.linearize(values).jacobian()
        assert np.allclose(A_n, A_p, atol=1e-9)
        assert np.allclose(b_n, b_p, atol=1e-9)


# ---- keyframe re-observations --------------------------------------------------

def test_keyframe_rows_reuse_landmark_keys():
    """Keyframe 0 re-observed from frames 2..4: every row must land on the landmark
    the keyframe pixel resolved to when pair (0,1) was stepped - no new landmarks,
    one extra p_k -> l factor per row, keyframe pose stamped alongside."""
    lms = landmarks()
    tracker = ISAM2FlowTracker(make_cfg())
    jobs = chained_jobs(4, lms)
    tracker.step(jobs[0])                                     # builds frame_lm[0]
    lm_of_kf_pixel = dict(tracker.frame_lm[0])
    assert len(lm_of_kf_pixel) == len(lms)
    for job in jobs[1:]:
        pose = tracker.step(with_kf(job, 0, lms))
        err = np.linalg.norm(_translation(pose) - pose_gt(job.frame_idx)[:3, 3])
        assert err < 0.02, f"frame {job.frame_idx}: translation error {err:.4f} m"
        assert tracker.stats[-1]["n_kf_obs"] == len(lms)
        assert tracker.stats[-1]["kf_idx"] == 0
    assert tracker.next_lm_id == len(lms)                     # nothing minted by keyframe rows
    assert tracker.frame_lm[0] == lm_of_kf_pixel              # table untouched, still live
    assert tracker.n_kf_total == 3 * len(lms)
    # the graph holds one chain factor AND one keyframe factor on p_k per landmark
    fg = tracker.isam.getFactorsUnsafe()
    p3 = gtsam.symbol("p", 3)
    on_p3 = [fg.at(i) for i in range(fg.size()) if fg.at(i) is not None and p3 in fg.at(i).keys()]
    assert len(on_p3) == 2 * len(lms)                          # motion prior off (sigma 0)


def test_kf_bearing_rows_leave_depth_free():
    """A biased keyframe depth sample must not move the pose under
    kf_factor_type='bearing' (depth-free by construction), while 'same' (the
    chain's own 3D factor family) is sensitive to it."""
    lms = landmarks()
    jobs = chained_jobs(4, lms)

    def run(kf_factor_type: str, biased: bool):
        tracker = ISAM2FlowTracker(make_cfg(kf_factor_type=kf_factor_type))
        tracker.step(jobs[0])
        errs = []
        for job in jobs[1:]:
            job = with_kf(job, 0, lms)
            assert job.kf is not None
            if biased:
                job.kf.pixel2_d = job.kf.pixel2_d * 1.3
            pose = tracker.step(job)
            errs.append(np.linalg.norm(_translation(pose) - pose_gt(job.frame_idx)[:3, 3]))
            assert tracker.stats[-1]["n_kf_obs"] == len(lms)
        return errs, tracker

    errs_same_clean, _ = run("same", False)
    errs_same_biased, _ = run("same", True)
    errs_bear_clean, _ = run("bearing", False)
    errs_bear_biased, tracker_bear = run("bearing", True)

    d_same = float(np.mean(np.abs(np.array(errs_same_biased) - np.array(errs_same_clean))))
    d_bear = float(np.mean(np.abs(np.array(errs_bear_biased) - np.array(errs_bear_clean))))
    assert d_bear < 1e-3, f"bearing kf rows should be ~insensitive to a keyframe depth bias: {d_bear}"
    assert d_same > 10 * d_bear, f"same-family kf rows should be far more sensitive: {d_same} vs {d_bear}"
    assert errs_bear_biased[-1] < 0.02

    fg = tracker_bear.isam.getFactorsUnsafe()
    p3 = gtsam.symbol("p", 3)
    on_p3 = [fg.at(i) for i in range(fg.size()) if fg.at(i) is not None and p3 in fg.at(i).keys()]
    assert sum(1 for f in on_p3 if f.dim() == 3) == len(lms)   # chain rows (pose2point)
    assert sum(1 for f in on_p3 if f.dim() == 2) == len(lms)   # bearing keyframe rows


def test_keyframe_rows_ignore_unknown_pixels_and_stale_tables():
    lms = landmarks()
    tracker = ISAM2FlowTracker(make_cfg())
    jobs = chained_jobs(3, lms)
    tracker.step(jobs[0])
    # rows whose keyframe pixel never resolved to a landmark are skipped
    job = with_kf(jobs[1], 0, lms)
    assert job.kf is not None
    job.kf.pixel1_uv = job.kf.pixel1_uv + 7.0
    tracker.step(job)
    assert tracker.stats[-1]["n_kf_obs"] == 0
    # a keyframe with no table (its pair never reached the backend) adds nothing, no exception
    tracker.step(with_kf(jobs[2], 99, lms))
    assert tracker.stats[-1]["n_kf_obs"] == 0
    assert tracker.next_lm_id == len(lms)


def test_keyframe_rows_help_against_drifting_chain():
    """Chain pixel2 carries a random walk (accumulated flow error); the drift-free
    keyframe re-observation must not make the readout worse and should help."""
    lms = landmarks()
    n_pairs = 8
    errors = {}
    for use_kf in (False, True):
        rng = np.random.default_rng(11)
        tracker = ISAM2FlowTracker(make_cfg(warmup_frames=0))
        uv_prev, d_prev = observe(lms, 0)
        uv1, d1 = np.rint(uv_prev), d_prev
        walk = np.zeros_like(uv_prev)
        err = []
        for k in range(1, n_pairs + 1):
            uv2, d2 = observe(lms, k)
            walk += rng.normal(scale=0.8, size=walk.shape)
            job = make_job(k, uv1, d1, uv2 + walk, d2)
            if use_kf and k >= 2:
                job = with_kf(job, 0, lms)
            pose = tracker.step(job)
            err.append(np.linalg.norm(_translation(pose) - pose_gt(k)[:3, 3]))
            uv1, d1 = np.rint(uv2 + walk), d2
        errors[use_kf] = float(np.mean(err))
    assert errors[True] <= errors[False] * 1.05, errors


def test_keyframe_pose_survives_marg_lag_until_keyframe_moves():
    """Under marg_lag the keyframe pose is re-stamped every frame it anchors rows
    (inverse of test_marg_lag_plateaus: p_0 must NOT expire), and expires once the
    keyframe moves on. Eight pairs: the camera closes on the landmark plane at frame 10
    and rows start failing the depth gates from frame 9."""
    pytest.importorskip("gtsam_unstable")
    lms = landmarks()
    tracker = ISAM2FlowTracker(make_cfg(marg_lag=2, warmup_frames=0))
    jobs = chained_jobs(8, lms)
    tracker.step(jobs[0])
    for job in jobs[1:5]:
        pose = tracker.step(with_kf(job, 0, lms))
        assert np.linalg.norm(_translation(pose) - pose_gt(job.frame_idx)[:3, 3]) < 0.05
        assert tracker.stats[-1]["n_kf_obs"] == len(lms)
    est = tracker.isam.calculateEstimate()
    assert est.exists(gtsam.symbol("p", 0))                  # keyframe kept alive (stamped 5)
    assert not est.exists(gtsam.symbol("p", 2))              # intermediate poses still expire
    for job in jobs[5:]:                                     # keyframe moves to frame 4
        tracker.step(with_kf(job, 4, lms))
        assert tracker.stats[-1]["n_kf_obs"] == len(lms)
    est = tracker.isam.calculateEstimate()
    assert not est.exists(gtsam.symbol("p", 0))              # old keyframe expired (5 < 8 - 2)
    assert est.exists(gtsam.symbol("p", 4))
    assert 0 not in tracker.frame_lm and 4 in tracker.frame_lm


@pytest.mark.parametrize("kf_factor_type", ["same", "bearing"])
def test_keyframe_rows_under_gnc(kf_factor_type):
    lms = landmarks()
    tracker = ISAM2FlowTracker(make_cfg(factor_type="bearingrange", kernel="none",
                                        gnc_rounds=3, gnc_c=0.4, gnc_mu_rate=5.0,
                                        kf_factor_type=kf_factor_type))
    jobs = chained_jobs(3, lms)
    tracker.step(jobs[0])
    for job in jobs[1:]:
        pose = tracker.step(with_kf(job, 0, lms))
        assert np.isfinite(pose.numpy()).all()
    assert tracker.n_gnc_rollback == 0
    assert tracker.n_kf_total == 2 * len(lms)


def test_frame_stats_rectangular_with_mixed_keyframe_frames():
    lms = landmarks()
    optimizer = ISAM2_Graph(make_cfg())
    jobs = chained_jobs(3, lms)
    optimizer.start_optimize(jobs[0])
    optimizer.start_optimize(with_kf(jobs[1], 0, lms))
    optimizer.start_optimize(jobs[2])
    stats = optimizer.frame_stats()
    assert stats is not None
    assert list(stats["n_kf_obs"]) == [0, len(lms), 0]
    assert list(stats["kf_idx"]) == [-1, 0, -1]
    assert all(len(v) == 3 for v in stats.values())


def test_kf_cov_scale_config():
    ISAM2_Graph.is_valid_config(make_cfg(kf_cov_scale=4.0))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_cov_scale=0.0))
    assert ISAM2FlowTracker(make_cfg(kf_cov_scale=4.0)).kf_cov_scale == 4.0


def test_kf_gates_skip_as_specified():
    lms = landmarks()
    jobs = chained_jobs(4, lms)

    # support gate: chain support (== len(lms)) never reaches kf_min_support -> every kf frame gated
    tracker = ISAM2FlowTracker(make_cfg(kf_min_support=len(lms) + 1))
    tracker.step(jobs[0])
    for job in jobs[1:]:
        tracker.step(with_kf(job, 0, lms))
        assert tracker.stats[-1]["kf_gate"] == 1
        assert tracker.stats[-1]["n_kf_obs"] == 0
    assert 0 in tracker.frame_lm                          # kept alive despite being gated every frame

    # gap gate: kf_min_gap=4 gates gaps 2-3, gaps 4-5 add rows
    jobs5 = chained_jobs(5, lms)
    tracker = ISAM2FlowTracker(make_cfg(kf_min_gap=4))
    tracker.step(jobs5[0])
    gates, gaps = [], []
    for job in jobs5[1:]:
        tracker.step(with_kf(job, 0, lms))
        gates.append(tracker.stats[-1]["kf_gate"])
        gaps.append(tracker.stats[-1]["kf_gap"])
    assert gaps == [2, 3, 4, 5]
    assert gates == [2, 2, 0, 0]
    assert tracker.stats[-1]["n_kf_obs"] == len(lms)

    # precedence: support gate wins even when the gap gate alone would have passed
    tracker = ISAM2FlowTracker(make_cfg(kf_min_support=len(lms) + 1, kf_min_gap=1))
    tracker.step(jobs[0])
    tracker.step(with_kf(jobs[1], 0, lms))                 # gap 2, satisfies kf_min_gap=1 alone
    assert tracker.stats[-1]["kf_gate"] == 1


def test_kf_factor_type_config():
    for v in ("same", "bearing", "bearingrange", "pose2point"):
        ISAM2_Graph.is_valid_config(make_cfg(kf_factor_type=v))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_factor_type="reprojection"))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_min_support=-1))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_min_gap=1.5))

    assert ISAM2FlowTracker(make_cfg(factor_type="bearingrange")).kf_factor_type == "bearingrange"
    assert ISAM2FlowTracker(make_cfg(factor_type="bearingrange",
                                     kf_factor_type="bearing")).kf_factor_type == "bearing"


def test_kf_max_obs_per_landmark_config():
    ISAM2_Graph.is_valid_config(make_cfg(kf_max_obs_per_landmark=3))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_max_obs_per_landmark=-1))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_max_obs_per_landmark=2.5))
    assert ISAM2FlowTracker(make_cfg(kf_max_obs_per_landmark=3)).kf_max_obs_per_landmark == 3
    assert ISAM2FlowTracker(make_cfg()).kf_max_obs_per_landmark == 0


def test_kf_max_obs_per_landmark_caps_factors():
    """4 keyframe frames (2..5) would each add one factor per landmark on the
    SAME keyframe (0); kf_max_obs_per_landmark=2 must let the first two through
    and cap every row on the last two, while the default (0 = unlimited) lets
    all four through with n_kf_capped == 0 everywhere."""
    lms = landmarks()

    tracker = ISAM2FlowTracker(make_cfg(kf_max_obs_per_landmark=2))
    jobs = chained_jobs(5, lms)
    tracker.step(jobs[0])                                    # frame 1: builds frame_lm[0]
    pose = None
    for job in jobs[1:]:                                      # frames 2..5
        pose = tracker.step(with_kf(job, 0, lms))
        if job.frame_idx <= 3:
            assert tracker.stats[-1]["n_kf_obs"] == len(lms)
            assert tracker.stats[-1]["n_kf_capped"] == 0
        else:
            assert tracker.stats[-1]["n_kf_obs"] == 0
            assert tracker.stats[-1]["n_kf_capped"] == len(lms)
    assert tracker.n_kf_total == 2 * len(lms)
    assert pose is not None
    err = np.linalg.norm(_translation(pose) - pose_gt(5)[:3, 3])
    assert err < 0.02, f"final pose translation error {err:.4f} m"

    tracker_default = ISAM2FlowTracker(make_cfg())
    jobs = chained_jobs(5, lms)
    tracker_default.step(jobs[0])
    for job in jobs[1:]:
        tracker_default.step(with_kf(job, 0, lms))
        assert tracker_default.stats[-1]["n_kf_obs"] == len(lms)
        assert tracker_default.stats[-1]["n_kf_capped"] == 0
    assert tracker_default.n_kf_total == 4 * len(lms)


@pytest.mark.skipif(_NATIVE_P2P is None,
                    reason="gtsam wheel lacks the PoseToPointFactor wrapper patch "
                           "(Scripts/patches/gtsam-posetopoint-wrapper.patch)")
def test_kf_factor_type_native_config():
    ISAM2_Graph.is_valid_config(make_cfg(kf_factor_type="pose2point_native"))
    assert ISAM2FlowTracker(
        make_cfg(kf_factor_type="pose2point_native")).kf_factor_type == "pose2point_native"


def test_kf_consistency_gate_rejects_disagreeing_rows():
    """3 of 12 keyframe rows land 25 px off the chain's own current pixel for the
    same landmark: kf_consistency_px=3.0 must reject exactly those rows and keep
    the pose near GT, while the gate off lets them corrupt the pose more."""
    lms = landmarks()
    jobs = chained_jobs(3, lms)

    def run(consistency_px: float) -> tuple[torch.Tensor, ISAM2FlowTracker]:
        tracker = ISAM2FlowTracker(make_cfg(kf_consistency_px=consistency_px))
        tracker.step(jobs[0])
        tracker.step(with_kf(jobs[1], 0, lms))
        job = with_kf(jobs[2], 0, lms)
        assert job.kf is not None
        job.kf.pixel2_uv[:3] += 25.0
        pose = tracker.step(job)
        return pose, tracker

    pose_gated, tracker_gated = run(3.0)
    pose_off, tracker_off = run(0.0)

    assert tracker_gated.stats[-1]["n_kf_inconsistent"] == 3
    assert tracker_gated.stats[-1]["n_kf_obs"] == len(lms) - 3
    assert tracker_off.stats[-1]["n_kf_inconsistent"] == 0
    assert tracker_off.stats[-1]["n_kf_obs"] == len(lms)

    err_gated = np.linalg.norm(_translation(pose_gated) - pose_gt(3)[:3, 3])
    err_off = np.linalg.norm(_translation(pose_off) - pose_gt(3)[:3, 3])
    assert err_gated < 0.02, f"gated translation error {err_gated:.4f} m"
    assert err_off > err_gated

    ISAM2_Graph.is_valid_config(make_cfg(kf_consistency_px=0.0))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_consistency_px=-1.0))


def test_kf_require_chain_alive_skips_dead_landmarks():
    """Landmark index 0's chain row is dropped from pair (2,3), killing its track;
    a keyframe row still tries to re-observe it. kf_require_chain_alive=True must
    skip that one row (n_kf_dead == 1); the default adds it."""
    lms = landmarks()
    jobs = chained_jobs(3, lms)

    def run(require_alive: bool) -> ISAM2FlowTracker:
        tracker = ISAM2FlowTracker(make_cfg(kf_require_chain_alive=require_alive))
        tracker.step(jobs[0])
        tracker.step(jobs[1])
        job = make_job(3, jobs[2].pixel1_uv.numpy()[1:], jobs[2].pixel1_d.numpy()[1:],
                       jobs[2].pixel2_uv.numpy()[1:], jobs[2].pixel2_d.numpy()[1:])
        tracker.step(with_kf(job, 0, lms))
        return tracker

    tracker_gated = run(True)
    tracker_off = run(False)

    assert tracker_gated.stats[-1]["n_kf_dead"] == 1
    assert tracker_gated.stats[-1]["n_kf_obs"] == len(lms) - 1
    assert tracker_off.stats[-1]["n_kf_dead"] == 0
    assert tracker_off.stats[-1]["n_kf_obs"] == len(lms)

    ISAM2_Graph.is_valid_config(make_cfg(kf_require_chain_alive=True))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_require_chain_alive="yes"))


def test_kf_max_gap_gates_block():
    """kf_max_gap=2 with keyframe 0: gap 2 (frame 2) still adds rows, gaps 3+
    (frames 3, 4) gate the whole block; the keyframe table stays alive."""
    lms = landmarks()
    tracker = ISAM2FlowTracker(make_cfg(kf_max_gap=2))
    jobs = chained_jobs(4, lms)
    tracker.step(jobs[0])
    for job in jobs[1:]:
        tracker.step(with_kf(job, 0, lms))
        if job.frame_idx == 2:
            assert tracker.stats[-1]["kf_gate"] == 0
            assert tracker.stats[-1]["n_kf_obs"] == len(lms)
        else:
            assert tracker.stats[-1]["kf_gate"] == 3
            assert tracker.stats[-1]["n_kf_obs"] == 0
    assert 0 in tracker.frame_lm

    ISAM2_Graph.is_valid_config(make_cfg(kf_max_gap=0))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_max_gap=-1))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_max_gap=1.5))


def test_kf_kernel_config_and_dispatch():
    for v in ("same", *_KERNELS):
        ISAM2_Graph.is_valid_config(make_cfg(kf_kernel=v))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(kf_kernel="biweight"))

    assert ISAM2FlowTracker(make_cfg(kernel="huber")).kf_kernel == "huber"     # "same" resolves
    assert ISAM2FlowTracker(make_cfg(kernel="huber", kf_kernel="none")).kf_kernel == "none"

    lms = landmarks()
    jobs = chained_jobs(2, lms)
    tracker = ISAM2FlowTracker(make_cfg(kernel="huber", kf_kernel="none"))
    tracker.step(jobs[0])
    tracker.step(with_kf(jobs[1], 0, lms))

    fg = tracker.isam.getFactorsUnsafe()
    p2 = gtsam.symbol("p", 2)
    on_p2 = [fg.at(i) for i in range(fg.size()) if fg.at(i) is not None and p2 in fg.at(i).keys()]
    assert len(on_p2) == 2 * len(lms)                          # chain + keyframe rows on p_2
    n_robust = sum(1 for f in on_p2 if isinstance(f.noiseModel(), gtsam.noiseModel.Robust))
    assert n_robust == len(lms)                                # chain rows: Huber-wrapped
    assert len(on_p2) - n_robust == len(lms)                   # keyframe rows: not wrapped


def make_map(n_frames: int) -> VisualMap:
    vmap = VisualMap()
    for i in range(n_frames):
        vmap.frames.push(FrameNode.init({
            "pose"       : pp.identity_SE3(1).tensor(),
            "T_BS"       : pp.identity_SE3(1).tensor(),
            "need_interp": torch.tensor([False]),
            "time_ns"    : torch.tensor([i], dtype=torch.long),
            "K"          : K_T.unsqueeze(0),
            "baseline"   : torch.tensor([0.1]),
        }))
    return vmap


def _match_rows(n: int, uv1: np.ndarray, uv2: np.ndarray, d2: np.ndarray) -> MatchObs:
    one = lambda v: torch.full((n, 1), v)
    return MatchObs.init({
        "pixel1_uv": torch.from_numpy(uv1).float(), "pixel2_uv": torch.from_numpy(uv2).float(),
        "pixel1_d": one(3.0), "pixel2_d": torch.from_numpy(d2).float().unsqueeze(-1),
        "pixel1_disp": one(-1.), "pixel2_disp": one(-1.),
        "pixel1_disp_cov": one(-1.), "pixel2_disp_cov": one(-1.),
        "pixel1_uv_cov": torch.tensor([0.09, 0.09, 0.0]).repeat(n, 1),
        "pixel2_uv_cov": torch.tensor([0.04, 0.04, 0.0]).repeat(n, 1),
        "pixel1_d_cov": one(0.01), "pixel2_d_cov": one(0.02),
        "obs1_covTc": torch.eye(3).double().repeat(n, 1, 1),
        "obs2_covTc": torch.eye(3).double().repeat(n, 1, 1),
    })


def test_get_graph_data_reads_keyframe_rows():
    """The odometry stores keyframe rows in VisualMap.kf_match with kfmatch2frame1 =
    keyframe; get_graph_data must surface them as ISAM2_GraphInput.kf (None when absent)."""
    lms = landmarks()
    optimizer = ISAM2_Graph(make_cfg())
    vmap = make_map(4)
    n = len(lms)
    uv0, _ = observe(lms, 0)
    uv3, d3 = observe(lms, 3)
    uv2, d2 = observe(lms, 2)
    frame3 = torch.tensor([3])

    # consecutive pair (2, 3) as MACVO.run_pair registers it
    n_orig = len(vmap.match)
    point_idx = vmap.points.push(PointNode.init({
        "pos_Tw": torch.zeros((n, 3)), "cov_Tw": torch.eye(3).double().repeat(n, 1, 1),
        "color": torch.zeros((n, 3), dtype=torch.uint8)}))
    match_idx = vmap.match.push(_match_rows(n, np.rint(uv2), uv3, d3))
    vmap.match2point.set(match_idx, point_idx)
    vmap.frame2match.add(torch.tensor([2]), torch.tensor([n_orig]), torch.tensor([n]))
    vmap.frame2match.add(frame3, torch.tensor([n_orig]), torch.tensor([n]))
    assert optimizer.get_graph_data(vmap, frame3).kf is None

    # keyframe rows (0 -> 3)
    kf_idx = vmap.kf_match.push(_match_rows(n, np.rint(uv0), uv3, d3))
    vmap.kfmatch2point.set(kf_idx, point_idx)
    vmap.kfmatch2frame1.set(kf_idx, torch.zeros(n, dtype=torch.long))
    vmap.kfmatch2frame2.set(kf_idx, torch.full((n,), 3, dtype=torch.long))
    vmap.frame2kfmatch.add(frame3, torch.tensor([0]), torch.tensor([n]))

    data = optimizer.get_graph_data(vmap, frame3)
    assert data.frame_idx == 3 and data.from_idx == 2
    assert data.kf is not None and data.kf.kf_idx == 0
    assert data.kf.pixel1_uv.shape == (n, 2) and data.kf.pixel2_d.shape == (n,)
    assert torch.allclose(data.kf.pixel1_uv, torch.from_numpy(np.rint(uv0)).float())
    assert torch.allclose(data.kf.pixel2_d_cov, torch.full((n,), 0.02))
    assert data.pixel1_uv.shape == (n, 2)                     # chain rows untouched


def test_write_graph_data_is_idempotent():
    optimizer = ISAM2_Graph(make_cfg())
    vmap = make_map(4)
    pose = _matrix_to_se3(pose_gt(2))

    optimizer.write_graph_data(ISAM2_GraphOutput(frame_idx=2, pose_estimate=pose), vmap)
    assert torch.allclose(vmap.frames.data["pose"][2], pose)

    # A re-delivered result (job skipped on a VOLostTrack frame) must be a no-op.
    vmap.frames.data["pose"][2] = pp.identity_SE3(1).tensor().reshape(7)
    optimizer.write_graph_data(ISAM2_GraphOutput(frame_idx=2, pose_estimate=pose), vmap)
    assert not torch.allclose(vmap.frames.data["pose"][2], pose)

    optimizer.write_graph_data(None, vmap)   # tolerate None


def test_config_validation():
    ISAM2_Graph.is_valid_config(make_cfg())
    ISAM2_Graph.is_valid_config(make_cfg(gnc_rounds=5, gnc_c=0.4, gnc_mu_rate=5.0))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(parallel=True))
    with pytest.raises(KeyError):
        ISAM2_Graph.is_valid_config(make_cfg(unexpected_knob=1))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(factor_type="reprojection"))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(readout="smoothed"))


# ---- fixed-lag marginalization (marg_lag) -------------------------------------

def _translation(pose: torch.Tensor) -> np.ndarray:
    return pp.SE3(pose.double()).matrix().numpy().reshape(4, 4)[:3, 3]


def test_marg_lag_zero_is_plain_isam2():
    assert type(ISAM2FlowTracker(make_cfg()).isam) is gtsam.ISAM2
    pytest.importorskip("gtsam_unstable")
    from Module.Optimization.GTSAM.Marginalization import FixedLagIsam2
    assert isinstance(ISAM2FlowTracker(make_cfg(marg_lag=3)).isam, FixedLagIsam2)


def test_marg_lag_plateaus_and_expires_old_poses():
    pytest.importorskip("gtsam_unstable")
    lms = landmarks()
    tracker = ISAM2FlowTracker(make_cfg(marg_lag=3, warmup_frames=0))
    for job in chained_jobs(8, lms):        # the camera reaches the landmark plane at frame 10
        pose = tracker.step(job)
        err = np.linalg.norm(_translation(pose) - pose_gt(job.frame_idx)[:3, 3])
        assert err < 0.05, f"frame {job.frame_idx}: translation error {err:.4f} m"

    est = tracker.isam.calculateEstimate()
    assert not est.exists(gtsam.symbol("p", 0))          # expired
    assert est.exists(gtsam.symbol("p", 7)) and est.exists(gtsam.symbol("p", 8))
    live = [s["live_vars"] for s in tracker.stats]
    assert live[-1] == live[-4] == 12 + 4, f"live variables did not plateau at 12 landmarks + 4 poses: {live}"
    assert live[-1] < len(tracker.pose_keys) + tracker.next_lm_id


def test_marg_lag_huge_reproduces_unbounded():
    """Lag 200 never expires anything over 5 pairs, so every readout must match
    the unbounded arm — proof that the extra updates run through the smoother
    itself (getISAM2() is a copy; wired that way the arms would diverge)."""
    pytest.importorskip("gtsam_unstable")
    lms = landmarks()
    poses = {}
    for lag in (0, 200):
        tracker = ISAM2FlowTracker(make_cfg(marg_lag=lag))
        poses[lag] = [tracker.step(job).numpy() for job in chained_jobs(5, lms)]
    for a, b in zip(poses[0], poses[200]):
        assert np.allclose(a, b, atol=1e-6)


def test_lost_track_gap_coasts_under_marg_lag():
    """Pairs (1,2)..(3,4) never reach the backend; pair (4,5) coasts p_4 in and
    the readout still reads p_1, which the gap rule re-stamps across the jump
    (the clock moves 1 -> 5, so with lag 2 an un-refreshed p_1 would expire).
    The coast itself is the same one-step extrapolation as the unbounded arm."""
    pytest.importorskip("gtsam_unstable")
    lms = landmarks()
    poses = {}
    for lag in (0, 2):
        tracker = ISAM2FlowTracker(make_cfg(marg_lag=lag, warmup_frames=0))
        tracker.step(chained_jobs(1, lms)[0])
        uv1, d1 = observe(lms, 4)
        uv2, d2 = observe(lms, 5)
        poses[lag] = tracker.step(make_job(5, np.rint(uv1), d1, uv2, d2)).numpy()
        assert tracker.pose_keys == {0, 1, 4, 5}
    est = tracker.isam.calculateEstimate()
    assert est.exists(gtsam.symbol("p", 1)) and not est.exists(gtsam.symbol("p", 0))
    assert np.isfinite(poses[2]).all()
    assert np.allclose(poses[0], poses[2], atol=1e-4)


def test_marg_lag_config_rules():
    ISAM2_Graph.is_valid_config(make_cfg(marg_lag=0))
    ISAM2_Graph.is_valid_config(make_cfg(marg_lag=5))
    for bad in (dict(marg_lag=1), dict(marg_lag=-1), dict(marg_lag=2.5),
                dict(marg_lag=3, gnc_rounds=2, gnc_c=0.4, gnc_mu_rate=5.0)):
        with pytest.raises(ValueError):
            ISAM2_Graph.is_valid_config(make_cfg(**bad))
    with pytest.raises(ValueError):
        ISAM2FlowTracker(make_cfg(marg_lag=1))


# ---- gtsam_unstable semantics the design rests on -----------------------------

def _fixedlag(lag: int):
    from Module.Optimization.GTSAM.Marginalization import FixedLagIsam2
    p = gtsam.ISAM2Params()
    p.setRelinearizeThreshold(0.0)
    p.relinearizeSkip = 1
    p.setFactorization("QR")
    return FixedLagIsam2(p, lag)


def test_timestamp_map_tuple_insert_and_refresh():
    pytest.importorskip("gtsam_unstable")
    from Module.Optimization.GTSAM.Marginalization import timestamp_map
    n6 = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    s = _fixedlag(3)
    x0 = gtsam.symbol("p", 0)
    g, v = gtsam.NonlinearFactorGraph(), gtsam.Values()
    v.insert(x0, gtsam.Pose3())
    g.add(gtsam.PriorFactorPose3(x0, gtsam.Pose3(), n6))
    s.update(g, v, timestamp_map({x0: 0.0}))
    for k in range(1, 9):
        xk = gtsam.symbol("p", k)
        g, v = gtsam.NonlinearFactorGraph(), gtsam.Values()
        v.insert(xk, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(k * 1.0, 0, 0)))
        g.add(gtsam.BetweenFactorPose3(gtsam.symbol("p", k - 1), xk,
                                       gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0)), n6))
        s.update(g, v, timestamp_map({xk: float(k), x0: float(k)}))   # x0 refreshed every frame
    assert s.calculateEstimate().exists(x0)                           # refresh honoured
    assert not s.calculateEstimate().exists(gtsam.symbol("p", 4))     # un-refreshed key expired


def test_bare_update_is_real_pass_and_expires_nothing():
    pytest.importorskip("gtsam_unstable")
    from Module.Optimization.GTSAM.Marginalization import timestamp_map
    n6 = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    s = _fixedlag(100)
    g, v, st = gtsam.NonlinearFactorGraph(), gtsam.Values(), {}
    x0 = gtsam.symbol("p", 0)
    g.add(gtsam.PriorFactorPose3(x0, gtsam.Pose3(), gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)))
    for k in range(3):
        xk = gtsam.symbol("p", k)
        v.insert(xk, gtsam.Pose3(gtsam.Rot3.Rodrigues(0.3, 0.2, 0.1), gtsam.Point3(k * 3.0, 1.7, -2.2)))
        st[xk] = float(k)
        if k:
            g.add(gtsam.BetweenFactorPose3(gtsam.symbol("p", k - 1), xk,
                                           gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0)), n6))
    s.update(g, v, timestamp_map(st))

    def snap():
        est = s.calculateEstimate()
        return np.concatenate([est.atPose3(gtsam.symbol("p", k)).translation() for k in range(3)])

    before = snap()
    for _ in range(3):
        s.smoother.getISAM2().update()                 # a COPY: must move nothing
    assert np.abs(snap() - before).max() == 0.0
    before = snap()
    for _ in range(3):
        s.update()                                     # bare pass: real work
    assert np.abs(snap() - before).max() > 1e-6
    assert s.calculateEstimate().size() == 3           # empty stamps expire nothing


def test_index_error_becomes_marginalization_failure(monkeypatch):
    pytest.importorskip("gtsam_unstable")
    from Module.Optimization.GTSAM.Marginalization import MarginalizationFailure
    s = _fixedlag(5)

    def boom(*_):
        raise IndexError("Requested variable 'p7' is not in this VectorValues")
    monkeypatch.setattr(s, "smoother", SimpleNamespace(update=boom))   # pybind methods are read-only
    with pytest.raises(MarginalizationFailure, match="marg_lag"):
        s.update()


# ---- marginalization mitigations -----------------------------------------------

def test_marg_touch_is_information_free():
    """A 1e4 m prior on every expiring key must not move the readout (it exists
    only to force the clique through the constrained re-elimination)."""
    pytest.importorskip("gtsam_unstable")
    lms = landmarks()
    poses = {}
    for sigma in (0.0, 1e4):
        tracker = ISAM2FlowTracker(make_cfg(marg_lag=3, warmup_frames=0, marg_touch_sigma=sigma))
        poses[sigma] = [tracker.step(job).numpy() for job in chained_jobs(8, lms)]
        assert not tracker.isam.calculateEstimate().exists(gtsam.symbol("p", 0))
    for a, b in zip(poses[0.0], poses[1e4]):
        assert np.allclose(a, b, atol=1e-4)


def test_marg_dead_at_birth_expires_dead_landmark_early():
    """A track that dies is re-stamped to its birth frame: once that frame is
    outside the window the landmark is gone, while the same landmark lingers
    under the plain policy until its own last stamp expires."""
    pytest.importorskip("gtsam_unstable")
    lms = landmarks()
    jobs = chained_jobs(8, lms)
    gone = {}
    for dab in (False, True):
        tracker = ISAM2FlowTracker(make_cfg(marg_lag=3, warmup_frames=0, marg_dead_at_birth=dab))
        for job in jobs[:4]:
            tracker.step(job)
        victim = tracker.tracks[(int(np.rint(jobs[4].pixel1_uv[0, 0])), int(np.rint(jobs[4].pixel1_uv[0, 1])))].lm_key
        for job in jobs[4:]:                        # drop row 0 from now on: its track dies at frame 5
            j = make_job(job.frame_idx, job.pixel1_uv.numpy()[1:], job.pixel1_d.numpy()[1:],
                         job.pixel2_uv.numpy()[1:], job.pixel2_d.numpy()[1:])
            tracker.step(j)
            if not tracker.isam.calculateEstimate().exists(victim):
                gone[dab] = job.frame_idx
                break
    assert gone[True] == 5                          # born at frame 0, already outside the window: leaves at once
    assert gone[False] == 8                         # last stamped 4 (lag 3): survives until the clock passes 7


def test_marg_mitigation_config_rules():
    ISAM2_Graph.is_valid_config(make_cfg(marg_lag=5, marg_dead_at_birth=True, marg_touch_sigma=1e4))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(marg_lag=5, marg_dead_at_birth="yes"))
    with pytest.raises(ValueError):
        ISAM2_Graph.is_valid_config(make_cfg(marg_lag=5, marg_touch_sigma=-1.0))


# ---- offline batch polish (final_lm) --------------------------------------------

def _lazy_cfg(**kw) -> SimpleNamespace:
    """Deliberately lazy online settings: one GN pass per frame, almost no relinearization."""
    return make_cfg(relin_threshold=1.0, relin_skip=10, extra_updates=0, warmup_frames=0,
                    readout="chain", final_lm=True, **kw)


def _noisy_jobs(n_pairs: int, lms: np.ndarray, rng: np.random.Generator) -> list[ISAM2_GraphInput]:
    jobs = chained_jobs(n_pairs, lms)
    for job in jobs:
        job.pixel2_uv += torch.from_numpy(rng.normal(scale=0.3, size=(lms.shape[0], 2))).float()
        job.pixel2_d  += torch.from_numpy(rng.normal(scale=0.05, size=lms.shape[0])).float()
    return jobs


def _traj_error(poses: dict[int, np.ndarray]) -> float:
    return max(float(np.linalg.norm(T[:3, 3] - pose_gt(k)[:3, 3])) for k, T in poses.items())


def test_final_lm_lowers_graph_error_and_covers_every_pose():
    rng = np.random.default_rng(0)
    lms = landmarks()
    tracker = ISAM2FlowTracker(_lazy_cfg())
    online = {}
    for job in _noisy_jobs(8, lms, rng):
        tracker.step(job)
        online[job.frame_idx] = tracker.isam.calculateEstimatePose3(gtsam.symbol("p", job.frame_idx)).matrix()
    online[0] = tracker.isam.calculateEstimatePose3(gtsam.symbol("p", 0)).matrix()

    polished = tracker.final_lm_solve()
    s = tracker.final_lm_stats
    assert s is not None
    assert set(polished) == set(range(9)) == tracker.pose_keys
    assert s["error_after"] <= s["error_before"] and s["iterations"] >= 1
    assert s["n_values"] == len(tracker.pose_keys) + tracker.next_lm_id
    assert _traj_error(polished) <= _traj_error(online) + 1e-9
    # the first pose carries the only gauge prior: LM must not move it
    assert np.allclose(polished[0], pose_gt(0), atol=1e-3)


def test_final_lm_default_off_and_finalize_writes_map():
    rng = np.random.default_rng(1)
    lms = landmarks()
    off = ISAM2_Graph(make_cfg())
    vmap_off = make_map(6)
    for job in _noisy_jobs(5, lms, rng):
        off.write_graph_data(off.sequential_optimize(job), vmap_off)
    before = vmap_off.frames.data["pose"].tensor.clone()
    off.finalize(vmap_off)
    assert torch.equal(vmap_off.frames.data["pose"].tensor, before)     # no-op when off
    assert off._tracker().final_lm_stats is None                          # type: ignore[union-attr]

    on = ISAM2_Graph(_lazy_cfg())
    vmap = make_map(6)
    for job in _noisy_jobs(5, lms, np.random.default_rng(1)):
        on.write_graph_data(on.sequential_optimize(job), vmap)
    online = vmap.frames.data["pose"].tensor.clone()
    on.finalize(vmap)
    polished = vmap.frames.data["pose"].tensor
    assert not torch.equal(polished, online)
    for k in range(6):      # every pose key rewritten with a pose near ground truth
        assert np.linalg.norm(_translation(polished[k]) - pose_gt(k)[:3, 3]) < 0.05
    # an empty tracker (no frames ever stepped) must be a no-op too
    ISAM2_Graph(_lazy_cfg()).finalize(make_map(2))


def test_final_lm_under_marg_lag_recovers_expired_poses():
    """The smoother has eliminated p_0..p_4 by the end; the shadow graph plus the
    frozen snapshots still let LM polish the WHOLE trajectory, and the result
    matches the unbounded arm's polish (same model, same measurements)."""
    pytest.importorskip("gtsam_unstable")
    lms = landmarks()
    polished = {}
    for lag in (0, 3):
        tracker = ISAM2FlowTracker(_lazy_cfg(marg_lag=lag))
        for job in _noisy_jobs(8, lms, np.random.default_rng(2)):
            tracker.step(job)
        if lag:
            live = tracker.isam.calculateEstimate()
            assert not live.exists(gtsam.symbol("p", 0)) and tracker._frozen.exists(gtsam.symbol("p", 0))
            assert tracker._shadow is not None and tracker._shadow.nrFactors() > 0 and tracker.final_lm_stats is None
        polished[lag] = tracker.final_lm_solve()
        assert set(polished[lag]) == set(range(9))
        assert tracker.final_lm_stats is not None
        assert tracker.final_lm_stats["n_values"] == len(tracker.pose_keys) + tracker.next_lm_id
    assert _traj_error(polished[3]) < 0.05
    for k in range(9):
        assert np.allclose(polished[0][k], polished[3][k], atol=1e-3), f"pose {k} differs between arms"


def test_final_lm_off_keeps_no_shadow_under_marg_lag():
    pytest.importorskip("gtsam_unstable")
    tracker = ISAM2FlowTracker(make_cfg(marg_lag=3))
    for job in chained_jobs(6, landmarks()):
        tracker.step(job)
    assert tracker._shadow is None and tracker._frozen.size() == 0


def test_final_lm_config_rules():
    ISAM2_Graph.is_valid_config(make_cfg(final_lm=True, final_lm_max_iters=20))
    ISAM2_Graph.is_valid_config(make_cfg(final_lm=False, marg_lag=5))
    for bad in (dict(final_lm=1), dict(final_lm_max_iters=0), dict(final_lm_max_iters=2.5)):
        with pytest.raises(ValueError):
            ISAM2_Graph.is_valid_config(make_cfg(**bad))
