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

from Module.Map import VisualMap, FrameNode
from Module.Optimization.GTSAM.ISAM2Optimizer import (
    ISAM2FlowTracker, ISAM2_Graph, ISAM2_GraphInput, ISAM2_GraphOutput,
    _matrix_to_se3, _NATIVE_P2P, gnc_weights, make_native_point_factor,
    make_native_pose_to_point_factor)
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
