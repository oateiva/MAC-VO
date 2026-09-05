"""
Test for the depth-validity-mask row filtering added to
`MACVO._observe_keyframe` (Odometry/MACVO.py): keyframe re-observations that
land on an invalid-mask pixel of `depth1` must be dropped before depth/cov
sampling, alongside the existing out-of-image drop - and every per-row field
must stay consistently subset.

Builds a MACVO instance via `object.__new__` (no GPU, no full config) with
fakes for `Frontend`/`ObsCovModel` and the real `IdentityFilter` + `VisualMap`,
mirroring the `_FakeDepthModel` pattern in `test_depth_seed.py` and the
`MatchObs`-construction pattern in `test_map_serialization.py`.
"""
from types import SimpleNamespace

import pypose as pp
import torch

from Module.Frontend.StereoDepth import IDepth
from Module.Frontend.Matching import IMatcher
from Module.Map import FrameNode, MatchObs, VisualMap
from Module.OutlierFilter import IdentityFilter
from Odometry.MACVO import MACVO, _KeyframeState

H, W = 20, 20
EDGE = 2
FRAME_IDX = 5


def push_frames(vmap: VisualMap, n: int) -> None:
    """frame2kfmatch auto-grows with `frames`, so `_observe_keyframe` needs the
    graph populated up through the frame index it registers rows against."""
    for i in range(n):
        vmap.frames.push(FrameNode.init({
            "pose": pp.identity_SE3(1).tensor(), "T_BS": pp.identity_SE3(1).tensor(),
            "need_interp": torch.tensor([False]), "time_ns": torch.tensor([i], dtype=torch.long),
            "K": torch.eye(3).unsqueeze(0), "baseline": torch.tensor([0.1]),
        }))


def make_kf_obs(pixel1_uv: torch.Tensor) -> MatchObs:
    n = pixel1_uv.size(0)
    return MatchObs.init({
        "pixel1_uv": pixel1_uv, "pixel2_uv": torch.zeros((n, 2)),
        "pixel1_d": torch.full((n, 1), 3.0), "pixel2_d": torch.zeros((n, 1)),
        "pixel1_disp": torch.full((n, 1), -1.0), "pixel2_disp": torch.zeros((n, 1)),
        "pixel1_disp_cov": torch.full((n, 1), -1.0), "pixel2_disp_cov": torch.zeros((n, 1)),
        "pixel1_uv_cov": torch.zeros((n, 3)), "pixel2_uv_cov": torch.zeros((n, 3)),
        "pixel1_d_cov": torch.full((n, 1), -1.0), "pixel2_d_cov": torch.zeros((n, 1)),
        "obs1_covTc": torch.eye(3).double().repeat(n, 1, 1),
        "obs2_covTc": torch.eye(3).double().repeat(n, 1, 1),
    })


def make_macvo() -> MACVO:
    macvo = object.__new__(MACVO)
    macvo.device = torch.device("cpu")
    macvo.edge_width = EDGE
    macvo.graph = VisualMap()
    push_frames(macvo.graph, FRAME_IDX + 1)
    macvo.OutlierFilter = IdentityFilter(SimpleNamespace())
    macvo.ObsCovModel = SimpleNamespace(  # type: ignore[reportAttributeAccessIssue]
        estimate=lambda camera, kp, depth_est, sigma_dd, sigma_uv: torch.eye(3).double().repeat(kp.size(0), 1, 1)
    )
    macvo.Frontend = SimpleNamespace(  # type: ignore[reportAttributeAccessIssue]
        estimate_match=lambda cam_a, cam_b: IMatcher.Output(flow=torch.zeros((1, 2, H, W)), cov=None),
        retrieve_pixels=lambda uv, scalar_map, interpolate=False: (
            None if scalar_map is None else scalar_map[0, ..., uv[..., 1].long(), uv[..., 0].long()]
        ),
    )
    return macvo


def test_observe_keyframe_drops_masked_and_out_of_bound_rows():
    # rows: in-bounds+unmasked, in-bounds+MASKED, out-of-bounds, in-bounds+unmasked
    pixel1_uv = torch.tensor([[5., 5.], [10., 10.], [1., 1.], [15., 7.]])
    kf_obs = make_kf_obs(pixel1_uv)
    kf_point_idx = torch.arange(pixel1_uv.size(0))

    mask = torch.ones((1, 1, H, W), dtype=torch.bool)
    mask[0, 0, 10, 10] = False    # [v, u]
    depth1 = IDepth.Output(depth=torch.full((1, 1, H, W), 3.0), mask=mask)

    macvo = make_macvo()
    frame1 = SimpleNamespace(camera=SimpleNamespace(width=W, height=H))
    kf = _KeyframeState(camera=SimpleNamespace(), frame_idx=0, depth=depth1)  # type: ignore[reportArgumentType]

    new_obs = macvo._observe_keyframe(frame1, FRAME_IDX, depth1, kf_obs, kf_point_idx, kf)

    assert len(new_obs) == 2
    surviving_uv = {tuple(row.tolist()) for row in new_obs.data["pixel2_uv"]}
    assert surviving_uv == {(5.0, 5.0), (15.0, 7.0)}
    assert len(macvo.graph.kf_match) == 2


def test_observe_keyframe_mask_none_keeps_only_inbound_filtering():
    """No mask on depth1: only the out-of-bound row is dropped (pre-existing
    behavior, unchanged)."""
    pixel1_uv = torch.tensor([[5., 5.], [10., 10.], [1., 1.], [15., 7.]])
    kf_obs = make_kf_obs(pixel1_uv)
    kf_point_idx = torch.arange(pixel1_uv.size(0))

    depth1 = IDepth.Output(depth=torch.full((1, 1, H, W), 3.0), mask=None)

    macvo = make_macvo()
    frame1 = SimpleNamespace(camera=SimpleNamespace(width=W, height=H))
    kf = _KeyframeState(camera=SimpleNamespace(), frame_idx=0, depth=depth1)  # type: ignore[reportArgumentType]

    new_obs = macvo._observe_keyframe(frame1, FRAME_IDX, depth1, kf_obs, kf_point_idx, kf)

    assert len(new_obs) == 3
    surviving_uv = {tuple(row.tolist()) for row in new_obs.data["pixel2_uv"]}
    assert surviving_uv == {(5.0, 5.0), (10.0, 10.0), (15.0, 7.0)}
