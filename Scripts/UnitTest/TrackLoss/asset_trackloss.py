"""
Shared fixtures & probes for the track-loss tests.

Named `asset_*` (not `test_*`) so pytest does not collect it, following the
convention set by `Scripts/UnitTest/asset_gedf.py`.

The tests in this directory answer one question: when MAC-VO is handed a second
frame that is *not* geometrically related to the first, does it notice? MAC-VO
already has the machinery to say so - `Odometry/MACVO.py` marks a frame
`need_interp=True` and skips optimization on either of two lost-track branches:

  A. no keypoint survived selection + the in-bounds gate  (MACVO.py:257-266)
  B. fewer than `min_num_point` observations survived     (MACVO.py:355, :380-384)

So `need_interp` is the authoritative "do not trust this pose" flag, and the
observation count tells us which branch fired.

Keypoint selection is `TrackingCovAwareSelector`. It is stateful, but each test
builds a fresh system, so `track_uv` is empty on the pair under test and only
the seeding path runs - which is deterministic (no `randperm`).
"""
import copy
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest
import torch

from DataLoader import Frame, SequenceBase, smart_transform
from DataLoader.Interface import CameraData
from DataLoader.Transform import SmartResizeFrame
from Odometry.MACVO import MACVO
from Utility.Config import load_config


# --- Configuration under test (monocular only) --------------------------------
ODOM_CFG   = Path("./Config/Experiment/MACVO/MACVO_MonoDAv2.yaml")
DATA_CFG   = Path("./Config/Sequence/EIVA_plane_nose_mono.yaml")
PLANE_NOSE = Path(r"D:\Datasets\EIVA\vobster_quay\plane_nose")
WEIGHTS    = (
    Path("./Model/MACVO_FrontendCov.pth"),                            # FlowFormerCovMatcher
    Path("C:/ai_checkpoints/monodepth/depth_anything_v2_vitl.pth"),   # DepthAnythingV2
)
FOREIGN    = Path("./Scripts/UnitTest/assets/test_trackloss/foreign_turbot.jpg")
REPORT_DIR = Path("./cache/trackloss")

SEED = 0       # seeded choice of the plane_nose anchor frame
GAP  = 400     # frame offset used for the "uncorrelated real frame" case

MIN_NUM_POINT = 10   # MACVO_MonoDAv2.yaml does not override it (Odometry/MACVO.py:59)

# Args as in Config/Experiment/MACVO/ISAM2/*.yaml; mask_width must stay >= edgewidth (32).
KEYPOINT_SELECTOR = SimpleNamespace(
    type="TrackingCovAwareSelector",
    args=SimpleNamespace(
        device="cuda",
        mask_width=32,
        kernel_size=7,
        max_match_cov=0.5,
        median_rel=1.5,
        seed_radius=90.0,
    ),
)


requires_assets = pytest.mark.skipif(
    not (PLANE_NOSE.exists() and FOREIGN.exists() and all(w.exists() for w in WEIGHTS)),
    reason="plane_nose data, turbot asset or monocular weights unavailable",
)


# --- System / sequence construction -------------------------------------------
def build_sequence() -> SequenceBase[Frame]:
    """The exact production data path (see MACVO.py:408-409)."""
    odom, _ = load_config(ODOM_CFG)
    data, _ = load_config(DATA_CFG)
    return smart_transform(
        SequenceBase[Frame].instantiate(data.type, data.args),
        odom.Preprocess,
    )


def build_system() -> MACVO[Frame]:
    """A fresh MACVO with an empty map graph.

    `from_config` wants a namespace whose `.Odometry` is the odometry config
    (Odometry/MACVO.py:103-131), which is exactly what `load_config` returns,
    so no Sandbox round-trip is needed.

    The optimizer is forced sequential: `parallel_timeout_s: 1.0` lets a
    parallel backend silently drop a result, which would make the pose readout
    flaky without changing what we assert on.

    Keypoint selection is overridden to `TrackingCovAwareSelector`, replacing
    the base config's `CovAwareSelector_NoDepth`.
    """
    odom, _ = load_config(ODOM_CFG)
    odom.Odometry.optimizer.args.parallel = False
    odom.Odometry.keypoint = KEYPOINT_SELECTOR

    system = MACVO[Frame].from_config(odom)
    assert type(system.KeypointSelector).__name__ == KEYPOINT_SELECTOR.type, (
        f"selector override did not take effect: got "
        f"{type(system.KeypointSelector).__name__}"
    )
    return system


def seeded_index(seq_len: int) -> int:
    """Seeded anchor frame, kept clear of the end so `i+1` and `i+GAP` exist."""
    assert seq_len > GAP + 2, f"plane_nose too short ({seq_len}) for GAP={GAP}"
    return random.Random(SEED).randrange(0, seq_len - GAP - 2)


# --- Synthetic second frames --------------------------------------------------
def black_like(frame: Frame) -> torch.Tensor:
    """A pure black image matching the real frame's shape/dtype/device exactly."""
    return torch.zeros_like(frame.camera.imageL)


def foreign_like(frame: Frame) -> torch.Tensor:
    """The turbot photo, brought to the frame's resolution.

    Reuses the production `SmartResizeFrame` (aspect-preserving scale + centre
    crop) with the target read off the *already transformed* real frame, so this
    can never drift from Config/Experiment/Common/Preprocess.yaml.
    """
    bgr = cv2.imread(str(FOREIGN), cv2.IMREAD_COLOR)
    assert bgr is not None, f"could not read {FOREIGN}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    img = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    resizer = SmartResizeFrame(SimpleNamespace(
        height=frame.camera.height, width=frame.camera.width, interp="nearest",
    ))
    probe_frame = Frame(
        idx=[0],
        time_ns=[0],
        camera=CameraData.from_mono(
            T_BS=frame.camera.T_BS,
            K=frame.camera.K,
            time_ns=[0],
            height=img.shape[-2],
            width=img.shape[-1],
            images=img,
        ),
    )
    out = resizer(probe_frame).camera.imageL

    # window_length: 2 makes the loader repeat one image, so B is 2 (EIVA.py:296-306).
    reference = frame.camera.imageL
    return out.repeat(reference.shape[0], 1, 1, 1).to(
        dtype=reference.dtype, device=reference.device,
    )


def swap_image(frame: Frame, image: torch.Tensor) -> Frame:
    """Same camera, different scene.

    Intrinsics, `T_BS` and `baseline` are deliberately left untouched: we are
    testing MAC-VO's reaction to unrelated *content*, not to a broken camera.
    """
    out = copy.deepcopy(frame)
    out.camera.imageL = image
    return out


# --- Probing the result -------------------------------------------------------
@dataclass
class Report:
    case            : str
    need_interp     : bool
    n_obs           : int
    branch          : str
    flow_cov_median : float | None
    step_translation: float


def probe(system: MACVO[Frame], case: str) -> Report:
    graph = system.graph
    obs   = graph.get_frame2match(graph.frames[-1:])
    n_obs = len(obs)

    if n_obs == 0:
        branch = "A/no-keypoints"
    elif n_obs < MIN_NUM_POINT:
        branch = "B/low-support"
    else:
        branch = "none"

    # -1 is the "not provided" sentinel written by Odometry/MACVO.py:313-323.
    flow_cov_median: float | None = None
    if n_obs > 0:
        cov = obs.data["pixel2_uv_cov"][:, :2]
        if bool((cov >= 0).any()):
            flow_cov_median = float(cov[cov >= 0].median())

    poses = graph.frames.data["pose"]
    step  = (
        float(torch.linalg.norm(poses[-1][:3] - poses[-2][:3]))
        if poses.size(0) >= 2 else 0.0
    )

    return Report(
        case=case,
        need_interp=bool(graph.frames.data["need_interp"][-1]),
        n_obs=n_obs,
        branch=branch,
        flow_cov_median=flow_cov_median,
        step_translation=step,
    )


def run_pair(system: MACVO[Frame], frame0: Frame, frame1: Frame, case: str) -> Report:
    """One single pass: initialize on frame0, then step onto frame1.

    Goes through `run` (not `run_pair`) so `prev_keyframe` and
    `OutlierFilter.set_meta` are set up exactly as in a real run.
    """
    system.run(frame0)      # initialize only - no observations yet
    system.run(frame1)      # -> MACVO.run_pair(frame0, frame1)
    # Normally written back at the top of the next run_pair (Odometry/MACVO.py:236).
    system.Optimizer.write_map(system.graph)
    return probe(system, case)


def report(r: Report) -> None:
    """Record the diagnostic payload - these numbers are the point of the suite."""
    cov = "n/a" if r.flow_cov_median is None else f"{r.flow_cov_median:.4f}"
    print(
        f"\n[track-loss] {r.case:<22} lost={str(r.need_interp):<5} "
        f"n_obs={r.n_obs:<5} branch={r.branch:<14} "
        f"flow_cov_med={cov:<8} step_t={r.step_translation:.5f}"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{r.case}.json").write_text(json.dumps(asdict(r), indent=2))


def teardown(system: MACVO[Frame]) -> None:
    """Release the frontend before the next test builds its own."""
    system.Optimizer.terminate()
    del system
    torch.cuda.empty_cache()
