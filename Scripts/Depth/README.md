# External depth-network integration (CVPR-2026 benchmark)

Benchmarks recent learned monocular / geometry depth networks as MAC-VO depth
backbones on the EIVA sequences, then compares trajectories.

## Why offline precompute (not live in-process wrappers)

The target networks pin **mutually incompatible stacks** — Flow3r `torch==2.5.1`,
DGGT `2.4.1`, GemDepth `2.3.1`+flash-attn, MD²E `torch==2.0`+`mmcv-1.7.1`, GGPT
`spconv`/`pointcept` custom CUDA — and MAC-VO runs its depth model *in-process*.
They cannot share MAC-VO's env (AITraining12, torch 2.6). Several are also
video / multi-view models that are degenerate on a single frame.

So each network runs **offline in its own isolated conda env** (`D:/envs/<net>`)
over the exact frames MAC-VO feeds its matcher, caching per-frame depth (+conf).
MAC-VO then consumes that cache through one config-driven frontend model,
`Module/Network/Depth/Cached/api.py` → `CachedDepth` (registered in
`Module/Network/ModelSelector.py`). This is faithful to MAC-VO's config style, is
dependency-isolated, reproducible, and lets video/multi-view nets run in their
natural batched mode.

## Pipeline

```
Scripts/Depth/dump_frames.py            # (AITraining12) export preprocessed 700x700 frames
   -> D:/macvo_depth/frames/<seq>/<frame_ns>.png  (+ meta.json)
Scripts/Depth/precompute_<net>.py       # (D:/envs/<net>) run net -> depth[+conf]
   -> D:/macvo_depth/cache/<net>/<frame_ns>.npz   {depth[, conf]}
CachedDepth (MAC-VO frontend)           # serve cache by frame_ns
Config/Experiment/MACVO/Custom/MACVO_<net>_GEDF.yaml   # GEDF backend, clip200 EIVA set
```

`frame_key(frame_ns) = str(int(round(float(ns))))` is the shared cache key (defined
identically in dump_frames.py, CachedDepth, and every precompute script).

Experiments use the **clip200** EIVA set (`Config/Sequence/EIVA_mono_datasets_clip200.yaml`,
first 200 frames/seq) — the full set is ~6,500 frames (Welland alone is 4,717) which
is impractical across many nets. The repo already benchmarks `plane_nose[80:160]`.

## Per-network status

| Net | Env | Input->depth | Metric? | Var | Status |
|---|---|---|---|---|---|
| **DGGT** | D:/envs/dggt (torch2.4.1) | S=1 forward, `pred["depth"]` | metric (driving-trained; OOD underwater) | `depth_conf` | RUNS ✅ |
| **Pixel-Perfect Depth** | D:/envs/ppd | `infer_image` (DA2 variant) | relative (→correct_scale) | heuristic | RUNS ✅ |
| **Flow3r** | D:/envs/flow3r (torch2.5.1) | N=1, `local_points[...,2]` | scale-invariant | `conf` | precompute staged |
| **GemDepth** | D:/envs/gemdepth (torch2.3.1) | video clips | relative | heuristic | RUNS ✅ (Windows workarounds) |
| **GGPT** | D:/envs/dggt (reused) | VGGT-1B `depth` (single-image backbone) | up-to-scale | conf (≥1) | RUNS ✅ as backbone only. Full GGPT (multi-view SfM + PTv3/spconv-CUDA refinement) is not monocular and not Windows-feasible; "ggpt" here = its stock VGGT-1B feedforward backbone (distinct from DGGT's Waymo-VGGT). |
| **MD²E / LR2Depth** | D:/envs/mde (torch1.13+cu117+mmcv1.7.1) | Swin+OneP, RGB-only | metric (KITTI-scale, OOD) | none | RUNS ✅ with user weight `1P-L-K.pth` (Z:/Research/weights/depth_estimation/mde) + `configs/LRDepth/1P_L_kitti.py`. The separate MD2EHead ckpt is still unreleased. |
| **MTD** | — | needs sparse-depth seeds + superpixels | metric | none | BLOCKED: not RGB-only (a completion method); full fitting code withheld |
| **PTC-Depth** | — | refines external depth using a metric baseline | via baseline | yes | BLOCKED: not a depth source; needs metric per-frame baseline + C++ build |

**GemDepth Windows notes:** flash/mem-efficient SDPA kernels aren't shipped in PyTorch-Windows,
and GemDepth uses global spatiotemporal attention. `precompute_gemdepth.py` uses
`attn_implementation="pytorch_naive"` + a pure-matmul SDPA monkeypatch, and must run at a reduced
`--input_size 252` (memory scales with size⁴; 518 OOMs at 57 GB). Depth quality is thus below the
net's native resolution — a Windows/hardware limitation, noted in results.

Run a net: `<env python> Scripts/Depth/precompute_<net>.py --frames D:/macvo_depth/frames
--out D:/macvo_depth/cache/<net> --ckpt <weights> --limit 200`, then
`/macvo_experiment` on `Config/Experiment/MACVO/Custom/MACVO_<net>_GEDF.yaml`, then
`/eiva_trajectory_evaluator` on the curated result folder.
