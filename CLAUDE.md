# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**MAC-VO** (Metrics-aware Covariance for Learning-based Stereo Visual Odometry) is an ICRA 2025 Best Conference Paper Award winner. It is a Python/CUDA stereo visual odometry system that uses learned depth and optical flow with explicit uncertainty (covariance) modeling for accurate camera pose estimation.

## Commands

### Run MAC-VO
```bash
python3 MACVO.py --odom Config/Experiment/MACVO/MACVO_Performant.yaml --data Config/Sequence/TartanAir_example.yaml
# Optional flags: --useRR (Rerun viz), --saveplt (matplotlib), --preload, --timing, --autoremove, --noeval
# --seq_from / --seq_to to clip the sequence
```

### Select Depth Model from Comet Model Registry
```bash
# Resolve the best registry version by validation loss and emit a runnable _FT config
python Scripts/select_depth_model.py --arch dav2 --workspace <ws> --registry-name <model> \
    --metric validation_loss
python MACVO.py --odom Config/Experiment/MACVO/MACVO_MonoDAv2_FT.yaml --data <seq>

python Scripts/select_depth_model.py --arch dav2 --original   # just point at the base (current) config
```

### Static Analysis
```bash
pyright                          # type-check via pyproject.toml config
```

### Tests
```bash
pytest                           # all non-local tests (CI mode)
pytest -m "not local"            # same as CI
pytest -m local                  # tests that require local GPU/data
pytest -m trt                    # tests requiring TensorRT
pytest Scripts/UnitTest/test_foo.py::test_bar   # single test
```

### Docker
```bash
docker build --network=host -t macvo:latest -f Docker/Dockerfile .
```

## Architecture

### Data Flow
```
SequenceBase (DataLoader) → MACVO.receive_frames()
  → Frontend (depth + optical flow + covariance estimation)
  → KeypointSelector  →  OutlierFilter
  → Covariance projection (2D pixel → 3D)
  → MotionModel prediction  →  IOptimizer (TwoFramePGO, possibly parallel)
  → Map graph update  →  Keyframe decision
  → Sandbox output (poses.npy, tensor_map.npz, *.rrd, ...)
```

### Key Directories
| Path | Purpose |
|---|---|
| `Odometry/MACVO.py` | Main orchestration loop |
| `Module/Frontend/` | Neural nets for depth (`StereoDepth.py`) and optical flow (`Matching.py`) |
| `Module/Covariance/` | Projects 2D pixel uncertainty → 3D covariance (`Project2to3.py`) |
| `Module/Map/Graph.py` | Map graph: `FrameNode`, `PointNode`, `MatchObs` stored as `TensorBundle` |
| `Module/Optimization/` | `TwoFramePGO` backend; supports parallel/non-blocking execution |
| `Module/KeypointSelector.py` | Strategies: Random, Grid, SparseGradient, CovAwareSelector |
| `DataLoader/` | Dataset adapters: TartanAir, KITTI, EuRoC, ZED, EIVA, Aqualoc, EiffelTower |
| `Config/Experiment/MACVO/` | YAML configs for Performant and Fast modes |
| `Config/Sequence/` | YAML configs per dataset sequence |
| `Utility/Config.py` | YAML loader with `!include` support |
| `Utility/Sandbox.py` | Timestamped result folders with metadata |
| `Evaluation/EvalSeq.py` | ATE / RTE / ROE metrics |
| `Baseline/` | DPVO and TartanVO stereo baselines for comparison |
| `Scripts/UnitTest/` | pytest test files |
| `stubs/` | Type stubs for third-party libraries |

### Configuration Pattern
Every major component uses **dynamic instantiation via YAML**:
```yaml
type: ClassName
args:
  key: value
```
`Utility/Config.py` resolves `type` to the actual class and calls it with `args`. Config files use `!include` to compose modular sub-configs. The top-level config keys are `Odometry`, `Frontend`, `MotionModel`, `KeypointSelector`, `Optimization`, `Preprocess`.

### Map Graph (`TensorBundle`)
`Module/Map/Graph.py` uses a **structure-of-arrays** design — data stored as `dict[str, Tensor]` rather than list of objects — for efficient batched GPU operations. `FrameNode`, `PointNode`, and `MatchObs` are all `TensorBundle` subclasses.

### Parallel Optimization
`IOptimizer` supports non-blocking job submission. The main loop submits an optimization job and retrieves the result for the *previous* frame, enabling pipelined execution. Configured via `parallel: true` in the optimizer YAML config.

### Depth Model Selection via Comet
The depth model is chosen purely through config: `monodepth.type` + `monodepth.args.weight` → `build_depth_model()` (`Module/Network/ModelSelector.py`) → `deepodo_initialize()`. Fine-tuned Depth Anything V2/V3 checkpoints are tracked in the **Comet Model Registry**; `Scripts/select_depth_model.py` is an *offline resolver* that lists a registered model's versions, picks the best by following each version to its source experiment and reading the validation-loss metric, downloads + normalizes that version into `Model/finetuned/<arch>/<registry>-<version>/`, writes a `selection.json` manifest, and emits a `_FT` config by copying the base config and patching `monodepth.args.weight`. (`--version` pins an explicit version and skips metric ranking.)

- **Inference stays Comet-free**: `comet_ml` is imported *only* in the resolver script, never on the VO path. Keep it that way.
- **Per-architecture checkpoint format**: V2 loads a bare `.pth` state_dict (`torch.load` + `load_state_dict`); V3 uses HF `from_pretrained`, which needs an HF dir (`config.json` + `model.safetensors`) — the resolver converts a state_dict to an HF dir via `save_pretrained` when needed.
- **"Original / current" = the unchanged base configs** (`MACVO_MonoDAv2.yaml`, `MACVO_MonoDAv3.yaml`); generated fine-tuned variants are the `_FT` siblings.
- Comet credentials come from `COMET_API_KEY` (env) or `~/.comet.config` — never hardcoded or passed on the CLI.

## Important Conventions

- **Python 3.10+ required** — codebase uses `match` syntax and PEP 604 union types (`X | Y`).
- **Coordinate system**: NED (North-East-Down) world frame; pixels in `(u, v)` format.
- **Type checking**: PyRight in `standard` mode is enforced in CI. Third-party network dirs under `Module/Network/` are excluded from checking.
- **Runtime type checking**: `jaxtyping` + `typeguard` are active during pytest for tensor shape validation.
- Pretrained models go in `Model/` (not committed). Download from the GitHub releases page.
- Results are written to `Results/<project_name>@<timestamp>/` by the `Sandbox` class.
- **gtsam**: optional dependency (guard-imported), validated at **4.3a2**. Linux/Docker: `pip install gtsam==4.3a2`. Windows has no PyPI wheels — build one with `Scripts/build_gtsam_windows.ps1` (see `Module/Optimization/README.md`). `gtsam` has no `__version__`; use `importlib.metadata.version("gtsam")`.
