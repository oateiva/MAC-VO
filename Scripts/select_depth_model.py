#!/usr/bin/env python3
"""
Select the depth-model checkpoint for a MAC-VO run from the Comet Model Registry.

MAC-VO chooses a depth model purely through config: ``monodepth.type`` +
``monodepth.args.weight`` -> ``build_depth_model()`` -> ``deepodo_initialize()``.
This script is an *offline resolver*: it looks up a registered model in the Comet
**Model Registry**, picks the best version by validation loss (by following each
version to its source experiment and reading the metric there), downloads + normalizes
that version's checkpoint into ``Model/finetuned/<arch>/<registry>-<version>/``, and
writes a ready-to-run ``_FT`` experiment YAML by copying a base config and patching
``monodepth.args.weight``.

Inference stays Comet-free: ``comet_ml`` is imported only here, never on the VO path.

Examples
--------
Run the original (current) weights -- no Comet needed, just confirms the base config::

    python Scripts/select_depth_model.py --arch dav2 --original

Resolve the best fine-tuned DA-V2 from the registry and emit a runnable config::

    python Scripts/select_depth_model.py --arch dav2 \
        --workspace my-ws --registry-name depth-anything-v2 \
        --metric validation_loss \
        --out-config Config/Experiment/MACVO/MACVO_MonoDAv2_FT.yaml

    python MACVO.py --odom Config/Experiment/MACVO/MACVO_MonoDAv2_FT.yaml --data <seq>

Pin an explicit registry version instead of ranking by metric::

    python Scripts/select_depth_model.py --arch dav3 \
        --workspace my-ws --registry-name depth-anything-v3 --version 1.2.0

Comet credentials are read from the ``COMET_API_KEY`` environment variable (or the
standard ``~/.comet.config`` file) -- this script never takes a key on the CLI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Per-architecture defaults. The base config is the "original / current" model; the
# generated config is a copy with monodepth.args.weight (and model_name for V3) patched.
ARCH = {
    "dav2": {
        "base_config":    REPO_ROOT / "Config/Experiment/MACVO/MACVO_MonoDAv2.yaml",
        "default_out":     REPO_ROOT / "Config/Experiment/MACVO/MACVO_MonoDAv2_FT.yaml",
        "format":         "state_dict",   # single .pth loaded via torch.load + load_state_dict
        "default_preset": "vitl",
    },
    "dav3": {
        "base_config":    REPO_ROOT / "Config/Experiment/MACVO/MACVO_MonoDAv3.yaml",
        "default_out":     REPO_ROOT / "Config/Experiment/MACVO/MACVO_MonoDAv3_FT.yaml",
        "format":         "hf_dir",       # from_pretrained() dir: config.json + model.safetensors
        "default_preset": "da3nested-giant-large",
    },
}

CKPT_ROOT = REPO_ROOT / "Model" / "finetuned"
HF_FILES = ("config.json", "model.safetensors")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------- Comet model registry

def _version_source_experiment_key(version_entry: dict) -> str | None:
    """Extract the source-experiment key from a registry version entry.

    Comet SDK versions nest this differently, so probe the known shapes.
    """
    return (
        version_entry.get("experimentKey")
        or (version_entry.get("experimentModel") or {}).get("experimentKey")
        or (version_entry.get("experiment") or {}).get("experimentKey")
    )


def _metric_value(summary, metric: str, mode: str) -> float | None:
    """Pull the best (min/max) metric value from a get_metrics_summary() result."""
    if isinstance(summary, list):
        summary = next((s for s in summary if s.get("name") == metric), None)
    if not summary:
        return None
    value = summary.get("valueMin") if mode == "min" else summary.get("valueMax")
    if value is None:
        value = summary.get("valueCurrent")
    return None if value is None else float(value)


def list_registry_versions(api, workspace: str, registry_name: str) -> list[dict]:
    details = api.get_registry_model_details(workspace, registry_name)
    if not details:
        names = api.get_registry_model_names(workspace) or []
        raise SystemExit(
            f"Registered model '{registry_name}' not found in workspace '{workspace}'. "
            f"Available: {names}")
    versions = details.get("versions") or []
    if not versions:
        raise SystemExit(f"Registered model '{registry_name}' has no versions.")
    return versions


def pick_best_version(api, workspace: str, registry_name: str, metric: str, mode: str):
    """Return (version_str, metric_value, source_experiment) for the best registry version.

    Ranks versions by following each to its source experiment and reading `metric`.
    """
    versions = list_registry_versions(api, workspace, registry_name)
    best = None  # (version_str, value, experiment)
    considered, skipped = 0, 0
    for entry in versions:
        version_str = entry.get("version")
        exp_key = _version_source_experiment_key(entry)
        if not exp_key:
            skipped += 1
            log(f"  - skip v{version_str}: no linked source experiment")
            continue
        exp = api.get_experiment_by_key(exp_key)
        value = _metric_value(exp.get_metrics_summary(metric), metric, mode)
        if value is None:
            skipped += 1
            log(f"  - skip v{version_str} ({exp_key}): no metric '{metric}'")
            continue
        considered += 1
        better = best is None or (value < best[1] if mode == "min" else value > best[1])
        if better:
            best = (version_str, value, exp)

    log(f"Considered {considered} version(s), skipped {skipped}.")
    if best is None:
        raise SystemExit(
            f"No version of '{registry_name}' had metric '{metric}' on its source experiment.")
    return best


def download_registry_version(api, workspace: str, registry_name: str, version: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    log(f"Downloading registry model '{registry_name}' v{version} -> {dest}")
    api.download_registry_model(
        workspace, registry_name, version=version, output_path=str(dest), expand=True)
    return dest


# --------------------------------------------------------------- format normalization

def _find_state_dict(download_dir: Path) -> Path:
    candidates = sorted(
        [p for p in download_dir.rglob("*") if p.suffix in (".pth", ".pt", ".ckpt", ".safetensors")],
        key=lambda p: p.stat().st_size, reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No checkpoint file found under {download_dir}.")
    return candidates[0]


def normalize_dav2(download_dir: Path) -> Path:
    """V2 wants a single .pth state_dict; return the path to it."""
    return _find_state_dict(download_dir)


def normalize_dav3(download_dir: Path, preset: str) -> Path:
    """V3 wants an HF dir (config.json + model.safetensors) for from_pretrained().

    If the download already contains those files, return that dir. Otherwise convert a
    bare state_dict once: load it into a DepthAnything3(preset) and save_pretrained().
    """
    for parent in {download_dir, *(p.parent for p in download_dir.rglob("config.json"))}:
        if all((parent / f).exists() for f in HF_FILES):
            log(f"Found HF-format directory: {parent}")
            return parent

    log("No HF directory in download; converting state_dict -> HF format via save_pretrained().")
    import torch
    from Module.Network.Depth.DepthAnythingV3.api import DepthAnything3

    ckpt_path = _find_state_dict(download_dir)
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(ckpt_path))
    else:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    model = DepthAnything3(model_name=preset)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)} key(s).")
    hf_dir = download_dir / "hf"
    model.save_pretrained(str(hf_dir))
    log(f"Saved HF-format model to {hf_dir}")
    return hf_dir


# ------------------------------------------------------------------- config patching

def patch_monodepth_args(base_config: Path, out_config: Path, updates: dict[str, str]) -> None:
    """Copy `base_config` to `out_config`, replacing keys under monodepth.args.

    Deliberately a line-level text edit (not load+reserialize) so YAML anchors (`*device`)
    and `!include` directives in the base config are preserved verbatim.
    """
    lines = base_config.read_text(encoding="utf-8").splitlines(keepends=True)

    # Locate the monodepth: block and its indentation.
    mono_idx = next((i for i, ln in enumerate(lines) if ln.strip().rstrip(":") == "monodepth"), None)
    if mono_idx is None:
        raise SystemExit(f"Could not find a 'monodepth:' block in {base_config}.")
    mono_indent = len(lines[mono_idx]) - len(lines[mono_idx].lstrip())

    remaining = dict(updates)
    i = mono_idx + 1
    while i < len(lines) and remaining:
        ln = lines[i]
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= mono_indent:
            break  # dedented out of the monodepth block
        stripped = ln.strip()
        for key in list(remaining):
            if stripped.startswith(f"{key}:"):
                indent = ln[: len(ln) - len(ln.lstrip())]
                lines[i] = f"{indent}{key}: {remaining.pop(key)}\n"
                break
        i += 1

    if remaining:
        raise SystemExit(f"Keys not found under monodepth.args in {base_config}: {list(remaining)}")

    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text("".join(lines), encoding="utf-8")
    log(f"Wrote config: {out_config}")


# ---------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=list(ARCH), help="Depth architecture to resolve.")
    ap.add_argument("--original", action="store_true",
                    help="Skip Comet; just confirm the base config is the original model.")
    ap.add_argument("--workspace", help="Comet workspace owning the Model Registry.")
    ap.add_argument("--registry-name", help="Registered model name in the Comet Model Registry.")
    ap.add_argument("--metric", default="validation_loss", help="Metric name to rank versions by.")
    ap.add_argument("--mode", default="min", choices=["min", "max"], help="Best = lowest (min) or highest (max).")
    ap.add_argument("--version", help="Pin an explicit registry version (skips metric ranking).")
    ap.add_argument("--model-name", help="V3 preset for state_dict->HF conversion (default per arch).")
    ap.add_argument("--out-config", help="Path for the generated _FT YAML (default per arch).")
    args = ap.parse_args()

    arch = ARCH[args.arch]
    base_config = arch["base_config"]
    out_config = Path(args.out_config) if args.out_config else arch["default_out"]
    if not base_config.exists():
        raise SystemExit(f"Base config not found: {base_config}")

    if args.original:
        log(f"Original / current model for {args.arch}: use the base config directly:\n  {base_config}")
        return

    if not (args.workspace and args.registry_name):
        raise SystemExit("--workspace and --registry-name are required unless --original is set.")

    from comet_ml import API  # imported lazily so non-Comet paths need no install

    api = API()  # reads COMET_API_KEY from env / ~/.comet.config

    source_exp = None
    if args.version:
        version, metric_value = args.version, None
        log(f"Using pinned registry version: {args.registry_name} v{version}")
    else:
        log(f"Querying Comet registry '{args.workspace}/{args.registry_name}' "
            f"for best '{args.metric}' ({args.mode})...")
        version, metric_value, source_exp = pick_best_version(
            api, args.workspace, args.registry_name, args.metric, args.mode)
        log(f"Best: v{version}  {args.metric}={metric_value:g}\n  "
            f"source experiment {source_exp.get_name()} ({source_exp.id})\n  {source_exp.url}")

    dest = CKPT_ROOT / args.arch / f"{args.registry_name}-{version}"
    download_registry_version(api, args.workspace, args.registry_name, version, dest)

    preset = args.model_name or arch["default_preset"]
    if arch["format"] == "state_dict":
        weight_path = normalize_dav2(dest)
        updates = {"weight": weight_path.resolve().as_posix()}
    else:  # hf_dir
        weight_path = normalize_dav3(dest, preset)
        updates = {"weight": weight_path.resolve().as_posix()}
        if args.model_name:
            updates["model_name"] = args.model_name

    manifest = {
        "arch": args.arch,
        "workspace": args.workspace,
        "registry_name": args.registry_name,
        "version": version,
        "metric": None if args.version else args.metric,
        "metric_mode": None if args.version else args.mode,
        "metric_value": metric_value,
        "source_experiment_key": source_exp.id if source_exp else None,
        "source_experiment_url": source_exp.url if source_exp else None,
        "resolved_weight": updates["weight"],
        "format": arch["format"],
    }
    (dest / "selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote manifest: {dest / 'selection.json'}")

    patch_monodepth_args(base_config, out_config, updates)
    log(f"\nDone. Run MAC-VO with:\n  python MACVO.py --odom {out_config} --data <sequence-config>")


if __name__ == "__main__":
    main()
