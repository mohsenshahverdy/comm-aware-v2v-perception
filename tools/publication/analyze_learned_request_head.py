#!/usr/bin/env python3
"""Analyze learned temporal receiver-request maps for publication evidence.

The script is intentionally artifact-driven: it computes statistics from the
learned temporal debug maps exported by CommunicationPolicy. If maps are not
already available, it can create a separate analysis run directory, patch the
config to enable learned debug export, symlink/copy the checkpoint, and run
inference to generate them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - handled at runtime
    plt = None

DEFAULT_CHECKPOINT_DIR = Path(
    "/home/ex-perception/runs/v2v_trajectory_carla_learned_lr1e3/"
    "smoke_carla_learned_temporal_receiver_request_10"
)
DEFAULT_VALIDATE_DIR = Path(
    "/home/ex-perception/Desktop/projects/data-all/test/carla_2021_only_validate"
)
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results/publication/learned_request_head_statistics.csv"
DEFAULT_FIGURES_DIR = REPO_ROOT / "results/publication/figures"
DEFAULT_ANALYSIS_RUN_DIR = REPO_ROOT / "results/publication/learned_request_head_analysis_run"
DEFAULT_DEBUG_SUBDIR = "learned_temporal_receiver_request_debug"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _deep_merge(destination: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _deep_merge(destination[key], value)
        else:
            destination[key] = value
    return destination


def _base_config_from_preset() -> Dict[str, Any]:
    base_path = REPO_ROOT / "src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml"
    preset_path = REPO_ROOT / "src/hypes_yaml/communication_approach_presets.yaml"
    base = _load_yaml(base_path)
    presets = (_load_yaml(preset_path).get("communication_presets", {}))
    preset = presets.get("learned_temporal_receiver_request_10")
    if not isinstance(preset, dict):
        raise KeyError("Preset learned_temporal_receiver_request_10 was not found")
    communication = base.setdefault("model", {}).setdefault("args", {}).setdefault("communication", {})
    _deep_merge(communication, preset)
    base.pop("communication_preset", None)
    return base


def _copy_or_link_checkpoint_files(source_dir: Path, run_dir: Path, *, mode: str) -> List[Path]:
    patterns = ["latest.pth", "net_epoch*.pth", "*epoch*.pth"]
    candidates: List[Path] = []
    for pattern in patterns:
        for path in sorted(source_dir.glob(pattern)):
            if path.is_file() and path not in candidates:
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint .pth files found in {source_dir}")

    created: List[Path] = []
    for src in candidates:
        dst = run_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)
        created.append(dst)
    return created


def _prepare_analysis_run(
    *,
    checkpoint_dir: Path,
    validate_dir: Path,
    root_dir: Optional[Path],
    run_dir: Path,
    debug_dir: Path,
    max_samples: int,
    force: bool,
    checkpoint_link_mode: str,
    logger,
) -> Path:
    if run_dir.exists() and force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_config = checkpoint_dir / "config.yaml"
    if source_config.exists():
        config = _load_yaml(source_config)
        logger.info("Using checkpoint config", config=source_config)
    else:
        config = _base_config_from_preset()
        logger.warn("Checkpoint config.yaml not found; using base config plus learned preset", checkpoint_dir=checkpoint_dir)

    config["validate_dir"] = str(validate_dir)
    config["root_dir"] = str(root_dir) if root_dir is not None else str(config.get("root_dir") or validate_dir)
    config["communication_preset"] = "learned_temporal_receiver_request_10"

    comm = config.setdefault("model", {}).setdefault("args", {}).setdefault("communication", {})
    rr_cfg = comm.setdefault("receiver_request", {})
    rr_cfg["keep_ratio"] = 0.10
    rr_cfg["strategy_variant"] = "learned_temporal"
    rr_cfg["enabled"] = True
    temporal_cfg = rr_cfg.setdefault("temporal", {})
    temporal_cfg["enabled"] = True
    learned_cfg = rr_cfg.setdefault("learned", {})
    learned_cfg["enabled"] = True
    learned_cfg["target_budget"] = 0.10
    debug_cfg = learned_cfg.setdefault("debug", {})
    debug_cfg["save_learned_maps"] = True
    debug_cfg["debug_dir"] = str(debug_dir)
    debug_cfg["debug_num_frames"] = int(max_samples) if max_samples > 0 else 1_000_000
    learned_cfg["debug_num_frames"] = debug_cfg["debug_num_frames"]

    _save_yaml(run_dir / "config.yaml", config)
    _save_yaml(run_dir / "config_resolved.yaml", config)
    created = _copy_or_link_checkpoint_files(checkpoint_dir, run_dir, mode=checkpoint_link_mode)
    logger.save("Analysis run prepared", run_dir=run_dir, config=run_dir / "config_resolved.yaml", checkpoint_files=len(created))
    return run_dir / "config.yaml"


def _run_inference(
    *,
    run_dir: Path,
    validate_dir: Path,
    python_bin: str,
    max_samples: int,
    skip_ap: bool,
    deterministic: bool,
    allow_untrained_request_head: bool,
    logger,
) -> None:
    command = [
        python_bin,
        "-m",
        "src.tools.inference",
        "--model_dir",
        str(run_dir),
        "--fusion_method",
        "intermediate",
        "--validate_dir",
        str(validate_dir),
        "--seed",
        "0",
    ]
    if max_samples > 0:
        command.extend(["--max_samples", str(max_samples)])
    if skip_ap:
        command.append("--skip_ap")
    if deterministic:
        command.append("--deterministic")
    if allow_untrained_request_head:
        command.append("--allow_untrained_request_head")

    log_path = run_dir / "learned_request_head_analysis_inference.log"
    logger.command("Running learned request-head analysis inference", cmd=" ".join(command), log=log_path)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"Inference failed with exit code {process.returncode}; see {log_path}")


def _map_from_npz(path: Path, key: str) -> np.ndarray:
    payload = np.load(path)
    if key not in payload:
        raise KeyError(f"{path} does not contain {key}")
    arr = np.asarray(payload[key], dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected {key} in {path} to become a 2D map after squeeze, got shape {arr.shape}")
    return arr


def _normalized_entropy(score_map: np.ndarray, eps: float = 1e-12) -> float:
    values = np.clip(np.asarray(score_map, dtype=np.float64).reshape(-1), 0.0, None)
    total = float(values.sum())
    if total <= eps or values.size <= 1:
        return 0.0
    probs = values / total
    entropy = -float(np.sum(probs * np.log(probs + eps)))
    return float(entropy / math.log(values.size))


def _top_fraction_mass(score_map: np.ndarray, fraction: float = 0.10, eps: float = 1e-12) -> float:
    values = np.clip(np.asarray(score_map, dtype=np.float64).reshape(-1), 0.0, None)
    total = float(values.sum())
    if total <= eps:
        return 0.0
    k = max(1, int(math.ceil(values.size * float(fraction))))
    return float(np.sort(values)[-k:].sum() / total)


def _discover_npz(debug_dir: Path) -> List[Path]:
    return sorted(debug_dir.glob("*.npz"))


def analyze_debug_maps(debug_dir: Path, logger) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    files = _discover_npz(debug_dir)
    if not files:
        raise FileNotFoundError(f"No learned debug .npz maps found in {debug_dir}")

    maps: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    per_file: List[Dict[str, Any]] = []
    target_shape: Optional[Tuple[int, int]] = None
    skipped = 0

    for path in files:
        try:
            prob = _map_from_npz(path, "learned_request_prob")
            mask = _map_from_npz(path, "final_request_mask")
        except Exception as exc:
            skipped += 1
            logger.warn("Skipping debug map", path=path, reason=str(exc))
            continue
        if target_shape is None:
            target_shape = prob.shape
        if prob.shape != target_shape:
            skipped += 1
            logger.warn("Skipping debug map with incompatible shape", path=path, shape=prob.shape, expected=target_shape)
            continue
        maps.append(prob)
        masks.append(mask)
        per_file.append({
            "file": path.name,
            "mean_request_score": float(np.mean(prob)),
            "std_request_score": float(np.std(prob)),
            "selected_cell_ratio": float(np.mean(mask > 0.0)),
            "normalized_entropy": _normalized_entropy(prob),
            "top10_percent_mass": _top_fraction_mass(prob, 0.10),
        })

    if not maps:
        raise RuntimeError(f"No valid learned_request_prob maps could be read from {debug_dir}")

    stack = np.stack(maps, axis=0)
    mask_stack = np.stack(masks, axis=0)
    mean_map = np.mean(stack, axis=0)
    std_map = np.std(stack, axis=0)
    per_map_stds = np.std(stack.reshape(stack.shape[0], -1), axis=1)
    per_cell_std = std_map.reshape(-1)

    summary = {
        "debug_dir": str(debug_dir),
        "num_debug_files": int(len(files)),
        "num_valid_maps": int(stack.shape[0]),
        "num_skipped_maps": int(skipped),
        "map_height": int(stack.shape[1]),
        "map_width": int(stack.shape[2]),
        "mean_request_score": float(np.mean(stack)),
        "std_request_score": float(np.std(stack)),
        "per_frame_request_map_std_mean": float(np.mean(per_map_stds)),
        "per_frame_request_map_std_std": float(np.std(per_map_stds)),
        "per_spatial_cell_std_over_frames_mean": float(np.mean(per_cell_std)),
        "per_spatial_cell_std_over_frames_median": float(np.median(per_cell_std)),
        "per_spatial_cell_std_over_frames_p90": float(np.percentile(per_cell_std, 90)),
        "selected_cell_ratio": float(np.mean(mask_stack > 0.0)),
        "selected_cell_ratio_std": float(np.std(np.mean(mask_stack.reshape(mask_stack.shape[0], -1) > 0.0, axis=1))),
        "normalized_entropy_mean": float(np.mean([row["normalized_entropy"] for row in per_file])),
        "normalized_entropy_std": float(np.std([row["normalized_entropy"] for row in per_file])),
        "top10_percent_mass_mean": float(np.mean([row["top10_percent_mass"] for row in per_file])),
        "top10_percent_mass_std": float(np.std([row["top10_percent_mass"] for row in per_file])),
    }
    return summary, mean_map, std_map, per_file


def _write_statistics_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _save_map_figure(map_data: np.ndarray, path_stem: Path, title: str, colorbar_label: str) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate figures")
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    im = ax.imshow(map_data, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("BEV x-cell")
    ax.set_ylabel("BEV y-cell")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(colorbar_label)
    for suffix in ("pdf", "png"):
        fig.savefig(path_stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _save_histogram(values: np.ndarray, path_stem: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate figures")
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    ax.hist(values, bins=40, color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.set_title("Spatial Cell Request-Score Variability", fontsize=12, fontweight="bold")
    ax.set_xlabel("Std. dev. over frames")
    ax.set_ylabel("Number of BEV cells")
    ax.grid(True, axis="y", alpha=0.25)
    for suffix in ("pdf", "png"):
        fig.savefig(path_stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def generate_figures(figures_dir: Path, mean_map: np.ndarray, std_map: np.ndarray, logger) -> None:
    _save_map_figure(
        mean_map,
        figures_dir / "learned_request_head_mean_map",
        "Learned Request-Head Mean Request Map",
        "Mean request probability",
    )
    _save_map_figure(
        std_map,
        figures_dir / "learned_request_head_std_map",
        "Learned Request-Head Spatial Variability Map",
        "Std. dev. over maps",
    )
    _save_histogram(std_map.reshape(-1), figures_dir / "learned_request_head_std_histogram")
    logger.save("Learned request-head figures saved", figures_dir=figures_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--validate_dir", type=Path, default=DEFAULT_VALIDATE_DIR)
    parser.add_argument("--root_dir", type=Path, default=None, help="Optional train/root directory override for inference config.")
    parser.add_argument("--analysis_run_dir", type=Path, default=DEFAULT_ANALYSIS_RUN_DIR)
    parser.add_argument("--debug_npz_dir", type=Path, default=None, help="Analyze existing learned debug maps instead of using the default analysis-run debug directory.")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--figures_dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--max_samples", type=int, default=0, help="Optional validation sample cap; 0 means full CARLA validation split.")
    parser.add_argument("--skip_inference", action="store_true", help="Only analyze existing debug maps.")
    parser.add_argument("--skip_ap", action="store_true", default=True, help="Skip AP during analysis inference; enabled by default.")
    parser.add_argument("--run_ap", dest="skip_ap", action="store_false", help="Run AP evaluation as part of analysis inference.")
    parser.add_argument("--force", action="store_true", help="Replace the analysis run directory before generating maps.")
    parser.add_argument("--checkpoint_link_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--allow_untrained_request_head", action="store_true", help="Debug only; normally this should stay false.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "WARN" if args.log_level == "WARNING" else args.log_level
    logger = get_logger("LearnedRequestHeadAnalysis", level="DEBUG" if args.debug else log_level, debug=args.debug)

    checkpoint_dir = _resolve(args.checkpoint_dir)
    validate_dir = _resolve(args.validate_dir)
    root_dir = _resolve(args.root_dir) if args.root_dir else None
    run_dir = _resolve(args.analysis_run_dir)
    output_csv = _resolve(args.output_csv)
    figures_dir = _resolve(args.figures_dir)
    debug_dir = _resolve(args.debug_npz_dir) if args.debug_npz_dir else run_dir / DEFAULT_DEBUG_SUBDIR

    try:
        if not args.skip_inference:
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            if not validate_dir.exists():
                raise FileNotFoundError(f"CARLA validation directory not found: {validate_dir}")
            _prepare_analysis_run(
                checkpoint_dir=checkpoint_dir,
                validate_dir=validate_dir,
                root_dir=root_dir,
                run_dir=run_dir,
                debug_dir=debug_dir,
                max_samples=args.max_samples,
                force=args.force,
                checkpoint_link_mode=args.checkpoint_link_mode,
                logger=logger,
            )
            _run_inference(
                run_dir=run_dir,
                validate_dir=validate_dir,
                python_bin=args.python_bin,
                max_samples=args.max_samples,
                skip_ap=args.skip_ap,
                deterministic=True,
                allow_untrained_request_head=args.allow_untrained_request_head,
                logger=logger,
            )

        summary, mean_map, std_map, _per_file = analyze_debug_maps(debug_dir, logger)
        summary.update({
            "checkpoint_dir": str(checkpoint_dir),
            "validate_dir": str(validate_dir),
            "analysis_run_dir": str(run_dir),
            "budget_percent": 10,
            "budget_ratio": 0.10,
        })
        _write_statistics_csv(output_csv, summary)
        generate_figures(figures_dir, mean_map, std_map, logger)
        logger.success("Learned request-head analysis completed", csv=output_csv, figures_dir=figures_dir)
        return 0
    except Exception as exc:
        logger.error("Learned request-head analysis failed", error=str(exc))
        if args.debug:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
