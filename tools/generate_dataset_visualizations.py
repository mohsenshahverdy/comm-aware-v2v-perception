#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate thesis-quality dataset visualizations from available result artifacts.

The script uses verified aggregate result CSV/YAML files by default. If per-frame
``danger_eval_boxes/frame_*.npz`` exports are supplied, it additionally computes
object geometry, distance, and per-frame count distributions from real GT boxes.
No synthetic dataset statistics are generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for dataset visualizations") from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LogNorm
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for dataset visualizations") from exc

def _find_repo_root(start: Path) -> Path:
    for parent in [start.parent, *start.parents]:
        if (parent / "src").exists() and (parent / "Classical_Format_Thesis").exists():
            return parent
    return start.resolve().parents[1]


ROOT = _find_repo_root(Path(__file__).resolve())
THESIS = ROOT / "Classical_Format_Thesis"
RESULTS = ROOT / "results"
OUT = THESIS / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f5f8b"
BLUE2 = "#4f8db3"
LIGHT_BLUE = "#dcebf4"
DARK = "#1f2933"
GRAY = "#667085"
MID_GRAY = "#98a2b3"
LIGHT_GRAY = "#f2f4f7"
GREEN = "#4f7f6a"
LIGHT_GREEN = "#e8f1e8"
ORANGE = "#b97832"
LIGHT_ORANGE = "#f4e6d3"
RED = "#a84d4d"
LIGHT_RED = "#f4dddd"
PURPLE = "#6f65a8"
LIGHT_PURPLE = "#e8e4f3"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 12,
    "axes.labelsize": 9.5,
    "figure.dpi": 180,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


@dataclass
class DatasetStats:
    key: str
    label: str
    frames: int
    skipped_frames: int
    trajectory_relevant_objects: int
    critical_objects: int
    future_pose: int
    constant_velocity: int
    danger_objects: Optional[int] = None
    sequences: Optional[int] = None
    scenarios: Optional[int] = None


@dataclass
class BoxStats:
    centers: np.ndarray
    distances: np.ndarray
    per_frame_counts: np.ndarray
    frames: int


def _read_csv_first(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _source_counts(raw: str) -> Dict[str, int]:
    try:
        return {str(k): int(v) for k, v in json.loads(raw).items()}
    except Exception:
        return {}


def load_dataset_stats() -> Dict[str, DatasetStats]:
    specs = {
        "carla": {
            "label": "CARLA 2021-only",
            "traj": RESULTS / "v2v_trajectory_carla_metrics.csv",
            "danger": RESULTS / "v2v_danger_carla_metrics.csv",
        },
        "culver": {
            "label": "Culver City",
            "traj": RESULTS / "v2v_trajectory_culver_metrics.csv",
            "danger": RESULTS / "v2v_danger_culver_metrics.csv",
        },
    }
    out: Dict[str, DatasetStats] = {}
    for key, spec in specs.items():
        row = _read_csv_first(spec["traj"])
        counts = _source_counts(row.get("trajectory_source_counts", "{}"))
        danger_objects = None
        if spec["danger"].exists():
            danger_row = _read_csv_first(spec["danger"])
            danger_objects = _safe_int(danger_row.get("danger_objects"), 0)
        out[key] = DatasetStats(
            key=key,
            label=spec["label"],
            frames=_safe_int(row.get("frames")),
            skipped_frames=_safe_int(row.get("skipped_frames")),
            trajectory_relevant_objects=_safe_int(row.get("trajectory_relevant_objects")),
            critical_objects=_safe_int(row.get("critical_objects")),
            future_pose=int(counts.get("future_pose", 0)),
            constant_velocity=int(counts.get("constant_velocity", 0)),
            danger_objects=danger_objects,
        )
    return out


def _save(fig: Any, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=240)
    plt.close(fig)


def _card(ax: Any, x: float, y: float, w: float, h: float, title: str, lines: Sequence[str], fc: str, ec: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.15, y + h - 0.22, title, ha="left", va="top", fontsize=12, weight="bold", color=DARK)
    for i, line in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.58 - i * 0.31, line, ha="left", va="top", fontsize=9.2, color=DARK)


def dataset_overview_figure(stats: Dict[str, DatasetStats]) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.text(0.2, 4.45, "Dataset split overview", fontsize=15, weight="bold", color=DARK, va="center")
    ax.text(0.2, 4.10, "Statistics are read from the verified evaluation result files.", fontsize=9.0, color=GRAY)
    configs = [("carla", 0.35, LIGHT_BLUE, BLUE), ("culver", 5.45, LIGHT_GREEN, GREEN)]
    for key, x, fc, ec in configs:
        s = stats[key]
        source_total = s.future_pose + s.constant_velocity
        future_pct = 100.0 * s.future_pose / source_total if source_total else 0.0
        lines = [
            f"Evaluated frames: {s.frames:,}",
            f"Static danger objects: {s.danger_objects:,}" if s.danger_objects is not None else "Static danger objects: not available",
            f"Trajectory-relevant objects: {s.trajectory_relevant_objects:,}",
            f"Critical trajectory objects: {s.critical_objects:,}",
            f"Trajectory source: {future_pct:.2f}% future-pose frames",
        ]
        _card(ax, x, 0.55, 4.65, 3.25, s.label, lines, fc, ec)
    _save(fig, "dataset_split_overview")


def trajectory_source_figure(stats: Dict[str, DatasetStats]) -> None:
    labels = [stats["carla"].label, stats["culver"].label]
    future = [stats["carla"].future_pose, stats["culver"].future_pose]
    fallback = [stats["carla"].constant_velocity, stats["culver"].constant_velocity]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x, future, 0.55, label="future pose", color=BLUE, edgecolor=BLUE)
    ax.bar(x, fallback, 0.55, bottom=future, label="constant-velocity fallback", color=ORANGE, edgecolor=ORANGE)
    ymax = max([f + b for f, b in zip(future, fallback)] or [1])
    for i, (fut, fb) in enumerate(zip(future, fallback)):
        total = fut + fb
        pct = 100 * fut / total if total else 0.0
        ax.text(x[i], total + ymax * 0.065, f"{pct:.2f}% future pose", ha="center", va="bottom", fontsize=8.0, color=DARK)
        ax.text(x[i], fut / 2, f"{fut:,}", ha="center", va="center", color="white", weight="bold", fontsize=8.5)
        # The fallback segment is very small; place its count just beside the
        # orange cap to keep the label readable without colliding with the
        # percentage annotation above the bar.
        ax.text(x[i] + 0.34, fut + max(fb / 2, ymax * 0.006), f"{fb} fallback",
                ha="left", va="center", color=DARK, fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Evaluated frames")
    ax.set_ylim(0, ymax * 1.16)
    ax.set_title("Trajectory-source availability", weight="bold", color=DARK)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    _save(fig, "dataset_trajectory_source_counts")


def safety_object_count_figure(stats: Dict[str, DatasetStats]) -> None:
    labels = [stats["carla"].label, stats["culver"].label]
    danger = [stats[k].danger_objects or 0 for k in ["carla", "culver"]]
    traj = [stats[k].trajectory_relevant_objects for k in ["carla", "culver"]]
    critical = [stats[k].critical_objects for k in ["carla", "culver"]]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.7, 4.4))
    groups = [
        ax.bar(x - width, danger, width, label="static danger objects", color=LIGHT_RED, edgecolor=RED),
        ax.bar(x, traj, width, label="trajectory-relevant objects", color=LIGHT_BLUE, edgecolor=BLUE),
        ax.bar(x + width, critical, width, label="critical trajectory objects", color=LIGHT_PURPLE, edgecolor=PURPLE),
    ]
    ymax = max(danger + traj + critical) if danger + traj + critical else 1
    for group in groups:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.012, f"{int(bar.get_height()):,}", ha="center", fontsize=7.8, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Object instances")
    ax.set_title("Object populations used by safety-oriented metrics", weight="bold", color=DARK)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.1)
    _save(fig, "dataset_safety_object_populations")


def _load_boxes_to_bev(boxes: Any) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] >= 8 and arr.shape[2] >= 3:
        # Server NPZ exports are usually (N, 8, 3). Collapse top/bottom corners
        # to an axis-aligned BEV footprint robustly for distribution plots.
        xy = arr[:, :8, :2]
        mins = np.min(xy, axis=1)
        maxs = np.max(xy, axis=1)
        return np.stack([
            np.stack([mins[:, 0], mins[:, 1]], axis=1),
            np.stack([maxs[:, 0], mins[:, 1]], axis=1),
            np.stack([maxs[:, 0], maxs[:, 1]], axis=1),
            np.stack([mins[:, 0], maxs[:, 1]], axis=1),
        ], axis=1).astype(np.float32)
    if arr.ndim == 3 and arr.shape[1] >= 4 and arr.shape[2] >= 2:
        return arr[:, :4, :2].astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 8:
        return arr.reshape(-1, 4, 2).astype(np.float32)
    raise ValueError(f"Unsupported GT box shape for dataset visualization: {arr.shape}")


def load_box_stats(box_dir: Optional[Path]) -> Optional[BoxStats]:
    if box_dir is None:
        return None
    if not box_dir.exists():
        print(f"[WARN] box directory not found, skipping geometry figures: {box_dir}")
        return None
    paths = sorted(box_dir.glob("frame_*.npz"))
    if not paths:
        print(f"[WARN] no frame_*.npz files in {box_dir}; skipping geometry figures")
        return None
    centers: List[np.ndarray] = []
    counts: List[int] = []
    for path in paths:
        try:
            data = np.load(path, allow_pickle=True)
            if "gt_boxes" not in data:
                counts.append(0)
                continue
            boxes = _load_boxes_to_bev(data["gt_boxes"])
            counts.append(int(boxes.shape[0]))
            if boxes.shape[0]:
                centers.append(boxes.mean(axis=1))
        except Exception as exc:
            print(f"[WARN] failed reading {path}: {exc}")
    if not centers:
        return BoxStats(np.zeros((0, 2)), np.zeros((0,)), np.asarray(counts), len(paths))
    all_centers = np.concatenate(centers, axis=0)
    distances = np.linalg.norm(all_centers, axis=1)
    return BoxStats(all_centers, distances, np.asarray(counts, dtype=np.int32), len(paths))


def geometry_heatmap(stats_by_dataset: Dict[str, BoxStats]) -> bool:
    if not stats_by_dataset:
        return False
    xlim = (-80.0, 80.0)
    ylim = (-50.0, 50.0)
    bins = (72, 50)
    histograms: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    positive_values: List[np.ndarray] = []
    for key, bs in stats_by_dataset.items():
        if bs.centers.size == 0:
            continue
        hist, xedges, yedges = np.histogram2d(
            bs.centers[:, 0],
            bs.centers[:, 1],
            bins=bins,
            range=[list(xlim), list(ylim)],
        )
        histograms[key] = (hist, xedges, yedges)
        positive = hist[hist > 0]
        if positive.size:
            positive_values.append(positive)

    if positive_values:
        all_positive = np.concatenate(positive_values)
        vmax = max(1.0, float(np.percentile(all_positive, 98)))
        norm = LogNorm(vmin=1.0, vmax=vmax)
    else:
        norm = None

    fig, axes = plt.subplots(1, len(stats_by_dataset), figsize=(5.9 * len(stats_by_dataset), 4.8), squeeze=False)
    mesh = None
    for ax, (key, bs) in zip(axes[0], stats_by_dataset.items()):
        if bs.centers.size == 0:
            ax.text(0.5, 0.5, "No GT centers available", ha="center", va="center", transform=ax.transAxes)
        else:
            hist, xedges, yedges = histograms[key]
            masked = np.ma.masked_where(hist.T <= 0, hist.T)
            mesh = ax.pcolormesh(xedges, yedges, masked, cmap="Blues", norm=norm, shading="auto")
            ax.scatter([0], [0], marker="^", color=GREEN, s=76, edgecolor="white", linewidth=0.8, zorder=5, label="ego")
        ax.set_title(f"{key.upper()} ego-frame object density", weight="bold", color=DARK)
        ax.set_xlabel("x in ego frame [m]")
        if ax is axes[0][0]:
            ax.set_ylabel("y in ego frame [m]")
        else:
            ax.set_ylabel("")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=list(axes[0]), fraction=0.026, pad=0.025)
        cbar.set_label("GT object count per BEV bin (log scale)", fontsize=8.2)
        cbar.ax.tick_params(labelsize=7.5)
    _save(fig, "dataset_object_density_xy")
    return True


def distance_histogram(stats_by_dataset: Dict[str, BoxStats]) -> bool:
    if not stats_by_dataset:
        return False
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bins = np.linspace(0, 100, 41)
    for key, bs in stats_by_dataset.items():
        if bs.distances.size:
            ax.hist(bs.distances, bins=bins, histtype="step", linewidth=2.0, label=f"{key.upper()} ({bs.distances.size:,} objects)")
    ax.set_xlabel("Object distance from ego [m]")
    ax.set_ylabel("Object count")
    ax.set_title("Object distance distribution", weight="bold", color=DARK)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    _save(fig, "dataset_object_distance_distribution")
    return True


def per_frame_count_histogram(stats_by_dataset: Dict[str, BoxStats]) -> bool:
    if not stats_by_dataset:
        return False
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    max_count = max(int(bs.per_frame_counts.max()) if bs.per_frame_counts.size else 0 for bs in stats_by_dataset.values())
    bins = np.arange(0, max_count + 2) - 0.5
    for key, bs in stats_by_dataset.items():
        if bs.per_frame_counts.size:
            ax.hist(bs.per_frame_counts, bins=bins, alpha=0.45, label=f"{key.upper()} ({bs.frames:,} frames)")
    ax.set_xlabel("GT objects per frame")
    ax.set_ylabel("Frame count")
    ax.set_title("Per-frame object-count distribution", weight="bold", color=DARK)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    _save(fig, "dataset_per_frame_object_count_distribution")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis dataset visualization figures.")
    parser.add_argument("--carla_box_dir", default=None, help="Optional CARLA danger_eval_boxes directory for geometry/distance distributions.")
    parser.add_argument("--culver_box_dir", default=None, help="Optional Culver danger_eval_boxes directory for geometry/distance distributions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = load_dataset_stats()
    dataset_overview_figure(stats)
    trajectory_source_figure(stats)
    safety_object_count_figure(stats)

    box_stats: Dict[str, BoxStats] = {}
    carla = load_box_stats(Path(args.carla_box_dir).expanduser().resolve()) if args.carla_box_dir else None
    culver = load_box_stats(Path(args.culver_box_dir).expanduser().resolve()) if args.culver_box_dir else None
    if carla is not None:
        box_stats["carla"] = carla
    if culver is not None:
        box_stats["culver"] = culver
    if box_stats:
        geometry_heatmap(box_stats)
        distance_histogram(box_stats)
        per_frame_count_histogram(box_stats)
    else:
        print("[INFO] No per-frame box directories supplied; geometry, distance, and per-frame-count figures were skipped.")
    print(f"[OK] Dataset visualization figures written to {OUT}")


if __name__ == "__main__":
    main()
