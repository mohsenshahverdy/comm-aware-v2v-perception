#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate qualitative BEV comparison scenes from saved inference outputs.

This script is intentionally server-runnable: it does not require the raw dataset,
only saved evaluation outputs such as ``danger_eval_boxes/frame_*.npz`` produced
by inference with box export enabled. It never fabricates communication masks; if
mask arrays are not present in the saved outputs, the generated report states that
only detections and missed objects are visualized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - allows --help without local numpy.
    np = None

def _find_repo_root(start: Path) -> Path:
    for parent in [start.parent, *start.parents]:
        if (parent / "src").exists() and (parent / "Classical_Format_Thesis").exists():
            return parent
    return start.resolve().parents[1]


# Allow running directly from tools/ or the thesis scripts directory without
# manually exporting PYTHONPATH, as long as the script remains inside the repo.
REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.tools.evaluate_danger_aware_metrics import bev_iou, boxes_to_numpy
except Exception:  # pragma: no cover - fallback is for standalone thesis use.
    bev_iou = None
    boxes_to_numpy = None

try:
    from src.utils.logging import get_logger
except Exception:  # pragma: no cover
    get_logger = None


class _FallbackLogger:
    def info(self, msg: str, **kw: Any) -> None:
        print(_fmt("INFO", msg, kw))

    def warn(self, msg: str, **kw: Any) -> None:
        print(_fmt("WARN", msg, kw))

    warning = warn

    def error(self, msg: str, **kw: Any) -> None:
        print(_fmt("ERROR", msg, kw))

    def success(self, msg: str, **kw: Any) -> None:
        print(_fmt("SUCCESS", msg, kw))


LOGGER = get_logger("QualitativeSceneAnalysis") if get_logger else _FallbackLogger()


def _fmt(level: str, msg: str, kw: Dict[str, Any]) -> str:
    suffix = " ".join(f"{k}={v}" for k, v in kw.items())
    return f"[{level}] {msg}" + (f" | {suffix}" if suffix else "")


METHOD_ROLE_HINTS = {
    "full": ["full", "baseline"],
    "topk": ["top", "top-k", "topk", "selective"],
    "receiver": ["receiver"],
    "temporal": ["temporal"],
    "learned": ["learned"],
}


@dataclass
class FrameData:
    key: str
    path: Path
    method: str
    pred_boxes: np.ndarray
    gt_boxes: np.ndarray
    pred_scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory: Optional[np.ndarray] = None
    collaborator_positions: Optional[np.ndarray] = None
    mask_available: bool = False


@dataclass
class CandidateRow:
    frame_key: str
    category: str
    score: float
    learned_success_risk: float
    temporal_gain_risk: float
    learned_failure_risk: float
    topk_learned_contrast_risk: float
    easy_score: float
    gt_count: int
    note: str = ""


@dataclass
class RunCollection:
    method_names: List[str]
    run_dirs: List[Path]
    frames_by_method: List[Dict[str, Path]]
    common_frame_keys: List[str]


def _as_text(value: Any, default: str = "") -> str:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
        if arr.size == 1:
            return str(arr.reshape(-1)[0].item())
    except Exception:
        pass
    return default


def _as_array(data: Any, dtype=None) -> np.ndarray:
    if dtype is None:
        dtype = np.float32
    try:
        arr = np.asarray(data, dtype=dtype)
        return arr
    except Exception:
        return np.zeros((0,), dtype=dtype)


def _metadata_from_npz(data: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in [
        "sample_idx",
        "scenario_id",
        "scenario_index",
        "timestamp",
        "timestamp_index",
        "frame_id",
        "frame_idx",
        "ego_id",
    ]:
        if key in data:
            metadata[key] = _as_text(data[key], "")
    if "metadata_json" in data:
        raw = _as_text(data["metadata_json"], "")
        metadata["metadata_json"] = raw
        try:
            metadata.update(json.loads(raw))
        except Exception:
            pass
    if "ego_lidar_pose" in data:
        metadata["ego_lidar_pose"] = _as_array(data["ego_lidar_pose"], dtype=np.float64).reshape(-1).tolist()
    return metadata


def _box_array_from_any(boxes: Any) -> np.ndarray:
    if boxes is None:
        return np.zeros((0, 4, 2), dtype=np.float32)
    if boxes_to_numpy is not None:
        try:
            return boxes_to_numpy(boxes).astype(np.float32)
        except Exception:
            pass
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] >= 4 and arr.shape[2] >= 2:
        return arr[:, :4, :2].astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 8:
        return arr[:, :8].reshape(-1, 4, 2).astype(np.float32)
    raise ValueError(f"Unsupported box shape: {arr.shape}")


def _first_existing_array(data: Any, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _load_frame(path: Path, method: str) -> FrameData:
    data = np.load(path, allow_pickle=True)
    pred_raw = _first_existing_array(data, ["pred_boxes", "pred_box", "pred_boxes_bev", "pred", "pred_boxes_lidar"])
    gt_raw = _first_existing_array(data, ["gt_boxes", "gt_box", "gt_boxes_bev", "gt", "gt_boxes_lidar"])
    score_raw = _first_existing_array(data, ["pred_scores", "scores", "score", "pred_score"])
    pred_boxes = _box_array_from_any(pred_raw)
    gt_boxes = _box_array_from_any(gt_raw)
    pred_scores = _as_array(score_raw, dtype=np.float32).reshape(-1) if score_raw is not None else np.zeros((pred_boxes.shape[0],), dtype=np.float32)
    trajectory_raw = _first_existing_array(
        data,
        ["ego_future_trajectory", "future_trajectory", "trajectory", "ego_trajectory", "future_pose_trajectory"],
    )
    trajectory = None
    if trajectory_raw is not None:
        traj = _as_array(trajectory_raw, dtype=np.float32)
        if traj.ndim == 2 and traj.shape[1] >= 2 and traj.shape[0] > 0:
            trajectory = traj[:, :2]
    collab_raw = _first_existing_array(data, ["collaborator_positions", "cav_positions", "agent_positions", "collaborator_xy"])
    collaborator_positions = None
    if collab_raw is not None:
        collab = _as_array(collab_raw, dtype=np.float32)
        if collab.ndim == 2 and collab.shape[1] >= 2:
            collaborator_positions = collab[:, :2]
    mask_available = any(k in data for k in ["request_mask", "comm_mask", "communication_mask", "mask"])
    return FrameData(
        key=path.stem,
        path=path,
        method=method,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        pred_scores=pred_scores,
        metadata=_metadata_from_npz(data),
        trajectory=trajectory,
        collaborator_positions=collaborator_positions,
        mask_available=mask_available,
    )


def _find_box_dir(run_dir: Path) -> Path:
    direct = run_dir / "danger_eval_boxes"
    if direct.exists():
        return direct
    matches = sorted(run_dir.glob("**/danger_eval_boxes"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No danger_eval_boxes directory found under {run_dir}. Expected files like danger_eval_boxes/frame_*.npz."
    )


def _collect_run_frames(run_dirs: Sequence[Path], method_names: Sequence[str]) -> RunCollection:
    frames_by_method: List[Dict[str, Path]] = []
    for run_dir, method in zip(run_dirs, method_names):
        box_dir = _find_box_dir(run_dir)
        paths = sorted(box_dir.glob("frame_*.npz"))
        if not paths:
            raise FileNotFoundError(f"No frame_*.npz files found in {box_dir} for method {method}.")
        frames_by_method.append({p.stem: p for p in paths})
        LOGGER.info("Found frame exports", method=method, frames=len(paths), box_dir=box_dir)
    common = set(frames_by_method[0].keys())
    for mapping in frames_by_method[1:]:
        common &= set(mapping.keys())
    if not common:
        raise RuntimeError("No common frame_*.npz keys across the supplied run directories.")
    common_keys = sorted(common)
    return RunCollection(list(method_names), list(run_dirs), frames_by_method, common_keys)


def _polygon_center(boxes: np.ndarray) -> np.ndarray:
    boxes = _box_array_from_any(boxes)
    if boxes.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return boxes[:, :4, :2].mean(axis=1)


def _max_iou_per_gt(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    pred = _box_array_from_any(pred_boxes)
    gt = _box_array_from_any(gt_boxes)
    if gt.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if pred.shape[0] == 0:
        return np.zeros((gt.shape[0],), dtype=np.float32)
    if bev_iou is None:
        raise RuntimeError("BEV IoU helper is unavailable. Run from the repository root or set PYTHONPATH to the repo.")
    out = np.zeros((gt.shape[0],), dtype=np.float32)
    for idx, gt_box in enumerate(gt):
        ious = bev_iou(gt_box, pred)
        out[idx] = float(np.max(ious)) if len(ious) else 0.0
    return out


def _trajectory_distance(centers: np.ndarray, trajectory: Optional[np.ndarray]) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if trajectory is None or len(trajectory) == 0:
        return np.linalg.norm(centers, axis=1).astype(np.float32)
    traj = np.asarray(trajectory, dtype=np.float32).reshape(-1, 2)
    dist = np.linalg.norm(centers[:, None, :] - traj[None, :, :], axis=2)
    return np.min(dist, axis=1).astype(np.float32)


def _risk_weights(gt_boxes: np.ndarray, trajectory: Optional[np.ndarray]) -> np.ndarray:
    centers = _polygon_center(gt_boxes)
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    d = _trajectory_distance(centers, trajectory)
    # A bounded proximity priority. This is not a thesis metric replacement; it
    # only ranks qualitative scenes when per-object risk exports are unavailable.
    return (1.0 / (1.0 + d)).astype(np.float32)


def _role_indices(method_names: Sequence[str]) -> Dict[str, int]:
    lower = [m.lower() for m in method_names]
    roles: Dict[str, int] = {}
    for role, hints in METHOD_ROLE_HINTS.items():
        if role == "receiver":
            candidates = [i for i, m in enumerate(lower) if "receiver" in m and "temporal" not in m and "learned" not in m]
        elif role == "temporal":
            candidates = [i for i, m in enumerate(lower) if "temporal" in m and "learned" not in m]
        elif role == "learned":
            candidates = [i for i, m in enumerate(lower) if "learned" in m]
        else:
            candidates = [i for i, m in enumerate(lower) if any(h in m for h in hints)]
        if candidates:
            roles[role] = candidates[0]
    # Conservative positional fallback for the expected order.
    fallback = {"full": 0, "topk": 1, "receiver": 2, "temporal": 3, "learned": 4}
    for role, idx in fallback.items():
        roles.setdefault(role, min(idx, len(method_names) - 1))
    return roles


def _frame_candidate(collection: RunCollection, frame_key: str, iou_threshold: float, roles: Dict[str, int]) -> CandidateRow:
    frames = [_load_frame(mapping[frame_key], method) for mapping, method in zip(collection.frames_by_method, collection.method_names)]
    gt = frames[roles["full"]].gt_boxes if frames[roles["full"]].gt_boxes.shape[0] else frames[0].gt_boxes
    trajectory = next((f.trajectory for f in frames if f.trajectory is not None), None)
    weights = _risk_weights(gt, trajectory)
    detected_by_method = []
    for f in frames:
        max_iou = _max_iou_per_gt(f.pred_boxes, gt)
        detected_by_method.append(max_iou >= float(iou_threshold))
    receiver = detected_by_method[roles["receiver"]]
    temporal = detected_by_method[roles["temporal"]]
    learned = detected_by_method[roles["learned"]]
    topk = detected_by_method[roles["topk"]]
    full = detected_by_method[roles["full"]]
    learned_success = float(weights[np.logical_and(learned, ~receiver)].sum())
    temporal_gain = float(weights[np.logical_and(temporal, ~receiver)].sum())
    learned_failure = float(weights[np.logical_and(~learned, np.logical_or(full, topk))].sum())
    topk_vs_learned = float(abs(weights[np.logical_and(learned, ~topk)].sum() - weights[np.logical_and(topk, ~learned)].sum()))
    easy_score = float(sum(int(mask.all()) for mask in detected_by_method)) if gt.shape[0] > 0 else 0.0
    score = max(learned_success, temporal_gain, learned_failure, topk_vs_learned, easy_score * 0.01)
    category = "ranked_candidate"
    if learned_failure > 0 and learned_failure >= max(learned_success, temporal_gain):
        category = "learned_temporal_failure"
    elif learned_success > 0 and learned_success >= max(temporal_gain, topk_vs_learned):
        category = "learned_temporal_success"
    elif temporal_gain > 0:
        category = "temporal_cache_benefit"
    elif topk_vs_learned > 0:
        category = "topk_vs_learned_contrast"
    elif easy_score > 0:
        category = "easy_case"
    return CandidateRow(
        frame_key=frame_key,
        category=category,
        score=score,
        learned_success_risk=learned_success,
        temporal_gain_risk=temporal_gain,
        learned_failure_risk=learned_failure,
        topk_learned_contrast_risk=topk_vs_learned,
        easy_score=easy_score,
        gt_count=int(gt.shape[0]),
        note="approximate ranking from NPZ boxes" + (" and trajectory" if trajectory is not None else " and ego-distance proxy"),
    )


def _load_metric_level_candidates(run_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    """Best-effort loader for future per-object risk exports.

    The current repository metric outputs are aggregate CSV/YAML files. If future
    server runs save per-object CSVs, this function recognizes common column
    names and lets those rows influence qualitative candidate ranking.
    """
    rows: List[Dict[str, Any]] = []
    patterns = ["*per*object*.csv", "*trajectory*object*.csv", "*risk*object*.csv"]
    for run_dir in run_dirs:
        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(run_dir.glob(f"**/{pattern}"))
        for path in sorted(set(paths)):
            try:
                with path.open("r", newline="") as f:
                    for row in csv.DictReader(f):
                        frame = row.get("frame_key") or row.get("frame") or row.get("frame_id") or row.get("sample_idx")
                        risk = row.get("missed_trajectory_risk") or row.get("trajectory_risk") or row.get("risk")
                        if frame is not None and risk is not None:
                            rows.append({"frame_key": str(frame), "risk": float(risk), "source": str(path)})
            except Exception as exc:
                LOGGER.warn("Could not parse per-object risk CSV", path=path, error=str(exc))
    if rows:
        LOGGER.info("Loaded metric-level candidate rows", rows=len(rows))
    return rows


def _select_candidates(
    collection: RunCollection,
    candidate_mode: str,
    max_scenes: int,
    iou_threshold: float,
    manual_frames: Sequence[str],
) -> Tuple[List[CandidateRow], List[CandidateRow]]:
    roles = _role_indices(collection.method_names)
    LOGGER.info("Resolved method roles", **roles)
    if candidate_mode == "manual":
        if not manual_frames:
            raise ValueError("--candidate_mode manual requires --manual_frames.")
        selected = []
        for raw in manual_frames:
            key = raw if raw.startswith("frame_") else f"frame_{int(raw):06d}" if raw.isdigit() else raw
            if key not in collection.common_frame_keys:
                raise ValueError(f"Manual frame {raw!r} resolved to {key!r}, which is not common to all runs.")
            selected.append(_frame_candidate(collection, key, iou_threshold, roles))
        return selected[:max_scenes], selected

    metric_rows = _load_metric_level_candidates(collection.run_dirs)
    ranked = [_frame_candidate(collection, key, iou_threshold, roles) for key in collection.common_frame_keys]
    metric_bonus: Dict[str, float] = {}
    for row in metric_rows:
        raw_key = str(row["frame_key"])
        key = raw_key if raw_key.startswith("frame_") else f"frame_{int(raw_key):06d}" if raw_key.isdigit() else raw_key
        metric_bonus[key] = metric_bonus.get(key, 0.0) + float(row.get("risk", 0.0))
    for row in ranked:
        if row.frame_key in metric_bonus:
            row.score += metric_bonus[row.frame_key]
            row.note = "includes metric-level per-object risk export"

    mode_to_category = {
        "learned_success": "learned_temporal_success",
        "failure_case": "learned_temporal_failure",
        "top_missed_risk": None,
        "easy_case": "easy_case",
    }
    if candidate_mode in mode_to_category and mode_to_category[candidate_mode] is not None:
        ranked_for_mode = [r for r in ranked if r.category == mode_to_category[candidate_mode]]
    elif candidate_mode == "top_missed_risk":
        ranked_for_mode = ranked
    else:
        ranked_for_mode = ranked
    ranked_for_mode = sorted(ranked_for_mode, key=lambda r: r.score, reverse=True)

    if candidate_mode == "auto":
        desired = [
            "learned_temporal_success",
            "topk_vs_learned_contrast",
            "temporal_cache_benefit",
            "learned_temporal_failure",
            "easy_case",
        ]
        selected: List[CandidateRow] = []
        used = set()
        for category in desired:
            choices = sorted([r for r in ranked if r.category == category and r.frame_key not in used], key=lambda r: r.score, reverse=True)
            if choices:
                selected.append(choices[0])
                used.add(choices[0].frame_key)
            if len(selected) >= max_scenes:
                break
        for row in sorted(ranked, key=lambda r: r.score, reverse=True):
            if len(selected) >= max_scenes:
                break
            if row.frame_key not in used:
                selected.append(row)
                used.add(row.frame_key)
    else:
        selected = ranked_for_mode[:max_scenes]

    return selected, sorted(ranked, key=lambda r: r.score, reverse=True)


def _write_candidates_csv(path: Path, rows: Sequence[CandidateRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_key",
        "category",
        "score",
        "learned_success_risk",
        "temporal_gain_risk",
        "learned_failure_risk",
        "topk_learned_contrast_risk",
        "easy_score",
        "gt_count",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def _plot_box(ax: Any, box: np.ndarray, color: str, linewidth: float, linestyle: str = "-", alpha: float = 1.0, fill: bool = False) -> None:
    pts = np.asarray(box, dtype=np.float32)[:4, :2]
    closed = np.vstack([pts, pts[0]])
    if fill:
        ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.18, linewidth=0)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def _axis_limits(frames: Sequence[FrameData]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts: List[np.ndarray] = []
    for f in frames:
        for boxes in [f.gt_boxes, f.pred_boxes]:
            boxes = _box_array_from_any(boxes)
            if boxes.shape[0]:
                pts.append(boxes.reshape(-1, 2))
        if f.trajectory is not None:
            pts.append(np.asarray(f.trajectory, dtype=np.float32).reshape(-1, 2))
        if f.collaborator_positions is not None:
            pts.append(np.asarray(f.collaborator_positions, dtype=np.float32).reshape(-1, 2))
    if not pts:
        return (-50.0, 70.0), (-40.0, 40.0)
    arr = np.concatenate(pts, axis=0)
    xmin, ymin = np.nanmin(arr, axis=0)
    xmax, ymax = np.nanmax(arr, axis=0)
    pad_x = max(10.0, 0.15 * float(xmax - xmin + 1e-6))
    pad_y = max(8.0, 0.15 * float(ymax - ymin + 1e-6))
    return (float(xmin - pad_x), float(xmax + pad_x)), (float(ymin - pad_y), float(ymax + pad_y))


def _generate_scene_figure(
    collection: RunCollection,
    candidate: CandidateRow,
    dataset_name: str,
    output_dir: Path,
    iou_threshold: float,
) -> Tuple[Path, Path, bool]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to generate qualitative figures on the server.") from exc

    frames = [_load_frame(mapping[candidate.frame_key], method) for mapping, method in zip(collection.frames_by_method, collection.method_names)]
    gt = frames[0].gt_boxes if frames[0].gt_boxes.shape[0] else next((f.gt_boxes for f in frames if f.gt_boxes.shape[0]), frames[0].gt_boxes)
    xlim, ylim = _axis_limits(frames)
    mask_available = any(f.mask_available for f in frames)

    fig, axes = plt.subplots(1, len(frames), figsize=(4.1 * len(frames), 4.4), sharex=True, sharey=True)
    if len(frames) == 1:
        axes = [axes]
    for ax, frame in zip(axes, frames):
        detected = _max_iou_per_gt(frame.pred_boxes, gt) >= float(iou_threshold)
        missed = ~detected
        for box in gt:
            _plot_box(ax, box, color="#303030", linewidth=1.0, linestyle="-", alpha=0.55)
        for box in gt[missed]:
            _plot_box(ax, box, color="#C00000", linewidth=1.8, linestyle="-", alpha=0.95, fill=True)
        for box in frame.pred_boxes:
            _plot_box(ax, box, color="#0072B2", linewidth=1.3, linestyle="--", alpha=0.9)
        ax.scatter([0.0], [0.0], marker="^", s=55, color="#009E73", label="Ego", zorder=5)
        if frame.trajectory is not None and len(frame.trajectory) > 0:
            traj = np.asarray(frame.trajectory, dtype=np.float32)
            ax.plot(traj[:, 0], traj[:, 1], color="#E69F00", linewidth=2.0, marker=".", markersize=3, label="Ego trajectory")
        if frame.collaborator_positions is not None and len(frame.collaborator_positions) > 0:
            collab = np.asarray(frame.collaborator_positions, dtype=np.float32)
            ax.scatter(collab[:, 0], collab[:, 1], marker="s", s=32, facecolor="none", edgecolor="#6A6A6A", label="Collaborators")
        ax.set_title(f"{frame.method}\nmissed GT: {int(missed.sum())}/{len(gt)}", fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.8)
        ax.set_xlabel("x in ego frame [m]")
    axes[0].set_ylabel("y in ego frame [m]")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), frameon=False, fontsize=9)
    note = "Sparse mask overlays were not available; detection outputs and missed GT objects are compared."
    if mask_available:
        note = "Communication mask arrays were present in NPZ files, but this qualitative figure overlays detections only."
    fig.suptitle(
        f"{dataset_name.upper()} qualitative scene {candidate.frame_key} ({candidate.category}, IoU={iou_threshold:.2f})",
        fontsize=12,
        y=0.98,
    )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.93])
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = candidate.frame_key.replace("frame_", "frame")
    pdf_path = output_dir / f"{dataset_name}_{scene_id}.pdf"
    png_path = output_dir / f"{dataset_name}_{scene_id}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path, mask_available


def _write_report(
    path: Path,
    dataset_name: str,
    selected: Sequence[CandidateRow],
    collection: RunCollection,
    figure_paths: Sequence[Tuple[Path, Path]],
    mask_any: bool,
    candidate_mode: str,
) -> None:
    lines = [
        f"# Qualitative Scene Report: {dataset_name}",
        "",
        "This report was generated from saved `danger_eval_boxes/frame_*.npz` evaluation outputs.",
        "No communication masks are fabricated. If mask arrays are absent, figures compare detections and missed ground-truth boxes only.",
        "",
        "## Inputs",
    ]
    for method, run_dir in zip(collection.method_names, collection.run_dirs):
        lines.append(f"- {method}: `{run_dir}`")
    lines.extend([
        "",
        f"Candidate mode: `{candidate_mode}`",
        f"Common frames across methods: {len(collection.common_frame_keys)}",
        f"Sparse mask arrays available in selected figures: {'yes' if mask_any else 'no'}",
        "",
        "## Selected Scenes",
        "",
        "| Scene | Category | Score | GT boxes | PDF | PNG |",
        "|---|---:|---:|---:|---|---|",
    ])
    for row, (pdf, png) in zip(selected, figure_paths):
        lines.append(f"| `{row.frame_key}` | {row.category} | {row.score:.4f} | {row.gt_count} | `{pdf}` | `{png}` |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Candidate scores are for qualitative ranking only and are not thesis evaluation metrics.",
        "- When per-object trajectory-risk exports are unavailable, ranking uses BEV IoU matches and proximity to the ego trajectory if present, otherwise proximity to the ego vehicle.",
        "- The same BEV axis limits are used across the five method subplots within each figure.",
    ])
    path.write_text("\n".join(lines) + "\n")


def _write_dynamic_latex_snippet(path: Path, dataset_name: str, selected: Sequence[CandidateRow]) -> None:
    lines = [
        "% Auto-generated qualitative scene figure snippet.",
        "% Review the generated figures before uncommenting in the thesis.",
        "% Do not include every generated scene; select only the clearest examples.",
        "",
    ]
    for row in selected:
        scene_id = row.frame_key.replace("frame_", "frame")
        fig_path = f"figures/qualitative/{dataset_name}_{scene_id}.pdf"
        label = f"fig:qualitative-{dataset_name}-{scene_id}"
        lines.extend([
            "%\\begin{figure}[t]",
            "%    \\centering",
            f"%    \\includegraphics[width=\\textwidth]{{{fig_path}}}",
            "%    \\caption{Qualitative BEV comparison for a selected scene. Ground-truth boxes, predicted boxes, and missed ground-truth objects are visualized for the compared communication policies. Sparse mask overlays are included only if saved by the evaluation pipeline.}",
            f"%    \\label{{{label}}}",
            "%\\end{figure}",
            "",
        ])
    path.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative BEV comparison scenes from saved evaluation outputs.")
    parser.add_argument("--dataset_name", choices=["carla", "culver"], required=True)
    parser.add_argument("--run_dirs", nargs=5, required=True, help="Five run directories in method comparison order.")
    parser.add_argument("--method_names", nargs=5, required=True, help="Five display names, e.g. Full Top-K Receiver Temporal Learned.")
    parser.add_argument("--output_dir", required=True, help="Output directory for figures and reports.")
    parser.add_argument(
        "--candidate_mode",
        default="auto",
        choices=["auto", "manual", "top_missed_risk", "learned_success", "failure_case", "easy_case"],
    )
    parser.add_argument("--max_scenes", type=int, default=5)
    parser.add_argument("--iou_threshold", type=float, default=0.7)
    parser.add_argument("--manual_frames", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if np is None:
        raise RuntimeError("numpy is required to run qualitative scene analysis. Install numpy in the evaluation environment.")
    run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    collection = _collect_run_frames(run_dirs, args.method_names)
    selected, ranked = _select_candidates(
        collection=collection,
        candidate_mode=args.candidate_mode,
        max_scenes=max(1, int(args.max_scenes)),
        iou_threshold=float(args.iou_threshold),
        manual_frames=args.manual_frames,
    )
    if not selected:
        raise RuntimeError("No qualitative candidate scenes could be selected.")
    candidates_csv = output_dir / f"qualitative_scene_candidates_{args.dataset_name}.csv"
    _write_candidates_csv(candidates_csv, ranked)
    figure_paths: List[Tuple[Path, Path]] = []
    mask_available = False
    for row in selected:
        pdf, png, mask = _generate_scene_figure(collection, row, args.dataset_name, output_dir, float(args.iou_threshold))
        figure_paths.append((pdf, png))
        mask_available = mask_available or mask
        LOGGER.info("Generated qualitative figure", frame=row.frame_key, pdf=pdf, png=png)
    report_path = output_dir / f"qualitative_scene_report_{args.dataset_name}.md"
    _write_report(report_path, args.dataset_name, selected, collection, figure_paths, mask_available, args.candidate_mode)
    snippet_path = output_dir / f"qualitative_scene_figures_{args.dataset_name}.tex"
    _write_dynamic_latex_snippet(snippet_path, args.dataset_name, selected)
    LOGGER.success("Qualitative scene analysis complete", candidates=candidates_csv, report=report_path, snippet=snippet_path)


if __name__ == "__main__":
    main()
