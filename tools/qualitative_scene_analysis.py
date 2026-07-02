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
    _NUMPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - allows --help without local numpy.
    np = None
    _NUMPY_IMPORT_ERROR = exc

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
    from src.utils.logging import get_logger
except Exception:  # pragma: no cover
    get_logger = None

try:
    from src.visualization.thesis_renderer import RendererConfig, RoadCarPanelRenderer, legend_handles as road_car_legend_handles
except Exception:  # pragma: no cover - keeps --help usable in minimal envs.
    RendererConfig = None
    RoadCarPanelRenderer = None
    road_car_legend_handles = None


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

FIGURE_MODE_CHOICES = ["legacy", "receiver_progression", "baseline_comparison", "roi_detail", "all"]

GROUPED_FIGURE_ROLES = {
    "receiver_progression": ("receiver", "temporal", "learned"),
    "baseline_comparison": ("full", "topk", "learned"),
}

GROUPED_FIGURE_TITLES = {
    "receiver_progression": "Receiver progression",
    "baseline_comparison": "Baseline comparison",
    "roi_detail": "Receiver-progression ROI detail",
}

ROI_MARGIN_METERS = 8.0
ROI_NEARBY_MATCH_RADIUS_METERS = 15.0
ROI_CLUSTER_RADIUS_METERS = 26.0
ROI_CLUSTER_TRIGGER_WIDTH_METERS = 55.0
ROI_CLUSTER_TRIGGER_HEIGHT_METERS = 40.0
ROI_MIN_SIDE_METERS = 18.0
ROI_MAX_SIDE_METERS = 58.0


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
    full_missed_gt: int = 0
    topk_missed_gt: int = 0
    receiver_missed_gt: int = 0
    temporal_missed_gt: int = 0
    learned_missed_gt: int = 0
    delta_learned_vs_receiver: int = 0
    delta_learned_vs_temporal: int = 0
    delta_learned_vs_topk: int = 0
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


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] <= 1:
        return points
    pts = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in points})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64)


def _order_rectangle_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 4:
        x0, y0 = np.min(points, axis=0) if points.size else (0.0, 0.0)
        x1, y1 = np.max(points, axis=0) if points.size else (0.0, 0.0)
        points = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    hull = _convex_hull(points)
    if hull.shape[0] >= 4:
        points = hull
    if points.shape[0] != 4:
        # Real box corner exports should have four unique BEV vertices. If tiny
        # numerical differences create more hull points, keep the four farthest
        # angularly separated points around the center.
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        ordered = points[np.argsort(angles)]
        if ordered.shape[0] > 4:
            picks = np.linspace(0, ordered.shape[0] - 1, 4, dtype=int)
            ordered = ordered[picks]
        points = ordered
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)].astype(np.float32)


def _center_boxes_to_bev(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"Center box conversion expects shape (N, >=7), got {arr.shape}.")
    out = []
    for box in arr:
        x, y = float(box[0]), float(box[1])
        length = abs(float(box[3]))
        width = abs(float(box[4]))
        yaw = float(box[6])
        # Accept either radians or degrees defensively.
        if abs(yaw) > 2 * math.pi:
            yaw = math.radians(yaw)
        local = np.asarray(
            [[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]],
            dtype=np.float32,
        )
        c, s_ = math.cos(yaw), math.sin(yaw)
        rot = np.asarray([[c, -s_], [s_, c]], dtype=np.float32)
        out.append(local @ rot.T + np.asarray([x, y], dtype=np.float32))
    return np.asarray(out, dtype=np.float32).reshape(-1, 4, 2)


def boxes_to_bev_corners(boxes: Any) -> np.ndarray:
    """Convert supported 3D/BEV box exports to ``(N, 4, 2)`` BEV corners.

    Supported inputs:
    - ``(N, 8, 3)``: 3D box corners. The x/y coordinates of all eight corners
      are projected to BEV; duplicate top/bottom vertices are collapsed into a
      four-corner 2D footprint independent of top/bottom corner ordering.
    - ``(N, 4, 2)``: BEV corners, used directly.
    - ``(N, 7)`` or ``(N, >=9)``: center format interpreted as
      ``x, y, z, length, width, height, yaw, ...``.
    - ``(N, 8)``: flattened BEV corners ``x1,y1,...,x4,y4``.
    - ``(N, 24)``: flattened 3D corners reshaped to ``(N, 8, 3)``.
    """
    if np is None:
        raise RuntimeError("numpy is required for box conversion.")
    if boxes is None:
        return np.zeros((0, 4, 2), dtype=np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (4, 2):
        arr = arr[None, :, :]
    if arr.ndim == 3 and arr.shape[1] == 4 and arr.shape[2] >= 2:
        return arr[:, :4, :2].astype(np.float32)
    if arr.ndim == 3 and arr.shape[1] >= 8 and arr.shape[2] >= 3:
        bev = [_order_rectangle_points(corners[:, :2]) for corners in arr[:, :8, :3]]
        return np.asarray(bev, dtype=np.float32).reshape(-1, 4, 2)
    if arr.ndim == 2 and arr.shape[1] == 24:
        return boxes_to_bev_corners(arr.reshape(-1, 8, 3))
    if arr.ndim == 2 and arr.shape[1] == 8:
        return arr.reshape(-1, 4, 2).astype(np.float32)
    if arr.ndim == 2 and (arr.shape[1] == 7 or arr.shape[1] >= 9):
        return _center_boxes_to_bev(arr)
    raise ValueError(
        "Unsupported box shape for qualitative BEV analysis: "
        f"{arr.shape}. Supported: (N,8,3), (N,4,2), (N,7), (N,>=9), (N,8), (N,24)."
    )


def _signed_polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _polygon_area(poly: np.ndarray) -> float:
    return abs(_signed_polygon_area(poly))


def _line_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r = p2 - p1
    q = q2 - q1
    denom = float(r[0] * q[1] - r[1] * q[0])
    if abs(denom) < 1e-8:
        return p2.copy()
    qp = q1 - p1
    t = float((qp[0] * q[1] - qp[1] * q[0]) / denom)
    return p1 + t * r


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray, orientation_sign: float) -> bool:
    edge = edge_end - edge_start
    rel = point - edge_start
    cross = float(edge[0] * rel[1] - edge[1] * rel[0])
    return orientation_sign * cross >= -1e-7


def _convex_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = np.asarray(subject, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    if output.shape[0] == 0 or clip.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)
    orientation_sign = 1.0 if _signed_polygon_area(clip) >= 0 else -1.0
    for i in range(clip.shape[0]):
        edge_start = clip[i]
        edge_end = clip[(i + 1) % clip.shape[0]]
        input_list = output
        output_points = []
        if input_list.shape[0] == 0:
            break
        prev = input_list[-1]
        prev_inside = _inside(prev, edge_start, edge_end, orientation_sign)
        for curr in input_list:
            curr_inside = _inside(curr, edge_start, edge_end, orientation_sign)
            if curr_inside:
                if not prev_inside:
                    output_points.append(_line_intersection(prev, curr, edge_start, edge_end))
                output_points.append(curr)
            elif prev_inside:
                output_points.append(_line_intersection(prev, curr, edge_start, edge_end))
            prev = curr
            prev_inside = curr_inside
        output = np.asarray(output_points, dtype=np.float64) if output_points else np.zeros((0, 2), dtype=np.float64)
    return output


def _bev_iou_single_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)[:4, :2]
    boxes = boxes_to_bev_corners(boxes).astype(np.float64)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    area_a = _polygon_area(box)
    out = []
    for other in boxes:
        other = other[:4, :2]
        area_b = _polygon_area(other)
        inter_poly = _convex_clip(box, other)
        inter = _polygon_area(inter_poly)
        union = area_a + area_b - inter
        out.append(0.0 if union <= 0 else inter / union)
    return np.asarray(out, dtype=np.float32)


def _box_array_from_any(boxes: Any) -> np.ndarray:
    return boxes_to_bev_corners(boxes)

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
    out = np.zeros((gt.shape[0],), dtype=np.float32)
    for idx, gt_box in enumerate(gt):
        ious = _bev_iou_single_to_many(gt_box, pred)
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


def _normalize_frame_key(raw: Any) -> str:
    text = str(raw).strip()
    if text.startswith("frame_"):
        return text
    if text.isdigit():
        return f"frame_{int(text):06d}"
    return text


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
    full_missed = int((~full).sum())
    topk_missed = int((~topk).sum())
    receiver_missed = int((~receiver).sum())
    temporal_missed = int((~temporal).sum())
    learned_missed = int((~learned).sum())
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
        full_missed_gt=full_missed,
        topk_missed_gt=topk_missed,
        receiver_missed_gt=receiver_missed,
        temporal_missed_gt=temporal_missed,
        learned_missed_gt=learned_missed,
        delta_learned_vs_receiver=receiver_missed - learned_missed,
        delta_learned_vs_temporal=temporal_missed - learned_missed,
        delta_learned_vs_topk=topk_missed - learned_missed,
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
            key = _normalize_frame_key(raw)
            if key not in collection.common_frame_keys:
                raise ValueError(f"Manual frame {raw!r} resolved to {key!r}, which is not common to all runs.")
            selected.append(_frame_candidate(collection, key, iou_threshold, roles))
        return selected[:max_scenes], selected

    metric_rows = _load_metric_level_candidates(collection.run_dirs)
    ranked = [_frame_candidate(collection, key, iou_threshold, roles) for key in collection.common_frame_keys]
    metric_bonus: Dict[str, float] = {}
    for row in metric_rows:
        raw_key = str(row["frame_key"])
        key = _normalize_frame_key(raw_key)
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
        "full_missed_gt",
        "topk_missed_gt",
        "receiver_missed_gt",
        "temporal_missed_gt",
        "learned_missed_gt",
        "delta_learned_vs_receiver",
        "delta_learned_vs_temporal",
        "delta_learned_vs_topk",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def _plot_box(
    ax: Any,
    box: np.ndarray,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
    fill: bool = False,
    zorder: int = 2,
) -> None:
    pts = np.asarray(box, dtype=np.float32)[:4, :2]
    closed = np.vstack([pts, pts[0]])
    if fill:
        ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.20, linewidth=0, zorder=zorder - 1)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder)


def _axis_limits(frames: Sequence[FrameData]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts: List[np.ndarray] = []
    for f in frames:
        for boxes in [f.gt_boxes, f.pred_boxes]:
            boxes = _box_array_from_any(boxes)
            if boxes.shape[0]:
                pts.append(boxes.reshape(-1, 2))
    if pts:
        # Keep the qualitative panels cropped around the first/last visible
        # vehicle boxes rather than around decorative road context. The ego
        # marker is still included so it is not clipped when the first detected
        # object is far from the receiver.
        pts.append(np.asarray([[0.0, 0.0]], dtype=np.float32))
    else:
        # Fall back to optional metadata only if no boxes are available.
        for f in frames:
            if f.trajectory is not None:
                pts.append(np.asarray(f.trajectory, dtype=np.float32).reshape(-1, 2))
            if f.collaborator_positions is not None:
                pts.append(np.asarray(f.collaborator_positions, dtype=np.float32).reshape(-1, 2))
    if not pts:
        return (-50.0, 70.0), (-40.0, 40.0)
    arr = np.concatenate(pts, axis=0)
    xmin, ymin = np.nanmin(arr, axis=0)
    xmax, ymax = np.nanmax(arr, axis=0)
    pad_x = max(3.0, 0.055 * float(xmax - xmin + 1e-6))
    pad_y = max(4.5, 0.120 * float(ymax - ymin + 1e-6))
    return (float(xmin - pad_x), float(xmax + pad_x)), (float(ymin - pad_y), float(ymax + pad_y))


def _iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    pred = _box_array_from_any(pred_boxes)
    gt = _box_array_from_any(gt_boxes)
    mat = np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float32)
    for pred_idx, pred_box in enumerate(pred):
        mat[pred_idx, :] = _bev_iou_single_to_many(pred_box, gt)
    return mat


def _match_predictions_to_gt(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, float]]]:
    """Greedy one-to-one matching for visualization.

    Returns ``gt_matched``, ``pred_matched`` and the selected ``(pred, gt, iou)``
    pairs. The same routine is used for full and zoomed panels, so the visual
    semantics remain deterministic across methods.
    """
    pred = _box_array_from_any(pred_boxes)
    gt = _box_array_from_any(gt_boxes)
    gt_matched = np.zeros((gt.shape[0],), dtype=bool)
    pred_matched = np.zeros((pred.shape[0],), dtype=bool)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return gt_matched, pred_matched, []
    mat = _iou_matrix(pred, gt)
    candidates = []
    for pred_idx in range(mat.shape[0]):
        for gt_idx in range(mat.shape[1]):
            iou = float(mat[pred_idx, gt_idx])
            if iou >= float(iou_threshold):
                candidates.append((iou, pred_idx, gt_idx))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    pairs: List[Tuple[int, int, float]] = []
    for iou, pred_idx, gt_idx in candidates:
        if pred_matched[pred_idx] or gt_matched[gt_idx]:
            continue
        pred_matched[pred_idx] = True
        gt_matched[gt_idx] = True
        pairs.append((pred_idx, gt_idx, iou))
    return gt_matched, pred_matched, pairs


def _compute_method_matches(frames: Sequence[FrameData], gt_boxes: np.ndarray, iou_threshold: float) -> List[Dict[str, Any]]:
    out = []
    for frame in frames:
        gt_matched, pred_matched, pairs = _match_predictions_to_gt(frame.pred_boxes, gt_boxes, iou_threshold)
        out.append({"gt_matched": gt_matched, "pred_matched": pred_matched, "pairs": pairs})
    return out


def _roi_from_disagreement(
    gt_boxes: np.ndarray,
    frames: Sequence[FrameData],
    matches: Sequence[Dict[str, Any]],
    full_xlim: Tuple[float, float],
    full_ylim: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    gt = _box_array_from_any(gt_boxes)
    roi_boxes: List[np.ndarray] = []
    focus_boxes: List[np.ndarray] = []
    if gt.shape[0] > 0 and matches:
        detected_stack = np.stack([m["gt_matched"] for m in matches], axis=0)
        disagreement = detected_stack.any(axis=0) != detected_stack.all(axis=0)
        missed_any = ~detected_stack.all(axis=0)
        focus_mask = np.logical_or(disagreement, missed_any)
        if focus_mask.any():
            focus = gt[focus_mask]
            focus_boxes.append(focus)
            roi_boxes.append(focus)
            # Add nearby matched context, but do not let far-away matched
            # objects stretch the ROI back toward the full-scene view.
            matched_all = detected_stack.all(axis=0)
            if matched_all.any():
                focus_centers = _polygon_center(focus)
                matched_boxes = gt[matched_all]
                matched_centers = _polygon_center(matched_boxes)
                dmat = np.linalg.norm(matched_centers[:, None, :] - focus_centers[None, :, :], axis=-1)
                nearby = dmat.min(axis=1) <= ROI_NEARBY_MATCH_RADIUS_METERS
                if nearby.any():
                    roi_boxes.append(matched_boxes[nearby])
    for frame, match in zip(frames, matches):
        pred = _box_array_from_any(frame.pred_boxes)
        pred_matched = match["pred_matched"]
        if pred.shape[0] and pred_matched.shape[0] == pred.shape[0] and (~pred_matched).any():
            fp_boxes = pred[~pred_matched]
            if focus_boxes:
                focus_centers = _polygon_center(np.concatenate(focus_boxes, axis=0))
                fp_centers = _polygon_center(fp_boxes)
                dmat = np.linalg.norm(fp_centers[:, None, :] - focus_centers[None, :, :], axis=-1)
                nearby_fp = dmat.min(axis=1) <= ROI_NEARBY_MATCH_RADIUS_METERS
                if nearby_fp.any():
                    roi_boxes.append(fp_boxes[nearby_fp])
            else:
                roi_boxes.append(fp_boxes)
    if not roi_boxes and gt.shape[0] > 0:
        centers = _polygon_center(gt)
        dist = np.linalg.norm(centers, axis=1)
        roi_boxes.append(gt[[int(np.argmin(dist))]])
    if not roi_boxes:
        return full_xlim, full_ylim
    boxes = np.concatenate(roi_boxes, axis=0)
    centers = _polygon_center(boxes)
    if boxes.shape[0] > 1:
        all_pts = boxes.reshape(-1, 2)
        raw_width = float(np.nanmax(all_pts[:, 0]) - np.nanmin(all_pts[:, 0]))
        raw_height = float(np.nanmax(all_pts[:, 1]) - np.nanmin(all_pts[:, 1]))
        # If all disagreements are spread across a long road segment, a single
        # "ROI" becomes visually identical to the full scene. Choose the
        # densest local cluster instead; this keeps the lower row useful for
        # thesis figures while preserving deterministic behavior.
        if raw_width > ROI_CLUSTER_TRIGGER_WIDTH_METERS or raw_height > ROI_CLUSTER_TRIGGER_HEIGHT_METERS:
            dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            radius = ROI_CLUSTER_RADIUS_METERS
            neighborhood = dmat <= radius
            # Prefer dense clusters, then regions closer to the ego vehicle.
            scores = neighborhood.sum(axis=1).astype(np.float32) - 0.015 * np.linalg.norm(centers, axis=1)
            anchor = int(np.argmax(scores))
            keep = neighborhood[anchor]
            if keep.any():
                boxes = boxes[keep]
    pts = boxes.reshape(-1, 2)
    xmin, ymin = np.nanmin(pts, axis=0)
    xmax, ymax = np.nanmax(pts, axis=0)
    width = max(float(xmax - xmin), 10.0)
    height = max(float(ymax - ymin), 8.0)
    pad = ROI_MARGIN_METERS
    cx = 0.5 * float(xmin + xmax)
    cy = 0.5 * float(ymin + ymax)
    # Use a square BEV viewport for the ROI row. This preserves metric geometry
    # while making the bottom panels visually inspectable instead of wide/flat.
    roi_side = min(max(max(width, height) + 2.0 * pad, ROI_MIN_SIDE_METERS), ROI_MAX_SIDE_METERS)
    xlim = (max(full_xlim[0], cx - 0.5 * roi_side), min(full_xlim[1], cx + 0.5 * roi_side))
    ylim = (max(full_ylim[0], cy - 0.5 * roi_side), min(full_ylim[1], cy + 0.5 * roi_side))
    if xlim[1] - xlim[0] < 8.0 or ylim[1] - ylim[0] < 8.0:
        return full_xlim, full_ylim
    return xlim, ylim


def _takeaway_text(method_names: Sequence[str], matches: Sequence[Dict[str, Any]]) -> str:
    roles = _role_indices(method_names)
    learned = matches[roles["learned"]]["gt_matched"]
    receiver = matches[roles["receiver"]]["gt_matched"]
    temporal = matches[roles["temporal"]]["gt_matched"]
    topk = matches[roles["topk"]]["gt_matched"]
    full = matches[roles["full"]]["gt_matched"]
    if learned.size == 0:
        return "No ground-truth boxes are available for this frame."
    learned_vs_receiver_temporal = int(np.logical_and(learned, np.logical_and(~receiver, ~temporal)).sum())
    if learned_vs_receiver_temporal > 0:
        return f"Learned recovers {learned_vs_receiver_temporal} object(s) missed by Receiver and Temporal."
    learned_vs_receiver = int(np.logical_and(learned, ~receiver).sum())
    if learned_vs_receiver > 0:
        return f"Learned recovers {learned_vs_receiver} object(s) missed by Receiver."
    temporal_vs_receiver = int(np.logical_and(temporal, ~receiver).sum())
    if temporal_vs_receiver > 0:
        return f"Temporal request recovers {temporal_vs_receiver} object(s) missed by snapshot Receiver."
    learned_failure = int(np.logical_and(~learned, np.logical_or(full, np.logical_or(topk, receiver))).sum())
    if learned_failure > 0:
        return f"Learned misses {learned_failure} object(s) detected by at least one comparison method."
    if int(np.logical_and(topk, ~learned).sum()) > 0:
        return "Top-K detects at least one object missed by Learned in this frame."
    return "The selected frame shows similar detections across the compared methods."


def _draw_roi_rectangle(ax: Any, xlim: Tuple[float, float], ylim: Tuple[float, float], color: str = "#333333") -> None:
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        fill=False,
        edgecolor=color,
        linewidth=2.2,
        linestyle="-",
        zorder=10,
    )
    ax.add_patch(rect)


def _draw_scene_panel(
    ax: Any,
    frame: FrameData,
    gt_boxes: np.ndarray,
    match: Dict[str, Any],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    show_roi: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
    anchor: str = "C",
    panel_renderer: Any = None,
    ego_marker_position: Optional[np.ndarray] = None,
) -> None:
    if panel_renderer is not None:
        panel_renderer.draw_panel(
            ax,
            frame=frame,
            gt_boxes=gt_boxes,
            match=match,
            xlim=xlim,
            ylim=ylim,
            title=title,
            show_roi=show_roi,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            anchor=anchor,
            ego_marker_position=ego_marker_position,
        )
        return

    gt = _box_array_from_any(gt_boxes)
    pred = _box_array_from_any(frame.pred_boxes)
    gt_matched = match["gt_matched"]
    pred_matched = match["pred_matched"]

    for box in gt[gt_matched]:
        _plot_box(ax, box, color="#B8B8B8", linewidth=1.45, linestyle="--", alpha=0.75, zorder=1)
    for box in gt[~gt_matched]:
        _plot_box(ax, box, color="#D62728", linewidth=2.95, linestyle="-", alpha=1.0, fill=True, zorder=4)
    if pred.shape[0] and pred_matched.shape[0] == pred.shape[0]:
        for box in pred[pred_matched]:
            _plot_box(ax, box, color="#0072B2", linewidth=2.45, linestyle="-", alpha=0.98, zorder=5)
        for box in pred[~pred_matched]:
            _plot_box(ax, box, color="#E69F00", linewidth=2.45, linestyle=":", alpha=0.98, zorder=5)
    else:
        for box in pred:
            _plot_box(ax, box, color="#0072B2", linewidth=1.5, linestyle="-", alpha=0.90, zorder=5)

    if frame.trajectory is not None and len(frame.trajectory) > 0:
        traj = np.asarray(frame.trajectory, dtype=np.float32)
        ax.plot(traj[:, 0], traj[:, 1], color="#4D4D4D", linewidth=1.75, marker=".", markersize=3.0, alpha=0.70, zorder=3)
    if frame.collaborator_positions is not None and len(frame.collaborator_positions) > 0:
        collab = np.asarray(frame.collaborator_positions, dtype=np.float32)
        ax.scatter(collab[:, 0], collab[:, 1], marker="s", s=26, facecolor="none", edgecolor="#4D4D4D", linewidth=1.0, zorder=6)
    if ego_marker_position is not None:
        ego_xy = np.asarray(ego_marker_position, dtype=np.float32).reshape(2)
        if xlim[0] <= float(ego_xy[0]) <= xlim[1] and ylim[0] <= float(ego_xy[1]) <= ylim[1]:
            ax.scatter([ego_xy[0]], [ego_xy[1]], marker="^", s=92, color="#009E73", edgecolor="white", linewidth=0.7, zorder=8)
    if show_roi is not None:
        _draw_roi_rectangle(ax, show_roi[0], show_roi[1])

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor(anchor)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=4)
    ax.grid(True, color="#E6E6E6", linewidth=0.45, alpha=0.85)
    ax.tick_params(axis="both", labelsize=10, length=3.0, pad=1.2)
    if show_xlabel:
        ax.set_xlabel("x [m]", fontsize=11)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("y [m]", fontsize=11)
    else:
        ax.set_ylabel("")


def _add_publication_legend(fig: Any, render_style: str = "classic") -> None:
    from matplotlib.lines import Line2D

    if render_style == "road_cars" and road_car_legend_handles is not None and RendererConfig is not None:
        handles = road_car_legend_handles(RendererConfig())
    else:
        handles = [
            Line2D([0], [0], color="#B8B8B8", linewidth=1.5, linestyle="--", label="matched GT"),
            Line2D([0], [0], color="#0072B2", linewidth=2.4, linestyle="-", label="true-positive prediction"),
            Line2D([0], [0], color="#E69F00", linewidth=2.4, linestyle=":", label="false-positive prediction"),
            Line2D([0], [0], color="#D62728", linewidth=2.8, linestyle="-", label="missed GT"),
            Line2D([0], [0], color="#009E73", marker="^", linestyle="None", markersize=8, label="ego vehicle"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10.6, bbox_to_anchor=(0.5, 0.012))


def _common_blue_ego_marker(gt_boxes: np.ndarray, matches: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Pick a GT center that is matched in every displayed method.

    This keeps the green ego marker visually attached to a car that is blue
    across all panels, avoiding the misleading impression that ego is a missed
    object. The marker is a visual anchor only; it does not alter matching.
    """
    gt = _box_array_from_any(gt_boxes)
    if gt.shape[0] == 0 or not matches:
        return None
    common: Optional[np.ndarray] = None
    for match in matches:
        matched = np.asarray(match.get("gt_matched", []), dtype=bool)
        if matched.shape[0] != gt.shape[0]:
            return None
        common = matched.copy() if common is None else np.logical_and(common, matched)
    if common is None or not common.any():
        return None
    centers = _polygon_center(gt[common])
    # Prefer a stable, visually central anchor among common true positives.
    idx = int(np.argmin(np.linalg.norm(centers, axis=1)))
    return centers[idx].astype(np.float32)


def _figure_scene_id(frame_key: str) -> str:
    """Return the compact scene id convention used by existing thesis figures."""
    return frame_key.replace("frame_", "frame")


def _method_title(frame: FrameData, match: Dict[str, Any]) -> str:
    tp_count = int(match["pred_matched"].sum())
    fp_count = int((~match["pred_matched"]).sum())
    missed_count = int((~match["gt_matched"]).sum())
    return f"{frame.method}\nTP={tp_count}  FP={fp_count}  Missed={missed_count}"


def _make_panel_renderer(render_style: str) -> Any:
    if render_style == "classic":
        return None
    if render_style == "road_cars":
        if RoadCarPanelRenderer is None or RendererConfig is None:
            raise RuntimeError("road_cars render style requires src.visualization.thesis_renderer and matplotlib/numpy.")
        return RoadCarPanelRenderer(RendererConfig())
    raise ValueError(f"Unsupported render style: {render_style}")


def _legacy_layout_for_dataset(dataset_name: str) -> Dict[str, Any]:
    """Return layout constants for the legacy five-method qualitative figure.

    CARLA scenes are usually wider in x and more compressed in y, so they need
    a slightly taller canvas and more bottom margin than Culver scenes.
    """
    if dataset_name.lower() == "carla":
        return {
            "figsize": (18.0, 6.25),
            "top": 0.825,
            "bottom": 0.225,
            "title_y": 0.982,
            "takeaway_y": 0.922,
            "note_y": 0.112,
            "note_fontsize": 8.8,
            "hspace": 0.200,
        }
    return {
        "figsize": (18.0, 5.65),
        "top": 0.815,
        "bottom": 0.205,
        "title_y": 0.982,
        "takeaway_y": 0.922,
        "note_y": 0.102,
        "note_fontsize": 8.8,
        "hspace": 0.185,
    }


def _resolve_role_indices(method_names: Sequence[str], role_names: Sequence[str]) -> List[int]:
    roles = _role_indices(method_names)
    indices: List[int] = []
    for role_name in role_names:
        if role_name not in roles:
            raise ValueError(f"Could not resolve method role {role_name!r} from method names: {list(method_names)}")
        idx = int(roles[role_name])
        if idx not in indices:
            indices.append(idx)
    if len(indices) != len(role_names):
        raise ValueError(f"Resolved duplicate method indices for roles {role_names}: {indices}")
    return indices


def _load_scene_frames_and_matches(
    collection: RunCollection,
    candidate: CandidateRow,
    iou_threshold: float,
) -> Tuple[List[FrameData], np.ndarray, List[Dict[str, Any]], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], bool]:
    frames = [_load_frame(mapping[candidate.frame_key], method) for mapping, method in zip(collection.frames_by_method, collection.method_names)]
    gt = frames[0].gt_boxes if frames[0].gt_boxes.shape[0] else next((f.gt_boxes for f in frames if f.gt_boxes.shape[0]), frames[0].gt_boxes)
    full_xlim, full_ylim = _axis_limits(frames)
    matches = _compute_method_matches(frames, gt, iou_threshold)
    roi_xlim, roi_ylim = _roi_from_disagreement(gt, frames, matches, full_xlim, full_ylim)
    mask_available = any(f.mask_available for f in frames)
    return frames, gt, matches, full_xlim, full_ylim, roi_xlim, roi_ylim, mask_available


def _save_figure(fig: Any, output_dir: Path, dataset_name: str, frame_key: str, suffix: Optional[str] = None) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = _figure_scene_id(frame_key) if suffix is None else frame_key
    stem = f"{dataset_name}_{scene_id}" if not suffix else f"{dataset_name}_{scene_id}_{suffix}"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.02)
    return pdf_path, png_path


def _generate_grouped_scene_figure(
    collection: RunCollection,
    candidate: CandidateRow,
    dataset_name: str,
    output_dir: Path,
    iou_threshold: float,
    mode: str,
    custom_takeaway: Optional[str] = None,
    render_style: str = "classic",
) -> Tuple[Path, Path, bool]:
    """Generate a larger 2x3 qualitative figure for one thesis comparison group."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to generate qualitative figures on the server.") from exc

    if mode not in GROUPED_FIGURE_ROLES:
        raise ValueError(f"Unsupported grouped figure mode: {mode}")

    frames, gt, matches, full_xlim, full_ylim, roi_xlim, roi_ylim, mask_available = _load_scene_frames_and_matches(
        collection, candidate, iou_threshold
    )
    group_indices = _resolve_role_indices(collection.method_names, GROUPED_FIGURE_ROLES[mode])
    grouped_frames = [frames[idx] for idx in group_indices]
    grouped_matches = [matches[idx] for idx in group_indices]
    ego_marker_position = _common_blue_ego_marker(gt, grouped_matches)
    takeaway = custom_takeaway or _takeaway_text(collection.method_names, matches)
    panel_renderer = _make_panel_renderer(render_style)

    with plt.rc_context({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 130,
        "savefig.dpi": 320,
    }):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(17.8, 7.35),
            gridspec_kw={"height_ratios": [0.52, 1.25], "wspace": 0.055, "hspace": 0.060},
            constrained_layout=False,
        )
        for col, (frame, match) in enumerate(zip(grouped_frames, grouped_matches)):
            _draw_scene_panel(
                axes[0, col],
                frame,
                gt,
                match,
                full_xlim,
                full_ylim,
                _method_title(frame, match),
                show_roi=(roi_xlim, roi_ylim),
                show_xlabel=False,
                show_ylabel=(col == 0),
                anchor="S",
                panel_renderer=panel_renderer,
                ego_marker_position=ego_marker_position,
            )
            axes[0, col].tick_params(labelbottom=False)
            _draw_scene_panel(
                axes[1, col],
                frame,
                gt,
                match,
                roi_xlim,
                roi_ylim,
                "",
                show_roi=None,
                show_xlabel=True,
                show_ylabel=(col == 0),
                anchor="N",
                panel_renderer=panel_renderer,
                ego_marker_position=ego_marker_position,
            )
        axes[0, 0].set_ylabel("Full scene\ny [m]", fontsize=12)
        axes[1, 0].set_ylabel("ROI zoom\ny [m]", fontsize=12)
        title = (
            f"{dataset_name.upper()} | {candidate.frame_key} | "
            f"{GROUPED_FIGURE_TITLES[mode]} | IoU={iou_threshold:.2f}"
        )
        fig.suptitle(title, fontsize=17.2, fontweight="bold", y=0.982)
        fig.text(0.5, 0.925, takeaway, ha="center", va="center", fontsize=13.1, color="#333333")
        note = "Sparse mask overlays unavailable; figure compares detections and missed ground-truth objects."
        if mask_available:
            note = "Communication mask arrays were present, but this thesis figure overlays detection outcomes only."
        fig.text(0.5, 0.092, note, ha="center", va="center", fontsize=9.6, color="#555555")
        _add_publication_legend(fig, render_style=render_style)
        fig.subplots_adjust(left=0.045, right=0.997, top=0.760, bottom=0.188, wspace=0.055, hspace=0.060)

        pdf_path, png_path = _save_figure(fig, output_dir, dataset_name, candidate.frame_key, mode)
        plt.close(fig)
    return pdf_path, png_path, mask_available


def _generate_roi_detail_figure(
    collection: RunCollection,
    candidate: CandidateRow,
    dataset_name: str,
    output_dir: Path,
    iou_threshold: float,
    custom_takeaway: Optional[str] = None,
    render_style: str = "classic",
) -> Tuple[Path, Path, bool]:
    """Generate a large ROI-only receiver-progression figure for close inspection."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to generate qualitative figures on the server.") from exc

    frames, gt, matches, _full_xlim, _full_ylim, roi_xlim, roi_ylim, mask_available = _load_scene_frames_and_matches(
        collection, candidate, iou_threshold
    )
    group_indices = _resolve_role_indices(collection.method_names, GROUPED_FIGURE_ROLES["receiver_progression"])
    grouped_frames = [frames[idx] for idx in group_indices]
    grouped_matches = [matches[idx] for idx in group_indices]
    ego_marker_position = _common_blue_ego_marker(gt, grouped_matches)
    takeaway = custom_takeaway or _takeaway_text(collection.method_names, matches)
    panel_renderer = _make_panel_renderer(render_style)

    with plt.rc_context({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 130,
        "savefig.dpi": 320,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(17.8, 6.15), gridspec_kw={"wspace": 0.055}, constrained_layout=False)
        for col, (frame, match) in enumerate(zip(grouped_frames, grouped_matches)):
            _draw_scene_panel(
                axes[col],
                frame,
                gt,
                match,
                roi_xlim,
                roi_ylim,
                _method_title(frame, match),
                show_roi=None,
                show_xlabel=True,
                show_ylabel=(col == 0),
                anchor="C",
                panel_renderer=panel_renderer,
                ego_marker_position=ego_marker_position,
            )
        axes[0].set_ylabel("ROI zoom\ny [m]", fontsize=13)
        title = f"{dataset_name.upper()} | {candidate.frame_key} | ROI detail | IoU={iou_threshold:.2f}"
        fig.suptitle(title, fontsize=17.2, fontweight="bold", y=0.982)
        fig.text(0.5, 0.920, takeaway, ha="center", va="center", fontsize=13.2, color="#333333")
        fig.text(
            0.5,
            0.092,
            "Large ROI-only view for manual inspection; sparse mask overlays are not visualized.",
            ha="center",
            va="center",
            fontsize=10.3,
            color="#555555",
        )
        _add_publication_legend(fig, render_style=render_style)
        fig.subplots_adjust(left=0.050, right=0.997, top=0.770, bottom=0.185, wspace=0.055)

        pdf_path, png_path = _save_figure(fig, output_dir, dataset_name, candidate.frame_key, "roi_detail")
        plt.close(fig)
    return pdf_path, png_path, mask_available


def _generate_scene_figure(
    collection: RunCollection,
    candidate: CandidateRow,
    dataset_name: str,
    output_dir: Path,
    iou_threshold: float,
    custom_takeaway: Optional[str] = None,
    render_style: str = "classic",
) -> Tuple[Path, Path, bool]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to generate qualitative figures on the server.") from exc

    frames, gt, matches, full_xlim, full_ylim, roi_xlim, roi_ylim, mask_available = _load_scene_frames_and_matches(
        collection, candidate, iou_threshold
    )
    takeaway = custom_takeaway or _takeaway_text(collection.method_names, matches)
    panel_renderer = _make_panel_renderer(render_style)
    layout = _legacy_layout_for_dataset(dataset_name)
    ego_marker_position = _common_blue_ego_marker(gt, matches)

    with plt.rc_context({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 240,
    }):
        fig, axes = plt.subplots(
            2,
            len(frames),
            figsize=layout["figsize"],
            gridspec_kw={"height_ratios": [1.0, 1.0], "wspace": 0.10, "hspace": layout["hspace"]},
            constrained_layout=False,
        )
        for col, (frame, match) in enumerate(zip(frames, matches)):
            _draw_scene_panel(
                axes[0, col],
                frame,
                gt,
                match,
                full_xlim,
                full_ylim,
                _method_title(frame, match),
                show_roi=(roi_xlim, roi_ylim),
                show_xlabel=False,
                show_ylabel=(col == 0),
                anchor="S",
                panel_renderer=panel_renderer,
                ego_marker_position=ego_marker_position,
            )
            _draw_scene_panel(
                axes[1, col],
                frame,
                gt,
                match,
                roi_xlim,
                roi_ylim,
                "",
                show_roi=None,
                show_xlabel=True,
                show_ylabel=(col == 0),
                anchor="N",
                panel_renderer=panel_renderer,
                ego_marker_position=ego_marker_position,
            )
        axes[0, 0].set_ylabel("Full scene\ny [m]", fontsize=9)
        axes[1, 0].set_ylabel("ROI zoom\ny [m]", fontsize=9)
        title = f"{dataset_name.upper()} | {candidate.frame_key} | {candidate.category.replace('_', ' ')} | IoU={iou_threshold:.2f}"
        fig.suptitle(title, fontsize=14, fontweight="bold", y=layout["title_y"])
        fig.text(0.5, layout["takeaway_y"], takeaway, ha="center", va="center", fontsize=11, color="#333333")
        note = "Sparse mask overlays unavailable; figure compares detections and missed ground-truth objects."
        if mask_available:
            note = "Communication mask arrays were present, but this thesis figure overlays detection outcomes only."
        fig.text(0.5, layout["note_y"], note, ha="center", va="center", fontsize=layout["note_fontsize"], color="#555555")
        _add_publication_legend(fig, render_style=render_style)
        fig.subplots_adjust(left=0.045, right=0.995, top=layout["top"], bottom=layout["bottom"])

        pdf_path, png_path = _save_figure(fig, output_dir, dataset_name, candidate.frame_key, None)
        plt.close(fig)
    return pdf_path, png_path, mask_available

def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "selected"}


def _load_selected_frames_csv(path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    frames: List[str] = []
    captions: Dict[str, str] = {}
    takeaways: Dict[str, str] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Selected-frame CSV has no header: {path}")
        for row in reader:
            if "selected" in row and row.get("selected") not in {None, ""} and not _truthy(row.get("selected")):
                continue
            raw = row.get("frame_key") or row.get("frame_id") or row.get("frame")
            if not raw:
                continue
            key = _normalize_frame_key(raw)
            frames.append(key)
            if row.get("caption"):
                captions[key] = str(row["caption"]).strip()
            if row.get("takeaway"):
                takeaways[key] = str(row["takeaway"]).strip()
    return frames, captions, takeaways


def _load_curation_config(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("YAML curation config requires PyYAML. Use JSON or install pyyaml.") from exc
        cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"Curation config must be a mapping: {path}")
    frames: List[str] = []
    captions: Dict[str, str] = {}
    takeaways: Dict[str, str] = {}
    for raw in cfg.get("selected_frame_ids", []) or cfg.get("frames", []) or []:
        if isinstance(raw, dict):
            frame_id = raw.get("frame_id") or raw.get("frame_key") or raw.get("frame")
            if frame_id is None:
                continue
            key = _normalize_frame_key(frame_id)
            frames.append(key)
            if raw.get("caption"):
                captions[key] = str(raw["caption"]).strip()
            if raw.get("takeaway"):
                takeaways[key] = str(raw["takeaway"]).strip()
        else:
            frames.append(_normalize_frame_key(raw))
    for key, value in (cfg.get("captions") or {}).items():
        captions[_normalize_frame_key(key)] = str(value).strip()
    for key, value in (cfg.get("takeaways") or {}).items():
        takeaways[_normalize_frame_key(key)] = str(value).strip()
    cfg["_selected_frames"] = frames
    cfg["_captions"] = captions
    cfg["_takeaways"] = takeaways
    return cfg


def _default_caption(dataset_name: str, row: CandidateRow) -> str:
    return (
        f"Qualitative BEV comparison for {dataset_name.upper()} frame {row.frame_key}. "
        "The panels compare full communication, sender-side Top-K, snapshot receiver-request, "
        "temporal receiver-request, and learned temporal receiver-request. Matched predictions, "
        "false positives, and missed ground-truth objects are shown at the selected IoU threshold."
    )


def _write_caption_stubs(output_dir: Path, dataset_name: str, rows: Sequence[CandidateRow], captions: Dict[str, str]) -> None:
    for row in rows:
        scene_id = row.frame_key.replace("frame_", "frame")
        caption = captions.get(row.frame_key, _default_caption(dataset_name, row))
        path = output_dir / f"caption_stub_{dataset_name}_{scene_id}.tex"
        path.write_text(
            "% LaTeX caption stub generated by qualitative_scene_analysis.py.\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{fig:qualitative-{dataset_name}-{scene_id}}}\n"
        )


def _write_report(
    path: Path,
    dataset_name: str,
    selected: Sequence[CandidateRow],
    collection: RunCollection,
    figure_paths: Dict[str, List[Tuple[Path, Path]]],
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
        "| Scene | Category | Score | GT boxes | Generated files |",
        "|---|---:|---:|---:|---|",
    ])
    for row in selected:
        paths = figure_paths.get(row.frame_key, [])
        if paths:
            files = "<br>".join(f"`{pdf.name}` / `{png.name}`" for pdf, png in paths)
        else:
            files = "dry run"
        lines.append(f"| `{row.frame_key}` | {row.category} | {row.score:.4f} | {row.gt_count} | {files} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Candidate scores are for qualitative ranking only and are not thesis evaluation metrics.",
        "- When per-object trajectory-risk exports are unavailable, ranking uses BEV IoU matches and proximity to the ego trajectory if present, otherwise proximity to the ego vehicle.",
        "- The same BEV axis limits are used across the five method subplots within each figure.",
    ])
    path.write_text("\n".join(lines) + "\n")


def _write_dynamic_latex_snippet(
    path: Path,
    dataset_name: str,
    selected: Sequence[CandidateRow],
    captions: Optional[Dict[str, str]] = None,
    figure_modes: Optional[Sequence[str]] = None,
) -> None:
    lines = [
        "% Auto-generated qualitative scene figure snippet.",
        "% Review the generated figures before uncommenting in the thesis.",
        "% Do not include every generated scene; select only the clearest examples.",
        "",
    ]
    captions = captions or {}
    modes = list(figure_modes or ["legacy"])
    if "all" in modes:
        modes = ["legacy", "receiver_progression", "baseline_comparison", "roi_detail"]
    for row in selected:
        caption = captions.get(row.frame_key, _default_caption(dataset_name, row))
        for mode in modes:
            suffix = "" if mode == "legacy" else f"_{mode}"
            scene_id = _figure_scene_id(row.frame_key) if mode == "legacy" else row.frame_key
            mode_label = "legacy-five-method" if mode == "legacy" else mode.replace("_", "-")
            fig_path = f"figures/qualitative/{dataset_name}_{scene_id}{suffix}.pdf"
            label = f"fig:qualitative-{dataset_name}-{scene_id}-{mode_label}"
            lines.extend([
                "%\\begin{figure}[t]",
                "%    \\centering",
                f"%    \\includegraphics[width=\\textwidth]{{{fig_path}}}",
                f"%    \\caption{{{caption}}}",
                f"%    \\label{{{label}}}",
                "%\\end{figure}",
                "",
            ])
    path.write_text("\n".join(lines))



def _inspect_npz(path: Path) -> None:
    if np is None:
        raise RuntimeError("numpy is required for --inspect_npz.")
    if not path.exists():
        raise FileNotFoundError(f"NPZ file does not exist: {path}")
    data = np.load(path, allow_pickle=True)
    print(f"NPZ: {path}")
    print("Keys:")
    for key in sorted(data.files):
        value = data[key]
        dtype = getattr(value, "dtype", "unknown")
        shape = getattr(value, "shape", "unknown")
        print(f"  - {key}: shape={shape} dtype={dtype}")
    for key in ["pred_boxes", "gt_boxes", "pred_box", "gt_box", "pred_boxes_bev", "gt_boxes_bev"]:
        if key not in data:
            continue
        try:
            bev = boxes_to_bev_corners(data[key])
            print(f"Box conversion: {key} -> {bev.shape} OK")
        except Exception as exc:
            print(f"Box conversion: {key} FAILED: {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative BEV comparison scenes from saved evaluation outputs.")
    parser.add_argument("--inspect_npz", default=None, help="Inspect one NPZ file: print keys, shapes, and BEV box conversion status, then exit.")
    parser.add_argument("--dry_run", action="store_true", help="Rank candidate frames and write reports without generating PDF/PNG figures.")
    parser.add_argument("--dataset_name", choices=["carla", "culver"], default=None)
    parser.add_argument("--run_dirs", nargs=5, default=None, help="Five run directories in method comparison order.")
    parser.add_argument("--method_names", nargs=5, default=None, help="Five display names, e.g. Full Top-K Receiver Temporal Learned.")
    parser.add_argument("--output_dir", default=None, help="Output directory for figures and reports.")
    parser.add_argument(
        "--candidate_mode",
        default="auto",
        choices=["auto", "manual", "top_missed_risk", "learned_success", "failure_case", "easy_case"],
    )
    parser.add_argument("--max_scenes", type=int, default=5)
    parser.add_argument("--iou_threshold", type=float, default=0.7)
    parser.add_argument("--manual_frames", nargs="*", default=[])
    parser.add_argument("--selected_frames_csv", default=None, help="CSV from a previous ranking run. Uses frame_key/frame_id/frame rows; optional selected, caption, and takeaway columns.")
    parser.add_argument("--curation_config", default=None, help="YAML/JSON file with dataset_name, selected_frame_ids/frames, and optional captions/takeaways.")
    parser.add_argument(
        "--render_style",
        default="classic",
        choices=["classic", "road_cars"],
        help=(
            "Visual renderer. 'classic' preserves the original scientific box overlays. "
            "'road_cars' uses a deterministic asphalt/lane background and top-down car icons "
            "while preserving exact box geometry and TP/FP/missed semantics."
        ),
    )
    parser.add_argument(
        "--figure_modes",
        nargs="+",
        default=["legacy", "receiver_progression", "baseline_comparison"],
        choices=FIGURE_MODE_CHOICES,
        help=(
            "Qualitative figure layouts to generate. 'legacy' keeps the original five-method 2x5 figure; "
            "'receiver_progression' creates Receiver/Temporal/Learned; 'baseline_comparison' creates "
            "Full/Top-K/Learned; 'roi_detail' creates a large ROI-only receiver progression. Use 'all' "
            "to generate every mode."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if np is None:
        raise RuntimeError(
            "numpy is required to run qualitative scene analysis. "
            "Install numpy in the active Python environment or run the script with the same Python used for evaluation. "
            f"Original import error: {_NUMPY_IMPORT_ERROR!r}"
        )
    if args.inspect_npz:
        _inspect_npz(Path(args.inspect_npz).expanduser().resolve())
        return

    selected_frames: List[str] = []
    caption_overrides: Dict[str, str] = {}
    takeaway_overrides: Dict[str, str] = {}

    if args.curation_config:
        cfg = _load_curation_config(Path(args.curation_config).expanduser().resolve())
        if args.dataset_name is None and cfg.get("dataset_name"):
            args.dataset_name = str(cfg["dataset_name"]).lower()
        selected_frames.extend(cfg.get("_selected_frames", []))
        caption_overrides.update(cfg.get("_captions", {}))
        takeaway_overrides.update(cfg.get("_takeaways", {}))

    if args.selected_frames_csv:
        frames, captions, takeaways = _load_selected_frames_csv(Path(args.selected_frames_csv).expanduser().resolve())
        selected_frames.extend(frames)
        caption_overrides.update(captions)
        takeaway_overrides.update(takeaways)

    if args.manual_frames:
        selected_frames.extend(_normalize_frame_key(frame) for frame in args.manual_frames)

    # Preserve order while removing duplicates. This makes CSV/config-driven
    # manual curation deterministic and easy to edit by hand.
    selected_frames = list(dict.fromkeys(selected_frames))

    missing = [name for name in ["dataset_name", "run_dirs", "method_names", "output_dir"] if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required arguments for scene generation: {missing}. Use --inspect_npz PATH for inspection-only mode.")
    if args.dataset_name not in {"carla", "culver"}:
        raise ValueError(f"Unsupported dataset_name={args.dataset_name!r}; expected 'carla' or 'culver'.")

    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    figure_modes = list(dict.fromkeys(args.figure_modes))
    if "all" in figure_modes:
        figure_modes = ["legacy", "receiver_progression", "baseline_comparison", "roi_detail"]

    collection = _collect_run_frames(run_dirs, args.method_names)
    selection_mode = "manual" if selected_frames else args.candidate_mode
    selected, ranked = _select_candidates(
        collection=collection,
        candidate_mode=selection_mode,
        max_scenes=max(1, int(args.max_scenes)),
        iou_threshold=float(args.iou_threshold),
        manual_frames=selected_frames,
    )
    if not selected:
        raise RuntimeError("No qualitative candidate scenes could be selected.")

    candidates_csv = output_dir / f"qualitative_scene_candidates_{args.dataset_name}.csv"
    scene_summary_csv = output_dir / f"qualitative_scene_summary_{args.dataset_name}.csv"
    _write_candidates_csv(candidates_csv, ranked)
    _write_candidates_csv(scene_summary_csv, ranked)

    figure_paths: Dict[str, List[Tuple[Path, Path]]] = {}
    mask_available = False
    if args.dry_run:
        LOGGER.info("Dry run enabled; skipping PDF/PNG figure generation", selected=len(selected))
    else:
        for row in selected:
            generated_for_frame: List[Tuple[Path, Path]] = []
            for mode in figure_modes:
                if mode == "legacy":
                    pdf, png, mask = _generate_scene_figure(
                        collection,
                        row,
                        args.dataset_name,
                        output_dir,
                        float(args.iou_threshold),
                        custom_takeaway=takeaway_overrides.get(row.frame_key),
                        render_style=args.render_style,
                    )
                elif mode in GROUPED_FIGURE_ROLES:
                    pdf, png, mask = _generate_grouped_scene_figure(
                        collection,
                        row,
                        args.dataset_name,
                        output_dir,
                        float(args.iou_threshold),
                        mode,
                        custom_takeaway=takeaway_overrides.get(row.frame_key),
                        render_style=args.render_style,
                    )
                elif mode == "roi_detail":
                    pdf, png, mask = _generate_roi_detail_figure(
                        collection,
                        row,
                        args.dataset_name,
                        output_dir,
                        float(args.iou_threshold),
                        custom_takeaway=takeaway_overrides.get(row.frame_key),
                        render_style=args.render_style,
                    )
                else:
                    raise ValueError(f"Unsupported figure mode: {mode}")
                generated_for_frame.append((pdf, png))
                mask_available = mask_available or mask
                LOGGER.info("Generated qualitative figure", frame=row.frame_key, mode=mode, pdf=pdf, png=png)
            figure_paths[row.frame_key] = generated_for_frame

    report_path = output_dir / f"qualitative_scene_report_{args.dataset_name}.md"
    _write_report(report_path, args.dataset_name, selected, collection, figure_paths, mask_available, selection_mode)
    snippet_path = output_dir / f"qualitative_scene_figures_{args.dataset_name}.tex"
    _write_dynamic_latex_snippet(snippet_path, args.dataset_name, selected, caption_overrides, figure_modes)
    _write_caption_stubs(output_dir, args.dataset_name, selected, caption_overrides)
    LOGGER.success(
        "Qualitative scene analysis complete",
        candidates=candidates_csv,
        scene_summary=scene_summary_csv,
        report=report_path,
        snippet=snippet_path,
    )


if __name__ == "__main__":
    main()
