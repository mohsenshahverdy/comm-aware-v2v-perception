# -*- coding: utf-8 -*-
"""Receiver trajectory-aware danger metrics for cooperative perception runs.

The evaluator consumes ``danger_eval_boxes/frame_*.npz`` produced by
``src.tools.inference --save_box_npz``. Newer exports may include ego pose and
scenario metadata. When future ego poses are unavailable, the evaluator falls
back to constant-velocity or ego-forward trajectory approximations and records
which source was actually used.
"""

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from src.tools.evaluate_danger_aware_metrics import boxes_to_numpy, bev_iou, _safe_div, _threshold_suffix
from src.utils.logging import get_logger

LOGGER = get_logger("TrajectoryDangerEval")


@dataclass
class FrameRecord:
    pred_boxes: np.ndarray
    gt_boxes: np.ndarray
    source_path: str
    frame_idx: int
    sample_idx: int = -1
    scenario_index: int = -1
    scenario_id: str = "unknown"
    timestamp: str = ""
    frame_id: str = ""
    timestamp_index: int = -1
    ego_id: str = ""
    ego_lidar_pose: Optional[np.ndarray] = None


def _as_scalar_string(value, default="") -> str:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
        if arr.size == 1:
            return str(arr.reshape(-1)[0].item())
    except Exception:
        pass
    return str(default)


def _as_scalar_int(value, default=-1) -> int:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return int(default)
        return int(arr.reshape(-1)[0])
    except Exception:
        return int(default)


def _load_npz_frame(path: Path) -> FrameRecord:
    data = np.load(path, allow_pickle=False)
    pred = data["pred_boxes"] if "pred_boxes" in data else np.zeros((0, 4, 2), dtype=np.float32)
    gt = data["gt_boxes"] if "gt_boxes" in data else np.zeros((0, 4, 2), dtype=np.float32)
    pose = None
    if "ego_lidar_pose" in data:
        try:
            pose_arr = np.asarray(data["ego_lidar_pose"], dtype=np.float64).reshape(-1)
            if pose_arr.size >= 6:
                pose = pose_arr[:6]
        except Exception:
            pose = None
    return FrameRecord(
        pred_boxes=pred,
        gt_boxes=gt,
        source_path=str(path),
        frame_idx=_as_scalar_int(data["frame_idx"] if "frame_idx" in data else path.stem.split("_")[-1], -1),
        sample_idx=_as_scalar_int(data["sample_idx"] if "sample_idx" in data else -1, -1),
        scenario_index=_as_scalar_int(data["scenario_index"] if "scenario_index" in data else -1, -1),
        scenario_id=_as_scalar_string(data["scenario_id"] if "scenario_id" in data else "unknown", "unknown"),
        timestamp=_as_scalar_string(data["timestamp"] if "timestamp" in data else "", ""),
        frame_id=_as_scalar_string(data["frame_id"] if "frame_id" in data else "", ""),
        timestamp_index=_as_scalar_int(data["timestamp_index"] if "timestamp_index" in data else -1, -1),
        ego_id=_as_scalar_string(data["ego_id"] if "ego_id" in data else "", ""),
        ego_lidar_pose=pose,
    )


def load_run_frames(run_dir: Path, max_frames: int = 0) -> List[FrameRecord]:
    box_dir = run_dir / "danger_eval_boxes"
    frames: List[FrameRecord] = []
    if not box_dir.exists():
        LOGGER.warn("No trajectory box export found", run_dir=run_dir, expected="danger_eval_boxes/frame_*.npz")
        return frames
    for idx, path in enumerate(sorted(box_dir.glob("frame_*.npz"))):
        if max_frames and idx >= int(max_frames):
            break
        try:
            frames.append(_load_npz_frame(path))
        except Exception as exc:
            LOGGER.warn("Skipping malformed frame", path=path, error=str(exc))
    return frames


def _pose_to_world_matrix(pose: np.ndarray) -> np.ndarray:
    x, y, z, roll, yaw, pitch = [float(v) for v in pose[:6]]
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = [x, y, z]
    mat[0, 0] = cp * cy
    mat[0, 1] = cy * sp * sr - sy * cr
    mat[0, 2] = -cy * sp * cr - sy * sr
    mat[1, 0] = sy * cp
    mat[1, 1] = sy * sp * sr + cy * cr
    mat[1, 2] = -sy * sp * cr + cy * sr
    mat[2, 0] = sp
    mat[2, 1] = -cp * sr
    mat[2, 2] = cp * cr
    return mat


def _world_xy_to_current_ego(current_pose: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    world_xy = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    points = np.ones((world_xy.shape[0], 4), dtype=np.float64)
    points[:, :2] = world_xy
    points[:, 2] = float(current_pose[2]) if len(current_pose) > 2 else 0.0
    world_to_ego = np.linalg.inv(_pose_to_world_matrix(current_pose))
    ego = (world_to_ego @ points.T).T
    return ego[:, :2]


def _frame_sort_key(frame: FrameRecord):
    ts = frame.timestamp_index if frame.timestamp_index >= 0 else frame.frame_idx
    return (frame.scenario_index, frame.scenario_id, ts, frame.frame_idx)


def group_frames_by_scenario(frames: Sequence[FrameRecord]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    indexed = sorted(range(len(frames)), key=lambda i: _frame_sort_key(frames[i]))
    for i in indexed:
        frame = frames[i]
        key = f"{frame.scenario_index}:{frame.scenario_id}"
        groups.setdefault(key, []).append(i)
    return groups


def infer_dt(frames: Sequence[FrameRecord], assumed_dt: float = 0.1) -> float:
    indices = [f.timestamp_index for f in frames if f.timestamp_index >= 0]
    if len(indices) >= 2:
        diffs = [b - a for a, b in zip(indices[:-1], indices[1:]) if b > a]
        if diffs:
            # Dataset frame ids are integer steps; OPV2V-style data commonly uses 0.1s.
            return float(assumed_dt * max(1, int(round(np.median(diffs)))))
    return float(assumed_dt)


def ego_forward_trajectory(k_steps: int, dt: float, default_speed: float) -> np.ndarray:
    steps = np.arange(1, k_steps + 1, dtype=np.float64)
    return np.stack([default_speed * steps * dt, np.zeros_like(steps)], axis=1)


def future_pose_trajectory(frames: Sequence[FrameRecord], group_indices: List[int], group_pos: int, k_steps: int) -> Optional[np.ndarray]:
    current = frames[group_indices[group_pos]]
    if current.ego_lidar_pose is None:
        return None
    future_world = []
    for j in range(group_pos + 1, min(group_pos + 1 + k_steps, len(group_indices))):
        pose = frames[group_indices[j]].ego_lidar_pose
        if pose is not None and len(pose) >= 2:
            future_world.append([pose[0], pose[1]])
    if not future_world:
        return None
    return _world_xy_to_current_ego(current.ego_lidar_pose, np.asarray(future_world, dtype=np.float64))


def constant_velocity_trajectory(
    frames: Sequence[FrameRecord],
    group_indices: List[int],
    group_pos: int,
    k_steps: int,
    dt: float,
    default_speed: float,
) -> Optional[np.ndarray]:
    current = frames[group_indices[group_pos]]
    if current.ego_lidar_pose is None:
        return None
    if group_pos > 0:
        prev = frames[group_indices[group_pos - 1]]
        if prev.ego_lidar_pose is not None:
            prev_xy_in_current = _world_xy_to_current_ego(current.ego_lidar_pose, np.asarray([[prev.ego_lidar_pose[0], prev.ego_lidar_pose[1]]]))[0]
            # Current ego is at (0,0); current - previous displacement in current ego frame.
            velocity_xy = -prev_xy_in_current / max(dt, 1e-6)
            if np.linalg.norm(velocity_xy) > 1e-3:
                steps = np.arange(1, k_steps + 1, dtype=np.float64)[:, None]
                return velocity_xy[None, :] * steps * dt
    return ego_forward_trajectory(k_steps, dt, default_speed)


def select_trajectory(
    frames: Sequence[FrameRecord],
    group_indices: List[int],
    group_pos: int,
    k_steps: int,
    dt: float,
    source: str,
    default_speed: float,
) -> Tuple[np.ndarray, str]:
    source = str(source).lower()
    if source in {"auto", "future_pose"}:
        traj = future_pose_trajectory(frames, group_indices, group_pos, k_steps)
        if traj is not None and len(traj) > 0:
            return traj, "future_pose"
        if source == "future_pose":
            LOGGER.warn("Future poses unavailable; falling back", frame=frames[group_indices[group_pos]].source_path, fallback="constant_velocity")
    if source in {"auto", "future_pose", "constant_velocity"}:
        traj = constant_velocity_trajectory(frames, group_indices, group_pos, k_steps, dt, default_speed)
        if traj is not None and len(traj) > 0:
            return traj, "constant_velocity"
        if source == "constant_velocity":
            LOGGER.warn("Ego poses unavailable; falling back", frame=frames[group_indices[group_pos]].source_path, fallback="ego_forward")
    return ego_forward_trajectory(k_steps, dt, default_speed), "ego_forward"


def max_iou_per_gt(pred_boxes, gt_boxes) -> np.ndarray:
    pred = boxes_to_numpy(pred_boxes)
    gt = boxes_to_numpy(gt_boxes)
    if gt.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if pred.shape[0] == 0:
        return np.zeros((gt.shape[0],), dtype=np.float32)
    out = np.zeros((gt.shape[0],), dtype=np.float32)
    for idx, gt_box in enumerate(gt):
        ious = bev_iou(gt_box, pred)
        out[idx] = float(np.max(ious)) if len(ious) else 0.0
    return out


def trajectory_object_stats(gt_boxes, trajectory_xy: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = boxes_to_numpy(gt_boxes)
    centers = gt[:, :4, :2].mean(axis=1) if gt.shape[0] else np.zeros((0, 2), dtype=np.float64)
    if centers.shape[0] == 0 or trajectory_xy.shape[0] == 0:
        n = centers.shape[0]
        return np.full((n,), np.inf), np.full((n,), np.inf), np.full((n,), -1, dtype=np.int64)
    diff = centers[:, None, :] - np.asarray(trajectory_xy, dtype=np.float64)[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    k_star = np.argmin(dist, axis=1)
    d_traj = dist[np.arange(dist.shape[0]), k_star]
    tca = (k_star.astype(np.float64) + 1.0) * float(dt)
    return d_traj.astype(np.float64), tca.astype(np.float64), k_star.astype(np.int64)


def _empty_accumulator(iou_thresholds: Sequence[float]) -> Dict:
    return {
        "frames": 0,
        "skipped_frames": 0,
        "trajectory_relevant_objects": 0,
        "critical_objects": 0,
        "total_spatial_weight": 0.0,
        "total_time_weight": 0.0,
        "total_traj_time_weight": 0.0,
        "trajectory_source_counts": {},
        "detected_trajectory": {float(q): 0 for q in iou_thresholds},
        "detected_critical": {float(q): 0 for q in iou_thresholds},
        "spatial_detected": {float(q): 0.0 for q in iou_thresholds},
        "time_detected": {float(q): 0.0 for q in iou_thresholds},
        "traj_time_detected": {float(q): 0.0 for q in iou_thresholds},
        "missed_trajectory_risk": {float(q): 0.0 for q in iou_thresholds},
    }


def update_accumulator(
    acc: Dict,
    frame: FrameRecord,
    trajectory_xy: np.ndarray,
    trajectory_source_used: str,
    iou_thresholds: Sequence[float],
    dt: float,
    d_traj_max: float,
    d_critical: float,
    t_critical: float,
    sigma_d: float,
    sigma_t: float,
) -> None:
    gt = boxes_to_numpy(frame.gt_boxes)
    acc["frames"] += 1
    acc["trajectory_source_counts"][trajectory_source_used] = acc["trajectory_source_counts"].get(trajectory_source_used, 0) + 1
    if gt.shape[0] == 0:
        return
    max_iou = max_iou_per_gt(frame.pred_boxes, gt)
    d_traj, tca, _ = trajectory_object_stats(gt, trajectory_xy, dt)
    relevant = d_traj < float(d_traj_max)
    critical = np.logical_and(d_traj < float(d_critical), tca < float(t_critical))
    w_spatial = np.exp(-d_traj / max(float(sigma_d), 1e-6))
    w_time = np.exp(-tca / max(float(sigma_t), 1e-6))
    w_traj_time = w_spatial * w_time
    w_spatial[~np.isfinite(w_spatial)] = 0.0
    w_time[~np.isfinite(w_time)] = 0.0
    w_traj_time[~np.isfinite(w_traj_time)] = 0.0

    acc["trajectory_relevant_objects"] += int(relevant.sum())
    acc["critical_objects"] += int(critical.sum())
    acc["total_spatial_weight"] += float(w_spatial.sum())
    acc["total_time_weight"] += float(w_time.sum())
    acc["total_traj_time_weight"] += float(w_traj_time.sum())

    for q in iou_thresholds:
        q = float(q)
        detected = max_iou >= q
        acc["detected_trajectory"][q] += int(np.logical_and(relevant, detected).sum())
        acc["detected_critical"][q] += int(np.logical_and(critical, detected).sum())
        acc["spatial_detected"][q] += float((w_spatial * detected.astype(np.float64)).sum())
        acc["time_detected"][q] += float((w_time * detected.astype(np.float64)).sum())
        acc["traj_time_detected"][q] += float((w_traj_time * detected.astype(np.float64)).sum())
        acc["missed_trajectory_risk"][q] += float((w_traj_time * (~detected).astype(np.float64)).sum())


def finalize_metrics(acc: Dict, method: str, iou_thresholds: Sequence[float]) -> Dict:
    out = {
        "method": method,
        "frames": int(acc["frames"]),
        "skipped_frames": int(acc.get("skipped_frames", 0)),
        "trajectory_relevant_objects": int(acc["trajectory_relevant_objects"]),
        "critical_objects": int(acc["critical_objects"]),
        "total_spatial_weight": float(acc["total_spatial_weight"]),
        "total_time_weight": float(acc["total_time_weight"]),
        "total_trajectory_time_risk_weight": float(acc["total_traj_time_weight"]),
        "trajectory_source_counts": dict(acc.get("trajectory_source_counts", {})),
    }
    for q in iou_thresholds:
        q = float(q)
        suffix = _threshold_suffix(q)
        out[f"detected_trajectory_objects@{suffix}"] = int(acc["detected_trajectory"][q])
        out[f"detected_critical_objects@{suffix}"] = int(acc["detected_critical"][q])
        out[f"trajectory_zone_recall@{suffix}"] = _safe_div(acc["detected_trajectory"][q], acc["trajectory_relevant_objects"])
        out[f"trajectory_risk_weighted_recall@{suffix}"] = _safe_div(acc["spatial_detected"][q], acc["total_spatial_weight"])
        out[f"time_to_closest_approach_weighted_recall@{suffix}"] = _safe_div(acc["time_detected"][q], acc["total_time_weight"])
        out[f"trajectory_time_risk_recall@{suffix}"] = _safe_div(acc["traj_time_detected"][q], acc["total_traj_time_weight"])
        out[f"missed_trajectory_risk@{suffix}"] = float(acc["missed_trajectory_risk"][q])
        out[f"critical_object_recall@{suffix}"] = _safe_div(acc["detected_critical"][q], acc["critical_objects"])
    return out


def add_missed_trajectory_risk_reduction(metrics: List[Dict], baseline_method: str, iou_thresholds: Sequence[float]) -> None:
    baseline = next((m for m in metrics if m.get("method") == baseline_method), None)
    if baseline is None:
        LOGGER.warn("Baseline method not found for trajectory-risk reduction", baseline_method=baseline_method)
        return
    for row in metrics:
        for q in iou_thresholds:
            suffix = _threshold_suffix(float(q))
            base = float(baseline.get(f"missed_trajectory_risk@{suffix}", 0.0) or 0.0)
            cur = float(row.get(f"missed_trajectory_risk@{suffix}", 0.0) or 0.0)
            value = _safe_div(base - cur, base)
            row[f"missed_trajectory_risk_reduction_vs_{baseline_method}@{suffix}"] = value
            if baseline_method == "receiver_request_energy_topk_10":
                row[f"missed_trajectory_risk_reduction_vs_receiver@{suffix}"] = value


def evaluate_run(
    run_dir: Path,
    method: str,
    iou_thresholds: Sequence[float],
    horizon_seconds: float,
    dt_arg: Optional[float],
    assumed_dt: float,
    d_traj_max: float,
    d_critical: float,
    t_critical: float,
    sigma_d: float,
    sigma_t: float,
    trajectory_source: str,
    default_speed: float,
    max_frames: int = 0,
) -> Dict:
    frames = load_run_frames(run_dir, max_frames=max_frames)
    acc = _empty_accumulator(iou_thresholds)
    if not frames:
        metrics = finalize_metrics(acc, method, iou_thresholds)
        metrics["run_dir"] = str(run_dir)
        return metrics
    dt = float(dt_arg) if dt_arg is not None else infer_dt(frames, assumed_dt=assumed_dt)
    k_steps = max(1, int(math.ceil(float(horizon_seconds) / max(dt, 1e-6))))
    groups = group_frames_by_scenario(frames)
    group_pos_by_frame = {}
    for _, indices in groups.items():
        for pos, frame_idx in enumerate(indices):
            group_pos_by_frame[frame_idx] = (indices, pos)

    LOGGER.info("Evaluating run", method=method, frames=len(frames), trajectory_source=trajectory_source, dt=dt, horizon_seconds=horizon_seconds, k_steps=k_steps)
    for frame_idx, frame in enumerate(frames):
        try:
            group_indices, group_pos = group_pos_by_frame[frame_idx]
            traj, used = select_trajectory(frames, group_indices, group_pos, k_steps, dt, trajectory_source, default_speed)
            update_accumulator(
                acc,
                frame,
                traj,
                used,
                iou_thresholds,
                dt,
                d_traj_max,
                d_critical,
                t_critical,
                sigma_d,
                sigma_t,
            )
        except Exception as exc:
            acc["skipped_frames"] += 1
            LOGGER.warn("Skipping frame", method=method, source=frame.source_path, error=str(exc))
    metrics = finalize_metrics(acc, method, iou_thresholds)
    metrics["run_dir"] = str(run_dir)
    metrics["dt_used"] = float(dt)
    metrics["horizon_seconds"] = float(horizon_seconds)
    metrics["k_steps"] = int(k_steps)
    return metrics


def _csv_columns(iou_thresholds: Sequence[float], baseline_method: str) -> List[str]:
    cols = [
        "method",
        "frames",
        "skipped_frames",
        "trajectory_relevant_objects",
        "critical_objects",
        "total_trajectory_time_risk_weight",
        "dt_used",
        "horizon_seconds",
        "trajectory_source_counts",
    ]
    for prefix in [
        "trajectory_zone_recall",
        "trajectory_risk_weighted_recall",
        "time_to_closest_approach_weighted_recall",
        "trajectory_time_risk_recall",
        "critical_object_recall",
        "missed_trajectory_risk",
    ]:
        for q in iou_thresholds:
            cols.append(f"{prefix}@{_threshold_suffix(float(q))}")
    for q in iou_thresholds:
        suffix = _threshold_suffix(float(q))
        cols.append(f"missed_trajectory_risk_reduction_vs_{baseline_method}@{suffix}")
        if baseline_method == "receiver_request_energy_topk_10":
            cols.append(f"missed_trajectory_risk_reduction_vs_receiver@{suffix}")
    return cols


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def save_outputs(metrics: List[Dict], output_path: Path, iou_thresholds: Sequence[float], baseline_method: str, params: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_method": baseline_method,
        "iou_thresholds": [float(q) for q in iou_thresholds],
        "parameters": params,
        "methods": metrics,
    }
    with open(output_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    csv_path = output_path.with_suffix(".csv")
    cols = _csv_columns(iou_thresholds, baseline_method)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in metrics:
            writer.writerow({k: _csv_value(v) for k, v in row.items()})
    LOGGER.save("Trajectory danger metrics saved", yaml=output_path, json=json_path, csv=csv_path)


def update_run_summaries(metrics: List[Dict]) -> None:
    for row in metrics:
        run_dir = Path(row.get("run_dir", ""))
        summary_path = run_dir / "summary_eval.yaml"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, "r") as f:
                summary = yaml.safe_load(f) or {}
            summary["trajectory_danger_metrics"] = {k: v for k, v in row.items() if k != "run_dir"}
            with open(summary_path, "w") as f:
                yaml.safe_dump(summary, f, sort_keys=False)
            LOGGER.save("Run summary updated", summary=summary_path)
        except Exception as exc:
            LOGGER.warn("Could not update run summary", summary=summary_path, error=str(exc))


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate receiver trajectory-aware danger metrics from inference boxes")
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--method_names", nargs="+", default=None)
    parser.add_argument("--baseline_method", default="receiver_request_energy_topk_10")
    parser.add_argument("--iou_thresholds", nargs="+", type=float, default=[0.5, 0.7])
    parser.add_argument("--horizon_seconds", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=None, help="Frame interval in seconds. If omitted, uses --assumed_dt with frame-step inference.")
    parser.add_argument("--assumed_dt", type=float, default=0.1)
    parser.add_argument("--d_traj_max", type=float, default=5.0)
    parser.add_argument("--d_critical", type=float, default=3.0)
    parser.add_argument("--t_critical", type=float, default=3.0)
    parser.add_argument("--sigma_d", type=float, default=5.0)
    parser.add_argument("--sigma_t", type=float, default=2.0)
    parser.add_argument("--trajectory_source", choices=["auto", "future_pose", "constant_velocity", "ego_forward"], default="auto")
    parser.add_argument("--default_speed", type=float, default=10.0, help="m/s used for ego-forward fallback")
    parser.add_argument("--output_path", default="trajectory_danger_metrics.yaml")
    parser.add_argument("--update_run_summaries", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    start = time.time()
    run_dirs = [Path(p).expanduser() for p in args.run_dirs]
    if args.method_names is None:
        method_names = [p.name.replace("smoke_carla_", "").replace("smoke_culver_", "") for p in run_dirs]
    else:
        method_names = list(args.method_names)
    if len(method_names) != len(run_dirs):
        raise ValueError("--method_names must have the same length as --run_dirs")

    params = {
        "horizon_seconds": float(args.horizon_seconds),
        "dt": None if args.dt is None else float(args.dt),
        "assumed_dt": float(args.assumed_dt),
        "d_traj_max": float(args.d_traj_max),
        "d_critical": float(args.d_critical),
        "t_critical": float(args.t_critical),
        "sigma_d": float(args.sigma_d),
        "sigma_t": float(args.sigma_t),
        "trajectory_source": str(args.trajectory_source),
        "default_speed": float(args.default_speed),
        "max_frames": int(args.max_frames),
    }
    LOGGER.run("Evaluating trajectory danger metrics", runs=len(run_dirs), baseline_method=args.baseline_method, **params)
    metrics = [
        evaluate_run(
            run_dir,
            method,
            args.iou_thresholds,
            args.horizon_seconds,
            args.dt,
            args.assumed_dt,
            args.d_traj_max,
            args.d_critical,
            args.t_critical,
            args.sigma_d,
            args.sigma_t,
            args.trajectory_source,
            args.default_speed,
            max_frames=args.max_frames,
        )
        for run_dir, method in zip(run_dirs, method_names)
    ]
    add_missed_trajectory_risk_reduction(metrics, args.baseline_method, args.iou_thresholds)
    save_outputs(metrics, Path(args.output_path).expanduser(), args.iou_thresholds, args.baseline_method, params)
    if args.update_run_summaries:
        update_run_summaries(metrics)
    for row in metrics:
        LOGGER.metric(
            "Trajectory result",
            method=row.get("method"),
            frames=row.get("frames"),
            trajectory_relevant=row.get("trajectory_relevant_objects"),
            critical=row.get("critical_objects"),
            source_counts=row.get("trajectory_source_counts"),
        )
    LOGGER.success("Trajectory danger evaluation complete", methods=len(metrics), elapsed_seconds=round(time.time() - start, 3))


if __name__ == "__main__":
    main()
