# -*- coding: utf-8 -*-
"""Danger-aware recall metrics for cooperative perception inference runs.

The evaluator consumes per-frame predicted and ground-truth BEV boxes exported by
``src.tools.inference --save_box_npz``. For older runs, it also supports the
legacy ``--save_npy`` folder containing ``*_pred.npy`` and ``*_gt.npy_test``.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from src.utils.logging import get_logger

LOGGER = get_logger("DangerAwareEval")


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if den > 0 else float(default)


def boxes_to_numpy(boxes) -> np.ndarray:
    if boxes is None:
        return np.zeros((0, 4, 2), dtype=np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (4, 2):
        arr = arr[None, :, :]
    if arr.ndim == 2 and arr.shape[1] >= 2:
        # Defensive fallback for flattened corners: (N, 8) -> (N, 4, 2).
        if arr.shape[1] >= 8:
            arr = arr[:, :8].reshape(-1, 4, 2)
    if arr.ndim == 3 and arr.shape[1] >= 4 and arr.shape[2] >= 2:
        return arr[:, :4, :2].astype(np.float32)
    raise ValueError(f"Unsupported box shape: {arr.shape}")


def box_centers_xy(boxes: np.ndarray) -> np.ndarray:
    boxes = boxes_to_numpy(boxes)
    if boxes.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return boxes[:, :4, :2].mean(axis=1)


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
    s = q2 - q1
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) < 1e-8:
        return p2.copy()
    qp = q1 - p1
    t = float((qp[0] * s[1] - qp[1] * s[0]) / denom)
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


def bev_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)[:4, :2]
    boxes = boxes_to_numpy(boxes).astype(np.float64)
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


def danger_weights(gt_boxes, x_max=40.0, y_max=10.0, tau=20.0) -> Tuple[np.ndarray, np.ndarray]:
    gt = boxes_to_numpy(gt_boxes)
    centers = box_centers_xy(gt)
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float64)
    x = centers[:, 0]
    y = centers[:, 1]
    d = np.sqrt(x ** 2 + y ** 2)
    danger_mask = (x > 0.0) & (x < float(x_max)) & (np.abs(y) < float(y_max))
    weights = np.exp(-d / float(tau)) * danger_mask.astype(np.float64)
    return danger_mask.astype(bool), weights.astype(np.float64)


def gt_detected_by_iou(pred_boxes, gt_boxes, iou_threshold: float) -> np.ndarray:
    pred = boxes_to_numpy(pred_boxes)
    gt = boxes_to_numpy(gt_boxes)
    detected = np.zeros((gt.shape[0],), dtype=bool)
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return detected
    for gt_idx, gt_box in enumerate(gt):
        ious = bev_iou(gt_box, pred)
        detected[gt_idx] = bool(len(ious) > 0 and np.max(ious) >= float(iou_threshold))
    return detected


def _empty_accumulator(iou_thresholds: Sequence[float]) -> Dict:
    return {
        "frames": 0,
        "skipped_frames": 0,
        "danger_objects": 0,
        "total_risk_weight": 0.0,
        "detected_danger": {float(q): 0 for q in iou_thresholds},
        "risk_detected": {float(q): 0.0 for q in iou_thresholds},
        "missed_risk": {float(q): 0.0 for q in iou_thresholds},
    }


def update_accumulator(acc: Dict, pred_boxes, gt_boxes, iou_thresholds: Sequence[float], x_max: float, y_max: float, tau: float) -> None:
    gt = boxes_to_numpy(gt_boxes)
    pred = boxes_to_numpy(pred_boxes)
    danger_mask, weights = danger_weights(gt, x_max=x_max, y_max=y_max, tau=tau)
    acc["frames"] += 1
    acc["danger_objects"] += int(danger_mask.sum())
    acc["total_risk_weight"] += float(weights.sum())
    for q in iou_thresholds:
        q = float(q)
        detected = gt_detected_by_iou(pred, gt, q)
        acc["detected_danger"][q] += int(np.logical_and(danger_mask, detected).sum())
        risk_detected = float((weights * detected.astype(np.float64)).sum())
        acc["risk_detected"][q] += risk_detected
        acc["missed_risk"][q] += float((weights * (~detected).astype(np.float64)).sum())


def finalize_metrics(acc: Dict, method: str, iou_thresholds: Sequence[float]) -> Dict:
    out = {
        "method": method,
        "frames": int(acc["frames"]),
        "skipped_frames": int(acc.get("skipped_frames", 0)),
        "danger_objects": int(acc["danger_objects"]),
        "total_danger_objects": int(acc["danger_objects"]),
        "total_risk_weight": float(acc["total_risk_weight"]),
    }
    for q in iou_thresholds:
        q = float(q)
        suffix = _threshold_suffix(q)
        out[f"detected_danger_objects@{suffix}"] = int(acc["detected_danger"][q])
        out[f"danger_zone_recall@{suffix}"] = _safe_div(acc["detected_danger"][q], acc["danger_objects"])
        out[f"risk_weighted_recall@{suffix}"] = _safe_div(acc["risk_detected"][q], acc["total_risk_weight"])
        out[f"missed_risk@{suffix}"] = float(acc["missed_risk"][q])
    return out


def add_missed_risk_reduction(metrics: List[Dict], baseline_method: str, iou_thresholds: Sequence[float]) -> None:
    baseline = next((m for m in metrics if m.get("method") == baseline_method), None)
    if baseline is None:
        LOGGER.warn("Baseline method not found for missed-risk reduction", baseline_method=baseline_method)
        return
    for m in metrics:
        for q in iou_thresholds:
            suffix = _threshold_suffix(float(q))
            base_missed = _safe_float(baseline.get(f"missed_risk@{suffix}", 0.0))
            method_missed = _safe_float(m.get(f"missed_risk@{suffix}", 0.0))
            m[f"missed_risk_reduction_vs_{baseline_method}@{suffix}"] = _safe_div(base_missed - method_missed, base_missed)
            # Short alias useful for CSVs/reports when baseline is receiver request.
            if baseline_method == "receiver_request_energy_topk_10":
                m[f"missed_risk_reduction_vs_receiver@{suffix}"] = m[f"missed_risk_reduction_vs_{baseline_method}@{suffix}"]


def _threshold_suffix(q: float) -> str:
    text = f"{q:.2f}".rstrip("0").rstrip(".")
    return text


def _load_npz_frame(path: Path):
    data = np.load(path, allow_pickle=False)
    pred = data["pred_boxes"] if "pred_boxes" in data else np.zeros((0, 4, 2), dtype=np.float32)
    gt = data["gt_boxes"] if "gt_boxes" in data else np.zeros((0, 4, 2), dtype=np.float32)
    return pred, gt


def _legacy_gt_path(pred_path: Path) -> Path:
    return pred_path.with_name(pred_path.name.replace("_pred.npy", "_gt.npy_test"))


def iter_run_frames(run_dir: Path) -> Iterable[Tuple[np.ndarray, np.ndarray, str]]:
    box_dir = run_dir / "danger_eval_boxes"
    if box_dir.exists():
        for path in sorted(box_dir.glob("frame_*.npz")):
            pred, gt = _load_npz_frame(path)
            yield pred, gt, str(path)
        return

    npy_dir = run_dir / "npy"
    if npy_dir.exists():
        for pred_path in sorted(npy_dir.glob("*_pred.npy")):
            gt_path = _legacy_gt_path(pred_path)
            if not gt_path.exists():
                LOGGER.warn("Legacy GT file missing", pred_path=pred_path, expected_gt=gt_path)
                continue
            yield np.load(pred_path, allow_pickle=False), np.load(gt_path, allow_pickle=False), str(pred_path)
        return

    LOGGER.warn("No box export found", run_dir=run_dir, expected="danger_eval_boxes/*.npz or npy/*_pred.npy")


def evaluate_run(run_dir: Path, method: str, iou_thresholds: Sequence[float], x_max: float, y_max: float, tau: float) -> Dict:
    acc = _empty_accumulator(iou_thresholds)
    for pred, gt, source in iter_run_frames(run_dir):
        try:
            update_accumulator(acc, pred, gt, iou_thresholds, x_max=x_max, y_max=y_max, tau=tau)
        except Exception as exc:
            acc["skipped_frames"] += 1
            LOGGER.warn("Skipping malformed frame", method=method, source=source, error=str(exc))
    metrics = finalize_metrics(acc, method, iou_thresholds)
    metrics["run_dir"] = str(run_dir)
    return metrics


def _csv_columns(iou_thresholds: Sequence[float], baseline_method: str) -> List[str]:
    cols = ["method", "frames", "skipped_frames", "danger_objects", "total_risk_weight"]
    for prefix in ["danger_zone_recall", "risk_weighted_recall", "missed_risk"]:
        for q in iou_thresholds:
            cols.append(f"{prefix}@{_threshold_suffix(float(q))}")
    for q in iou_thresholds:
        suffix = _threshold_suffix(float(q))
        cols.append(f"missed_risk_reduction_vs_{baseline_method}@{suffix}")
        if baseline_method == "receiver_request_energy_topk_10":
            cols.append(f"missed_risk_reduction_vs_receiver@{suffix}")
    return cols


def save_outputs(metrics: List[Dict], output_path: Path, iou_thresholds: Sequence[float], baseline_method: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_method": baseline_method,
        "iou_thresholds": [float(q) for q in iou_thresholds],
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
            writer.writerow(row)
    LOGGER.save("Danger-aware metrics saved", yaml=output_path, json=json_path, csv=csv_path)


def update_run_summaries(metrics: List[Dict]) -> None:
    for row in metrics:
        run_dir = Path(row.get("run_dir", ""))
        summary_path = run_dir / "summary_eval.yaml"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, "r") as f:
                summary = yaml.safe_load(f) or {}
            summary["danger_aware_metrics"] = {k: v for k, v in row.items() if k != "run_dir"}
            with open(summary_path, "w") as f:
                yaml.safe_dump(summary, f, sort_keys=False)
            LOGGER.save("Run summary updated", summary=summary_path)
        except Exception as exc:
            LOGGER.warn("Could not update run summary", summary=summary_path, error=str(exc))


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate danger-aware missed-risk metrics from inference boxes")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="Run directories to evaluate")
    parser.add_argument("--method_names", nargs="+", default=None, help="Method names matching --run_dirs")
    parser.add_argument("--baseline_method", default="receiver_request_energy_topk_10")
    parser.add_argument("--iou_thresholds", nargs="+", type=float, default=[0.5, 0.7])
    parser.add_argument("--x_max", type=float, default=40.0)
    parser.add_argument("--y_max", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--output_path", default="danger_aware_metrics.yaml")
    parser.add_argument("--update_run_summaries", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_dirs = [Path(p).expanduser() for p in args.run_dirs]
    if args.method_names is None:
        method_names = [p.name.replace("smoke_carla_", "").replace("smoke_culver_", "") for p in run_dirs]
    else:
        method_names = list(args.method_names)
    if len(method_names) != len(run_dirs):
        raise ValueError("--method_names must have the same length as --run_dirs")

    LOGGER.run(
        "Evaluating danger-aware metrics",
        runs=len(run_dirs),
        baseline_method=args.baseline_method,
        iou_thresholds=args.iou_thresholds,
        x_max=args.x_max,
        y_max=args.y_max,
        tau=args.tau,
    )
    metrics = [
        evaluate_run(run_dir, method, args.iou_thresholds, args.x_max, args.y_max, args.tau)
        for run_dir, method in zip(run_dirs, method_names)
    ]
    add_missed_risk_reduction(metrics, args.baseline_method, args.iou_thresholds)
    save_outputs(metrics, Path(args.output_path).expanduser(), args.iou_thresholds, args.baseline_method)
    if args.update_run_summaries:
        update_run_summaries(metrics)
    LOGGER.success("Danger-aware evaluation complete", methods=len(metrics))


if __name__ == "__main__":
    main()
