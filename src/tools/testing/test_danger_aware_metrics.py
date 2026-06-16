import math
import tempfile
from pathlib import Path

import numpy as np

from src.tools.evaluate_danger_aware_metrics import (
    add_missed_risk_reduction,
    danger_weights,
    evaluate_run,
    gt_detected_by_iou,
    update_accumulator,
    _empty_accumulator,
    finalize_metrics,
)
from src.utils.logging import get_logger

logger = get_logger("TestDangerAwareMetrics")


def box(cx, cy, l=4.0, w=2.0):
    return np.asarray([
        [cx - l / 2, cy - w / 2],
        [cx + l / 2, cy - w / 2],
        [cx + l / 2, cy + w / 2],
        [cx - l / 2, cy + w / 2],
    ], dtype=np.float32)


def test_detection_and_danger_recall():
    gt = np.stack([
        box(10, 0),    # danger detected
        box(20, 0),    # danger missed
        box(-5, 0),    # behind ego, ignored
        box(15, 20),   # lateral outside danger zone, ignored
    ])
    pred = np.stack([box(10, 0)])
    acc = _empty_accumulator([0.5, 0.7])
    update_accumulator(acc, pred, gt, [0.5, 0.7], x_max=40, y_max=10, tau=20)
    metrics = finalize_metrics(acc, "method", [0.5, 0.7])
    assert metrics["danger_objects"] == 2
    assert metrics["detected_danger_objects@0.5"] == 1
    assert abs(metrics["danger_zone_recall@0.5"] - 0.5) < 1e-6
    assert abs(metrics["danger_zone_recall@0.7"] - 0.5) < 1e-6


def test_close_object_has_higher_risk_weight():
    gt = np.stack([box(5, 0), box(30, 0)])
    danger_mask, weights = danger_weights(gt, x_max=40, y_max=10, tau=20)
    assert danger_mask.tolist() == [True, True]
    assert weights[0] > weights[1]
    assert abs(weights[0] - math.exp(-5 / 20)) < 1e-6


def test_iou_matching():
    gt = np.stack([box(10, 0), box(20, 0)])
    pred = np.stack([box(10, 0), box(100, 0)])
    detected = gt_detected_by_iou(pred, gt, 0.7)
    assert detected.tolist() == [True, False]


def test_missed_risk_reduction_positive():
    baseline = {
        "method": "receiver_request_energy_topk_10",
        "missed_risk@0.5": 2.0,
        "missed_risk@0.7": 3.0,
    }
    better = {
        "method": "temporal_receiver_request_energy_topk_10",
        "missed_risk@0.5": 1.0,
        "missed_risk@0.7": 2.0,
    }
    metrics = [baseline, better]
    add_missed_risk_reduction(metrics, "receiver_request_energy_topk_10", [0.5, 0.7])
    assert better["missed_risk_reduction_vs_receiver@0.5"] > 0
    assert abs(better["missed_risk_reduction_vs_receiver@0.5"] - 0.5) < 1e-6


def test_npz_run_loading():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        box_dir = run_dir / "danger_eval_boxes"
        box_dir.mkdir()
        np.savez_compressed(
            box_dir / "frame_000000.npz",
            pred_boxes=np.stack([box(10, 0)]),
            pred_scores=np.asarray([0.9], dtype=np.float32),
            gt_boxes=np.stack([box(10, 0), box(20, 0)]),
        )
        metrics = evaluate_run(run_dir, "method", [0.5, 0.7], x_max=40, y_max=10, tau=20)
        assert metrics["frames"] == 1
        assert metrics["danger_objects"] == 2
        assert metrics["detected_danger_objects@0.5"] == 1


def main():
    test_detection_and_danger_recall()
    test_close_object_has_higher_risk_weight()
    test_iou_matching()
    test_missed_risk_reduction_positive()
    test_npz_run_loading()
    logger.success("Danger-aware metric tests passed")


if __name__ == "__main__":
    main()
