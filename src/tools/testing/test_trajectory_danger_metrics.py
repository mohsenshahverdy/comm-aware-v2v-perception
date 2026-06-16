import tempfile
from pathlib import Path

import numpy as np

from src.tools.evaluate_trajectory_danger_metrics import (
    FrameRecord,
    add_missed_trajectory_risk_reduction,
    ego_forward_trajectory,
    evaluate_run,
    finalize_metrics,
    max_iou_per_gt,
    trajectory_object_stats,
    update_accumulator,
    _empty_accumulator,
)
from src.utils.logging import get_logger

logger = get_logger("TestTrajectoryDangerMetrics")


def box(cx, cy, l=4.0, w=2.0):
    return np.asarray([
        [cx - l / 2, cy - w / 2],
        [cx + l / 2, cy - w / 2],
        [cx + l / 2, cy + w / 2],
        [cx - l / 2, cy + w / 2],
    ], dtype=np.float32)


def frame(pred, gt):
    return FrameRecord(
        pred_boxes=np.stack(pred).astype(np.float32) if pred else np.zeros((0, 4, 2), dtype=np.float32),
        gt_boxes=np.stack(gt).astype(np.float32) if gt else np.zeros((0, 4, 2), dtype=np.float32),
        source_path="synthetic",
        frame_idx=0,
        scenario_id="s0",
        timestamp_index=0,
    )


def test_object_on_path_detected_and_missed():
    f = frame(pred=[box(5, 0)], gt=[box(5, 0), box(15, 0), box(10, 10)])
    traj = ego_forward_trajectory(k_steps=30, dt=0.1, default_speed=10.0)
    acc = _empty_accumulator([0.5, 0.7])
    update_accumulator(acc, f, traj, "ego_forward", [0.5, 0.7], 0.1, 5.0, 3.0, 3.0, 5.0, 2.0)
    m = finalize_metrics(acc, "method", [0.5, 0.7])
    assert m["trajectory_relevant_objects"] == 2
    assert m["detected_trajectory_objects@0.5"] == 1
    assert abs(m["trajectory_zone_recall@0.5"] - 0.5) < 1e-6
    assert m["missed_trajectory_risk@0.5"] > 0


def test_far_from_path_has_lower_weight():
    gt = np.stack([box(10, 0), box(10, 12)]).astype(np.float32)
    traj = ego_forward_trajectory(k_steps=30, dt=0.1, default_speed=10.0)
    d_traj, tca, _ = trajectory_object_stats(gt, traj, 0.1)
    w = np.exp(-d_traj / 5.0)
    assert d_traj[0] < d_traj[1]
    assert w[0] > w[1]


def test_soon_object_has_higher_time_weight_than_later_object():
    gt = np.stack([box(5, 0), box(25, 0)]).astype(np.float32)
    traj = ego_forward_trajectory(k_steps=30, dt=0.1, default_speed=10.0)
    _, tca, _ = trajectory_object_stats(gt, traj, 0.1)
    w_time = np.exp(-tca / 2.0)
    assert tca[0] < tca[1]
    assert w_time[0] > w_time[1]


def test_critical_object_recall():
    f = frame(pred=[box(5, 0)], gt=[box(5, 0), box(25, 0)])
    traj = ego_forward_trajectory(k_steps=30, dt=0.1, default_speed=10.0)
    acc = _empty_accumulator([0.5])
    update_accumulator(acc, f, traj, "ego_forward", [0.5], 0.1, 5.0, 3.0, 1.0, 5.0, 2.0)
    m = finalize_metrics(acc, "method", [0.5])
    assert m["critical_objects"] == 1
    assert abs(m["critical_object_recall@0.5"] - 1.0) < 1e-6


def test_missed_trajectory_risk_reduction_positive():
    baseline = {"method": "receiver_request_energy_topk_10", "missed_trajectory_risk@0.5": 2.0, "missed_trajectory_risk@0.7": 3.0}
    better = {"method": "temporal_receiver_request_energy_topk_10", "missed_trajectory_risk@0.5": 1.0, "missed_trajectory_risk@0.7": 1.5}
    rows = [baseline, better]
    add_missed_trajectory_risk_reduction(rows, "receiver_request_energy_topk_10", [0.5, 0.7])
    assert abs(better["missed_trajectory_risk_reduction_vs_receiver@0.5"] - 0.5) < 1e-6
    assert abs(better["missed_trajectory_risk_reduction_vs_receiver@0.7"] - 0.5) < 1e-6


def test_npz_run_with_future_pose_metadata():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        box_dir = run_dir / "danger_eval_boxes"
        box_dir.mkdir()
        for idx in range(4):
            np.savez_compressed(
                box_dir / f"frame_{idx:06d}.npz",
                frame_idx=np.asarray([idx], dtype=np.int64),
                sample_idx=np.asarray([idx], dtype=np.int64),
                scenario_index=np.asarray([0], dtype=np.int64),
                timestamp_index=np.asarray([idx], dtype=np.int64),
                scenario_id=np.asarray("s0"),
                timestamp=np.asarray(f"{idx:06d}"),
                ego_id=np.asarray("ego"),
                ego_lidar_pose=np.asarray([idx, 0, 0, 0, 0, 0], dtype=np.float32),
                pred_boxes=np.stack([box(1, 0)]),
                pred_scores=np.asarray([0.9], dtype=np.float32),
                gt_boxes=np.stack([box(1, 0), box(3, 0)]),
            )
        metrics = evaluate_run(run_dir, "method", [0.5], 3.0, 1.0, 0.1, 5.0, 3.0, 3.0, 5.0, 2.0, "future_pose", 10.0)
        assert metrics["frames"] == 4
        assert metrics["trajectory_relevant_objects"] > 0
        assert metrics["trajectory_source_counts"].get("future_pose", 0) > 0


def test_max_iou_per_gt_handles_empty_predictions():
    gt = np.stack([box(5, 0)])
    out = max_iou_per_gt(np.zeros((0, 4, 2), dtype=np.float32), gt)
    assert out.shape == (1,)
    assert out[0] == 0.0


def main():
    test_object_on_path_detected_and_missed()
    test_far_from_path_has_lower_weight()
    test_soon_object_has_higher_time_weight_than_later_object()
    test_critical_object_recall()
    test_missed_trajectory_risk_reduction_positive()
    test_npz_run_with_future_pose_metadata()
    test_max_iou_per_gt_handles_empty_predictions()
    logger.success("Trajectory danger metric tests passed")


if __name__ == "__main__":
    main()
