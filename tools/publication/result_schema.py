"""Unified row schema for publication experiments."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List


RESULT_COLUMNS: List[str] = [
    "experiment_name",
    "dataset",
    "method",
    "budget_percent",
    "seed",
    "loss_type",
    "loss_probability",
    "monte_carlo_run",
    "num_frames",
    "num_collaborators",
    "num_transmitted_units",
    "num_lost_units",
    "actual_loss_rate",
    "ap_05",
    "ap_07",
    "feature_comm_ratio",
    "total_comm_ratio",
    "bytes_per_frame",
    "static_danger_recall_07",
    "static_risk_weighted_recall_07",
    "trajectory_time_risk_recall_07",
    "missed_trajectory_risk_07",
    "missed_trajectory_risk_reduction_07",
    "result_path",
    "checkpoint_path",
    "config_path",
    "timestamp",
    "status",
    "notes",
]

NUMERIC_COLUMNS = {
    "budget_percent",
    "seed",
    "loss_probability",
    "monte_carlo_run",
    "num_frames",
    "num_collaborators",
    "num_transmitted_units",
    "num_lost_units",
    "actual_loss_rate",
    "ap_05",
    "ap_07",
    "feature_comm_ratio",
    "total_comm_ratio",
    "bytes_per_frame",
    "static_danger_recall_07",
    "static_risk_weighted_recall_07",
    "trajectory_time_risk_recall_07",
    "missed_trajectory_risk_07",
    "missed_trajectory_risk_reduction_07",
}

# Existing inference/evaluator names mapped to publication names.
ALIASES = {
    "processed_frames": "num_frames",
    "max_samples": "num_frames",
    "frames": "num_frames",
    "ap_50": "ap_05",
    "ap50": "ap_05",
    "ap_70": "ap_07",
    "ap70": "ap_07",
    "comm_feature_normalized_ratio": "feature_comm_ratio",
    "comm_total_normalized_ratio": "total_comm_ratio",
    "comm_normalized_ratio": "total_comm_ratio",
    "comm_total_bytes_per_frame": "bytes_per_frame",
    "comm_bytes_per_frame": "bytes_per_frame",
    "comm_packet_loss_rate": "actual_loss_rate",
    "danger_zone_recall_07": "static_danger_recall_07",
    "danger_zone_recall@0.7": "static_danger_recall_07",
    "risk_weighted_recall_07": "static_risk_weighted_recall_07",
    "risk_weighted_recall@0.7": "static_risk_weighted_recall_07",
    "trajectory_time_risk_recall_07": "trajectory_time_risk_recall_07",
    "trajectory_time_risk_recall@0.7": "trajectory_time_risk_recall_07",
    "missed_trajectory_risk_07": "missed_trajectory_risk_07",
    "missed_trajectory_risk@0.7": "missed_trajectory_risk_07",
    "missed_trajectory_risk_reduction_vs_receiver_07": "missed_trajectory_risk_reduction_07",
    "missed_trajectory_risk_reduction_vs_receiver@0.7": "missed_trajectory_risk_reduction_07",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_result(**values: Any) -> Dict[str, Any]:
    row = {column: math.nan if column in NUMERIC_COLUMNS else "" for column in RESULT_COLUMNS}
    row.update(values)
    if not row.get("timestamp"):
        row["timestamp"] = utc_timestamp()
    return row


def normalize_result(payload: Dict[str, Any], **defaults: Any) -> Dict[str, Any]:
    """Normalize an existing summary or publication record without dropping schema columns."""
    flattened = dict(payload or {})
    metrics = flattened.pop("metrics", None)
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            flattened.setdefault(key, value)
    for section_name in ("danger_aware_metrics", "trajectory_danger_metrics"):
        section = flattened.pop(section_name, None)
        if isinstance(section, dict):
            for key, value in section.items():
                flattened.setdefault(key, value)

    row = empty_result(**defaults)
    for key, value in flattened.items():
        target = ALIASES.get(key, key)
        if (
            key.startswith("missed_trajectory_risk_reduction_vs_")
            and key.endswith("@0.7")
            and ("receiver" in key or "snapshot_receiver_request" in key)
        ):
            target = "missed_trajectory_risk_reduction_07"
        if target in row and value is not None:
            row[target] = value
    return {column: row[column] for column in RESULT_COLUMNS}


def numeric_value(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate_columns(columns: Iterable[str]) -> List[str]:
    present = set(columns)
    return [column for column in RESULT_COLUMNS if column not in present]
