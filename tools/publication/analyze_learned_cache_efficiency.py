#!/usr/bin/env python3
"""Analyze learned temporal request-head training, cache, and efficiency artifacts.

This script is deliberately read-only with respect to model artifacts: it parses
existing training/evaluation files and publication CSVs, then writes summary CSVs,
figures, and a Markdown report for paper preparation. It never launches training
or inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


DEFAULT_TRAINING_RUN = Path("/home/ex-perception/runs/learned_temporal_receiver_request_10_lr1e3_full")
DEFAULT_EVAL_RUN = Path(
    "/home/ex-perception/runs/v2v_trajectory_carla_learned_lr1e3/"
    "smoke_carla_learned_temporal_receiver_request_10"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/publication"
DEFAULT_FIGURES_DIR = DEFAULT_OUTPUT_DIR / "figures"
DEFAULT_PUBLICATION_TABLES = [
    DEFAULT_OUTPUT_DIR / "carla_2021_full_completed_results.csv",
    DEFAULT_OUTPUT_DIR / "carla_2021_paper_summary_table.csv",
    DEFAULT_OUTPUT_DIR / "all_experiments_raw.csv",
    DEFAULT_OUTPUT_DIR / "all_experiments_summary.csv",
]

METHOD_ORDER = [
    "full_communication",
    "selective_topk",
    "snapshot_receiver_request",
    "temporal_receiver_request",
    "learned_temporal_receiver_request",
]
METHOD_LABELS = {
    "full_communication": "Full communication",
    "selective_topk": "Selective Top-K",
    "snapshot_receiver_request": "Snapshot receiver-request",
    "temporal_receiver_request": "Temporal receiver-request",
    "learned_temporal_receiver_request": "Learned temporal receiver-request",
}
METHOD_ALIASES = {
    "full": "full_communication",
    "full communication": "full_communication",
    "baseline_full_communication": "full_communication",
    "full_communication": "full_communication",
    "selective top-k 10%": "selective_topk",
    "selective top-k": "selective_topk",
    "top-k": "selective_topk",
    "topk": "selective_topk",
    "selective_topk": "selective_topk",
    "selective_topk_energy_10": "selective_topk",
    "snapshot receiver-request 10%": "snapshot_receiver_request",
    "snapshot receiver-request": "snapshot_receiver_request",
    "receiver": "snapshot_receiver_request",
    "receiver_request": "snapshot_receiver_request",
    "snapshot_receiver_request": "snapshot_receiver_request",
    "receiver_request_energy_topk_10": "snapshot_receiver_request",
    "temporal receiver-request 10%": "temporal_receiver_request",
    "temporal receiver-request": "temporal_receiver_request",
    "temporal_receiver_request": "temporal_receiver_request",
    "temporal_receiver_request_energy_topk_10": "temporal_receiver_request",
    "learned temporal receiver-request 10%": "learned_temporal_receiver_request",
    "learned temporal receiver-request": "learned_temporal_receiver_request",
    "learned temporal": "learned_temporal_receiver_request",
    "learned_temporal_receiver_request": "learned_temporal_receiver_request",
    "learned_temporal_receiver_request_10": "learned_temporal_receiver_request",
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _num(value: Any, default: float = math.nan) -> float:
    if value in (None, "", "None", "nan", "NaN"):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _str(value: Any) -> str:
    return "" if value is None else str(value)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _nested_get(mapping: Dict[str, Any], dotted: str, default: Any = math.nan) -> Any:
    def _walk(cursor: Any, remaining: str) -> Any:
        if not isinstance(cursor, dict):
            return default
        if remaining in cursor:
            return cursor[remaining]
        if "." not in remaining:
            return default
        head, tail = remaining.split(".", 1)
        if head not in cursor:
            return default
        return _walk(cursor[head], tail)

    return _walk(mapping, dotted)


def _first_present(mapping: Dict[str, Any], keys: Iterable[str], default: Any = math.nan) -> Any:
    for key in keys:
        if "." in key:
            value = _nested_get(mapping, key, None)
        else:
            value = mapping.get(key, None)
        if value not in (None, ""):
            return value
    return default


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered = OrderedDict()
        for row in rows:
            for key in row:
                ordered[key] = None
        fieldnames = list(ordered.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_training_comm_metrics(training_run: Path, logger) -> Tuple[List[Dict[str, Any]], List[str]]:
    path = training_run / "comm_metrics_epoch.csv"
    warnings: List[str] = []
    rows = _read_csv(path)
    if not rows:
        warnings.append(f"Missing or empty training comm metrics: {path}")
        return [], warnings
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        epoch = _num(row.get("epoch"), math.nan)
        if math.isnan(epoch):
            continue
        parsed.append({
            "epoch": int(epoch),
            "train_loss": _num(row.get("train_loss")),
            "val_loss": _num(row.get("val_loss")),
            "normalized_ratio": _num(row.get("normalized_ratio")),
            "feature_normalized_ratio": _num(row.get("feature_normalized_ratio")),
            "context_normalized_ratio": _num(row.get("context_normalized_ratio")),
            "metadata_normalized_ratio": _num(row.get("metadata_normalized_ratio")),
            "receiver_request_keep_ratio": _num(row.get("receiver_request_keep_ratio")),
            "fps": _num(row.get("fps")),
        })
    parsed.sort(key=lambda item: item["epoch"])
    logger.info("Training comm metrics parsed", rows=len(parsed), path=path)
    return parsed, warnings


def _load_loss_dicts(training_run: Path, logger) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    files = sorted(training_run.glob("loss_dict_epoch_*.json"))
    if not files:
        warnings.append(f"No loss_dict_epoch_*.json files found in {training_run}")
        return [], warnings
    rows: List[Dict[str, Any]] = []
    for path in files:
        payload = _read_json(path)
        match = re.search(r"epoch_(\d+)", path.name)
        epoch = int(match.group(1)) if match else int(_num(payload.get("epoch"), -1))
        row = {"epoch": epoch, "loss_dict_path": str(path)}
        for key, value in payload.items():
            if isinstance(value, (int, float, str)):
                numeric = _num(value)
                if not math.isnan(numeric):
                    row[key] = numeric
        rows.append(row)
    rows.sort(key=lambda item: item["epoch"])
    logger.info("Loss dicts parsed", files=len(rows), run=training_run)
    return rows, warnings


def _merge_training_rows(comm_rows: List[Dict[str, Any]], loss_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_epoch: Dict[int, Dict[str, Any]] = {int(row["epoch"]): dict(row) for row in comm_rows if "epoch" in row}
    for row in loss_rows:
        epoch = int(row.get("epoch", -1))
        target = by_epoch.setdefault(epoch, {"epoch": epoch})
        for key, value in row.items():
            if key not in target:
                target[key] = value
    return [by_epoch[key] for key in sorted(by_epoch)]


def _extract_comm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    comm = _nested_get(config, "model.args.communication", {})
    if not isinstance(comm, dict):
        comm = config.get("communication", {}) if isinstance(config.get("communication"), dict) else {}
    rr = comm.get("receiver_request", {}) if isinstance(comm.get("receiver_request"), dict) else {}
    temporal = rr.get("temporal", {}) if isinstance(rr.get("temporal"), dict) else {}
    learned = rr.get("learned", {}) if isinstance(rr.get("learned"), dict) else {}
    loss = learned.get("loss", {}) if isinstance(learned.get("loss"), dict) else {}
    optimizer = learned.get("optimizer", {}) if isinstance(learned.get("optimizer"), dict) else {}
    return {
        "trainable": rr.get("trainable", math.nan),
        "keep_ratio": rr.get("keep_ratio", math.nan),
        "cache_type": temporal.get("cache_type", ""),
        "cache_momentum": temporal.get("cache_momentum", math.nan),
        "cache_confidence_decay": temporal.get("cache_confidence_decay", math.nan),
        "max_cache_age": temporal.get("max_cache_age", math.nan),
        "cache_confidence_weight": temporal.get("cache_confidence_weight", math.nan),
        "periodic_refresh_interval": temporal.get("periodic_refresh_interval", math.nan),
        "periodic_refresh_keep_ratio": temporal.get("periodic_refresh_keep_ratio", math.nan),
        "use_soft_mask_train": learned.get("use_soft_mask_train", math.nan),
        "target_budget": learned.get("target_budget", math.nan),
        "budget_loss_enabled": loss.get("budget_enabled", loss.get("enabled", math.nan)),
        "budget_lambda": loss.get("budget_lambda", math.nan),
        "target_ratio": loss.get("target_ratio", math.nan),
        "request_head_lr": optimizer.get("request_head_lr", math.nan),
        "train_request_head_only": optimizer.get("train_request_head_only", math.nan),
    }


def _load_config_summary(training_run: Path, eval_run: Path) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    for path in (training_run / "config.yaml", eval_run / "config.yaml"):
        config = _read_yaml(path)
        if config:
            return _extract_comm_config(config), warnings
        warnings.append(f"Config not found or empty: {path}")
    return _extract_comm_config({}), warnings


def _extract_final_eval(eval_run: Path) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    summary_path = eval_run / "summary_eval.yaml"
    inference_path = eval_run / "inference_summary.json"
    summary = _read_yaml(summary_path)
    inference = _read_json(inference_path)
    if not summary:
        warnings.append(f"Missing or empty final evaluation summary: {summary_path}")
    if not inference:
        warnings.append(f"Missing or empty final inference summary: {inference_path}")
    merged = {**inference, **summary}
    row = {
        "ap_07": _num(_first_present(merged, ["ap_70", "ap70", "ap_07"])),
        "total_communication_ratio": _num(_first_present(merged, ["comm_total_normalized_ratio", "comm_normalized_ratio", "total_comm_ratio"])),
        "feature_bytes_per_frame": _num(_first_present(merged, ["comm_feature_bytes_per_frame"])),
        "context_bytes_per_frame": _num(_first_present(merged, ["comm_context_bytes_per_frame"])),
        "metadata_bytes_per_frame": _num(_first_present(merged, ["comm_metadata_bytes_per_frame"])),
        "total_bytes_per_frame": _num(_first_present(merged, ["comm_total_bytes_per_frame", "comm_bytes_per_frame"])),
        "receiver_request_keep_ratio": _num(_first_present(merged, ["receiver_request_keep_ratio"])),
        "temporal_cache_hit_ratio": _num(_first_present(merged, ["temporal_cache_hit_ratio"])),
        "temporal_cache_age_mean": _num(_first_present(merged, ["temporal_cache_age_mean"])),
        "temporal_cache_confidence_mean": _num(_first_present(merged, ["temporal_cache_confidence_mean"])),
        "temporal_cache_entries": _num(_first_present(merged, ["temporal_cache_entries"])),
        "temporal_init_frame_ratio": _num(_first_present(merged, ["temporal_init_frame_ratio"])),
        "temporal_refresh_ratio": _num(_first_present(merged, ["temporal_refresh_ratio"])),
        "packet_loss_rate": _num(_first_present(merged, ["comm_packet_loss_rate", "packet_loss_rate"])),
        "trajectory_time_risk_recall_07": _num(_first_present(merged, [
            "trajectory_time_risk_recall_07",
            "trajectory_danger_metrics.trajectory_time_risk_recall@0.7",
        ])),
        "missed_trajectory_risk_07": _num(_first_present(merged, [
            "missed_trajectory_risk_07",
            "trajectory_danger_metrics.missed_trajectory_risk@0.7",
        ])),
        "missed_trajectory_risk_reduction_vs_receiver_07": _num(_first_present(merged, [
            "missed_trajectory_risk_reduction_vs_receiver_07",
            "missed_trajectory_risk_reduction_07",
            "trajectory_danger_metrics.missed_trajectory_risk_reduction_vs_receiver@0.7",
            "trajectory_danger_metrics.missed_trajectory_risk_reduction_vs_snapshot_receiver_request@0.7",
        ])),
    }
    return row, warnings


def _canonical_method(row: Dict[str, Any]) -> Optional[str]:
    candidates = [
        row.get("method"), row.get("Method"), row.get("approach"), row.get("Approach"),
        row.get("public_name"), row.get("name"), row.get("Name"),
    ]
    for candidate in candidates:
        key = _str(candidate).strip().lower().replace("_10%", "").replace("  ", " ")
        if key in METHOD_ALIASES:
            return METHOD_ALIASES[key]
        key_no_pct = key.replace("%", "").strip()
        if key_no_pct in METHOD_ALIASES:
            return METHOD_ALIASES[key_no_pct]
    return None


def _row_value(row: Dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        if key in row:
            value = _num(row.get(key))
            if not math.isnan(value):
                return value
    return math.nan


def _load_publication_comparison(paths: List[Path], final_eval: Dict[str, Any], logger) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    selected: Dict[str, Dict[str, Any]] = {}
    source_by_method: Dict[str, str] = {}
    for path in paths:
        rows = _read_csv(path)
        if not rows:
            warnings.append(f"Missing or empty publication comparison CSV: {path}")
            continue
        for row in rows:
            method = _canonical_method(row)
            if method not in METHOD_ORDER:
                continue
            dataset = _str(row.get("dataset") or row.get("Dataset") or row.get("split") or row.get("Split")).lower()
            if dataset and "carla" not in dataset and dataset != "":
                continue
            budget = _row_value(row, ["budget_percent", "Budget", "budget", "communication_budget"])
            if not math.isnan(budget) and abs(budget - 10.0) > 1e-9 and method != "full_communication":
                continue
            record = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "ap_07": _row_value(row, ["ap_07", "ap_07_mean", "AP@0.7", "AP70", "ap_70", "ap70"]),
                "total_communication_ratio": _row_value(row, [
                    "total_comm_ratio", "total_comm_ratio_mean", "total_communication_ratio",
                    "comm_total_normalized_ratio", "communication_ratio", "Comm. ratio",
                ]),
                "trajectory_time_risk_recall_07": _row_value(row, [
                    "trajectory_time_risk_recall_07", "trajectory_time_risk_recall_07_mean",
                    "TTRR@0.7", "trajectory_time_risk_recall@0.7",
                ]),
                "missed_trajectory_risk_07": _row_value(row, [
                    "missed_trajectory_risk_07", "missed_trajectory_risk_07_mean",
                    "missed_trajectory_risk@0.7", "MTR@0.7",
                ]),
                "missed_trajectory_risk_reduction_07": _row_value(row, [
                    "missed_trajectory_risk_reduction_07", "missed_trajectory_risk_reduction_07_mean",
                    "missed_trajectory_risk_reduction_vs_receiver_07",
                    "missed_trajectory_risk_reduction_vs_receiver@0.7",
                    "risk_reduction", "Risk reduction",
                ]),
                "source_path": str(path),
            }
            current = selected.get(method)
            current_score = sum(not math.isnan(_num(current.get(k))) for k in record if current) if current else -1
            record_score = sum(not math.isnan(_num(record.get(k))) for k in record)
            if current is None or record_score >= current_score:
                selected[method] = record
                source_by_method[method] = str(path)

    # Make sure the learned row exists using final eval if publication CSVs are incomplete.
    learned = selected.setdefault("learned_temporal_receiver_request", {
        "method": "learned_temporal_receiver_request",
        "method_label": METHOD_LABELS["learned_temporal_receiver_request"],
        "source_path": str(DEFAULT_EVAL_RUN / "summary_eval.yaml"),
    })
    learned.setdefault("ap_07", final_eval.get("ap_07", math.nan))
    learned.setdefault("total_communication_ratio", final_eval.get("total_communication_ratio", math.nan))
    learned.setdefault("trajectory_time_risk_recall_07", final_eval.get("trajectory_time_risk_recall_07", math.nan))
    learned.setdefault("missed_trajectory_risk_07", final_eval.get("missed_trajectory_risk_07", math.nan))
    learned.setdefault("missed_trajectory_risk_reduction_07", final_eval.get("missed_trajectory_risk_reduction_vs_receiver_07", math.nan))
    for key in ("ap_07", "total_communication_ratio", "trajectory_time_risk_recall_07", "missed_trajectory_risk_07", "missed_trajectory_risk_reduction_07"):
        if math.isnan(_num(learned.get(key))):
            learned[key] = final_eval.get(key, learned.get(key, math.nan))

    full_ratio = _num(selected.get("full_communication", {}).get("total_communication_ratio"))
    rows_out: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        row = selected.get(method)
        if not row:
            warnings.append(f"No budget-10 comparison row found for {method}")
            continue
        ratio = _num(row.get("total_communication_ratio"))
        row["communication_saving_vs_full"] = float(1.0 - ratio / full_ratio) if full_ratio and not math.isnan(full_ratio) and not math.isnan(ratio) else math.nan
        rows_out.append(row)
    logger.info("Publication comparison parsed", rows=len(rows_out))
    return rows_out, warnings


def _plot_training_loss(rows: List[Dict[str, Any]], figures_dir: Path, logger) -> bool:
    if plt is None or not rows:
        return False
    xs = [row["epoch"] for row in rows]
    train = [_num(row.get("train_loss")) for row in rows]
    val = [_num(row.get("val_loss")) for row in rows]
    if all(math.isnan(v) for v in train + val):
        logger.warn("Training loss plot skipped; train_loss/val_loss unavailable")
        return False
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    if not all(math.isnan(v) for v in train):
        ax.plot(xs, train, marker="o", linewidth=2.0, label="train loss")
    if not all(math.isnan(v) for v in val):
        ax.plot(xs, val, marker="s", linewidth=2.0, label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learned Temporal Request-Head Training Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"learned_training_loss_curve.{ext}", dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def _plot_training_comm(rows: List[Dict[str, Any]], figures_dir: Path, logger) -> bool:
    if plt is None or not rows:
        return False
    xs = [row["epoch"] for row in rows]
    specs = [
        ("normalized_ratio", "total normalized ratio", "o"),
        ("feature_normalized_ratio", "feature normalized ratio", "s"),
        ("receiver_request_keep_ratio", "receiver request keep ratio", "^"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    plotted = False
    for key, label, marker in specs:
        ys = [_num(row.get(key)) for row in rows]
        if all(math.isnan(v) for v in ys):
            continue
        ax.plot(xs, ys, marker=marker, linewidth=2.0, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        logger.warn("Communication efficiency plot skipped; ratio columns unavailable")
        return False
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ratio")
    ax.set_title("Learned Temporal Communication Ratio During Training")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"learned_comm_efficiency_curve.{ext}", dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def _plot_cache_summary(final_eval: Dict[str, Any], figures_dir: Path, logger) -> bool:
    if plt is None:
        return False
    metrics = OrderedDict([
        ("cache hit", _num(final_eval.get("temporal_cache_hit_ratio"))),
        ("init frame", _num(final_eval.get("temporal_init_frame_ratio"))),
        ("refresh", _num(final_eval.get("temporal_refresh_ratio"))),
        ("confidence", _num(final_eval.get("temporal_cache_confidence_mean"))),
    ])
    labels = [k for k, v in metrics.items() if not math.isnan(v)]
    values = [v for v in metrics.values() if not math.isnan(v)]
    if not values:
        logger.warn("Cache behavior plot skipped; temporal cache metrics unavailable")
        return False
    fig, ax = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#1f77b4", "#8c8c8c", "#ff7f0e", "#2ca02c"][:len(values)])
    ax.set_ylabel("Ratio / mean value")
    ax.set_title("Learned Temporal Cache Behavior Summary")
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    age = _num(final_eval.get("temporal_cache_age_mean"))
    entries = _num(final_eval.get("temporal_cache_entries"))
    note = []
    if not math.isnan(age):
        note.append(f"mean age={age:.2f}")
    if not math.isnan(entries):
        note.append(f"entries={entries:.0f}")
    if note:
        ax.text(0.01, 0.96, ", ".join(note), transform=ax.transAxes, va="top", fontsize=8)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"learned_cache_behavior_summary.{ext}", dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def _plot_efficiency_comparison(rows: List[Dict[str, Any]], figures_dir: Path, logger) -> bool:
    if plt is None or not rows:
        return False
    labels = [METHOD_LABELS.get(row["method"], row["method"]) for row in rows]
    comm = [_num(row.get("total_communication_ratio")) for row in rows]
    ap = [_num(row.get("ap_07")) for row in rows]
    ttrr = [_num(row.get("trajectory_time_risk_recall_07")) for row in rows]
    x = list(range(len(rows)))
    width = 0.28
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    if not all(math.isnan(v) for v in comm):
        ax.bar([i - width for i in x], comm, width=width, label="total comm. ratio", color="#9ecae1")
    if not all(math.isnan(v) for v in ap):
        ax.bar(x, ap, width=width, label="AP@0.7", color="#3182bd")
    if not all(math.isnan(v) for v in ttrr):
        ax.bar([i + width for i in x], ttrr, width=width, label="TTRR@0.7", color="#31a354")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("CARLA Budget-10 Learned Temporal Efficiency Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"learned_efficiency_comparison_budget10.{ext}", dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def _fmt(value: Any, digits: int = 4) -> str:
    value = _num(value)
    return "missing" if math.isnan(value) else f"{value:.{digits}f}"


def _write_report(
    path: Path,
    *,
    training_rows: List[Dict[str, Any]],
    final_eval: Dict[str, Any],
    config_summary: Dict[str, Any],
    comparison_rows: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in training_rows if "epoch" in row]
    first_epoch = min(epochs) if epochs else "missing"
    last_epoch = max(epochs) if epochs else "missing"
    train_start = next((_num(row.get("train_loss")) for row in training_rows if not math.isnan(_num(row.get("train_loss")))), math.nan)
    train_end = next((_num(row.get("train_loss")) for row in reversed(training_rows) if not math.isnan(_num(row.get("train_loss")))), math.nan)
    val_end = next((_num(row.get("val_loss")) for row in reversed(training_rows) if not math.isnan(_num(row.get("val_loss")))), math.nan)

    lines = [
        "# Learned Temporal Cache and Efficiency Analysis",
        "",
        "## Training-Loss Summary",
        "",
        f"Training metrics were parsed for epochs `{first_epoch}` to `{last_epoch}`.",
        "The intended training interval is epochs 45--54 starting from checkpoint `net_epoch45.pth`; the final checkpoint is `net_epoch55.pth`.",
        f"First available train loss: `{_fmt(train_start)}`; final available train loss: `{_fmt(train_end)}`; final available validation loss: `{_fmt(val_end)}`.",
        "",
        "## Cache-Behavior Summary",
        "",
        f"Final AP@0.7: `{_fmt(final_eval.get('ap_07'))}`.",
        f"Final total communication ratio: `{_fmt(final_eval.get('total_communication_ratio'))}`.",
        f"Receiver-request keep ratio: `{_fmt(final_eval.get('receiver_request_keep_ratio'))}`.",
        f"Temporal cache hit ratio: `{_fmt(final_eval.get('temporal_cache_hit_ratio'))}`.",
        f"Temporal cache age mean: `{_fmt(final_eval.get('temporal_cache_age_mean'))}`.",
        f"Temporal cache confidence mean: `{_fmt(final_eval.get('temporal_cache_confidence_mean'))}`.",
        f"Temporal cache entries: `{_fmt(final_eval.get('temporal_cache_entries'), 0)}`.",
        f"Trajectory-time risk recall@0.7: `{_fmt(final_eval.get('trajectory_time_risk_recall_07'))}`.",
        f"Missed trajectory risk@0.7: `{_fmt(final_eval.get('missed_trajectory_risk_07'))}`.",
        f"Missed trajectory-risk reduction vs receiver@0.7: `{_fmt(final_eval.get('missed_trajectory_risk_reduction_vs_receiver_07'))}`.",
        "",
        "## Learned Configuration Summary",
        "",
    ]
    for key in sorted(config_summary):
        lines.append(f"- `{key}`: `{config_summary[key]}`")
    lines.extend(["", "## Budget-10 Comparison Against Baselines", ""])
    if comparison_rows:
        lines.append("| Method | AP@0.7 | Total comm. ratio | TTRR@0.7 | Missed trajectory risk | Risk reduction | Saving vs full |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in comparison_rows:
            lines.append(
                f"| {row.get('method_label', row.get('method'))} | {_fmt(row.get('ap_07'))} | "
                f"{_fmt(row.get('total_communication_ratio'))} | {_fmt(row.get('trajectory_time_risk_recall_07'))} | "
                f"{_fmt(row.get('missed_trajectory_risk_07'))} | {_fmt(row.get('missed_trajectory_risk_reduction_07'))} | "
                f"{_fmt(row.get('communication_saving_vs_full'))} |"
            )
    else:
        lines.append("No compatible publication comparison rows were found.")
    lines.extend(["", "## Warnings / Missing Inputs", ""])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("No missing inputs were detected.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=DEFAULT_TRAINING_RUN)
    parser.add_argument("--eval-run", type=Path, default=DEFAULT_EVAL_RUN)
    parser.add_argument("--publication-csv", type=Path, action="append", default=None, help="Additional/override publication CSV input. Can be repeated.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--strict", action="store_true", help="Fail if required server files are missing.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "WARN" if args.log_level == "WARNING" else args.log_level
    logger = get_logger("LearnedCacheEfficiency", level="DEBUG" if args.debug else log_level, debug=args.debug)
    training_run = _resolve(args.training_run)
    eval_run = _resolve(args.eval_run)
    output_dir = _resolve(args.output_dir)
    figures_dir = _resolve(args.figures_dir)
    publication_paths = [_resolve(path) for path in (args.publication_csv or DEFAULT_PUBLICATION_TABLES)]

    warnings: List[str] = []
    comm_rows, comm_warnings = _load_training_comm_metrics(training_run, logger)
    loss_rows, loss_warnings = _load_loss_dicts(training_run, logger)
    warnings.extend(comm_warnings + loss_warnings)
    training_rows = _merge_training_rows(comm_rows, loss_rows)
    final_eval, eval_warnings = _extract_final_eval(eval_run)
    config_summary, config_warnings = _load_config_summary(training_run, eval_run)
    warnings.extend(eval_warnings + config_warnings)
    comparison_rows, comparison_warnings = _load_publication_comparison(publication_paths, final_eval, logger)
    warnings.extend(comparison_warnings)

    if args.strict and warnings:
        for warning in warnings:
            logger.error("Required input missing", warning=warning)
        return 1

    training_loss_csv = output_dir / "learned_training_loss_summary.csv"
    cache_summary_csv = output_dir / "learned_cache_efficiency_summary.csv"
    report_path = output_dir / "learned_cache_efficiency_report.md"

    _write_csv(training_loss_csv, training_rows or [{"status": "missing_training_metrics"}])
    cache_row = {**final_eval, **config_summary, "training_run": str(training_run), "eval_run": str(eval_run)}
    _write_csv(cache_summary_csv, [cache_row])

    generated = []
    if _plot_training_loss(training_rows, figures_dir, logger):
        generated.append("learned_training_loss_curve")
    if _plot_training_comm(training_rows, figures_dir, logger):
        generated.append("learned_comm_efficiency_curve")
    if _plot_cache_summary(final_eval, figures_dir, logger):
        generated.append("learned_cache_behavior_summary")
    if _plot_efficiency_comparison(comparison_rows, figures_dir, logger):
        generated.append("learned_efficiency_comparison_budget10")

    _write_report(
        report_path,
        training_rows=training_rows,
        final_eval=final_eval,
        config_summary=config_summary,
        comparison_rows=comparison_rows,
        warnings=warnings,
    )
    logger.save("Training-loss summary saved", path=training_loss_csv)
    logger.save("Cache efficiency summary saved", path=cache_summary_csv)
    logger.save("Learned cache efficiency report saved", path=report_path)
    if generated:
        logger.save("Figures generated", figures=generated, figures_dir=figures_dir)
    if warnings:
        logger.warn("Analysis completed with missing/partial inputs", warnings=len(warnings))
    logger.success("Learned cache efficiency analysis completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
