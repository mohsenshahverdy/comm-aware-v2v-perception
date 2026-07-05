#!/usr/bin/env python3
"""Aggregate publication result records into raw and grouped CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.publication.result_schema import (  # noqa: E402
    NUMERIC_COLUMNS,
    RESULT_COLUMNS,
    normalize_result,
    numeric_value,
)
from src.utils.logging import get_logger  # noqa: E402

GROUP_COLUMNS = ["dataset", "method", "budget_percent", "loss_type", "loss_probability"]


def _read_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return [payload] if isinstance(payload, dict) else []


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_yaml(path: Path) -> List[Dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [payload] if isinstance(payload, dict) else []


def discover_result_files(input_dir: Path) -> List[Path]:
    preferred = sorted(input_dir.rglob("publication_result.json"))
    standalone = [
        path for path in sorted(input_dir.glob("*.json")) + sorted(input_dir.glob("*.csv"))
        if path.name not in {"all_experiments_raw.csv", "all_experiments_summary.csv"}
    ]
    # Existing inference summaries are accepted only when no publication record
    # exists beside them, preventing duplicate rows for one run.
    existing = []
    for name in ("summary_eval.yaml", "inference_summary.json"):
        for path in sorted(input_dir.rglob(name)):
            if not (path.parent / "publication_result.json").exists():
                existing.append(path)
    return list(dict.fromkeys(preferred + standalone + existing))


def read_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _read_json(path)
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _read_yaml(path)
    return []


def normalize_files(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for payload in read_records(path):
            if path.name == "publication_result.json":
                summary_path = path.parent / "summary_eval.yaml"
                if summary_path.exists():
                    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
                    if isinstance(summary, dict):
                        # Identity/status remain owned by publication_result;
                        # metrics are refreshed from the latest run summary.
                        identity_keys = {
                            "experiment_name", "dataset", "method", "budget_percent", "seed",
                            "loss_type", "loss_probability", "monte_carlo_run", "result_path",
                            "checkpoint_path", "config_path", "timestamp", "status", "notes",
                        }
                        identity = {key: payload[key] for key in identity_keys if key in payload}
                        payload = {**payload, **summary, **identity}
            row = normalize_result(payload, result_path=str(path))
            if not row.get("result_path"):
                row["result_path"] = str(path)
            rows.append(row)
    return rows


def write_raw(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_group_value(row: Dict[str, Any], column: str) -> str:
    value = row.get(column, "")
    if column in {"budget_percent", "loss_probability"}:
        number = numeric_value(value)
        return "" if number is None else f"{number:g}"
    return str(value)


def _group_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(_canonical_group_value(row, column) for column in GROUP_COLUMNS)


def summarize(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status", "")).lower() not in {"completed", "pass", "success"}:
            continue
        groups[_group_key(row)].append(row)

    metric_columns = [column for column in RESULT_COLUMNS if column in NUMERIC_COLUMNS and column not in {"budget_percent", "loss_probability"}]
    fields = GROUP_COLUMNS + ["num_runs"]
    for metric in metric_columns:
        fields.extend([f"{metric}_mean", f"{metric}_std"])

    output: List[Dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        summary: Dict[str, Any] = dict(zip(GROUP_COLUMNS, key))
        summary["num_runs"] = len(members)
        for metric in metric_columns:
            values = [value for value in (numeric_value(row.get(metric)) for row in members) if value is not None]
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else math.nan
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else math.nan)
        output.append(summary)
    return fields, output


def write_summary(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/publication"))
    parser.add_argument("--raw-output", type=Path, default=Path("results/publication/all_experiments_raw.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/publication/all_experiments_summary.csv"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "WARN" if args.log_level == "WARNING" else args.log_level
    logger = get_logger("PublicationAggregate", level="DEBUG" if args.debug else log_level, debug=args.debug)
    input_dir = args.input_dir if args.input_dir.is_absolute() else REPO_ROOT / args.input_dir
    raw_output = args.raw_output if args.raw_output.is_absolute() else REPO_ROOT / args.raw_output
    summary_output = args.summary_output if args.summary_output.is_absolute() else REPO_ROOT / args.summary_output
    try:
        files = discover_result_files(input_dir) if input_dir.exists() else []
        logger.info("Result discovery completed", files=len(files), input_dir=input_dir)
        logger.debug("Discovered result files", paths=files)
        rows = normalize_files(files)
        write_raw(raw_output, rows)
        fields, summary_rows = summarize(rows)
        write_summary(summary_output, fields, summary_rows)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        logger.error("Result aggregation failed", error=str(exc))
        if args.debug:
            raise
        return 1
    logger.metric("Results normalized", raw_rows=len(rows), summary_rows=len(summary_rows))
    logger.save("Raw results saved", path=raw_output)
    logger.save("Summary results saved", path=summary_output)
    if not rows:
        logger.warn("No compatible results found; header-only CSV files written")
    else:
        logger.success("Publication result aggregation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
