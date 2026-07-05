#!/usr/bin/env python3
"""Generate per-dataset publication budget curves from the summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger  # noqa: E402


PLOT_SPECS = [
    ("ap_07_mean", "AP@0.7", "ap07_vs_budget"),
    ("trajectory_time_risk_recall_07_mean", "Trajectory-time risk recall@0.7", "ttrr07_vs_budget"),
    ("missed_trajectory_risk_07_mean", "Missed trajectory risk@0.7", "mtr07_vs_budget"),
    ("total_comm_ratio_mean", "Measured total communication ratio", "total_comm_vs_budget"),
]

METHOD_LABELS = {
    "full_communication": "Full",
    "selective_topk": "Top-K",
    "snapshot_receiver_request": "Snapshot receiver",
    "temporal_receiver_request": "Temporal receiver",
    "learned_temporal_receiver_request": "Learned temporal",
}


def _number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _series(rows: Iterable[Dict[str, str]], dataset: str, metric: str) -> Dict[str, List[Tuple[float, float]]]:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        budget = _number(row.get("budget_percent", ""))
        value = _number(row.get(metric, ""))
        if budget is not None and value is not None:
            grouped[row.get("method", "unknown")].append((budget, value))
    return grouped


def plot_metric(
    rows: List[Dict[str, str]],
    dataset: str,
    metric: str,
    ylabel: str,
    stem: str,
    output_dir: Path,
    logger=None,
) -> bool:
    logger = logger or get_logger("PublicationPlot")
    grouped = _series(rows, dataset, metric)
    if not grouped:
        logger.warn("Plot skipped; required metric unavailable", dataset=dataset, metric=metric, figure=stem)
        return False

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for method, points in sorted(grouped.items()):
        points = sorted(points)
        axis.plot(
            [x for x, _ in points],
            [y for _, y in points],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label=METHOD_LABELS.get(method, method.replace("_", " ")),
        )
    axis.set_xlabel("Nominal communication budget [%]")
    axis.set_ylabel(ylabel)
    axis.set_title(dataset.replace("_", " ").title())
    axis.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 50, 75, 100])
    axis.tick_params(axis="x", labelrotation=45)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{dataset}_{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{dataset}_{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    logger.save("Publication figure saved", dataset=dataset, pdf=output_dir / f"{dataset}_{stem}.pdf")
    return True


def plot_trajectory_efficiency(rows: List[Dict[str, str]], dataset: str, output_dir: Path, logger=None) -> bool:
    enriched: List[Dict[str, str]] = []
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        ttrr = _number(row.get("trajectory_time_risk_recall_07_mean", ""))
        ratio = _number(row.get("total_comm_ratio_mean", ""))
        if ttrr is None or ratio is None or ratio <= 0:
            continue
        item = dict(row)
        item["trajectory_efficiency_mean"] = str(ttrr / ratio)
        enriched.append(item)
    return plot_metric(
        enriched,
        dataset,
        "trajectory_efficiency_mean",
        "TTRR@0.7 / total communication ratio",
        "trajectory_efficiency_vs_budget",
        output_dir,
        logger,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/publication/all_experiments_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/publication/figures"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "WARN" if args.log_level == "WARNING" else args.log_level
    logger = get_logger("PublicationPlot", level="DEBUG" if args.debug else log_level, debug=args.debug)
    summary = args.summary if args.summary.is_absolute() else REPO_ROOT / args.summary
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    if not summary.exists():
        logger.warn("Summary CSV not found; run aggregation first", path=summary)
        return 0
    try:
        rows = load_rows(summary)
    except (OSError, ValueError) as exc:
        logger.error("Could not read publication summary", path=summary, error=str(exc))
        if args.debug:
            raise
        return 1
    if not rows:
        logger.warn("Summary CSV contains no experiment rows", path=summary)
        return 0

    datasets = sorted({row.get("dataset", "") for row in rows if row.get("dataset")})
    logger.info("Generating publication curves", datasets=datasets, rows=len(rows))
    generated = 0
    for dataset in datasets:
        generated += sum(int(plot_metric(rows, dataset, *spec, output_dir, logger)) for spec in PLOT_SPECS)
        generated += int(plot_trajectory_efficiency(rows, dataset, output_dir, logger))
    logger.success("Publication plotting completed", generated_families=generated, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
