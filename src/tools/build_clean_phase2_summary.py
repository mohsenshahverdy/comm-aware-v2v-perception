import argparse
import csv
import os
import re

import src.hypes_yaml.yaml_utils as yaml_utils
from src.utils.logging import get_logger


def _safe_get(dct, key, default=None):
    return dct.get(key, default) if isinstance(dct, dict) else default


def _infer_strategy(cfg):
    comm = _safe_get(_safe_get(_safe_get(cfg, "model", {}), "args", {}), "communication", {})
    strategy = _safe_get(comm, "strategy", "unknown")
    keep_ratio = None
    if strategy == "topk_energy":
        keep_ratio = _safe_get(_safe_get(comm, "topk_energy", {}), "keep_ratio", None)
    elif strategy == "receiver_request_topk":
        keep_ratio = _safe_get(_safe_get(comm, "receiver_request", {}), "keep_ratio", None)
    elif strategy in ("random_drop_comm_only", "random_drop_all_features", "random_drop"):
        keep_ratio = _safe_get(_safe_get(comm, "drop_random", {}), "keep_ratio", None)
    packet_loss_rate = _safe_get(_safe_get(comm, "packet_loss", {}), "loss_rate", 0.0)
    return strategy, keep_ratio, packet_loss_rate


def _infer_keep_ratio_from_run_name(run_name):
    m = re.search(r"_(\d{2})$", run_name)
    if not m:
        return None
    return float(m.group(1)) / 100.0


def _is_legacy_or_stress(run_name, strategy):
    # Keep in CSV, but mark for filtering in clean comparisons.
    stress = ("phase2_random_drop_all_features" in run_name) or (strategy == "random_drop_all_features")
    legacy = (
        "phase2_random_drop" in run_name and "phase2_random_drop_comm_only" not in run_name and "phase2_random_drop_all_features" not in run_name
    ) or (
        "phase2_topk_energy" in run_name and not re.search(r"phase2_topk_energy_\d{2}$", run_name)
    )
    return legacy, stress


def main():
    logger = get_logger("CleanSummary")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="/kaggle/working/phase_runs")
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    rows = []
    logger.run("Building clean summary", runs_root=args.runs_root)
    for run_name in sorted(os.listdir(args.runs_root)):
        run_dir = os.path.join(args.runs_root, run_name)
        if not os.path.isdir(run_dir):
            continue
        summary_path = os.path.join(run_dir, "summary_eval.yaml")
        cfg_path = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(summary_path):
            continue
        summary = yaml_utils.load_yaml(summary_path)
        cfg = yaml_utils.load_yaml(cfg_path) if os.path.exists(cfg_path) else {}
        strategy, keep_ratio, packet_loss_rate = _infer_strategy(cfg)
        if keep_ratio is None:
            keep_ratio = _infer_keep_ratio_from_run_name(run_name)

        is_legacy, is_stress = _is_legacy_or_stress(run_name, strategy)
        include_in_clean = (not is_legacy) and (not is_stress)
        comm_norm = _safe_get(summary, "comm_normalized_ratio", None)
        if ("phase0_baseline" in run_name) or ("phase1_measurement" in run_name):
            comm_norm = 1.0

        row = {
            "run": run_name,
            "strategy": strategy,
            "keep_ratio": keep_ratio,
            "packet_loss_rate": packet_loss_rate,
            "AP@0.3": _safe_get(summary, "ap30", _safe_get(summary, "ap_30", None)),
            "AP@0.5": _safe_get(summary, "ap_50", None),
            "AP@0.7": _safe_get(summary, "ap_70", None),
            "comm_active_ratio": _safe_get(summary, "comm_active_ratio", None),
            "comm_active_neighbors_ratio": _safe_get(summary, "comm_active_neighbors_ratio", None),
            "comm_feature_bytes_per_frame": _safe_get(summary, "comm_feature_bytes_per_frame", _safe_get(summary, "comm_bytes_per_frame", None)),
            "comm_context_bytes_per_frame": _safe_get(summary, "comm_context_bytes_per_frame", 0.0),
            "comm_metadata_bytes_per_frame": _safe_get(summary, "comm_metadata_bytes_per_frame", 0.0),
            "comm_total_bytes_per_frame": _safe_get(summary, "comm_total_bytes_per_frame", _safe_get(summary, "comm_bytes_per_frame", None)),
            "comm_normalized_ratio": comm_norm,
            "comm_feature_normalized_ratio": _safe_get(summary, "comm_feature_normalized_ratio", comm_norm),
            "comm_context_normalized_ratio": _safe_get(summary, "comm_context_normalized_ratio", 0.0),
            "comm_metadata_normalized_ratio": _safe_get(summary, "comm_metadata_normalized_ratio", 0.0),
            "comm_total_normalized_ratio": _safe_get(summary, "comm_total_normalized_ratio", comm_norm),
            "receiver_request_keep_ratio": _safe_get(summary, "receiver_request_keep_ratio", None),
            "receiver_request_context_ratio": _safe_get(summary, "receiver_request_context_ratio", None),
            "receiver_request_mask_metadata_ratio": _safe_get(summary, "receiver_request_mask_metadata_ratio", None),
            "legacy_run": bool(is_legacy),
            "stress_test": bool(is_stress),
            "include_in_clean": bool(include_in_clean),
        }
        rows.append(row)
        logger.debug(
            "Collected run summary",
            run=run_name,
            strategy=strategy,
            keep_ratio=keep_ratio,
            ap70=row["AP@0.7"],
            comm_normalized_ratio=row["comm_normalized_ratio"],
        )

    out_csv = args.out_csv or os.path.join(args.runs_root, "clean_summary.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "run", "strategy", "keep_ratio", "packet_loss_rate",
            "AP@0.3", "AP@0.5", "AP@0.7",
            "comm_active_ratio", "comm_active_neighbors_ratio",
            "comm_feature_bytes_per_frame", "comm_context_bytes_per_frame", "comm_metadata_bytes_per_frame",
            "comm_total_bytes_per_frame", "comm_normalized_ratio",
            "comm_feature_normalized_ratio", "comm_context_normalized_ratio",
            "comm_metadata_normalized_ratio", "comm_total_normalized_ratio",
            "receiver_request_keep_ratio", "receiver_request_context_ratio",
            "receiver_request_mask_metadata_ratio",
            "legacy_run", "stress_test", "include_in_clean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    yaml_utils.save_yaml({"runs": rows}, os.path.join(os.path.dirname(out_csv), "clean_summary.yaml"))
    logger.save("Clean summary written", csv_path=out_csv, yaml_path=os.path.join(os.path.dirname(out_csv), "clean_summary.yaml"), runs=len(rows))


if __name__ == "__main__":
    main()
