import argparse
import csv
import os
import re

import src.hypes_yaml.yaml_utils as yaml_utils
from src.tools.reporting.approach_name_mapping import (
    canonical_public_name,
    infer_public_name_from_run,
    parse_public_name,
)
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
    metadata = _safe_get(comm, "metadata", {})
    return strategy, keep_ratio, packet_loss_rate, metadata


def _infer_keep_ratio_from_run_name(run_name):
    m = re.search(r"_(\d{2})(?:_(?:train|test))?$", run_name)
    if not m:
        return None
    return float(m.group(1)) / 100.0


def _split_and_name(run_name):
    m = re.match(r"^(carla|culver)_(.+)$", run_name)
    if not m:
        return "unknown", run_name
    split = m.group(1)
    core = m.group(2)
    core = re.sub(r"_(train|test)$", "", core)
    return split, core


def _is_stress(run_name, strategy):
    return ("stress_" in run_name) or (strategy == "random_drop_all_features")


def _approach_metadata(run_name, metadata):
    public_name = metadata.get("public_name") if isinstance(metadata, dict) else None
    if not public_name:
        public_name = infer_public_name_from_run(run_name)
    public_name = canonical_public_name(public_name)

    family = metadata.get("approach_family") if isinstance(metadata, dict) else None
    name = metadata.get("approach_name") if isinstance(metadata, dict) else None
    setting = metadata.get("approach_setting") if isinstance(metadata, dict) else None

    if not (family and name and setting):
        pf, pn, ps = parse_public_name(public_name)
        family = family or pf or "unknown"
        name = name or pn or "unknown"
        setting = setting or ps or "unknown"

    return family, name, str(setting), public_name


def main():
    logger = get_logger("CleanSummary")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="/kaggle/working/approach_runs")
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
        strategy, keep_ratio, packet_loss_rate, metadata = _infer_strategy(cfg)
        if keep_ratio is None:
            keep_ratio = _infer_keep_ratio_from_run_name(run_name)

        split, run_core = _split_and_name(run_name)
        approach_family, approach_name, approach_setting, public_name = _approach_metadata(run_name, metadata)

        comm_total_norm = _safe_get(summary, "comm_total_normalized_ratio", _safe_get(summary, "comm_normalized_ratio", None))
        if ("baseline_full_communication" in run_core) or ("measurement_full_communication" in run_core):
            comm_total_norm = 1.0

        row = {
            "run_name": run_name,
            "split": split,
            "approach_family": approach_family,
            "approach_name": approach_name,
            "approach_setting": approach_setting,
            "public_name": public_name,
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
            "comm_feature_normalized_ratio": _safe_get(summary, "comm_feature_normalized_ratio", comm_total_norm),
            "comm_context_normalized_ratio": _safe_get(summary, "comm_context_normalized_ratio", 0.0),
            "comm_metadata_normalized_ratio": _safe_get(summary, "comm_metadata_normalized_ratio", 0.0),
            "comm_total_normalized_ratio": comm_total_norm,
            "comm_normalized_ratio": comm_total_norm,
            "receiver_request_keep_ratio": _safe_get(summary, "receiver_request_keep_ratio", None),
            "receiver_request_context_ratio": _safe_get(summary, "receiver_request_context_ratio", None),
            "receiver_request_mask_metadata_ratio": _safe_get(summary, "receiver_request_mask_metadata_ratio", None),
            "temporal_novelty_mean": _safe_get(summary, "temporal_novelty_mean", None),
            "temporal_cache_age_mean": _safe_get(summary, "temporal_cache_age_mean", None),
            "temporal_cache_hit_ratio": _safe_get(summary, "temporal_cache_hit_ratio", None),
            "temporal_refresh_ratio": _safe_get(summary, "temporal_refresh_ratio", None),
            "temporal_init_frame_ratio": _safe_get(summary, "temporal_init_frame_ratio", None),
            "comm_cumulative_bytes_per_scenario": _safe_get(summary, "comm_cumulative_bytes_per_scenario", None),
            "comm_average_bytes_per_frame": _safe_get(summary, "comm_average_bytes_per_frame", None),
            "comm_total_bytes_per_frame_after_init": _safe_get(summary, "comm_total_bytes_per_frame_after_init", None),
            "comm_total_normalized_ratio_after_init": _safe_get(summary, "comm_total_normalized_ratio_after_init", None),
            "stress_test": bool(_is_stress(run_name, strategy)),
        }
        rows.append(row)

        logger.debug(
            "Collected run summary",
            run=run_name,
            public_name=public_name,
            strategy=strategy,
            keep_ratio=keep_ratio,
            ap70=row["AP@0.7"],
            comm_total_norm=row["comm_total_normalized_ratio"],
        )

    out_csv = args.out_csv or os.path.join(args.runs_root, "clean_summary.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "run_name", "split", "approach_family", "approach_name", "approach_setting", "public_name",
            "strategy", "keep_ratio", "packet_loss_rate",
            "AP@0.3", "AP@0.5", "AP@0.7",
            "comm_active_ratio", "comm_active_neighbors_ratio",
            "comm_feature_bytes_per_frame", "comm_context_bytes_per_frame", "comm_metadata_bytes_per_frame",
            "comm_total_bytes_per_frame",
            "comm_feature_normalized_ratio", "comm_context_normalized_ratio",
            "comm_metadata_normalized_ratio", "comm_total_normalized_ratio", "comm_normalized_ratio",
            "receiver_request_keep_ratio", "receiver_request_context_ratio", "receiver_request_mask_metadata_ratio",
            "temporal_novelty_mean", "temporal_cache_age_mean", "temporal_cache_hit_ratio",
            "temporal_refresh_ratio", "temporal_init_frame_ratio",
            "comm_cumulative_bytes_per_scenario", "comm_average_bytes_per_frame",
            "comm_total_bytes_per_frame_after_init", "comm_total_normalized_ratio_after_init",
            "stress_test",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    yaml_utils.save_yaml({"runs": rows}, os.path.join(os.path.dirname(out_csv), "clean_summary.yaml"))
    logger.save(
        "Clean summary written",
        csv_path=out_csv,
        yaml_path=os.path.join(os.path.dirname(out_csv), "clean_summary.yaml"),
        runs=len(rows),
    )


if __name__ == "__main__":
    main()
