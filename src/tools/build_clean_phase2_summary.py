import argparse
import csv
import os

import src.hypes_yaml.yaml_utils as yaml_utils


def _safe_get(dct, key, default=None):
    return dct.get(key, default) if isinstance(dct, dict) else default


def _infer_strategy(cfg):
    comm = _safe_get(_safe_get(_safe_get(cfg, "model", {}), "args", {}), "communication", {})
    return _safe_get(comm, "strategy", "unknown"), _safe_get(_safe_get(comm, "topk_energy", {}), "keep_ratio", _safe_get(_safe_get(comm, "drop_random", {}), "keep_ratio", None)), _safe_get(_safe_get(comm, "packet_loss", {}), "loss_rate", 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="/kaggle/working/phase_runs")
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    rows = []
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
        row = {
            "run": run_name,
            "strategy": strategy,
            "keep_ratio": keep_ratio,
            "packet_loss_rate": packet_loss_rate,
            "AP@0.3": _safe_get(summary, "ap_30", None),
            "AP@0.5": _safe_get(summary, "ap_50", None),
            "AP@0.7": _safe_get(summary, "ap_70", None),
            "comm_active_ratio": _safe_get(summary, "comm_active_ratio", None),
            "comm_active_neighbors_ratio": _safe_get(summary, "comm_active_neighbors_ratio", None),
            "comm_feature_bytes_per_frame": _safe_get(summary, "comm_feature_bytes_per_frame", _safe_get(summary, "comm_bytes_per_frame", None)),
            "comm_metadata_bytes_per_frame": _safe_get(summary, "comm_metadata_bytes_per_frame", 0.0),
            "comm_total_bytes_per_frame": _safe_get(summary, "comm_total_bytes_per_frame", _safe_get(summary, "comm_bytes_per_frame", None)),
            "comm_normalized_ratio": _safe_get(summary, "comm_normalized_ratio", None),
        }
        rows.append(row)

    out_csv = args.out_csv or os.path.join(args.runs_root, "clean_summary.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "run", "strategy", "keep_ratio", "packet_loss_rate",
            "AP@0.3", "AP@0.5", "AP@0.7",
            "comm_active_ratio", "comm_active_neighbors_ratio",
            "comm_feature_bytes_per_frame", "comm_metadata_bytes_per_frame",
            "comm_total_bytes_per_frame", "comm_normalized_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    yaml_utils.save_yaml({"runs": rows}, os.path.join(os.path.dirname(out_csv), "clean_summary.yaml"))
    print("wrote:", out_csv)


if __name__ == "__main__":
    main()
