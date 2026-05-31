import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logging import get_logger


def main():
    logger = get_logger("PlotCommMetrics")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to comm_metrics_epoch.csv")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = args.out_dir or os.path.dirname(args.csv)
    os.makedirs(out_dir, exist_ok=True)
    logger.run("Plotting communication metrics", csv=args.csv, out_dir=out_dir, rows=len(df))
    if "total_bytes_per_frame" not in df.columns and "bytes_per_frame" in df.columns:
        df["total_bytes_per_frame"] = df["bytes_per_frame"]
    if "total_bytes_per_frame" not in df.columns and "comm_total_bytes_per_frame" in df.columns:
        df["total_bytes_per_frame"] = df["comm_total_bytes_per_frame"]
    if "normalized_ratio" not in df.columns:
        if "comm_normalized_ratio" in df.columns:
            df["normalized_ratio"] = df["comm_normalized_ratio"]
        else:
            df["normalized_ratio"] = 1.0
    if "ap_70" not in df.columns and "AP@0.7" in df.columns:
        df["ap_70"] = df["AP@0.7"]
    if "ap_50" not in df.columns and "AP@0.5" in df.columns:
        df["ap_50"] = df["AP@0.5"]
    if "packet_loss_rate" not in df.columns and "comm_packet_loss_rate" in df.columns:
        df["packet_loss_rate"] = df["comm_packet_loss_rate"]
    if "active_neighbors_ratio" not in df.columns and "comm_active_neighbors_ratio" in df.columns:
        df["active_neighbors_ratio"] = df["comm_active_neighbors_ratio"]

    if "epoch" in df.columns:
        df_inf = df[df["epoch"].astype(str) == "inference"]
    else:
        # clean summary tables are already inference-only rows
        df_inf = df.copy()
        if "include_in_clean" in df_inf.columns:
            df_inf = df_inf[df_inf["include_in_clean"] == True]
    if len(df_inf) > 0:
        plt.figure()
        plt.scatter(df_inf["normalized_ratio"], df_inf["ap_70"])
        plt.xlabel("Communication Ratio (normalized)")
        plt.ylabel("AP@0.7")
        plt.title("AP@0.7 vs Communication Ratio")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "ap70_vs_comm_ratio.png"), dpi=200)
        logger.save("Plot saved", path=os.path.join(out_dir, "ap70_vs_comm_ratio.png"))
        plt.close()

        plt.figure()
        plt.scatter(df_inf["normalized_ratio"], df_inf["ap_50"])
        plt.xlabel("Communication Ratio (normalized)")
        plt.ylabel("AP@0.5")
        plt.title("AP@0.5 vs Communication Ratio")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "ap50_vs_comm_ratio.png"), dpi=200)
        logger.save("Plot saved", path=os.path.join(out_dir, "ap50_vs_comm_ratio.png"))
        plt.close()

        plt.figure()
        plt.scatter(df_inf["total_bytes_per_frame"], df_inf["ap_70"])
        plt.xlabel("Communication Cost (bytes/frame)")
        plt.ylabel("AP@0.7")
        plt.title("AP@0.7 vs Total Communication Bytes")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "ap70_vs_total_bytes.png"), dpi=200)
        logger.save("Plot saved", path=os.path.join(out_dir, "ap70_vs_total_bytes.png"))
        plt.close()

        plt.figure()
        plt.scatter(df_inf["packet_loss_rate"], df_inf["ap_70"])
        plt.xlabel("Packet Loss Rate")
        plt.ylabel("AP@0.7")
        plt.title("AP@0.7 vs Packet Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "ap70_vs_packet_loss.png"), dpi=200)
        logger.save("Plot saved", path=os.path.join(out_dir, "ap70_vs_packet_loss.png"))
        plt.close()

        plt.figure()
        plt.scatter(df_inf["active_neighbors_ratio"], df_inf["total_bytes_per_frame"])
        plt.xlabel("Active Neighbors Ratio")
        plt.ylabel("Communication Cost (bytes/frame)")
        plt.title("Communication Cost vs Number of Vehicles")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "comm_cost_vs_neighbors.png"), dpi=200)
        logger.save("Plot saved", path=os.path.join(out_dir, "comm_cost_vs_neighbors.png"))
        plt.close()
    else:
        logger.warn("No inference rows found; no plots generated")

    logger.success("Plot generation completed", out_dir=out_dir)


if __name__ == "__main__":
    main()
