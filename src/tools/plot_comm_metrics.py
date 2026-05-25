import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to comm_metrics_epoch.csv")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = args.out_dir or os.path.dirname(args.csv)
    os.makedirs(out_dir, exist_ok=True)

    df_inf = df[df["epoch"].astype(str) == "inference"]
    if len(df_inf) > 0:
        plt.figure()
        plt.scatter(df_inf["bytes_per_frame"], df_inf["ap_70"])
        plt.xlabel("Communication Cost (bytes/frame)")
        plt.ylabel("AP@0.7")
        plt.title("AP@0.7 vs Communication Cost")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "ap70_vs_comm_cost.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(df_inf["bytes_per_frame"], df_inf["ap_50"])
        plt.xlabel("Communication Cost (bytes/frame)")
        plt.ylabel("AP@0.5")
        plt.title("AP@0.5 vs Communication Cost")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "ap50_vs_comm_cost.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(df_inf["packet_loss_rate"], df_inf["ap_70"])
        plt.xlabel("Packet Loss Rate")
        plt.ylabel("AP@0.7")
        plt.title("AP@0.7 vs Packet Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "ap70_vs_packet_loss.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(df_inf["active_neighbors_ratio"], df_inf["bytes_per_frame"])
        plt.xlabel("Active Neighbors Ratio")
        plt.ylabel("Communication Cost (bytes/frame)")
        plt.title("Communication Cost vs Number of Vehicles")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "comm_cost_vs_neighbors.png"), dpi=200)
        plt.close()

    print("plots written to", out_dir)


if __name__ == "__main__":
    main()

