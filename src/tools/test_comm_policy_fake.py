import torch

from src.models.fuse_modules.communication_policy import CommunicationPolicy


def run_case(name, cfg, features, record_len):
    policy = CommunicationPolicy(in_channels=features.shape[1], comm_cfg=cfg)
    out = policy(features, record_len, pairwise_t_matrix=None)
    print(f"\n=== {name} ===")
    print("output shape:", tuple(out.features.shape))
    print("comm_stats:", out.stats)
    print("aux keys:", list(out.aux.keys()))


def main():
    torch.manual_seed(0)
    features = torch.randn(5, 256, 100, 352)
    record_len = torch.tensor([3, 2])

    base = {
        "enabled": True,
        "phase": "phase2",
        "strategy": "none",
        "seed": 42,
        "measurement": {
            "track_bytes": True,
            "track_active_cells": True,
            "track_active_neighbors": True,
            "track_latency": False,
        },
        "drop_random": {"keep_ratio": 0.5},
        "topk_energy": {"keep_ratio": 0.1, "score_type": "l2"},
        "neighbor_selection": {"mode": "all", "k": 0, "distance_metric": "euclidean"},
        "packet_loss": {"enabled": False, "loss_rate": 0.0, "unit": "cell"},
        "learnable_mask": {
            "enabled": False,
            "mask_channels": 16,
            "sparsity_lambda": 0.0,
            "temperature": 1.0,
            "hard_mask": False,
        },
        "repair_network": {"enabled": False, "type": "conv", "hidden_dim": 128, "loss_weight": 0.1},
    }

    # 1) random_drop
    cfg = dict(base)
    cfg["strategy"] = "random_drop"
    cfg["drop_random"] = {"keep_ratio": 0.2}
    run_case("random_drop", cfg, features, record_len)

    # 2) topk_energy
    cfg = dict(base)
    cfg["strategy"] = "topk_energy"
    cfg["topk_energy"] = {"keep_ratio": 0.1, "score_type": "l2"}
    run_case("topk_energy", cfg, features, record_len)

    # 3) neighbor_selection only
    cfg = dict(base)
    cfg["strategy"] = "none"
    cfg["neighbor_selection"] = {"mode": "topk_importance", "k": 1, "distance_metric": "euclidean"}
    run_case("neighbor_selection(topk_importance)", cfg, features, record_len)

    # 4) packet_loss
    cfg = dict(base)
    cfg["strategy"] = "topk_energy"
    cfg["topk_energy"] = {"keep_ratio": 0.25, "score_type": "l2"}
    cfg["packet_loss"] = {"enabled": True, "loss_rate": 0.3, "unit": "cell"}
    run_case("packet_loss", cfg, features, record_len)


if __name__ == "__main__":
    main()

