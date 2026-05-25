import copy

import torch

from src.models.fuse_modules.communication_policy import CommunicationPolicy


def run_case(name, cfg, features, record_len):
    policy = CommunicationPolicy(in_channels=features.shape[1], comm_cfg=cfg)
    out = policy(features, record_len, pairwise_t_matrix=None)
    print(f"\n=== {name} ===")
    print("output shape:", tuple(out.features.shape))
    print("comm_stats:", out.stats)
    print("aux keys:", list(out.aux.keys()))
    return out


def main():
    torch.manual_seed(0)
    features = torch.randn(5, 256, 100, 352)
    record_len = torch.tensor([3, 2])
    ego_idx = [0, 3]

    base = {
        "enabled": True,
        "phase": "phase2",
        "strategy": "none",
        "seed": 42,
        "drop_ego": False,
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

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "random_drop_all_features"
    cfg["drop_random"] = {"keep_ratio": 0.0}
    out_all = run_case("random_drop_all_features", cfg, features.clone(), record_len)

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "random_drop_comm_only"
    cfg["drop_random"] = {"keep_ratio": 0.0}
    out_comm = run_case("random_drop_comm_only", cfg, features.clone(), record_len)

    for i in ego_idx:
        assert torch.equal(out_comm.features[i], features[i]), f"ego index {i} changed in random_drop_comm_only"

    changed_any_ego = any(not torch.equal(out_all.features[i], features[i]) for i in ego_idx)
    assert changed_any_ego, "expected at least one ego feature to change in random_drop_all_features"

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "topk_energy"
    cfg["topk_energy"] = {"keep_ratio": 0.1, "score_type": "l2"}
    run_case("topk_energy", cfg, features.clone(), record_len)

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "none"
    cfg["neighbor_selection"] = {"mode": "topk_importance", "k": 1, "distance_metric": "euclidean"}
    run_case("neighbor_selection(topk_importance)", cfg, features.clone(), record_len)

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "topk_energy"
    cfg["topk_energy"] = {"keep_ratio": 0.25, "score_type": "l2"}
    cfg["packet_loss"] = {"enabled": True, "loss_rate": 0.3, "unit": "cell"}
    run_case("packet_loss", cfg, features.clone(), record_len)

    print("\nAll communication policy fake tests passed.")


if __name__ == "__main__":
    main()
