import copy

import torch

from src.models.fuse_modules.communication_policy import CommunicationPolicy
from src.utils.logging import get_logger


logger = get_logger("TestCommPolicy")


def run_case(name, cfg, features, record_len):
    policy = CommunicationPolicy(in_channels=features.shape[1], comm_cfg=cfg)
    out = policy(features, record_len, pairwise_t_matrix=None)
    logger.step("Running fake case", case=name)
    logger.info("Output shape", case=name, shape=tuple(out.features.shape))
    logger.metric("Communication stats", case=name, stats=out.stats)
    logger.debug("Aux keys", case=name, keys=list(out.aux.keys()))
    return out


def _assert_ego_unchanged(out, features, ego_idx, case_name):
    for i in ego_idx:
        assert torch.equal(out.features[i], features[i]), f"ego index {i} changed in {case_name}"


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
        "bytes_per_value": 4.0,
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
        "receiver_request": {
            "enabled": True,
            "strategy_variant": "energy_topk",
            "trainable": False,
            "keep_ratio": 0.10,
            "score_type": "multiplicative",
            "normalize_scores": True,
            "ego_need_type": "inverse_energy",
            "collaborator_context_type": "l2",
            "context_resolution": "full",
            "context_quantization_bits": 32,
            "count_context_overhead": True,
            "count_mask_metadata": True,
            "metadata_encoding": "dense_binary",
            "alignment_mode": "ego_aligned",
        },
    }

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "random_drop_all_features"
    cfg["drop_random"] = {"keep_ratio": 0.0}
    out_all = run_case("random_drop_all_features", cfg, features.clone(), record_len)

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "random_drop_comm_only"
    cfg["drop_random"] = {"keep_ratio": 0.0}
    out_comm = run_case("random_drop_comm_only", cfg, features.clone(), record_len)

    _assert_ego_unchanged(out_comm, features, ego_idx, "random_drop_comm_only")

    changed_any_ego = any(not torch.equal(out_all.features[i], features[i]) for i in ego_idx)
    assert changed_any_ego, "expected at least one ego feature to change in random_drop_all_features"

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "topk_energy"
    cfg["topk_energy"] = {"keep_ratio": 0.1, "score_type": "l2"}
    out_topk = run_case("topk_energy", cfg, features.clone(), record_len)
    _assert_ego_unchanged(out_topk, features, ego_idx, "topk_energy")

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "none"
    cfg["neighbor_selection"] = {"mode": "topk_importance", "k": 1, "distance_metric": "euclidean"}
    out_neighbor = run_case("neighbor_selection(topk_importance)", cfg, features.clone(), record_len)
    _assert_ego_unchanged(out_neighbor, features, ego_idx, "neighbor selection")

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "topk_energy"
    cfg["topk_energy"] = {"keep_ratio": 0.25, "score_type": "l2"}
    cfg["packet_loss"] = {"enabled": True, "loss_rate": 0.3, "unit": "cell"}
    out_packet = run_case("packet_loss", cfg, features.clone(), record_len)
    _assert_ego_unchanged(out_packet, features, ego_idx, "packet_loss")

    cfg = copy.deepcopy(base)
    cfg["strategy"] = "receiver_request_topk"
    cfg["receiver_request"]["keep_ratio"] = 0.10
    out_rr = run_case("receiver_request_topk", cfg, features.clone(), record_len)
    _assert_ego_unchanged(out_rr, features, ego_idx, "receiver_request_topk")

    # collaborator-only metric sanity checks
    # record_len [3,2] -> collaborators are indices [1,2,4] => 3 collaborators
    c, h, w = features.shape[1], features.shape[2], features.shape[3]
    per_collab_full = c * h * w * 4.0
    expected_selected_collab = 2 * per_collab_full
    assert abs(out_neighbor.stats["feature_bytes_per_frame"] - expected_selected_collab) < 1e-3
    assert abs(out_neighbor.stats["active_ratio"] - 1.0) < 1e-6
    assert abs(out_neighbor.stats["active_neighbors_ratio"] - (2.0 / 3.0)) < 1e-6

    # receiver-request metrics should exist
    for key in [
        "context_bytes_per_frame",
        "metadata_bytes_per_frame",
        "total_bytes_per_frame",
        "normalized_ratio",
        "receiver_request_keep_ratio",
        "receiver_request_context_ratio",
        "receiver_request_mask_metadata_ratio",
    ]:
        assert key in out_rr.stats, f"missing metric key: {key}"

    # keep-ratio approximation for receiver-request collaborator cells only
    rr_ratio = float(out_rr.stats["active_ratio"])
    assert 0.07 <= rr_ratio <= 0.13, f"receiver_request keep ratio off target: {rr_ratio}"

    # receiver-request should produce a different mask pattern than sender-only top-k
    diff = (out_rr.features - out_topk.features).abs().sum().item()
    assert diff > 0.0, "receiver_request_topk output unexpectedly identical to topk_energy"

    # grouping check for record_len=[3,2]: first in each group is ego and must be unchanged.
    assert torch.equal(out_rr.features[0], features[0]) and torch.equal(out_rr.features[3], features[3])

    logger.success("All communication policy fake tests passed")


if __name__ == "__main__":
    main()
