import copy
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models.fuse_modules.communication_policy import CommunicationPolicy
from src.utils.logging import get_logger


logger = get_logger("TestTemporalReceiverRequest")


def _base_cfg(debug_dir=None, init_mode="full_request", init_keep_ratio=1.0, keep_ratio=0.25):
    temporal = {
        "enabled": True,
        "init_mode": init_mode,
        "init_frames": 1,
        "init_keep_ratio": init_keep_ratio,
        "cache_type": "context_energy",
        "cache_momentum": 0.5,
        "cache_confidence_decay": 0.8,
        "max_cache_age": 5,
        "novelty_type": "absolute_diff",
        "novelty_weight": 1.0,
        "age_weight": 0.05,
        "cache_confidence_weight": 0.5,
        "min_temporal_factor": 0.25,
        "max_temporal_factor": 3.0,
        "periodic_refresh_enabled": False,
        "periodic_refresh_interval": 10,
        "periodic_refresh_keep_ratio": 0.5,
        "save_temporal_maps": debug_dir is not None,
        "debug_num_frames": 4,
    }
    if debug_dir is not None:
        temporal["debug_dir"] = str(debug_dir)
    return {
        "enabled": True,
        "strategy": "receiver_request_topk",
        "seed": 42,
        "drop_ego": False,
        "bytes_per_value": 4.0,
        "neighbor_selection": {"mode": "all", "k": 0},
        "packet_loss": {"enabled": False, "loss_rate": 0.0},
        "learnable_mask": {"enabled": False, "mask_channels": 4},
        "repair_network": {"enabled": False, "hidden_dim": 8},
        "receiver_request": {
            "enabled": True,
            "implementation_status": "implemented",
            "strategy_variant": "temporal_energy_topk",
            "trainable": False,
            "keep_ratio": keep_ratio,
            "score_type": "multiplicative",
            "normalize_scores": True,
            "ego_need_type": "inverse_energy",
            "collaborator_context_type": "l2",
            "alignment_mode": "ego_aligned",
            "context_resolution": "full",
            "context_quantization_bits": 32,
            "count_context_overhead": True,
            "count_mask_metadata": True,
            "metadata_encoding": "dense_binary",
            "temporal": temporal,
            "loss": {"enabled": False},
            "optimizer": {"separate_lr": False},
        },
    }


def _metadata(scenario_id="scenario_a", timestamp="000001"):
    return [{
        "sample_idx": 0,
        "scenario_index": 0,
        "scenario_id": scenario_id,
        "timestamp": timestamp,
        "frame_id": timestamp,
        "ego_id": "ego",
        "cav_ids": ["ego", "cav_1"],
        "record_len": 2,
    }]


def _features(scale=1.0):
    torch.manual_seed(7)
    feat = torch.randn(2, 4, 4, 4)
    feat[1, :, 2:, 2:] *= scale
    return feat


def main():
    record_len = torch.tensor([2])

    with tempfile.TemporaryDirectory() as tmp:
        debug_dir = Path(tmp) / "temporal_receiver_request_debug"
        policy = CommunicationPolicy(in_channels=4, comm_cfg=_base_cfg(debug_dir=debug_dir, init_mode="full_request", keep_ratio=0.25))
        first = _features(scale=1.0)
        out1 = policy(first.clone(), record_len, metadata=_metadata("scenario_a", "000001"))

        assert torch.equal(out1.features[0], first[0]), "ego changed on init frame"
        assert torch.equal(out1.features[1], first[1]), "full_request init should keep collaborator unchanged"
        assert out1.stats["temporal_init_frame_ratio"] == 1.0
        assert out1.stats["temporal_cache_hit_ratio"] == 0.0
        assert out1.stats["temporal_cache_entries"] == 1

        second = _features(scale=4.0)
        out2 = policy(second.clone(), record_len, metadata=_metadata("scenario_a", "000002"))
        assert torch.equal(out2.features[0], second[0]), "ego changed after cache exists"
        assert not torch.equal(out2.features[1], second[1]), "non-init temporal top-k should mask collaborator"
        assert out2.stats["temporal_cache_hit_ratio"] == 1.0
        assert out2.stats["temporal_init_frame_ratio"] == 0.0
        assert out2.stats["temporal_novelty_mean"] >= 0.0
        assert out2.stats["comm_average_bytes_per_frame" if "comm_average_bytes_per_frame" in out2.stats else "average_bytes_per_frame"] >= 0.0

        entry_a = policy.temporal_cache.get("scenario_a", "ego", "cav_1")
        assert entry_a is not None
        assert entry_a.update_count == 2

        third = _features(scale=2.0)
        policy(third.clone(), record_len, metadata=_metadata("scenario_b", "000001"))
        assert policy.temporal_cache.get("scenario_b", "ego", "cav_1") is not None
        assert len(policy.temporal_cache) == 2, "scenario cache entries leaked/overwrote each other"

        debug_files = sorted(debug_dir.glob("*.npz"))
        assert debug_files, "temporal debug npz was not written"
        payload = np.load(debug_files[0])
        for key in [
            "ego_need_map",
            "collaborator_context_map",
            "previous_cache_map",
            "novelty_map",
            "temporal_factor_map",
            "request_score_map",
            "request_mask",
            "cache_age_map",
            "cache_confidence_map",
        ]:
            assert key in payload, f"missing debug map: {key}"

    high_cfg = _base_cfg(init_mode="high_ratio", init_keep_ratio=0.5, keep_ratio=0.25)
    high_policy = CommunicationPolicy(in_channels=4, comm_cfg=copy.deepcopy(high_cfg))
    high_features = _features(scale=1.0)
    high_out = high_policy(high_features.clone(), record_len, metadata=_metadata("scenario_high", "000001"))
    assert torch.equal(high_out.features[0], high_features[0]), "ego changed in high-ratio init"
    assert abs(high_out.stats["active_ratio"] - 0.5) < 1e-6

    logger.success("Temporal receiver-request tests passed")


if __name__ == "__main__":
    main()
