import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models.fuse_modules.communication_policy import CommunicationPolicy
from src.models.fuse_modules.learned_temporal_request import DEFAULT_INPUT_MAPS
from src.tools.communication_losses import compute_comm_losses
from src.utils.logging import get_logger


logger = get_logger("TestLearnedTemporalPolicy")


def _base_cfg(debug_dir=None, keep_ratio=0.25, use_soft_mask_train=True):
    learned_debug = {
        "save_learned_maps": debug_dir is not None,
        "debug_num_frames": 4,
    }
    if debug_dir is not None:
        learned_debug["debug_dir"] = str(debug_dir)

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
            "strategy_variant": "learned_temporal",
            "trainable": True,
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
            "temporal": {
                "enabled": True,
                "cache_momentum": 0.5,
                "cache_confidence_decay": 0.8,
                "novelty_type": "absolute_diff",
                "save_temporal_maps": False,
            },
            "learned": {
                "enabled": True,
                "head_type": "small_cnn",
                "input_maps": list(DEFAULT_INPUT_MAPS),
                "hidden_channels": 8,
                "use_soft_mask_train": use_soft_mask_train,
                "use_hard_topk_inference": True,
                "straight_through": False,
                "target_budget": keep_ratio,
                "loss": {
                    "enabled": True,
                    "target_budget": keep_ratio,
                    "lambda_budget": 0.1,
                    "lambda_sparse": 0.01,
                },
                "debug": learned_debug,
            },
            "loss": {"enabled": False},
            "optimizer": {"separate_lr": False},
        },
    }


def _metadata(scenario_id="scenario_a", timestamp="000001", cav_ids=None):
    cav_ids = cav_ids or ["ego", "cav_1"]
    return [{
        "sample_idx": 0,
        "scenario_index": 0,
        "scenario_id": scenario_id,
        "timestamp": timestamp,
        "frame_id": timestamp,
        "ego_id": "ego",
        "cav_ids": cav_ids,
        "record_len": len(cav_ids),
    }]


def _features(num_cav=2, scale=1.0):
    torch.manual_seed(17)
    feat = torch.randn(num_cav, 4, 4, 4)
    feat[1:, :, 2:, 2:] *= scale
    return feat


def _assert_grad_exists(policy):
    grad_sum = 0.0
    for param in policy.learned_temporal_request_head.parameters():
        if param.grad is not None:
            grad_sum += float(param.grad.detach().abs().sum().cpu().item())
    assert grad_sum > 0.0, "learned temporal request head did not receive gradients"


def _hypes_for_policy(comm_cfg):
    return {
        "model": {
            "args": {
                "communication": comm_cfg,
            }
        }
    }


def main():
    record_len = torch.tensor([2])

    # Training mode: soft mask is differentiable and keeps ego unchanged.
    train_policy = CommunicationPolicy(in_channels=4, comm_cfg=_base_cfg(keep_ratio=0.25, use_soft_mask_train=True))
    train_policy.train()
    train_features = _features(scale=2.0)
    train_out = train_policy(train_features.clone(), record_len, metadata=_metadata("scenario_train", "000001"))

    assert torch.equal(train_out.features[0], train_features[0]), "ego changed in learned temporal training path"
    assert not torch.equal(train_out.features[1], train_features[1]), "collaborator was not masked in learned temporal training path"
    for key in [
        "learned_request_prob",
        "learned_request_logits",
        "learned_request_mask",
        "learned_request_prob_mean_tensor",
        "learned_effective_keep_ratio_tensor",
        "learned_budget_error_tensor",
        "learned_mask_entropy_tensor",
    ]:
        assert key in train_out.aux, f"missing learned aux tensor: {key}"
    assert train_out.aux["learned_request_prob_mean_tensor"].requires_grad, "soft probability mean lost gradients"
    train_out.aux["learned_request_prob_mean_tensor"].backward()
    _assert_grad_exists(train_policy)

    # Full training-style loss: detector feature loss + communication aux loss.
    # This catches in-place view writes that simple aux-only backward can miss.
    backward_policy = CommunicationPolicy(in_channels=4, comm_cfg=_base_cfg(keep_ratio=0.25, use_soft_mask_train=True))
    backward_policy.train()
    backward_features = _features(scale=2.0).requires_grad_(True)
    backward_out = backward_policy(
        backward_features,
        record_len,
        metadata=_metadata("scenario_backward", "000001"),
    )
    aux_total, aux_breakdown, _ = compute_comm_losses(
        _hypes_for_policy(_base_cfg(keep_ratio=0.25, use_soft_mask_train=True)),
        {"comm_aux": backward_out.aux, "comm_stats": backward_out.stats},
        backward_features.device,
    )
    assert aux_breakdown.get("learned_total_loss", 0.0) > 0.0, "learned temporal aux loss was not active"
    full_loss = backward_out.features.square().mean() + aux_total
    full_loss.backward()
    _assert_grad_exists(backward_policy)

    for key in [
        "learned_request_prob_mean",
        "learned_request_prob_std",
        "learned_effective_keep_ratio",
        "learned_budget_target",
        "learned_budget_error",
        "learned_mask_entropy",
        "temporal_cache_hit_ratio",
        "temporal_novelty_mean",
    ]:
        assert key in train_out.stats, f"missing learned/temporal stat: {key}"

    # Inference mode: hard top-k mask should approximately match keep_ratio.
    eval_policy = CommunicationPolicy(in_channels=4, comm_cfg=_base_cfg(keep_ratio=0.25, use_soft_mask_train=True))
    eval_policy.eval()
    eval_features = _features(scale=3.0)
    eval_out = eval_policy(eval_features.clone(), record_len, metadata=_metadata("scenario_eval", "000001"))
    assert torch.equal(eval_out.features[0], eval_features[0]), "ego changed in learned temporal inference path"
    assert not torch.equal(eval_out.features[1], eval_features[1]), "collaborator was not masked in learned temporal inference path"
    assert abs(eval_out.stats["learned_effective_keep_ratio"] - 0.25) < 1e-6
    assert abs(eval_out.stats["active_ratio"] - 0.25) < 1e-6

    # Cache keys stay isolated by scenario.
    eval_policy(_features(scale=4.0), record_len, metadata=_metadata("scenario_eval", "000002"))
    eval_policy(_features(scale=4.0), record_len, metadata=_metadata("scenario_other", "000001"))
    assert eval_policy.temporal_cache.get("scenario_eval", "ego", "cav_1") is not None
    assert eval_policy.temporal_cache.get("scenario_other", "ego", "cav_1") is not None
    assert len(eval_policy.temporal_cache) == 2, "learned temporal cache keys leaked across scenarios"

    # Debug export is off by default and writes when explicitly enabled.
    default_debug_policy = CommunicationPolicy(in_channels=4, comm_cfg=_base_cfg(keep_ratio=0.25))
    with tempfile.TemporaryDirectory() as tmp:
        debug_dir = Path(tmp) / "learned_temporal_receiver_request_debug"
        debug_policy = CommunicationPolicy(in_channels=4, comm_cfg=_base_cfg(debug_dir=debug_dir, keep_ratio=0.25))
        debug_policy.eval()
        debug_policy(_features(scale=1.0), record_len, metadata=_metadata("scenario_debug", "000001"))
        debug_files = sorted(debug_dir.glob("*.npz"))
        assert debug_files, "learned temporal debug npz was not written"
        payload = np.load(debug_files[0])
        for key in [
            "learned_request_logits",
            "learned_request_prob",
            "final_request_mask",
            "novelty_map",
            "previous_cache_map",
            "cache_age_map",
            "cache_confidence_map",
        ]:
            assert key in payload, f"missing learned debug map: {key}"

    assert default_debug_policy.learned_temporal_cfg.get("debug", {}).get("save_learned_maps", False) is False
    logger.success("Learned temporal policy tests passed")


if __name__ == "__main__":
    main()
