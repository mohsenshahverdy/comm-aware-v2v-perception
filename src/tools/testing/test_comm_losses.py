import copy

import torch

from src.models.fuse_modules.communication_policy import CommunicationPolicy
from src.tools.communication_losses import compute_comm_losses
from src.tools.testing.test_learned_temporal_policy import _base_cfg as _learned_policy_cfg
from src.tools.testing.test_learned_temporal_policy import _features, _metadata
from src.utils.logging import get_logger


logger = get_logger("TestCommLosses")


def _hypes(comm_cfg):
    return {"model": {"args": {"communication": comm_cfg}}}


def _learned_comm_cfg(prob_target=0.10, sparse_lambda=0.05, budget_lambda=1.0, penalty_type="symmetric"):
    cfg = _learned_policy_cfg(keep_ratio=prob_target, use_soft_mask_train=True)
    cfg["receiver_request"]["learned"]["loss"] = {
        "enabled": True,
        "sparse_lambda": sparse_lambda,
        "budget_lambda": budget_lambda,
        "target_ratio": prob_target,
        "budget_penalty_type": penalty_type,
        "apply_to": "collaborators_only",
    }
    return cfg


def _assert_close(actual, expected, name, atol=1e-6):
    assert abs(float(actual) - float(expected)) <= atol, f"{name}: expected {expected}, got {actual}"


def test_disabled_learned_zero_aux(device):
    hypes = _hypes({"enabled": False, "strategy": "none"})
    aux_total, losses, _ = compute_comm_losses(hypes, {"comm_aux": {}, "comm_stats": {}}, device)
    assert aux_total.item() == 0.0
    assert losses["total_aux_loss"] == 0.0


def test_missing_learned_tensor_warns_and_zero(device):
    hypes = _hypes(_learned_comm_cfg())
    aux_total, losses, _ = compute_comm_losses(hypes, {"comm_aux": {}, "comm_stats": {}}, device)
    assert aux_total.item() == 0.0
    assert losses["total_aux_loss"] == 0.0
    assert "learned_total_loss" not in losses


def test_learned_sparse_and_symmetric_budget_math(device):
    prob_mean = torch.tensor(0.20, device=device, requires_grad=True)
    hypes = _hypes(_learned_comm_cfg(prob_target=0.10, sparse_lambda=0.05, budget_lambda=1.0, penalty_type="symmetric"))
    aux_total, losses, _ = compute_comm_losses(
        hypes,
        {"comm_aux": {"learned_request_prob_mean_tensor": prob_mean}, "comm_stats": {}},
        device,
    )
    _assert_close(losses["learned_sparse_loss"], 0.010000, "learned_sparse_loss")
    _assert_close(losses["learned_budget_loss"], 0.010000, "learned_budget_loss")
    _assert_close(losses["learned_total_loss"], 0.020000, "learned_total_loss")
    aux_total.backward()
    assert prob_mean.grad is not None and prob_mean.grad.item() > 0.0


def test_upper_bound_budget_math(device):
    prob_mean = torch.tensor(0.05, device=device, requires_grad=True)
    hypes = _hypes(_learned_comm_cfg(prob_target=0.10, sparse_lambda=0.05, budget_lambda=1.0, penalty_type="upper_bound"))
    aux_total, losses, _ = compute_comm_losses(
        hypes,
        {"comm_aux": {"learned_request_prob_mean_tensor": prob_mean}, "comm_stats": {}},
        device,
    )
    _assert_close(losses["learned_sparse_loss"], 0.0025, "learned_sparse_loss")
    _assert_close(losses["learned_budget_loss"], 0.0, "learned_budget_loss")
    _assert_close(losses["learned_total_loss"], 0.0025, "learned_total_loss")
    aux_total.backward()
    assert prob_mean.grad is not None


def test_old_learnable_mask_loss_unchanged(device):
    mask_mean = torch.tensor(0.20, device=device, requires_grad=True)
    comm_cfg = {
        "enabled": True,
        "strategy": "learnable_mask",
        "learnable_mask": {
            "enabled": True,
            "sparsity_lambda": 0.10,
            "target_ratio": 0.10,
            "budget_lambda": 1.0,
            "use_budget_loss": True,
        },
    }
    aux_total, losses, _ = compute_comm_losses(
        _hypes(comm_cfg),
        {"comm_aux": {"mask_mean": mask_mean}, "comm_stats": {}},
        device,
    )
    _assert_close(losses["sparse_loss"], 0.020000, "sparse_loss")
    _assert_close(losses["budget_loss"], 0.010000, "budget_loss")
    _assert_close(losses["total_aux_loss"], 0.030000, "total_aux_loss")
    aux_total.backward()
    assert mask_mean.grad is not None


def test_repair_loss_still_works(device):
    pred = torch.zeros(1, 1, 2, 2, device=device, requires_grad=True)
    target = torch.ones(1, 1, 2, 2, device=device)
    comm_cfg = {
        "enabled": True,
        "strategy": "topk_energy",
        "repair_network": {"enabled": True, "loss_weight": 0.5},
    }
    aux_total, losses, _ = compute_comm_losses(
        _hypes(comm_cfg),
        {"comm_aux": {"repair_pred": pred, "repair_target": target}, "comm_stats": {}},
        device,
    )
    _assert_close(losses["repair_loss"], 0.5, "repair_loss")
    aux_total.backward()
    assert pred.grad is not None


def test_learned_temporal_policy_loss_is_differentiable(device):
    torch.manual_seed(9)
    comm_cfg = _learned_comm_cfg(prob_target=0.25, sparse_lambda=0.05, budget_lambda=1.0, penalty_type="symmetric")
    policy = CommunicationPolicy(in_channels=4, comm_cfg=copy.deepcopy(comm_cfg)).to(device)
    policy.train()
    record_len = torch.tensor([2], device=device)
    features = _features(scale=2.0).to(device)
    output = policy(features, record_len, metadata=_metadata("scenario_loss", "000001"))
    aux_total, losses, _ = compute_comm_losses(
        _hypes(comm_cfg),
        {"comm_aux": output.aux, "comm_stats": output.stats},
        device,
    )
    assert losses["learned_total_loss"] >= 0.0
    aux_total.backward()
    grad_sum = 0.0
    for param in policy.learned_temporal_request_head.parameters():
        if param.grad is not None:
            grad_sum += float(param.grad.detach().abs().sum().cpu().item())
    assert grad_sum > 0.0, "learned request head did not receive gradients from compute_comm_losses"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_disabled_learned_zero_aux(device)
    test_missing_learned_tensor_warns_and_zero(device)
    test_learned_sparse_and_symmetric_budget_math(device)
    test_upper_bound_budget_math(device)
    test_old_learnable_mask_loss_unchanged(device)
    test_repair_loss_still_works(device)
    test_learned_temporal_policy_loss_is_differentiable(device)
    logger.success("Communication loss tests passed")


if __name__ == "__main__":
    main()
