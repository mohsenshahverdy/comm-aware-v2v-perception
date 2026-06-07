# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch

from src.utils.logging import get_logger
from src.utils.runtime_config import get_communication_cfg


LOGGER = get_logger("CommLoss")
_WARNED_MISSING_LEARNED_TENSOR = False


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item()) if value.numel() else 0.0
    return float(value)


def _learned_prob_mean(aux: dict):
    prob_mean = aux.get("learned_request_prob_mean_tensor", None)
    if prob_mean is not None:
        return prob_mean
    prob = aux.get("learned_request_prob", None)
    if prob is not None:
        return prob.mean()
    return None


def compute_comm_losses(hypes, output_dict, device, logger=None):
    """Compute optional communication auxiliary losses.

    This function is deliberately config-gated. If learned temporal request loss
    is disabled, existing training behavior remains unchanged.
    """

    global _WARNED_MISSING_LEARNED_TENSOR
    comm_cfg = get_communication_cfg(hypes)
    stats = output_dict.get("comm_stats", {})
    aux = output_dict.get("comm_aux", {})
    losses = {}
    total_aux = torch.tensor(0.0, device=device)
    log = logger or LOGGER

    learn_cfg = comm_cfg.get("learnable_mask", {})
    if bool(comm_cfg.get("enabled", False)) and bool(learn_cfg.get("enabled", False)):
        lam_sparse = float(learn_cfg.get("sparsity_lambda", 0.0))
        target_ratio = float(learn_cfg.get("target_ratio", 0.10))
        lam_budget = float(learn_cfg.get("budget_lambda", 0.0))
        use_budget = bool(learn_cfg.get("use_budget_loss", False))
        mask_mean = aux.get("mask_mean", None)
        if mask_mean is not None:
            if lam_sparse > 0:
                l_sparse = lam_sparse * mask_mean
                losses["sparse_loss"] = _to_float(l_sparse)
                # Backward-compatible name
                losses["comm_loss"] = losses["sparse_loss"]
                total_aux = total_aux + l_sparse
            if use_budget and lam_budget > 0:
                l_budget = lam_budget * torch.relu(mask_mean - target_ratio) ** 2
                losses["budget_loss"] = _to_float(l_budget)
                total_aux = total_aux + l_budget

    rr_cfg = comm_cfg.get("receiver_request", {}) if isinstance(comm_cfg.get("receiver_request", {}), dict) else {}
    learned_cfg = rr_cfg.get("learned", {}) if isinstance(rr_cfg.get("learned", {}), dict) else {}
    learned_loss_cfg = learned_cfg.get("loss", {}) if isinstance(learned_cfg.get("loss", {}), dict) else {}
    learned_loss_enabled = (
        bool(comm_cfg.get("enabled", False))
        and bool(learned_cfg.get("enabled", False))
        and bool(learned_loss_cfg.get("enabled", False))
    )
    if learned_loss_enabled:
        prob_mean = _learned_prob_mean(aux)
        if prob_mean is None:
            if not _WARNED_MISSING_LEARNED_TENSOR:
                log.warn(
                    "Learned temporal loss enabled but request probability tensor is missing",
                    expected="learned_request_prob_mean_tensor|learned_request_prob",
                )
                _WARNED_MISSING_LEARNED_TENSOR = True
        else:
            lam_sparse = float(learned_loss_cfg.get("sparse_lambda", learned_loss_cfg.get("lambda_sparse", 0.0)))
            lam_budget = float(learned_loss_cfg.get("budget_lambda", learned_loss_cfg.get("lambda_budget", 0.0)))
            target_ratio = float(learned_loss_cfg.get("target_ratio", learned_loss_cfg.get("target_budget", 0.10)))
            penalty_type = str(learned_loss_cfg.get("budget_penalty_type", "symmetric")).lower()
            target = torch.as_tensor(target_ratio, device=prob_mean.device, dtype=prob_mean.dtype)

            l_sparse = prob_mean * lam_sparse
            budget_error = prob_mean - target
            if penalty_type == "upper_bound":
                l_budget = lam_budget * torch.relu(budget_error) ** 2
            else:
                l_budget = lam_budget * budget_error ** 2
                penalty_type = "symmetric"
            l_total = l_sparse + l_budget

            losses["learned_sparse_loss"] = _to_float(l_sparse)
            losses["learned_budget_loss"] = _to_float(l_budget)
            losses["learned_total_loss"] = _to_float(l_total)
            losses["learned_request_prob_mean"] = _to_float(prob_mean)
            losses["learned_budget_target"] = float(target_ratio)
            losses["learned_budget_error"] = _to_float(budget_error)
            losses["learned_budget_penalty_type"] = penalty_type
            entropy = aux.get("learned_mask_entropy_tensor", None)
            if entropy is not None:
                losses["learned_mask_entropy"] = _to_float(entropy)
            effective_keep = aux.get("learned_effective_keep_ratio_tensor", None)
            if effective_keep is not None:
                losses["learned_effective_keep_ratio"] = _to_float(effective_keep)
            total_aux = total_aux + l_total

    repair_cfg = comm_cfg.get("repair_network", {})
    if bool(comm_cfg.get("enabled", False)) and bool(repair_cfg.get("enabled", False)):
        pred = aux.get("repair_pred", None)
        target = aux.get("repair_target", None)
        w = float(repair_cfg.get("loss_weight", 0.0))
        if pred is not None and target is not None and w > 0:
            l_rep = w * torch.nn.functional.mse_loss(pred, target)
            losses["repair_loss"] = _to_float(l_rep)
            total_aux = total_aux + l_rep

    losses["total_aux_loss"] = _to_float(total_aux)
    return total_aux, losses, stats
