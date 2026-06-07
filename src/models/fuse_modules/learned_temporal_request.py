from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn


DEFAULT_INPUT_MAPS = (
    "ego_need",
    "collaborator_context",
    "previous_cache",
    "novelty",
    "cache_age",
    "cache_confidence",
)


class LearnedTemporalRequestHead(nn.Module):
    """Small CNN that predicts receiver-side request logits from temporal maps.

    This module is intentionally standalone. The communication policy will call
    it in a later integration stage; keeping it separate makes the learned
    request mechanism testable without changing current non-learned policies.
    """

    def __init__(self, in_channels: int = 6, hidden_channels: int = 16):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_request_input(
    maps: Mapping[str, torch.Tensor],
    input_maps: Iterable[str] = DEFAULT_INPUT_MAPS,
) -> torch.Tensor:
    """Concatenate named [B, 1, H, W] request maps into [B, C, H, W]."""

    tensors = []
    reference_shape: Tuple[int, int, int, int] = None
    for name in input_maps:
        if name not in maps:
            raise KeyError(f"Missing learned temporal request input map: {name}")
        tensor = maps[name]
        if tensor.dim() != 4:
            raise ValueError(f"Input map must be 4D [B, C, H, W]: {name} shape={tuple(tensor.shape)}")
        if tensor.shape[1] != 1:
            raise ValueError(f"Input map must have one channel: {name} shape={tuple(tensor.shape)}")
        if reference_shape is None:
            reference_shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != reference_shape:
            raise ValueError(
                f"All input maps must have the same shape. "
                f"Expected {reference_shape}, got {tuple(tensor.shape)} for {name}"
            )
        tensors.append(tensor)
    return torch.cat(tensors, dim=1)


def request_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def topk_mask_from_prob(prob: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Build a hard top-k mask per sample using the existing policy convention."""

    keep_ratio = float(max(min(float(keep_ratio), 1.0), 0.0))
    if prob.dim() != 4:
        raise ValueError(f"Request probability must be 4D [B, 1, H, W], got {tuple(prob.shape)}")
    if prob.shape[0] == 0:
        return torch.ones_like(prob)
    if keep_ratio >= 1.0:
        return torch.ones_like(prob)

    flat = prob.view(prob.shape[0], -1)
    if keep_ratio <= 0.0:
        k = 1
    else:
        k = max(1, int(flat.shape[1] * keep_ratio))
    topk_vals, _ = torch.topk(flat, k=k, dim=1)
    threshold = topk_vals[:, -1].view(-1, 1, 1, 1)
    return (prob >= threshold).to(prob.dtype)


def request_mask_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.clamp(prob, eps, 1.0 - eps)
    entropy = -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))
    return entropy.mean()


def compute_budget_sparse_losses(
    request_prob: torch.Tensor,
    *,
    target_budget: float = 0.10,
    lambda_budget: float = 0.0,
    lambda_sparse: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Return differentiable learned-request loss terms.

    This helper is not wired into train.py yet. It exists so Stage 1 can verify
    the loss math and gradient flow before policy/training integration.
    """

    prob_mean = request_prob.mean()
    target = torch.as_tensor(float(target_budget), device=request_prob.device, dtype=request_prob.dtype)
    budget_loss = float(lambda_budget) * (prob_mean - target).pow(2)
    sparse_loss = float(lambda_sparse) * prob_mean
    total = budget_loss + sparse_loss
    return {
        "learned_request_prob_mean": prob_mean,
        "learned_budget_loss": budget_loss,
        "learned_sparse_loss": sparse_loss,
        "learned_comm_loss_total": total,
        "learned_budget_error": prob_mean - target,
        "learned_mask_entropy": request_mask_entropy(request_prob),
    }
