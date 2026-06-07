import torch

from src.models.fuse_modules.learned_temporal_request import (
    DEFAULT_INPUT_MAPS,
    LearnedTemporalRequestHead,
    build_request_input,
    compute_budget_sparse_losses,
    request_prob_from_logits,
    topk_mask_from_prob,
)
from src.utils.logging import get_logger


LOGGER = get_logger("TestLearnedTemporalRequest")


def _fake_maps(batch_size=2, height=8, width=10):
    torch.manual_seed(11)
    maps = {}
    for idx, name in enumerate(DEFAULT_INPUT_MAPS):
        maps[name] = torch.randn(batch_size, 1, height, width) + float(idx) * 0.01
    return maps


def test_head_shape_and_probability_range():
    maps = _fake_maps()
    x = build_request_input(maps)
    assert x.shape == (2, len(DEFAULT_INPUT_MAPS), 8, 10)

    head = LearnedTemporalRequestHead(in_channels=x.shape[1], hidden_channels=16)
    logits = head(x)
    assert logits.shape == (2, 1, 8, 10)

    prob = request_prob_from_logits(logits)
    assert prob.shape == logits.shape
    assert torch.all(prob >= 0.0)
    assert torch.all(prob <= 1.0)


def test_build_request_input_validation():
    maps = _fake_maps()
    missing = dict(maps)
    missing.pop("novelty")
    try:
        build_request_input(missing)
    except KeyError as exc:
        assert "novelty" in str(exc)
    else:
        raise AssertionError("Missing map did not raise KeyError")

    bad_shape = dict(maps)
    bad_shape["novelty"] = torch.randn(2, 1, 7, 10)
    try:
        build_request_input(bad_shape)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Shape mismatch did not raise ValueError")


def test_hard_topk_mask_exact_ratio_for_unique_scores():
    # Unique scores avoid threshold ties, so the exact selected count is stable.
    prob = torch.arange(2 * 1 * 4 * 5, dtype=torch.float32).view(2, 1, 4, 5)
    prob = prob / prob.max()
    mask = topk_mask_from_prob(prob, keep_ratio=0.25)
    assert mask.shape == prob.shape
    assert torch.all(mask.view(2, -1).sum(dim=1) == 5)

    zero_ratio_mask = topk_mask_from_prob(prob, keep_ratio=0.0)
    assert torch.all(zero_ratio_mask.view(2, -1).sum(dim=1) == 1)

    full_mask = topk_mask_from_prob(prob, keep_ratio=1.0)
    assert torch.all(full_mask == 1.0)


def test_budget_sparse_loss_and_gradients():
    maps = _fake_maps(batch_size=1, height=6, width=6)
    x = build_request_input(maps)
    head = LearnedTemporalRequestHead(in_channels=x.shape[1], hidden_channels=8)

    logits = head(x)
    prob = request_prob_from_logits(logits)
    losses = compute_budget_sparse_losses(
        prob,
        target_budget=0.10,
        lambda_budget=0.2,
        lambda_sparse=0.01,
    )

    assert losses["learned_comm_loss_total"].requires_grad
    assert losses["learned_budget_loss"].item() >= 0.0
    assert losses["learned_sparse_loss"].item() >= 0.0
    assert "learned_mask_entropy" in losses

    losses["learned_comm_loss_total"].backward()
    grad_norm = 0.0
    for param in head.parameters():
        assert param.grad is not None
        grad_norm += float(param.grad.detach().abs().sum().item())
    assert grad_norm > 0.0


def main():
    test_head_shape_and_probability_range()
    test_build_request_input_validation()
    test_hard_topk_mask_exact_ratio_for_unique_scores()
    test_budget_sparse_loss_and_gradients()
    LOGGER.success("Learned temporal request head tests passed")


if __name__ == "__main__":
    main()
