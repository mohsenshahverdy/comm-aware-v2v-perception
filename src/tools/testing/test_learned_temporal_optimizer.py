import torch
import torch.nn as nn

from src.tools import train_utils
from src.tools.learned_request_training_utils import (
    build_learned_temporal_param_groups,
    configure_learned_temporal_freezing,
    enforce_request_head_lr,
    get_learned_request_head,
    is_learned_temporal_training_enabled,
    should_disable_learned_temporal_scheduler,
)
from src.utils.logging import get_logger


logger = get_logger("TestLearnedTemporalOptimizer")


class FakeCommPolicy(nn.Module):
    def __init__(self, with_head=True):
        super().__init__()
        self.learned_temporal_request_head = nn.Sequential(
            nn.Conv2d(6, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1),
        ) if with_head else None


class FakeModel(nn.Module):
    def __init__(self, with_head=True):
        super().__init__()
        self.pillar_vfe = nn.Linear(2, 2)
        self.scatter = nn.Linear(2, 2)
        self.backbone = nn.Linear(2, 2)
        self.shrink_conv = nn.Linear(2, 2)
        self.cls_head = nn.Linear(2, 1)
        self.reg_head = nn.Linear(2, 1)
        self.other = nn.Linear(2, 2)
        self.comm_policy = FakeCommPolicy(with_head=with_head)


class FakeNoHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.other = nn.Linear(2, 2)


def _hypes(optimizer_cfg=None, learned_enabled=True):
    learned = {"enabled": learned_enabled}
    if optimizer_cfg is not None:
        learned["optimizer"] = optimizer_cfg
    return {
        "optimizer": {"core_method": "Adam", "lr": 0.001},
        "lr_scheduler": {"core_method": "multistep", "step_size": [1], "gamma": 0.1},
        "model": {
            "args": {
                "communication": {
                    "enabled": True,
                    "strategy": "receiver_request_topk",
                    "receiver_request": {
                        "enabled": True,
                        "strategy_variant": "learned_temporal",
                        "learned": learned,
                    },
                }
            }
        },
    }


def _all_params(model):
    return list(model.parameters())


def _trainable_names(model):
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _param_ids(params):
    return {id(p) for p in params}


def _assert_no_duplicate_params(groups):
    seen = set()
    for group in groups:
        for p in group["params"]:
            pid = id(p)
            assert pid not in seen, "duplicate parameter in optimizer groups"
            seen.add(pid)


def test_default_behavior_unchanged():
    model = FakeModel(with_head=True)
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    hypes = _hypes(optimizer_cfg=None)
    assert not is_learned_temporal_training_enabled(hypes)
    configure_learned_temporal_freezing(model, hypes)
    after = {name: p.requires_grad for name, p in model.named_parameters()}
    assert before == after, "default path changed requires_grad flags"
    groups = build_learned_temporal_param_groups(model, hypes, base_lr=0.001)
    assert groups is None, "default path should not create special param groups"
    optimizer = train_utils.setup_optimizer(hypes, model)
    assert len(optimizer.param_groups) == 1


def test_disabled_learned_optimizer_flags_do_nothing():
    model = FakeModel(with_head=True)
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    hypes = _hypes({"separate_lr": True, "train_request_head_only": True}, learned_enabled=False)
    assert not is_learned_temporal_training_enabled(hypes)
    configure_learned_temporal_freezing(model, hypes)
    groups = build_learned_temporal_param_groups(model, hypes, base_lr=0.001)
    after = {name: p.requires_grad for name, p in model.named_parameters()}
    assert groups is None
    assert before == after, "learned.enabled=false should ignore learned optimizer flags"


def test_train_request_head_only():
    model = FakeModel(with_head=True)
    hypes = _hypes({"train_request_head_only": True, "request_head_lr": 0.0001})
    configure_learned_temporal_freezing(model, hypes)
    head = get_learned_request_head(model)
    head_ids = _param_ids(head.parameters())
    for name, p in model.named_parameters():
        assert p.requires_grad == (id(p) in head_ids), f"bad requires_grad for {name}"
    groups = build_learned_temporal_param_groups(model, hypes, base_lr=0.001)
    assert len(groups) == 1
    assert float(groups[0]["lr"]) == 0.0001
    assert groups[0]["name"] == "learned_temporal_request_head"
    assert groups[0]["is_learned_temporal_request_head"] is True


def test_separate_lr_parameter_groups():
    model = FakeModel(with_head=True)
    hypes = _hypes({"separate_lr": True, "request_head_lr": 0.0001})
    configure_learned_temporal_freezing(model, hypes)
    groups = build_learned_temporal_param_groups(model, hypes, base_lr=0.001)
    assert len(groups) == 2
    _assert_no_duplicate_params(groups)
    assert float(groups[0]["lr"]) == 0.001
    assert float(groups[1]["lr"]) == 0.0001
    assert groups[1]["name"] == "learned_temporal_request_head"
    head_ids = _param_ids(get_learned_request_head(model).parameters())
    assert all(id(p) in head_ids for p in groups[1]["params"])


def test_request_head_only_disables_scheduler_and_keeps_lr():
    model = FakeModel(with_head=True)
    hypes = _hypes({"train_request_head_only": True, "request_head_lr": 0.001})
    optimizer = train_utils.setup_optimizer(hypes, model)
    assert should_disable_learned_temporal_scheduler(hypes)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, n_iter_per_epoch=1)
    assert scheduler is None
    optimizer.param_groups[0]["lr"] = 1e-7
    changed = enforce_request_head_lr(optimizer, hypes)
    assert changed
    assert float(optimizer.param_groups[0]["lr"]) == 0.001
    assert float(optimizer.param_groups[0]["initial_lr"]) == 0.001


def test_freeze_backbone():
    model = FakeModel(with_head=True)
    hypes = _hypes({"freeze_backbone": True})
    configure_learned_temporal_freezing(model, hypes)
    for module_name in ["pillar_vfe", "scatter", "backbone", "shrink_conv"]:
        assert all(not p.requires_grad for p in getattr(model, module_name).parameters())
    assert any(p.requires_grad for p in get_learned_request_head(model).parameters())
    assert any(p.requires_grad for p in model.cls_head.parameters())


def test_freeze_detector():
    model = FakeModel(with_head=True)
    hypes = _hypes({"freeze_detector": True})
    configure_learned_temporal_freezing(model, hypes)
    for module_name in ["cls_head", "reg_head"]:
        assert all(not p.requires_grad for p in getattr(model, module_name).parameters())
    assert any(p.requires_grad for p in get_learned_request_head(model).parameters())
    assert any(p.requires_grad for p in model.backbone.parameters())


def test_missing_request_head_error():
    for cfg in [{"separate_lr": True}, {"train_request_head_only": True}]:
        model = FakeNoHeadModel()
        hypes = _hypes(cfg)
        try:
            configure_learned_temporal_freezing(model, hypes)
            build_learned_temporal_param_groups(model, hypes, base_lr=0.001)
        except ValueError as exc:
            assert "request head" in str(exc).lower() or "learned temporal" in str(exc).lower()
        else:
            raise AssertionError("missing request head did not raise")


def main():
    test_default_behavior_unchanged()
    test_disabled_learned_optimizer_flags_do_nothing()
    test_train_request_head_only()
    test_separate_lr_parameter_groups()
    test_request_head_only_disables_scheduler_and_keeps_lr()
    test_freeze_backbone()
    test_freeze_detector()
    test_missing_request_head_error()
    logger.success("Learned temporal optimizer tests passed")


if __name__ == "__main__":
    main()
