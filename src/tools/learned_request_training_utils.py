# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib

from typing import Dict, Iterable, List, Optional, Tuple

import torch.nn as nn

from src.utils.logging import get_logger
from src.utils.runtime_config import get_communication_cfg


LOGGER = get_logger("LearnedRequestTraining")
BACKBONE_MODULE_NAMES = ("pillar_vfe", "scatter", "backbone", "shrink_conv", "naive_compressor")
DETECTOR_MODULE_NAMES = ("cls_head", "reg_head", "dir_head")


def get_learned_temporal_optimizer_cfg(hypes) -> dict:
    comm_cfg = get_communication_cfg(hypes)
    rr_cfg = comm_cfg.get("receiver_request", {}) if isinstance(comm_cfg.get("receiver_request", {}), dict) else {}
    learned_cfg = rr_cfg.get("learned", {}) if isinstance(rr_cfg.get("learned", {}), dict) else {}
    optimizer_cfg = learned_cfg.get("optimizer", {}) if isinstance(learned_cfg.get("optimizer", {}), dict) else {}
    return optimizer_cfg


def is_learned_temporal_training_enabled(hypes) -> bool:
    comm_cfg = get_communication_cfg(hypes)
    rr_cfg = comm_cfg.get("receiver_request", {}) if isinstance(comm_cfg.get("receiver_request", {}), dict) else {}
    learned_cfg = rr_cfg.get("learned", {}) if isinstance(rr_cfg.get("learned", {}), dict) else {}
    optimizer_cfg = get_learned_temporal_optimizer_cfg(hypes)
    return bool(learned_cfg.get("enabled", False)) and any([
        bool(optimizer_cfg.get("separate_lr", False)),
        bool(optimizer_cfg.get("freeze_detector", False)),
        bool(optimizer_cfg.get("freeze_backbone", False)),
        bool(optimizer_cfg.get("train_request_head_only", False)),
    ])


def get_learned_request_head(model) -> Optional[nn.Module]:
    comm_policy = getattr(model, "comm_policy", None)
    if comm_policy is None:
        return None
    return getattr(comm_policy, "learned_temporal_request_head", None)


def _parameters(module: nn.Module) -> List[nn.Parameter]:
    return list(module.parameters()) if module is not None else []


def _parameter_count(params: Iterable[nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params))


def _trainable_parameter_count(params: Iterable[nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params if p.requires_grad))


def _freeze_module(model, name: str) -> Tuple[bool, int]:
    module = getattr(model, name, None)
    if module is None:
        return False, 0
    frozen = 0
    for p in module.parameters():
        if p.requires_grad:
            frozen += p.numel()
        p.requires_grad = False
    return True, int(frozen)


def _log_param_summary(model, optimizer_cfg: dict, logger=None, optimizer_groups=None):
    log = logger or LOGGER
    all_params = list(model.parameters())
    total_params = _parameter_count(all_params)
    trainable_params = _trainable_parameter_count(all_params)
    frozen_params = total_params - trainable_params
    fields = {
        "train_request_head_only": bool(optimizer_cfg.get("train_request_head_only", False)),
        "freeze_backbone": bool(optimizer_cfg.get("freeze_backbone", False)),
        "freeze_detector": bool(optimizer_cfg.get("freeze_detector", False)),
        "separate_lr": bool(optimizer_cfg.get("separate_lr", False)),
        "request_head_lr": optimizer_cfg.get("request_head_lr", None),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }
    if optimizer_groups is not None:
        fields["optimizer_groups"] = len(optimizer_groups)
        fields["group_lrs"] = [g.get("lr", None) for g in optimizer_groups]
        fields["group_param_counts"] = [_parameter_count(g.get("params", [])) for g in optimizer_groups]
    log.config("Learned temporal request training config", **fields)


def configure_learned_temporal_freezing(model, hypes, logger=None):
    optimizer_cfg = get_learned_temporal_optimizer_cfg(hypes)
    log = logger or LOGGER
    if not is_learned_temporal_training_enabled(hypes):
        return model

    request_head = get_learned_request_head(model)
    needs_request_head = bool(optimizer_cfg.get("train_request_head_only", False)) or bool(optimizer_cfg.get("separate_lr", False))
    if request_head is None and needs_request_head:
        raise ValueError("Learned temporal request head is required but was not found on model.comm_policy")

    if bool(optimizer_cfg.get("train_request_head_only", False)):
        if request_head is None:
            raise ValueError("train_request_head_only=true requires model.comm_policy.learned_temporal_request_head")
        for p in model.parameters():
            p.requires_grad = False
        for p in request_head.parameters():
            p.requires_grad = True
        if _trainable_parameter_count(request_head.parameters()) <= 0:
            raise ValueError("train_request_head_only=true but no request-head parameters are trainable")
        _log_param_summary(model, optimizer_cfg, logger=log)
        return model

    skipped = []
    if bool(optimizer_cfg.get("freeze_backbone", False)):
        for name in BACKBONE_MODULE_NAMES:
            found, _ = _freeze_module(model, name)
            if not found:
                skipped.append(name)
    if bool(optimizer_cfg.get("freeze_detector", False)):
        for name in DETECTOR_MODULE_NAMES:
            found, _ = _freeze_module(model, name)
            if not found:
                skipped.append(name)
    if skipped:
        log.warn("Optional modules not found while applying learned temporal freezing", skipped=skipped)

    _log_param_summary(model, optimizer_cfg, logger=log)
    return model


def build_learned_temporal_param_groups(model, hypes, base_lr, logger=None):
    if not is_learned_temporal_training_enabled(hypes):
        return None

    optimizer_cfg = get_learned_temporal_optimizer_cfg(hypes)
    separate_lr = bool(optimizer_cfg.get("separate_lr", False))
    request_head_only = bool(optimizer_cfg.get("train_request_head_only", False))
    if not separate_lr and not request_head_only:
        return None

    request_head = get_learned_request_head(model)
    if request_head is None:
        raise ValueError("Learned temporal separate LR/request-head-only mode requires model.comm_policy.learned_temporal_request_head")

    request_params = [p for p in request_head.parameters() if p.requires_grad]
    if len(request_params) == 0:
        raise ValueError("Learned temporal request head exists but has no trainable parameters")
    request_param_ids = {id(p) for p in request_head.parameters()}

    groups = []
    if not request_head_only:
        main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in request_param_ids]
        if main_params:
            groups.append({"params": main_params, "lr": base_lr})
    request_lr = float(optimizer_cfg.get("request_head_lr", base_lr))
    groups.append({"params": request_params, "lr": request_lr})

    seen = set()
    for group in groups:
        for p in group["params"]:
            pid = id(p)
            if pid in seen:
                raise ValueError("Duplicate parameter detected in learned temporal optimizer groups")
            seen.add(pid)

    _log_param_summary(model, optimizer_cfg, logger=logger, optimizer_groups=groups)
    return groups
