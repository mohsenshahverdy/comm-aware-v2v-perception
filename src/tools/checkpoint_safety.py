# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib

"""Checkpoint safety checks for learned temporal receiver-request inference."""

import glob
import os
import re
from typing import Dict, Tuple

import torch

from src.utils.runtime_config import get_communication_cfg


LEARNED_REQUEST_HEAD_ERROR = (
    "Learned temporal receiver-request requires a trained learned request head, "
    "but checkpoint does not contain comm_policy.learned_temporal_request_head.* weights. "
    "Refusing reportable inference. Use --allow_untrained_request_head only for debug smoke tests."
)

UNTRAINED_REQUEST_HEAD_WARNING = (
    "Untrained learned request head allowed by debug override; result is not reportable."
)


def _as_comm_cfg(hypes_or_comm_cfg) -> dict:
    if not isinstance(hypes_or_comm_cfg, dict):
        return {}
    if "receiver_request" in hypes_or_comm_cfg or "strategy" in hypes_or_comm_cfg:
        return hypes_or_comm_cfg
    return get_communication_cfg(hypes_or_comm_cfg)


def requires_learned_temporal_request_head(hypes_or_comm_cfg) -> bool:
    """Return True only for learned temporal receiver-request configurations."""
    comm_cfg = _as_comm_cfg(hypes_or_comm_cfg)
    if not isinstance(comm_cfg, dict) or not comm_cfg:
        return False

    strategy = str(comm_cfg.get("strategy", "none")).lower()
    rr_cfg = comm_cfg.get("receiver_request", {})
    if not isinstance(rr_cfg, dict):
        return False

    temporal_cfg = rr_cfg.get("temporal", {}) if isinstance(rr_cfg.get("temporal", {}), dict) else {}
    learned_cfg = rr_cfg.get("learned", {}) if isinstance(rr_cfg.get("learned", {}), dict) else {}
    variant = str(rr_cfg.get("strategy_variant", "")).lower()

    learned_temporal_variant = (
        variant in {"learned_temporal", "learned_temporal_topk"}
        or variant.startswith("learned_temporal_")
    )
    learned_temporal_enabled = bool(temporal_cfg.get("enabled", False)) and bool(learned_cfg.get("enabled", False))

    return strategy == "receiver_request_topk" and (learned_temporal_variant or learned_temporal_enabled)


def resolve_checkpoint_path(model_dir: str) -> Tuple[int, str]:
    """Resolve the checkpoint path that train_utils.load_saved_model will load."""
    if not model_dir or not os.path.exists(model_dir):
        return 0, None

    latest_path = os.path.join(model_dir, "latest.pth")
    if os.path.exists(latest_path):
        epoch = 10000
        try:
            checkpoint = torch.load(latest_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "epoch" in checkpoint:
                epoch = int(checkpoint["epoch"])
        except Exception:
            # Let the actual loader report detailed checkpoint load errors later.
            pass
        return epoch, latest_path

    file_list = glob.glob(os.path.join(model_dir, "*epoch*.pth"))
    epochs = []
    for path in file_list:
        match = re.findall(r".*epoch(\d+)\.pth.*", path)
        if match:
            epochs.append((int(match[0]), path))
    if not epochs:
        return 0, None

    epoch, path = max(epochs, key=lambda item: item[0])
    return int(epoch), path


def load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint path was resolved.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    return state_dict


def normalize_checkpoint_key(key: str) -> str:
    key = str(key)
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def learned_request_head_key_match_type(state_dict: dict) -> str:
    strict_prefixes = (
        "comm_policy.learned_temporal_request_head.",
        "model.comm_policy.learned_temporal_request_head.",
    )
    normalized_keys = [normalize_checkpoint_key(k) for k in state_dict.keys()]

    for key in normalized_keys:
        if key.startswith(strict_prefixes) or ".comm_policy.learned_temporal_request_head." in key:
            return "prefix"

    for key in normalized_keys:
        if "learned_temporal_request_head" in key:
            return "fallback"

    return "none"


def checkpoint_has_learned_request_head(state_dict: dict) -> bool:
    return learned_request_head_key_match_type(state_dict) != "none"


def validate_learned_request_checkpoint(
    hypes: dict,
    checkpoint_path: str,
    allow_untrained_request_head: bool = False,
    logger=None,
) -> Dict[str, object]:
    """Validate learned request-head checkpoint safety and return metadata."""
    required = requires_learned_temporal_request_head(hypes)
    metadata = {
        "checkpoint_path": checkpoint_path,
        "requires_learned_request_head": bool(required),
        "learned_request_head_trained": None,
        "allow_untrained_request_head": bool(allow_untrained_request_head),
        "reportable_result": True,
        "checkpoint_safety_warning": None,
        "learned_request_head_key_match": "not_required",
    }

    if not required:
        if logger is not None:
            logger.info("Checkpoint safety", requires_learned_request_head=False, reportable_result=True)
        return metadata

    if checkpoint_path is None:
        metadata.update({
            "learned_request_head_trained": False,
            "reportable_result": False,
            "checkpoint_safety_warning": LEARNED_REQUEST_HEAD_ERROR,
            "learned_request_head_key_match": "none",
        })
        if allow_untrained_request_head:
            metadata["checkpoint_safety_warning"] = UNTRAINED_REQUEST_HEAD_WARNING
            if logger is not None:
                logger.warn("Untrained learned request head allowed", checkpoint_path="None", reportable_result=False)
            return metadata
        raise RuntimeError(LEARNED_REQUEST_HEAD_ERROR)

    state_dict = load_checkpoint_state_dict(checkpoint_path)
    match_type = learned_request_head_key_match_type(state_dict)
    has_head = match_type != "none"
    metadata["learned_request_head_key_match"] = match_type

    if has_head:
        metadata.update({
            "learned_request_head_trained": True,
            "reportable_result": True,
        })
        if logger is not None:
            logger.success(
                "Learned request-head checkpoint verified",
                checkpoint_path=checkpoint_path,
                key_match=match_type,
                reportable_result=True,
            )
        return metadata

    metadata.update({
        "learned_request_head_trained": False,
        "reportable_result": False,
        "checkpoint_safety_warning": UNTRAINED_REQUEST_HEAD_WARNING if allow_untrained_request_head else LEARNED_REQUEST_HEAD_ERROR,
    })

    if allow_untrained_request_head:
        if logger is not None:
            logger.warn(
                "Untrained learned request head allowed",
                checkpoint_path=checkpoint_path,
                key_match=match_type,
                reportable_result=False,
            )
        return metadata

    raise RuntimeError(LEARNED_REQUEST_HEAD_ERROR)
