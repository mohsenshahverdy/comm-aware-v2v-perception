# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import tempfile

import torch

from src.tools.checkpoint_safety import (
    LEARNED_REQUEST_HEAD_ERROR,
    UNTRAINED_REQUEST_HEAD_WARNING,
    checkpoint_has_learned_request_head,
    learned_request_head_key_match_type,
    load_checkpoint_state_dict,
    normalize_checkpoint_key,
    requires_learned_temporal_request_head,
    resolve_checkpoint_path,
    validate_learned_request_checkpoint,
)
from src.utils.logging import get_logger

LOGGER = get_logger("TestCheckpointSafety")


def _learned_hypes():
    return {
        "communication": {
            "enabled": True,
            "strategy": "receiver_request_topk",
            "receiver_request": {
                "enabled": True,
                "strategy_variant": "learned_temporal",
                "temporal": {"enabled": True},
                "learned": {"enabled": True},
            },
        }
    }


def _learned_by_enabled_flags_hypes():
    return {
        "communication": {
            "enabled": True,
            "strategy": "receiver_request_topk",
            "receiver_request": {
                "enabled": True,
                "strategy_variant": "temporal_energy_topk",
                "temporal": {"enabled": True},
                "learned": {"enabled": True},
            },
        }
    }


def _receiver_non_learned_hypes():
    return {
        "communication": {
            "enabled": True,
            "strategy": "receiver_request_topk",
            "receiver_request": {
                "enabled": True,
                "strategy_variant": "energy_topk",
                "temporal": {"enabled": False},
                "learned": {"enabled": False},
            },
        }
    }


def _temporal_non_learned_hypes():
    return {
        "communication": {
            "enabled": True,
            "strategy": "receiver_request_topk",
            "receiver_request": {
                "enabled": True,
                "strategy_variant": "temporal_energy_topk",
                "temporal": {"enabled": True},
                "learned": {"enabled": False},
            },
        }
    }


def _selective_hypes():
    return {
        "communication": {
            "enabled": True,
            "strategy": "topk_energy",
            "receiver_request": {},
        }
    }


def _save_checkpoint(folder, state_dict, epoch=7, name="net_epoch7.pth", wrapped=True):
    path = os.path.join(folder, name)
    payload = {"epoch": epoch, "model_state_dict": state_dict} if wrapped else state_dict
    torch.save(payload, path)
    return path


def test_detection_helper():
    assert requires_learned_temporal_request_head(_learned_hypes())
    assert requires_learned_temporal_request_head(_learned_by_enabled_flags_hypes())
    assert requires_learned_temporal_request_head(_learned_hypes()["communication"])
    assert not requires_learned_temporal_request_head(_receiver_non_learned_hypes())
    assert not requires_learned_temporal_request_head(_temporal_non_learned_hypes())
    assert not requires_learned_temporal_request_head(_selective_hypes())
    assert not requires_learned_temporal_request_head({"communication": {"enabled": False, "strategy": "none"}})


def test_key_detection():
    assert normalize_checkpoint_key("module.module.comm_policy.x") == "comm_policy.x"
    assert learned_request_head_key_match_type({
        "comm_policy.learned_temporal_request_head.net.0.weight": torch.ones(1)
    }) == "prefix"
    assert learned_request_head_key_match_type({
        "module.comm_policy.learned_temporal_request_head.net.0.weight": torch.ones(1)
    }) == "prefix"
    assert learned_request_head_key_match_type({
        "outer.comm_policy.learned_temporal_request_head.net.0.weight": torch.ones(1)
    }) == "prefix"
    assert learned_request_head_key_match_type({
        "some.learned_temporal_request_head.weight": torch.ones(1)
    }) == "fallback"
    assert learned_request_head_key_match_type({"backbone.weight": torch.ones(1)}) == "none"
    assert checkpoint_has_learned_request_head({
        "model.comm_policy.learned_temporal_request_head.net.0.weight": torch.ones(1)
    })


def test_learned_missing_checkpoint_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _save_checkpoint(tmp, {"backbone.weight": torch.ones(1)})
        try:
            validate_learned_request_checkpoint(_learned_hypes(), ckpt, allow_untrained_request_head=False)
        except RuntimeError as exc:
            assert LEARNED_REQUEST_HEAD_ERROR in str(exc)
        else:
            raise AssertionError("Expected missing learned request-head checkpoint to fail")


def test_learned_missing_checkpoint_allow_flag_passes_with_warning():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _save_checkpoint(tmp, {"backbone.weight": torch.ones(1)})
        meta = validate_learned_request_checkpoint(_learned_hypes(), ckpt, allow_untrained_request_head=True)
        assert meta["requires_learned_request_head"] is True
        assert meta["learned_request_head_trained"] is False
        assert meta["allow_untrained_request_head"] is True
        assert meta["reportable_result"] is False
        assert meta["checkpoint_safety_warning"] == UNTRAINED_REQUEST_HEAD_WARNING
        assert meta["learned_request_head_key_match"] == "none"


def test_learned_checkpoint_with_request_head_passes():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _save_checkpoint(tmp, {
            "comm_policy.learned_temporal_request_head.net.0.weight": torch.ones(1),
            "backbone.weight": torch.ones(1),
        })
        meta = validate_learned_request_checkpoint(_learned_hypes(), ckpt)
        assert meta["requires_learned_request_head"] is True
        assert meta["learned_request_head_trained"] is True
        assert meta["reportable_result"] is True
        assert meta["checkpoint_safety_warning"] is None
        assert meta["learned_request_head_key_match"] == "prefix"


def test_dataparallel_checkpoint_with_request_head_passes():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _save_checkpoint(tmp, {
            "module.comm_policy.learned_temporal_request_head.net.0.bias": torch.ones(1),
        })
        meta = validate_learned_request_checkpoint(_learned_hypes(), ckpt)
        assert meta["learned_request_head_trained"] is True
        assert meta["learned_request_head_key_match"] == "prefix"


def test_non_learned_configs_do_not_require_head_keys():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _save_checkpoint(tmp, {"backbone.weight": torch.ones(1)})
        for hypes in [_receiver_non_learned_hypes(), _temporal_non_learned_hypes(), _selective_hypes()]:
            meta = validate_learned_request_checkpoint(hypes, ckpt)
            assert meta["requires_learned_request_head"] is False
            assert meta["learned_request_head_trained"] is None
            assert meta["reportable_result"] is True
            assert meta["learned_request_head_key_match"] == "not_required"


def test_checkpoint_resolution_and_state_loading():
    with tempfile.TemporaryDirectory() as tmp:
        _save_checkpoint(tmp, {"a": torch.ones(1)}, epoch=2, name="net_epoch2.pth")
        latest = _save_checkpoint(tmp, {"b": torch.ones(1)}, epoch=9, name="latest.pth")
        epoch, path = resolve_checkpoint_path(tmp)
        assert epoch == 9
        assert path == latest
        state = load_checkpoint_state_dict(path)
        assert "b" in state


def main():
    test_detection_helper()
    test_key_detection()
    test_learned_missing_checkpoint_fails()
    test_learned_missing_checkpoint_allow_flag_passes_with_warning()
    test_learned_checkpoint_with_request_head_passes()
    test_dataparallel_checkpoint_with_request_head_passes()
    test_non_learned_configs_do_not_require_head_keys()
    test_checkpoint_resolution_and_state_loading()
    LOGGER.success("Learned temporal checkpoint safety tests passed")


if __name__ == "__main__":
    main()
