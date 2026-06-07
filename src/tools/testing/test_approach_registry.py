# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib

from pathlib import Path

import yaml

from src.tools.checkpoint_safety import requires_learned_temporal_request_head
from src.tools.testing.smoke_test_pipeline import (
    is_planned_placeholder,
    list_approaches_to_run,
    should_skip_in_all_approaches,
)
from src.utils.logging import get_logger

LOGGER = get_logger("TestApproachRegistry")
PRESET_PATH = Path("src/hypes_yaml/communication_approach_presets.yaml")
LEARNED_NAME = "learned_temporal_receiver_request_10"


def _presets():
    with open(PRESET_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("communication_presets", {})


def test_learned_temporal_preset_exists_and_is_trainable_experimental():
    presets = _presets()
    assert LEARNED_NAME in presets
    cfg = presets[LEARNED_NAME]
    rr = cfg["receiver_request"]
    learned = rr["learned"]
    md = cfg["metadata"]

    assert cfg["enabled"] is True
    assert cfg["strategy"] == "receiver_request_topk"
    assert rr["strategy_variant"] == "learned_temporal"
    assert rr["temporal"]["enabled"] is True
    assert learned["enabled"] is True
    assert learned["loss"]["enabled"] is True
    assert rr["loss"]["enabled"] is False
    assert rr["trainable"] is True
    assert abs(float(rr["keep_ratio"]) - 0.10) < 1e-8
    assert learned["optimizer"]["train_request_head_only"] is True
    assert learned["optimizer"]["separate_lr"] is True

    assert md["approach_family"] == "learned_temporal_receiver_request"
    assert md["approach_name"] == "learned_temporal_request"
    assert str(md["approach_setting"]) == "10"
    assert md["public_name"] == LEARNED_NAME
    assert md["implementation_status"] == "trainable_experimental"
    assert md["reportable_without_training"] is False
    assert md["include_in_all_approaches"] is False
    assert md["requires_train_data"] is True
    assert md["requires_trained_request_head"] is True


def test_learned_temporal_not_in_default_all_approaches():
    presets = _presets()
    selected, skipped = list_approaches_to_run(presets, allow_planned=False, skip_planned=True)
    skipped_map = dict(skipped)
    assert LEARNED_NAME not in selected
    assert skipped_map[LEARNED_NAME] == "trainable_experimental"
    skip, reason = should_skip_in_all_approaches(presets[LEARNED_NAME])
    assert skip is True
    assert reason == "trainable_experimental"


def test_planned_placeholders_still_skip():
    presets = _presets()
    selected, skipped = list_approaches_to_run(presets, allow_planned=False, skip_planned=True)
    skipped_map = dict(skipped)
    for name in [
        "receiver_request_uncertainty_topk_10",
        "receiver_request_visibility_topk",
        "receiver_request_learned",
        "receiver_request_learned_budget",
        "receiver_request_warped",
    ]:
        assert name not in selected
        assert is_planned_placeholder(presets[name])
        assert skipped_map[name] == "planned_placeholder"


def test_existing_runnable_approaches_still_selected():
    presets = _presets()
    selected, _ = list_approaches_to_run(presets, allow_planned=False, skip_planned=True)
    for name in [
        "selective_topk_energy_10",
        "receiver_request_energy_topk_10",
        "temporal_receiver_request_energy_topk_10",
    ]:
        assert name in selected
        assert not should_skip_in_all_approaches(presets[name])[0]


def test_checkpoint_safety_requirement_detection():
    presets = _presets()
    assert requires_learned_temporal_request_head({"communication": presets[LEARNED_NAME]})
    assert requires_learned_temporal_request_head(presets[LEARNED_NAME])
    assert not requires_learned_temporal_request_head({"communication": presets["receiver_request_energy_topk_10"]})
    assert not requires_learned_temporal_request_head({"communication": presets["temporal_receiver_request_energy_topk_10"]})
    assert not requires_learned_temporal_request_head({"communication": presets["selective_topk_energy_10"]})


def test_explicit_selection_by_name_is_allowed_by_registry():
    presets = _presets()
    assert LEARNED_NAME in presets
    assert not is_planned_placeholder(presets[LEARNED_NAME])
    assert presets[LEARNED_NAME]["metadata"]["implementation_status"] == "trainable_experimental"


def main():
    test_learned_temporal_preset_exists_and_is_trainable_experimental()
    test_learned_temporal_not_in_default_all_approaches()
    test_planned_placeholders_still_skip()
    test_existing_runnable_approaches_still_selected()
    test_checkpoint_safety_requirement_detection()
    test_explicit_selection_by_name_is_allowed_by_registry()
    LOGGER.success("Approach registry tests passed")


if __name__ == "__main__":
    main()
