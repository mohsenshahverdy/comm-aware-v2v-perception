import os

import yaml

from src.tools.approach_name_mapping import infer_public_name_from_run


PRESET_PATH = os.path.join("src", "hypes_yaml", "communication_approach_presets.yaml")


def _load_presets():
    with open(PRESET_PATH, "r") as f:
        return yaml.safe_load(f)["communication_presets"]


def test_primary_presets_exist():
    presets = _load_presets()
    assert "selective_topk_energy_10" in presets
    assert "receiver_request_energy_topk_10" in presets
    assert "baseline_full_communication" in presets
    assert "learned_mask_lam01_temp05_soft" in presets


def test_summary_name_inference_from_runs():
    assert infer_public_name_from_run("carla_selective_topk_energy_10") == "selective_topk_energy_10"
    assert infer_public_name_from_run("carla_receiver_request_energy_topk_10_test") == "receiver_request_energy_topk_10"


def test_metadata_fields_present():
    presets = _load_presets()
    md = presets["receiver_request_energy_topk_10"].get("metadata", {})
    assert md.get("approach_family") == "receiver_request"
    assert md.get("approach_name") == "energy_topk"
    assert md.get("public_name") == "receiver_request_energy_topk_10"
