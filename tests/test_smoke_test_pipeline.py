import json
from pathlib import Path

import yaml

from src.tools.testing.smoke_test_pipeline import (
    build_smoke_run_name,
    list_approaches_to_run,
    load_yaml,
    patch_config_for_smoke,
    required_metric_keys,
    validate_required_metrics,
    write_report,
)


PRESET_PATH = Path("src/hypes_yaml/communication_approach_presets.yaml")


def _load_presets():
    with open(PRESET_PATH, "r") as f:
        return yaml.safe_load(f)["communication_presets"]


def test_run_folder_naming_uses_smoke_prefix():
    assert build_smoke_run_name("carla", "receiver_request_energy_topk_10") == "smoke_carla_receiver_request_energy_topk_10"


def test_default_all_approach_selection_excludes_planned_and_disabled():
    presets = _load_presets()
    selected, skipped = list_approaches_to_run(presets, allow_planned=False, skip_planned=True)
    assert "baseline_full_communication" in selected
    assert "receiver_request_uncertainty_topk_10" not in selected
    assert "receiver_request_visibility_topk" not in selected
    assert "receiver_request_learned" not in selected
    assert any(name == "receiver_request_uncertainty_topk_10" for name, _ in skipped)


def test_required_metrics_validation_catches_missing_keys():
    summary = {"ap_50": 0.9}
    missing = validate_required_metrics(summary, is_receiver_request=True, require_ap=True)
    assert "comm_total_bytes_per_frame" in missing
    assert "receiver_request_keep_ratio" in missing
    assert "ap_70" in missing
    assert "ap30_or_ap_30" in missing


def test_required_metric_key_sets_include_receiver_request_fields():
    keys = required_metric_keys(is_receiver_request=True, require_ap=True)
    assert "comm_feature_bytes_per_frame" in keys
    assert "comm_total_normalized_ratio" in keys
    assert "receiver_request_keep_ratio" in keys


def test_smoke_report_json_is_written(tmp_path):
    payload = {"status": "pass", "approach": "selective_topk_energy_10"}
    report_path = tmp_path / "smoke_test_report.json"
    write_report(report_path, payload)
    assert report_path.exists()
    with open(report_path, "r") as f:
        loaded = json.load(f)
    assert loaded["status"] == "pass"


def test_config_patching_does_not_modify_base_yaml_file(tmp_path):
    base_cfg = {
        "root_dir": "X",
        "validate_dir": "Y",
        "model": {"args": {"communication": {"receiver_request": {"enabled": False}}}},
    }
    base_path = tmp_path / "base.yaml"
    with open(base_path, "w") as f:
        yaml.safe_dump(base_cfg, f)

    original_text = base_path.read_text()
    loaded = load_yaml(base_path)
    patch_config_for_smoke(
        loaded,
        split="carla",
        root_dir_override=None,
        validate_dir_override=None,
        save_debug_maps=True,
        debug_num_frames=3,
        debug_dir=tmp_path / "receiver_request_debug",
        approach_name="receiver_request_energy_topk_10",
    )

    # Base file is unchanged until explicitly re-saved.
    assert base_path.read_text() == original_text
    assert loaded["communication_preset"] == "receiver_request_energy_topk_10"
    rr = loaded["model"]["args"]["communication"]["receiver_request"]
    assert rr["save_request_maps"] is True
    assert rr["debug_num_frames"] == 3
