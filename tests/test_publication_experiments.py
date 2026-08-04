import copy
import csv
import json
import os
from pathlib import Path

import yaml
import torch

from tools.publication.result_schema import RESULT_COLUMNS, empty_result, normalize_result
from tools.publication import aggregate_publication_results as aggregate
from tools.publication import plot_publication_curves as plots
from tools.publication.check_publication_environment import _path_status
from tools.publication import run_publication_experiments as runner
from tools.publication.run_publication_experiments import (
    build_command,
    build_run_config,
    expand_jobs,
    load_config,
    run_dir_for,
)
from src.models.fuse_modules.communication_policy import CommunicationPolicy


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments/publication/publication_sweep_config.yaml"


def test_publication_config_loads_and_grid_is_stable():
    config = load_config(CONFIG_PATH)
    assert config["post_evaluation"]["enabled"] is True
    assert config["post_evaluation"]["static_danger"]["enabled"] is True
    assert config["post_evaluation"]["trajectory"]["enabled"] is True
    jobs = expand_jobs(config)
    assert len(jobs) == 122
    assert len({job.experiment_name for job in jobs}) == len(jobs)
    assert {job.dataset for job in jobs} == {"carla_2021", "culver_city"}
    assert not any(job.method == "external_baseline" for job in jobs)
    sparse_budgets = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 50, 75, 100}
    for method in {
        "selective_topk",
        "snapshot_receiver_request",
        "temporal_receiver_request",
        "learned_temporal_receiver_request",
    }:
        assert {job.budget_percent for job in jobs if job.method == method} == sparse_budgets


def test_existing_preset_names_resolve():
    config = load_config(CONFIG_PATH)
    preset_path = REPO_ROOT / config["preset_file"]
    import yaml

    presets = (yaml.safe_load(preset_path.read_text()) or {})["communication_presets"]
    missing = sorted({job.preset for job in expand_jobs(config)} - set(presets))
    assert missing == []


def test_result_schema_is_complete_and_ordered():
    row = empty_result(experiment_name="smoke")
    assert list(row) == RESULT_COLUMNS
    assert row["experiment_name"] == "smoke"
    assert row["timestamp"]


def test_result_schema_reads_empirical_packet_loss_counters():
    row = normalize_result({
        "comm_packet_loss_rate": 0.01,
        "comm_num_transmitted_units": 1000,
        "comm_num_lost_units": 12,
        "comm_actual_loss_rate": 0.012,
    })
    assert row["num_transmitted_units"] == 1000
    assert row["num_lost_units"] == 12
    assert row["actual_loss_rate"] == 0.012


def test_result_schema_does_not_copy_configured_packet_probability_as_actual_rate():
    row = normalize_result({"comm_packet_loss_rate": 0.01})
    assert str(row["actual_loss_rate"]) == "nan"


def test_budget_overrides_resolve_to_ratios(monkeypatch):
    config = load_config(CONFIG_PATH)
    monkeypatch.setenv("CARLA_TRAIN_ROOT", "/tmp/train")
    monkeypatch.setenv("CARLA_2021_VALIDATE_ROOT", "/tmp/carla")
    jobs = expand_jobs(config)

    expected_paths = {
        "selective_topk": ("topk_energy", "keep_ratio"),
        "snapshot_receiver_request": ("receiver_request", "keep_ratio"),
        "temporal_receiver_request": ("receiver_request", "keep_ratio"),
        "learned_temporal_receiver_request": ("receiver_request", "keep_ratio"),
    }
    for method, path in expected_paths.items():
        job = next(job for job in jobs if job.dataset == "carla_2021" and job.method == method and job.budget_percent == 1)
        resolved = build_run_config(config, job)
        comm = resolved["model"]["args"]["communication"]
        assert comm[path[0]][path[1]] == 0.01
        if method == "learned_temporal_receiver_request":
            assert comm["receiver_request"]["learned"]["target_budget"] == 0.01


def test_packet_loss_seed_uses_monte_carlo_run(monkeypatch):
    config = load_config(CONFIG_PATH)
    monkeypatch.setenv("CARLA_TRAIN_ROOT", "/tmp/train")
    monkeypatch.setenv("CARLA_2021_VALIDATE_ROOT", "/tmp/carla")
    jobs = expand_jobs(config)
    packet_jobs = [
        job for job in jobs
        if job.dataset == "carla_2021"
        and job.method == "selective_topk"
        and job.budget_percent == 10
        and job.loss_type == "none"
    ]
    # The default publication grid keeps lossy scenarios disabled, so create
    # representative packet jobs by overriding the immutable dataclass fields.
    from dataclasses import replace

    base = packet_jobs[0]
    mc0 = replace(base, loss_type="packet", loss_probability=0.01, monte_carlo_run=0)
    mc1 = replace(base, loss_type="packet", loss_probability=0.01, monte_carlo_run=1)
    cfg0 = build_run_config(config, mc0)["model"]["args"]["communication"]["packet_loss"]
    cfg0_repeat = build_run_config(config, mc0)["model"]["args"]["communication"]["packet_loss"]
    cfg1 = build_run_config(config, mc1)["model"]["args"]["communication"]["packet_loss"]
    assert cfg0["seed"] == cfg0_repeat["seed"]
    assert cfg0["seed"] != cfg1["seed"]


def test_explicit_packet_loss_filter_expands_disabled_scenario():
    config = load_config(CONFIG_PATH)
    jobs = expand_jobs(
        config,
        include_disabled_loss_scenarios=True,
        only_loss_type="packet",
        extra_loss_probability=0.001,
    )
    selected = [
        job for job in jobs
        if job.dataset == "carla_2021"
        and job.method == "selective_topk"
        and job.budget_percent == 10
        and abs(job.loss_probability - 0.001) < 1e-12
    ]
    assert len(selected) == 10
    assert {job.monte_carlo_run for job in selected} == set(range(10))


def test_run_layout_and_smoke_command():
    config = load_config(CONFIG_PATH)
    job = next(job for job in expand_jobs(config) if job.dataset == "carla_2021" and job.method == "selective_topk" and job.budget_percent == 1)
    run_dir = run_dir_for(config, job, smoke=True)
    assert "carla_2021/selective_topk/budget_1" in run_dir.as_posix()
    assert run_dir.name == "smoke"
    command = build_command(config, job, run_dir, smoke=True)
    assert command[command.index("--max_samples") + 1] == "20"


def test_packet_loss_masks_are_reproducible_and_counted():
    torch.manual_seed(123)
    features = torch.randn(3, 4, 64, 64)
    record_len = torch.tensor([3])
    base_cfg = {
        "enabled": True,
        "strategy": "none",
        "seed": 0,
        "packet_loss": {"enabled": True, "loss_rate": 0.1, "unit": "packet", "seed": 31415},
    }

    policy_a = CommunicationPolicy(in_channels=4, comm_cfg=copy.deepcopy(base_cfg))
    out_a = policy_a(features, record_len)
    policy_a_repeat = CommunicationPolicy(in_channels=4, comm_cfg=copy.deepcopy(base_cfg))
    out_a_repeat = policy_a_repeat(features, record_len)

    cfg_b = copy.deepcopy(base_cfg)
    cfg_b["packet_loss"]["seed"] = 31416
    policy_b = CommunicationPolicy(in_channels=4, comm_cfg=cfg_b)
    out_b = policy_b(features, record_len)

    assert out_a.stats["num_transmitted_units"] == 2 * 64 * 64
    assert out_a.stats["num_lost_units"] > 0
    assert out_a.stats["num_lost_units"] == out_a_repeat.stats["num_lost_units"]
    assert torch.equal(out_a.features, out_a_repeat.features)
    assert out_a.stats["num_lost_units"] != out_b.stats["num_lost_units"]
    assert not torch.equal(out_a.features, out_b.features)
    assert abs(out_a.stats["actual_loss_rate"] - 0.1) < 0.02


def test_mock_execution_stages_reproducibility_artifacts(tmp_path, monkeypatch):
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["results_root"] = str(tmp_path / "publication")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "net_epoch1.pth").write_bytes(b"mock checkpoint")
    config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    train_dir = tmp_path / "train"
    validate_dir = tmp_path / "validate"
    train_dir.mkdir()
    validate_dir.mkdir()
    monkeypatch.setenv("CARLA_TRAIN_ROOT", str(train_dir))
    monkeypatch.setenv("CARLA_2021_VALIDATE_ROOT", str(validate_dir))

    job = next(
        job for job in expand_jobs(config)
        if job.dataset == "carla_2021" and job.method == "selective_topk" and job.budget_percent == 1
    )

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        module = command[command.index("-m") + 1]
        if module == "src.tools.inference":
            run_dir = Path(command[command.index("--model_dir") + 1])
            (run_dir / "summary_eval.yaml").write_text(
                yaml.safe_dump({
                    "ap_70": 0.5,
                    "comm_total_normalized_ratio": 0.01,
                    "comm_total_bytes_per_frame": 1000,
                })
            )
            (run_dir / "inference_summary.json").write_text("{}")
            boxes = run_dir / "danger_eval_boxes"
            boxes.mkdir()
            (boxes / "frame_000000.npz").write_bytes(b"mock")
        else:
            run_dir = Path(command[command.index("--run_dirs") + 1])
            summary_path = run_dir / "summary_eval.yaml"
            summary = yaml.safe_load(summary_path.read_text()) or {}
            if module == "src.tools.evaluate_danger_aware_metrics":
                summary["danger_aware_metrics"] = {
                    "danger_zone_recall@0.7": 0.6,
                    "risk_weighted_recall@0.7": 0.7,
                }
            elif module == "src.tools.evaluate_trajectory_danger_metrics":
                summary["trajectory_danger_metrics"] = {
                    "trajectory_time_risk_recall@0.7": 0.8,
                    "missed_trajectory_risk@0.7": 2.0,
                }
            summary_path.write_text(yaml.safe_dump(summary))
        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.execute_job(config, job, resume=False, overwrite=False, smoke=True) == "smoke_completed"
    run_dir = runner.run_dir_for(config, job, smoke=True)
    for name in (
        "config.yaml", "config_resolved.yaml", "command.txt", "run_metadata.json",
        "publication_result.json", "publication_run.log", "publication_inference.log",
        "publication_post_evaluation.log", "post_evaluation_commands.json", "net_epoch1.pth",
    ):
        assert (run_dir / name).exists()
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "smoke_completed"
    assert metadata["nominal_budget_ratio"] == 0.01
    assert metadata["post_evaluation"]["status"] == "completed"
    result = json.loads((run_dir / "publication_result.json").read_text())
    assert result["static_danger_recall_07"] == 0.6
    assert result["trajectory_time_risk_recall_07"] == 0.8
    assert runner.post_evaluate_existing_job(config, job, smoke=True) == "post_evaluated"


def test_command_exports_have_expected_counts(tmp_path):
    config = load_config(CONFIG_PATH)
    jobs = expand_jobs(config)
    carla_jobs = [job for job in jobs if job.dataset == "carla_2021"]
    smoke_job = [
        job for job in jobs
        if job.dataset == "carla_2021" and job.method == "selective_topk" and job.budget_percent == 10
    ]
    smoke_path = tmp_path / "smoke.sh"
    runner.export_commands(smoke_path, smoke_job, config_path=CONFIG_PATH, smoke=True, overwrite=False)
    assert sum(line.startswith("./env/bin/python") for line in smoke_path.read_text().splitlines()) == 1
    assert "--smoke" in smoke_path.read_text()

    carla_path = tmp_path / "carla.sh"
    runner.export_commands(carla_path, carla_jobs, config_path=CONFIG_PATH, smoke=False, overwrite=False)
    assert sum(line.startswith("./env/bin/python") for line in carla_path.read_text().splitlines()) == 61

    full_path = tmp_path / "full.sh"
    runner.export_commands(full_path, jobs, config_path=CONFIG_PATH, smoke=False, overwrite=False)
    assert sum(line.startswith("./env/bin/python") for line in full_path.read_text().splitlines()) == 122
    assert "WARNING: This launches the full publication sweep" in full_path.read_text()


def test_missing_environment_is_reported_without_exception(monkeypatch):
    monkeypatch.delenv("CARLA_TRAIN_ROOT", raising=False)
    row = _path_status("CARLA_TRAIN_ROOT", "CARLA training data")
    assert row[1] == "<unset>"
    assert row[2] == "no"
    assert "missing" in row[3]


def test_missing_native_outputs_never_mark_run_completed(tmp_path, monkeypatch):
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["results_root"] = str(tmp_path / "publication")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "net_epoch1.pth").write_bytes(b"mock checkpoint")
    config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    train_dir, validate_dir = tmp_path / "train", tmp_path / "validate"
    train_dir.mkdir()
    validate_dir.mkdir()
    monkeypatch.setenv("CARLA_TRAIN_ROOT", str(train_dir))
    monkeypatch.setenv("CARLA_2021_VALIDATE_ROOT", str(validate_dir))
    job = next(
        job for job in expand_jobs(config)
        if job.dataset == "carla_2021" and job.method == "selective_topk" and job.budget_percent == 10
    )

    class Completed:
        returncode = 0

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Completed())
    try:
        runner.execute_job(config, job, resume=False, overwrite=False, smoke=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing native outputs must fail publication execution")
    run_dir = runner.run_dir_for(config, job, smoke=True)
    result = json.loads((run_dir / "publication_result.json").read_text())
    assert result["status"] == "failed"
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "summary_eval.yaml" in metadata["missing_outputs"]


def test_mock_result_aggregation_and_plotting(tmp_path):
    run_dir = tmp_path / "runs" / "mock"
    run_dir.mkdir(parents=True)
    (run_dir / "publication_result.json").write_text(json.dumps({
        "experiment_name": "mock",
        "dataset": "carla_2021",
        "method": "selective_topk",
        "budget_percent": 10,
        "seed": 0,
        "loss_type": "none",
        "loss_probability": 0,
        "monte_carlo_run": 0,
        "status": "completed",
    }))
    (run_dir / "summary_eval.yaml").write_text(yaml.safe_dump({
        "ap_70": 0.8,
        "comm_total_normalized_ratio": 0.1,
        "comm_total_bytes_per_frame": 1000,
        "trajectory_danger_metrics": {
            "trajectory_time_risk_recall@0.7": 0.9,
            "missed_trajectory_risk@0.7": 10.0,
        },
    }))
    rows = aggregate.normalize_files(aggregate.discover_result_files(tmp_path))
    raw_path, summary_path = tmp_path / "raw.csv", tmp_path / "summary.csv"
    aggregate.write_raw(raw_path, rows)
    fields, summary_rows = aggregate.summarize(rows)
    aggregate.write_summary(summary_path, fields, summary_rows)
    summary_csv_rows = plots.load_rows(summary_path)
    output_dir = tmp_path / "figures"
    assert plots.plot_metric(summary_csv_rows, "carla_2021", "ap_07_mean", "AP@0.7", "ap07_vs_budget", output_dir)
    assert plots.plot_trajectory_efficiency(summary_csv_rows, "carla_2021", output_dir)
    assert (output_dir / "carla_2021_ap07_vs_budget.pdf").exists()
    assert (output_dir / "carla_2021_trajectory_efficiency_vs_budget.pdf").exists()
