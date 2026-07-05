import copy
import csv
import json
import os
from pathlib import Path

import yaml

from tools.publication.result_schema import RESULT_COLUMNS, empty_result
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


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments/publication/publication_sweep_config.yaml"


def test_publication_config_loads_and_grid_is_stable():
    config = load_config(CONFIG_PATH)
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


def test_run_layout_and_smoke_command():
    config = load_config(CONFIG_PATH)
    job = next(job for job in expand_jobs(config) if job.dataset == "carla_2021" and job.method == "selective_topk" and job.budget_percent == 1)
    run_dir = run_dir_for(config, job, smoke=True)
    assert "carla_2021/selective_topk/budget_1" in run_dir.as_posix()
    assert run_dir.name == "smoke"
    command = build_command(config, job, run_dir, smoke=True)
    assert command[command.index("--max_samples") + 1] == "20"


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
        run_dir = Path(command[command.index("--model_dir") + 1])
        (run_dir / "summary_eval.yaml").write_text(
            yaml.safe_dump({
                "ap_70": 0.5,
                "comm_total_normalized_ratio": 0.01,
                "comm_total_bytes_per_frame": 1000,
            })
        )
        (run_dir / "inference_summary.json").write_text("{}")
        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.execute_job(config, job, resume=False, overwrite=False, smoke=True) == "smoke_completed"
    run_dir = runner.run_dir_for(config, job, smoke=True)
    for name in (
        "config.yaml", "config_resolved.yaml", "command.txt", "run_metadata.json",
        "publication_result.json", "publication_run.log", "publication_inference.log", "net_epoch1.pth",
    ):
        assert (run_dir / name).exists()
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "smoke_completed"
    assert metadata["nominal_budget_ratio"] == 0.01


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
