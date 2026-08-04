#!/usr/bin/env python3
"""Plan and later execute publication experiment jobs without mutating presets."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.publication.result_schema import normalize_result  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


@dataclass(frozen=True)
class ExperimentJob:
    experiment_name: str
    dataset: str
    method: str
    budget_percent: float
    seed: int
    loss_type: str
    loss_probability: float
    monte_carlo_run: int
    preset: str
    checkpoint_key: str
    requires_trained_request_head: bool
    notes: str


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = ["base_config", "preset_file", "results_root", "datasets", "methods"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Publication config is missing required keys: {missing}")
    return config


def _budget_label(value: float) -> str:
    number = float(value)
    return f"{int(number):03d}" if number.is_integer() else str(number).replace(".", "p")


def _preset_for(method_cfg: Dict[str, Any], budget: float) -> str:
    mapping = method_cfg.get("preset_by_budget", {}) or {}
    for key in (budget, int(budget) if float(budget).is_integer() else budget, str(int(budget)) if float(budget).is_integer() else str(budget)):
        if key in mapping:
            return str(mapping[key])
    preset = method_cfg.get("preset")
    if not preset:
        raise ValueError(f"No preset mapping for budget {budget}: {method_cfg}")
    return str(preset)


def expand_jobs(
    config: Dict[str, Any],
    *,
    include_disabled_loss_scenarios: bool = False,
    only_loss_type: Optional[str] = None,
    extra_loss_probability: Optional[float] = None,
) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    loss_scenarios = [
        x for x in config.get("loss_scenarios", [])
        if (include_disabled_loss_scenarios or x.get("enabled", True))
        and (only_loss_type is None or str(x.get("loss_type", "none")) == only_loss_type)
    ]
    if not loss_scenarios and only_loss_type is None:
        loss_scenarios = [{"loss_type": "none", "probabilities": [0.0], "monte_carlo_runs": 1}]

    for dataset in config["datasets"]:
        for method, method_cfg in config["methods"].items():
            if not method_cfg.get("enabled", True):
                continue
            budgets = method_cfg.get("budgets_percent", config.get("default_budgets_percent", []))
            for budget in budgets:
                preset = _preset_for(method_cfg, float(budget))
                for seed in config.get("seeds", [0]):
                    for scenario in loss_scenarios:
                        loss_type = str(scenario.get("loss_type", "none"))
                        probabilities = list(scenario.get("probabilities", [0.0]))
                        if extra_loss_probability is not None and all(abs(float(p) - float(extra_loss_probability)) > 1e-12 for p in probabilities):
                            probabilities.append(float(extra_loss_probability))
                        for probability in probabilities:
                            runs = int(scenario.get("monte_carlo_runs", 1))
                            for mc_run in range(runs):
                                name = (
                                    f"{dataset}__{method}__b{_budget_label(float(budget))}"
                                    f"__s{int(seed)}__{loss_type}_p{float(probability):.3f}__mc{mc_run:03d}"
                                )
                                jobs.append(ExperimentJob(
                                    experiment_name=name,
                                    dataset=str(dataset),
                                    method=str(method),
                                    budget_percent=float(budget),
                                    seed=int(seed),
                                    loss_type=loss_type,
                                    loss_probability=float(probability),
                                    monte_carlo_run=mc_run,
                                    preset=preset,
                                    checkpoint_key=str(method_cfg.get("checkpoint_key", "checkpoint_dir")),
                                    requires_trained_request_head=bool(method_cfg.get("requires_trained_request_head", False)),
                                    notes=str(method_cfg.get("notes", "")),
                                ))
    return jobs


def filter_jobs(jobs: Iterable[ExperimentJob], args: argparse.Namespace) -> List[ExperimentJob]:
    selected = []
    for job in jobs:
        if args.dataset and job.dataset != args.dataset:
            continue
        if args.method and job.method != args.method:
            continue
        if args.budget is not None and abs(job.budget_percent - args.budget) > 1e-9:
            continue
        if args.seed is not None and job.seed != args.seed:
            continue
        if args.loss_type and job.loss_type != args.loss_type:
            continue
        if args.loss_probability is not None and abs(job.loss_probability - args.loss_probability) > 1e-12:
            continue
        if args.monte_carlo_run is not None and job.monte_carlo_run != args.monte_carlo_run:
            continue
        selected.append(job)
    return selected


def export_commands(
    output_path: Path,
    jobs: List[ExperimentJob],
    *,
    config_path: Path,
    smoke: bool,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Command export exists; use --overwrite true to replace it: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config_arg = config_path.relative_to(REPO_ROOT)
    except ValueError:
        config_arg = config_path

    lines = [
        "#!/usr/bin/env bash",
        "set -e",
        "",
        "# Generated publication experiment commands. Run from the repository root.",
        "# Required for CARLA/non-learned runs:",
        "#   export PUBLICATION_CHECKPOINT_DIR=\"<checkpoint-directory>\"",
        "#   export CARLA_TRAIN_ROOT=\"<training-data-directory>\"",
        "#   export CARLA_2021_VALIDATE_ROOT=\"<carla-validation-directory>\"",
        "# Required for Culver runs:",
        "#   export CULVER_VALIDATE_ROOT=\"<culver-validation-directory>\"",
        "# Required for learned temporal runs:",
        "#   export LEARNED_PUBLICATION_CHECKPOINT_DIR=\"<learned-checkpoint-directory>\"",
        "",
    ]
    if len(jobs) == 122:
        lines.extend([
            "# WARNING: This launches the full publication sweep. Run only after smoke and single full validation succeed.",
            "",
        ])
    else:
        lines.extend([f"# Selected publication jobs: {len(jobs)}", ""])

    for index, job in enumerate(jobs, start=1):
        command = [
            "./env/bin/python",
            "tools/publication/run_publication_experiments.py",
            "--config",
            str(config_arg),
            "--dataset",
            job.dataset,
            "--method",
            job.method,
            "--budget",
            f"{job.budget_percent:g}",
            "--seed",
            str(job.seed),
            "--loss-type",
            job.loss_type,
            "--loss-probability",
            f"{job.loss_probability:g}",
            "--monte-carlo-run",
            str(job.monte_carlo_run),
            "--execute",
            "--resume",
            "--overwrite",
            "false",
        ]
        if smoke:
            command.append("--smoke")
        lines.extend([f"# Job {index}/{len(jobs)}: {job.experiment_name}", shlex.join(command), ""])
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    output_path.chmod(output_path.stat().st_mode | 0o111)
    return output_path


def deep_merge(destination: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            deep_merge(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)
    return destination


def set_nested(mapping: Dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = mapping
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def resolve_env(value: str, *, required: bool) -> str:
    expanded = os.path.expandvars(str(value))
    if required and "${" in expanded:
        import re

        names = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(value))
        if names:
            raise ValueError(f"{', '.join(names)} is not set. Please export it before running publication experiments.")
        raise ValueError(f"Unresolved environment variable in path: {value}")
    return expanded


def build_run_config(config: Dict[str, Any], job: ExperimentJob) -> Dict[str, Any]:
    base_path = REPO_ROOT / config["base_config"]
    preset_path = REPO_ROOT / config["preset_file"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    presets = (yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}).get("communication_presets", {})
    if job.preset not in presets:
        raise KeyError(f"Preset '{job.preset}' was not found in {preset_path}")

    communication = base.setdefault("model", {}).setdefault("args", {}).setdefault("communication", {})
    deep_merge(communication, presets[job.preset])
    base.pop("communication_preset", None)

    method_cfg = config["methods"][job.method]
    ratio = job.budget_percent / 100.0
    for path in method_cfg.get("budget_override_paths", []):
        set_nested(communication, str(path), ratio)

    if job.loss_type != "none":
        packet_loss = communication.setdefault("packet_loss", {})
        packet_seed = _packet_loss_seed(job)
        packet_loss.update({
            "enabled": True,
            "loss_rate": job.loss_probability,
            "unit": "cell" if job.loss_type == "feature_cell" else "packet",
            "seed": packet_seed,
        })

    dataset_paths = config["paths"]["datasets"][job.dataset]
    base["root_dir"] = resolve_env(dataset_paths["root_dir"], required=True)
    base["validate_dir"] = resolve_env(dataset_paths["validate_dir"], required=True)
    base["seed"] = job.seed
    return base


def _packet_loss_seed(job: ExperimentJob) -> int:
    """Stable packet-loss seed: same Monte Carlo id is reproducible; different ids differ."""
    loss_offset = 100000 if job.loss_type == "feature_cell" else 200000
    probability_offset = int(round(float(job.loss_probability) * 1_000_000))
    return int(job.seed) * 1_000_003 + int(job.monte_carlo_run) * 9_176 + probability_offset + loss_offset


def _plain_budget_label(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number).replace(".", "p")


def run_dir_for(config: Dict[str, Any], job: ExperimentJob, *, smoke: bool = False) -> Path:
    mode = "smoke" if smoke else "full"
    loss_label = f"{job.loss_type}_p{job.loss_probability:.3f}"
    return (
        REPO_ROOT
        / config["results_root"]
        / "runs"
        / job.dataset
        / job.method
        / f"budget_{_plain_budget_label(job.budget_percent)}"
        / f"seed_{job.seed}"
        / loss_label
        / f"mc_{job.monte_carlo_run:03d}"
        / mode
    )


def checkpoint_dir_for(config: Dict[str, Any], job: ExperimentJob, *, required: bool) -> Path:
    raw = config["paths"].get(job.checkpoint_key, "")
    resolved = resolve_env(raw, required=required)
    return Path(resolved) if resolved else Path("<unset>")


def build_command(config: Dict[str, Any], job: ExperimentJob, run_dir: Path, *, smoke: bool = False) -> List[str]:
    execution = config.get("execution", {})
    configured_python = str(execution.get("python", "auto"))
    python_bin = sys.executable if configured_python == "auto" else resolve_env(configured_python, required=False)
    command = [
        python_bin,
        "-m",
        "src.tools.inference",
        "--model_dir",
        str(run_dir),
        "--fusion_method",
        str(execution.get("fusion_method", "intermediate")),
        "--seed",
        str(job.seed),
    ]
    max_samples = int(execution.get("smoke_samples", 20) if smoke else execution.get("max_samples", 0))
    if max_samples > 0:
        command.extend(["--max_samples", str(max_samples)])
    if execution.get("deterministic", True):
        command.append("--deterministic")
    if execution.get("save_box_npz", False):
        command.append("--save_box_npz")
    return command


def _checkpoint_epoch(path: Path) -> int:
    import re

    match = re.search(r"epoch(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def _copy_checkpoints(source: Path, destination: Path, patterns: Iterable[str]) -> List[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {source}")
    candidates: List[Path] = []
    for pattern in patterns:
        for checkpoint in source.glob(pattern):
            if checkpoint not in candidates:
                candidates.append(checkpoint)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files matching {list(patterns)} in {source}")
    latest = source / "latest.pth"
    selected = latest if latest.exists() else max(candidates, key=_checkpoint_epoch)
    target = destination / selected.name
    shutil.copy2(selected, target)
    return [target]


def validate_execution_prerequisites(config: Dict[str, Any], job: ExperimentJob) -> tuple[Dict[str, Any], Path]:
    """Resolve all external inputs before creating a run directory."""
    staged_config = build_run_config(config, job)
    missing_data = [
        path for path in (Path(staged_config["root_dir"]), Path(staged_config["validate_dir"]))
        if not path.exists()
    ]
    if missing_data:
        raise FileNotFoundError(f"Dataset path(s) not found: {missing_data}")

    checkpoint_dir = checkpoint_dir_for(config, job, required=True)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    patterns = config.get("execution", {}).get("checkpoint_patterns", ["net_epoch*.pth"])
    if not any(any(checkpoint_dir.glob(pattern)) for pattern in patterns):
        raise FileNotFoundError(f"No checkpoint matching {patterns} in {checkpoint_dir}")
    return staged_config, checkpoint_dir


def validate_native_outputs(run_dir: Path) -> List[str]:
    """Return missing required output/metric names for a real inference run."""
    missing = []
    summary_path = run_dir / "summary_eval.yaml"
    inference_summary_path = run_dir / "inference_summary.json"
    for path in (summary_path, inference_summary_path):
        if not path.exists():
            missing.append(path.name)
    if summary_path.exists():
        summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
        for alternatives in (
            ("ap_70", "ap70"),
            ("comm_total_normalized_ratio", "comm_normalized_ratio"),
            ("comm_total_bytes_per_frame", "comm_bytes_per_frame"),
        ):
            if not any(summary.get(key) is not None for key in alternatives):
                missing.append("|".join(alternatives))
    return missing


def validate_post_evaluation_outputs(run_dir: Path, post_cfg: Dict[str, Any]) -> List[str]:
    summary_path = run_dir / "summary_eval.yaml"
    if not summary_path.exists():
        return ["summary_eval.yaml"]
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
    missing = []
    if post_cfg.get("static_danger", {}).get("enabled", True):
        static = summary.get("danger_aware_metrics", {})
        if not isinstance(static, dict) or static.get("danger_zone_recall@0.7") is None:
            missing.append("danger_aware_metrics.danger_zone_recall@0.7")
        if not isinstance(static, dict) or static.get("risk_weighted_recall@0.7") is None:
            missing.append("danger_aware_metrics.risk_weighted_recall@0.7")
    if post_cfg.get("trajectory", {}).get("enabled", True):
        trajectory = summary.get("trajectory_danger_metrics", {})
        if not isinstance(trajectory, dict) or trajectory.get("trajectory_time_risk_recall@0.7") is None:
            missing.append("trajectory_danger_metrics.trajectory_time_risk_recall@0.7")
        if not isinstance(trajectory, dict) or trajectory.get("missed_trajectory_risk@0.7") is None:
            missing.append("trajectory_danger_metrics.missed_trajectory_risk@0.7")
    return missing


def _peer_run_dirs(config: Dict[str, Any], job: ExperimentJob, run_dir: Path, *, smoke: bool) -> List[tuple[Path, str]]:
    peers = [(run_dir, job.method)]
    post_cfg = config.get("post_evaluation", {})
    if not post_cfg.get("include_completed_peer_runs", True):
        return peers
    mode = "smoke" if smoke else "full"
    root = REPO_ROOT / config["results_root"] / "runs" / job.dataset
    if not root.exists():
        return peers
    budget = f"budget_{_plain_budget_label(job.budget_percent)}"
    loss = f"{job.loss_type}_p{job.loss_probability:.3f}"
    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        candidate = method_dir / budget / f"seed_{job.seed}" / loss / f"mc_{job.monte_carlo_run:03d}" / mode
        if candidate == run_dir or not candidate.exists():
            continue
        result_path = candidate / "publication_result.json"
        boxes = candidate / "danger_eval_boxes"
        if not result_path.exists() or not boxes.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(result.get("status", "")) not in {"completed", "smoke_completed"}:
            continue
        peers.append((candidate, str(result.get("method") or method_dir.name)))
    baseline = str(post_cfg.get("baseline_method", "snapshot_receiver_request"))
    return sorted(peers, key=lambda item: (item[1] != baseline, item[1]))


def _append_option(command: List[str], option: str, value: Any) -> None:
    command.extend([option, str(value)])


def build_post_evaluation_commands(
    config: Dict[str, Any],
    job: ExperimentJob,
    run_dir: Path,
    *,
    smoke: bool,
) -> tuple[List[List[str]], List[tuple[Path, str]]]:
    post_cfg = config.get("post_evaluation", {})
    peers = _peer_run_dirs(config, job, run_dir, smoke=smoke)
    run_dirs = [str(path) for path, _ in peers]
    methods = [method for _, method in peers]
    python_bin = sys.executable
    output_dir = run_dir / "post_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = str(post_cfg.get("baseline_method", "snapshot_receiver_request"))
    commands: List[List[str]] = []

    static_cfg = post_cfg.get("static_danger", {})
    if static_cfg.get("enabled", True):
        command = [
            python_bin, "-m", "src.tools.evaluate_danger_aware_metrics",
            "--run_dirs", *run_dirs,
            "--method_names", *methods,
            "--baseline_method", baseline,
            "--iou_thresholds", *[str(value) for value in static_cfg.get("iou_thresholds", [0.5, 0.7])],
            "--output_path", str(output_dir / "static_danger_metrics.yaml"),
            "--update_run_summaries",
        ]
        for option, key, default in (("--x_max", "x_max", 40.0), ("--y_max", "y_max", 10.0), ("--tau", "tau", 20.0)):
            _append_option(command, option, static_cfg.get(key, default))
        commands.append(command)

    trajectory_cfg = post_cfg.get("trajectory", {})
    if trajectory_cfg.get("enabled", True):
        command = [
            python_bin, "-m", "src.tools.evaluate_trajectory_danger_metrics",
            "--run_dirs", *run_dirs,
            "--method_names", *methods,
            "--baseline_method", baseline,
            "--iou_thresholds", *[str(value) for value in trajectory_cfg.get("iou_thresholds", [0.5, 0.7])],
            "--output_path", str(output_dir / "trajectory_danger_metrics.yaml"),
            "--update_run_summaries",
        ]
        options = (
            ("--horizon_seconds", "horizon_seconds", 3.0),
            ("--assumed_dt", "assumed_dt", 0.1),
            ("--d_traj_max", "d_traj_max", 5.0),
            ("--d_critical", "d_critical", 3.0),
            ("--t_critical", "t_critical", 3.0),
            ("--sigma_d", "sigma_d", 5.0),
            ("--sigma_t", "sigma_t", 2.0),
            ("--trajectory_source", "trajectory_source", "auto"),
            ("--default_speed", "default_speed", 10.0),
            ("--max_frames", "max_frames", 0),
        )
        for option, key, default in options:
            _append_option(command, option, trajectory_cfg.get(key, default))
        commands.append(command)
    return commands, peers


def refresh_publication_result(run_dir: Path) -> None:
    result_path = run_dir / "publication_result.json"
    summary_path = run_dir / "summary_eval.yaml"
    if not result_path.exists() or not summary_path.exists():
        return
    existing = json.loads(result_path.read_text(encoding="utf-8"))
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
    identity = {key: value for key, value in existing.items() if key not in {
        "ap_05", "ap_07", "feature_comm_ratio", "total_comm_ratio", "bytes_per_frame",
        "static_danger_recall_07", "static_risk_weighted_recall_07",
        "trajectory_time_risk_recall_07", "missed_trajectory_risk_07",
        "missed_trajectory_risk_reduction_07",
    }}
    refreshed = normalize_result(summary, **identity)
    result_path.write_text(json.dumps(refreshed, indent=2, allow_nan=True), encoding="utf-8")


def run_post_evaluation(
    config: Dict[str, Any],
    job: ExperimentJob,
    run_dir: Path,
    *,
    smoke: bool,
    logger,
) -> Dict[str, Any]:
    post_cfg = config.get("post_evaluation", {})
    if not post_cfg.get("enabled", False):
        logger.info("Post-evaluation disabled by config")
        return {"enabled": False, "status": "disabled", "commands": [], "peer_runs": 0}
    box_files = list((run_dir / "danger_eval_boxes").glob("frame_*.npz"))
    if not box_files:
        raise RuntimeError(f"Post-evaluation requires danger_eval_boxes/frame_*.npz in {run_dir}")

    commands, peers = build_post_evaluation_commands(config, job, run_dir, smoke=smoke)
    log_path = run_dir / "publication_post_evaluation.log"
    command_records = []
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            command_text = shlex.join(command)
            logger.command("Executing post-evaluation", cmd=command_text, peer_runs=len(peers))
            process = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            command_records.append({"command": command_text, "exit_code": process.returncode})
            if process.returncode != 0:
                raise RuntimeError(f"Post-evaluation failed with exit code {process.returncode}; see {log_path}")

    for peer_dir, _ in peers:
        refresh_publication_result(peer_dir)
    missing = validate_post_evaluation_outputs(run_dir, post_cfg)
    if missing:
        raise RuntimeError(f"Post-evaluation finished but metrics were missing: {', '.join(missing)}")
    payload = {
        "enabled": True,
        "status": "completed",
        "commands": command_records,
        "peer_runs": len(peers),
        "peer_run_dirs": [str(path) for path, _ in peers],
        "log_path": str(log_path),
        "completed_at": _utc_now(),
    }
    (run_dir / "post_evaluation_commands.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.success("Post-evaluation completed", peer_runs=len(peers), log=log_path)
    return payload


def collect_result(config: Dict[str, Any], job: ExperimentJob, run_dir: Path, checkpoint_path: str, status: str, notes: str = "") -> Path:
    summary_path = run_dir / "summary_eval.yaml"
    payload = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    row = normalize_result(payload or {}, **asdict(job))
    row.update({
        "result_path": str(summary_path),
        "checkpoint_path": checkpoint_path,
        "config_path": str(run_dir / "config_resolved.yaml"),
        "status": status,
        "notes": "; ".join(x for x in (job.notes, notes) if x),
    })
    output = run_dir / "publication_result.json"
    output.write_text(json.dumps(row, indent=2, allow_nan=True), encoding="utf-8")
    return output


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def execute_job(
    config: Dict[str, Any],
    job: ExperimentJob,
    *,
    resume: bool,
    overwrite: bool,
    smoke: bool,
    log_level: str = "INFO",
    debug: bool = False,
) -> str:
    run_dir = run_dir_for(config, job, smoke=smoke)
    result_file = run_dir / "publication_result.json"
    if result_file.exists() and resume:
        return "resumed-skip"
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory exists; use --resume or --overwrite true: {run_dir}")
    staged_config, checkpoint_dir = validate_execution_prerequisites(config, job)
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_dir / "publication_run.log"
    logger = get_logger(
        "PublicationRun",
        level="DEBUG" if debug else log_level,
        debug=debug,
        log_to_file=True,
        file_path=str(run_log_path),
    )
    logger.run(
        "Publication run started",
        experiment=job.experiment_name,
        mode="smoke" if smoke else "full",
        run_dir=run_dir,
    )

    resolved_yaml = yaml.safe_dump(staged_config, sort_keys=False)
    # inference.py expects config.yaml; config_resolved.yaml is the immutable,
    # explicitly named publication artifact requested for reproducibility.
    (run_dir / "config.yaml").write_text(resolved_yaml, encoding="utf-8")
    (run_dir / "config_resolved.yaml").write_text(resolved_yaml, encoding="utf-8")
    logger.config(
        "Resolved config saved",
        config_path=run_dir / "config_resolved.yaml",
        budget_percent=job.budget_percent,
    )
    copied = _copy_checkpoints(checkpoint_dir, run_dir, config.get("execution", {}).get("checkpoint_patterns", ["net_epoch*.pth"]))
    logger.info("Checkpoint staged", checkpoint=copied[0])
    command = build_command(config, job, run_dir, smoke=smoke)
    command_text = shlex.join(command)
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    logger.command("Executing inference", cmd=command_text)
    method_cfg = config["methods"][job.method]
    metadata = {
        **asdict(job),
        "nominal_budget_ratio": job.budget_percent / 100.0,
        "packet_loss_seed": _packet_loss_seed(job) if job.loss_type != "none" else None,
        "budget_override_paths": method_cfg.get("budget_override_paths", []),
        "run_mode": "smoke" if smoke else "full",
        "smoke_samples": int(config.get("execution", {}).get("smoke_samples", 20)) if smoke else None,
        "run_dir": str(run_dir),
        "resolved_config_path": str(run_dir / "config_resolved.yaml"),
        "command_path": str(run_dir / "command.txt"),
        "publication_run_log": str(run_log_path),
        "command": command_text,
        "checkpoint_source": str(checkpoint_dir),
        "checkpoint_files": [str(path) for path in copied],
        "started_at": _utc_now(),
        "status": "running",
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.save("Run metadata initialized", path=metadata_path)
    log_path = run_dir / "publication_inference.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    checkpoint_text = ";".join(str(path) for path in copied)
    missing_outputs = validate_native_outputs(run_dir) if process.returncode == 0 else []
    inference_success = process.returncode == 0 and not missing_outputs
    post_payload = {"enabled": False, "status": "not_run"}
    post_error = None
    if inference_success and config.get("post_evaluation", {}).get("enabled", False):
        try:
            post_payload = run_post_evaluation(config, job, run_dir, smoke=smoke, logger=logger)
        except RuntimeError as exc:
            post_error = str(exc)
            logger.error("Post-evaluation failed", error=post_error)
            if config.get("post_evaluation", {}).get("required", True):
                inference_success = False
            else:
                logger.warn("Continuing because post-evaluation is not required")

    success = inference_success
    status = "smoke_completed" if smoke and success else ("completed" if success else "failed")
    detail = f"run_mode={'smoke' if smoke else 'full'}; exit_code={process.returncode}"
    if missing_outputs:
        detail += f"; missing_outputs={','.join(missing_outputs)}"
    if post_error:
        detail += f"; post_evaluation_error={post_error}"
    collect_result(config, job, run_dir, checkpoint_text, status, detail)
    metadata.update({
        "completed_at": _utc_now(),
        "status": status,
        "exit_code": process.returncode,
        "missing_outputs": missing_outputs,
        "post_evaluation": post_payload,
        "post_evaluation_error": post_error,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not success:
        logger.error(
            "Publication run failed",
            exit_code=process.returncode,
            missing_outputs=missing_outputs,
            inference_log=log_path,
        )
        if process.returncode == 0 and missing_outputs:
            raise RuntimeError(
                "Inference finished but required native output(s) were missing: "
                f"{', '.join(missing_outputs)}. Run is marked failed."
            )
        if post_error:
            raise RuntimeError(f"Inference succeeded but post-evaluation failed: {post_error}. Run is marked failed.")
        raise RuntimeError(f"Inference failed for {job.experiment_name}; see {log_path}")
    logger.success(
        "Publication run completed",
        status=status,
        result=run_dir / "publication_result.json",
        inference_log=log_path,
    )
    return status


def post_evaluate_existing_job(
    config: Dict[str, Any],
    job: ExperimentJob,
    *,
    smoke: bool,
    log_level: str = "INFO",
    debug: bool = False,
) -> str:
    run_dir = run_dir_for(config, job, smoke=smoke)
    if not run_dir.exists():
        raise FileNotFoundError(f"Completed publication run directory not found: {run_dir}")
    missing_native = validate_native_outputs(run_dir)
    if missing_native:
        raise RuntimeError(f"Existing run is missing required native outputs: {', '.join(missing_native)}")

    logger = get_logger(
        "PublicationRun",
        level="DEBUG" if debug else log_level,
        debug=debug,
        log_to_file=True,
        file_path=str(run_dir / "publication_run.log"),
    )
    logger.run("Post-evaluating existing publication run", run_dir=run_dir, experiment=job.experiment_name)
    payload = run_post_evaluation(config, job, run_dir, smoke=smoke, logger=logger)
    refresh_publication_result(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else asdict(job)
    metadata["post_evaluation"] = payload
    metadata["post_evaluation_error"] = None
    metadata["post_evaluated_at"] = _utc_now()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.success("Existing run post-evaluation completed", result=run_dir / "publication_result.json")
    return "post_evaluated"


def str_to_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=["carla_2021", "culver_city"])
    parser.add_argument("--method")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--loss-type")
    parser.add_argument("--loss-probability", type=float)
    parser.add_argument("--monte-carlo-run", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print jobs without creating run directories (default).")
    mode.add_argument("--execute", action="store_true", help="Execute selected jobs sequentially.")
    mode.add_argument("--export-commands", type=Path, metavar="PATH", help="Write one safe execution command per selected job without running inference.")
    mode.add_argument("--post-evaluate-only", action="store_true", help="Run configured danger/trajectory evaluators on an existing completed run without inference.")
    parser.add_argument("--smoke", action="store_true", help="With --execute, use inference --max_samples on an isolated smoke run directory.")
    parser.add_argument("--allow-multiple", action="store_true", help="Explicitly permit execution of more than one selected job.")
    parser.add_argument("--resume", action="store_true", help="Skip jobs that already contain publication_result.json.")
    parser.add_argument("--overwrite", type=str_to_bool, default=False, metavar="BOOL")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"])
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and tracebacks for expected execution errors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "WARN" if args.log_level == "WARNING" else args.log_level
    logger = get_logger("PublicationRunner", level="DEBUG" if args.debug else log_level, debug=args.debug)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    try:
        config = load_config(config_path)
        jobs = filter_jobs(
            expand_jobs(
                config,
                include_disabled_loss_scenarios=bool(args.loss_type),
                only_loss_type=args.loss_type,
                extra_loss_probability=args.loss_probability,
            ),
            args,
        )
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        logger.error("Could not prepare publication grid", error=str(exc), config=config_path)
        if args.debug:
            raise
        return 2
    if not jobs:
        logger.error("No publication jobs match selected filters")
        return 2

    mode_name = (
        "execute" if args.execute else
        ("post-evaluate-only" if args.post_evaluate_only else
         ("export-commands" if args.export_commands else "dry-run"))
    )
    logger.config("Publication config loaded", config=config_path)
    logger.info("Publication plan", mode=mode_name, planned_jobs=len(jobs))
    if not args.export_commands:
        for index, job in enumerate(jobs, start=1):
            run_dir = run_dir_for(config, job, smoke=args.smoke)
            logger.info(
                "Planned job",
                index=f"{index}/{len(jobs)}",
                experiment=job.experiment_name,
                preset=job.preset,
            )
            if not args.post_evaluate_only:
                command = build_command(config, job, run_dir, smoke=args.smoke)
                logger.command("Planned command", cmd=shlex.join(command))
            logger.debug(
                "Planned job details",
                run_dir=run_dir,
                budget_percent=job.budget_percent,
                checkpoint_key=job.checkpoint_key,
                loss_type=job.loss_type,
            )

    if args.export_commands:
        output_path = args.export_commands if args.export_commands.is_absolute() else REPO_ROOT / args.export_commands
        try:
            exported = export_commands(
                output_path,
                jobs,
                config_path=config_path,
                smoke=args.smoke,
                overwrite=args.overwrite,
            )
        except FileExistsError as exc:
            logger.error("Command export failed", error=str(exc))
            if args.debug:
                raise
            return 1
        logger.save("Commands exported", path=exported, jobs=len(jobs))
        return 0

    if not args.execute and not args.post_evaluate_only:
        return 0

    if len(jobs) != 1 and not args.allow_multiple:
        logger.error(
            "Multiple-job execution requires explicit opt-in",
            selected_jobs=len(jobs),
            hint="Filter to one job or add --allow-multiple",
        )
        return 2

    for job in jobs:
        try:
            if args.post_evaluate_only:
                status = post_evaluate_existing_job(
                    config,
                    job,
                    smoke=args.smoke,
                    log_level=log_level,
                    debug=args.debug,
                )
            else:
                status = execute_job(
                    config,
                    job,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    smoke=args.smoke,
                    log_level=log_level,
                    debug=args.debug,
                )
        except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
            if isinstance(exc, FileExistsError):
                logger.warn(
                    "Run directory already exists",
                    experiment=job.experiment_name,
                    hint="Use --resume or --overwrite true",
                )
            else:
                logger.error("Publication job failed", experiment=job.experiment_name, error=str(exc))
            if args.debug:
                raise
            return 1
        logger.success("Publication job finished", experiment=job.experiment_name, status=status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
