import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.utils.logging import get_logger


LOGGER = get_logger("SmokeTest")


def default_runs_root() -> Path:
    kaggle_root = Path("/kaggle/working/approach_runs")
    if Path("/kaggle").exists():
        return kaggle_root
    return Path("./approach_runs")


def default_split_paths(split: str) -> Tuple[str, str]:
    split = str(split).lower()
    root_dir = "/kaggle/input/data-all/train"
    if split == "culver":
        validate_dir = "/kaggle/input/data-all/test/test_culver_city/test_culver_city"
    else:
        validate_dir = "/kaggle/input/data-all/test/test"
    return root_dir, validate_dir


def build_smoke_run_name(split: str, approach_name: str) -> str:
    safe = str(approach_name).strip().replace(" ", "_")
    return f"smoke_{split}_{safe}"


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(path: Path, data: dict):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_comm_cfg(cfg: dict, create: bool = False) -> dict:
    model = cfg.setdefault("model", {}) if create else cfg.get("model", {})
    args = model.setdefault("args", {}) if create else model.get("args", {})
    comm = args.setdefault("communication", {}) if create else args.get("communication", {})
    if not isinstance(comm, dict):
        if create:
            args["communication"] = {}
            return args["communication"]
        return {}
    return comm


def resolve_checkpoint_source(args) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    if args.model_dir:
        return Path(args.model_dir)
    raise ValueError("Provide --checkpoint_dir or --model_dir so smoke test can load checkpoint files.")


def copy_checkpoint_files(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        raise FileNotFoundError(f"Checkpoint source not found: {src_dir}")
    copied = 0
    patterns = ["net_epoch*.pth", "latest.pth"]
    for pat in patterns:
        for p in src_dir.glob(pat):
            shutil.copy2(p, dst_dir / p.name)
            copied += 1
    if copied == 0:
        raise FileNotFoundError(f"No checkpoint files found in: {src_dir}")
    return copied


def copy_support_files(dst_dir: Path):
    preset_src = Path("src/hypes_yaml/communication_approach_presets.yaml")
    if preset_src.exists():
        shutil.copy2(preset_src, dst_dir / preset_src.name)


def patch_config_for_smoke(
    cfg: dict,
    *,
    split: str,
    root_dir_override: Optional[str],
    validate_dir_override: Optional[str],
    save_debug_maps: bool,
    debug_num_frames: int,
    debug_dir: Path,
    approach_name: Optional[str],
):
    root_dir_default, validate_dir_default = default_split_paths(split)
    cfg["root_dir"] = root_dir_override or root_dir_default
    cfg["validate_dir"] = validate_dir_override or validate_dir_default

    if approach_name:
        cfg["communication_preset"] = approach_name

    comm_cfg = get_comm_cfg(cfg, create=True)
    rr_cfg = comm_cfg.setdefault("receiver_request", {})
    if isinstance(rr_cfg, dict):
        rr_cfg["save_request_maps"] = bool(save_debug_maps)
        rr_cfg["debug_num_frames"] = int(debug_num_frames)
        rr_cfg["debug_dir"] = str(debug_dir)
        temporal_cfg = rr_cfg.setdefault("temporal", {})
        if isinstance(temporal_cfg, dict):
            temporal_cfg["save_temporal_maps"] = bool(save_debug_maps)
            temporal_cfg["debug_num_frames"] = int(debug_num_frames)
            temporal_cfg["debug_dir"] = str(debug_dir.parent / "temporal_receiver_request_debug")


def is_planned_placeholder(preset_cfg: dict) -> bool:
    md = preset_cfg.get("metadata", {}) if isinstance(preset_cfg.get("metadata", {}), dict) else {}
    rr = preset_cfg.get("receiver_request", {}) if isinstance(preset_cfg.get("receiver_request", {}), dict) else {}
    status = str(md.get("implementation_status", rr.get("implementation_status", "implemented"))).lower()
    return status in {"planned_placeholder", "planned"}


def is_enabled_preset(preset_cfg: dict) -> bool:
    return bool(preset_cfg.get("enabled", False))


def list_approaches_to_run(
    presets: Dict[str, dict],
    *,
    allow_planned: bool,
    skip_planned: bool,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    selected: List[str] = []
    skipped: List[Tuple[str, str]] = []
    for name in sorted(presets.keys()):
        cfg = presets[name]
        planned = is_planned_placeholder(cfg)
        enabled = is_enabled_preset(cfg)
        strategy = str(cfg.get("strategy", "none")).lower()
        if planned and skip_planned and not allow_planned:
            skipped.append((name, "planned_placeholder"))
            continue
        if not enabled and strategy != "none":
            skipped.append((name, "disabled"))
            continue
        selected.append(name)
    return selected, skipped


def required_metric_keys(is_receiver_request: bool, is_temporal: bool, require_ap: bool) -> List[str]:
    keys = [
        "comm_feature_bytes_per_frame",
        "comm_total_bytes_per_frame",
        "comm_normalized_ratio",
    ]
    if require_ap:
        keys.extend(["ap_50", "ap_70"])
    if is_receiver_request:
        keys.extend([
            "comm_context_bytes_per_frame",
            "comm_metadata_bytes_per_frame",
            "comm_feature_normalized_ratio",
            "comm_context_normalized_ratio",
            "comm_metadata_normalized_ratio",
            "comm_total_normalized_ratio",
            "receiver_request_keep_ratio",
        ])
    if is_temporal:
        keys.extend([
            "temporal_novelty_mean",
            "temporal_cache_age_mean",
            "temporal_cache_hit_ratio",
            "temporal_refresh_ratio",
            "temporal_init_frame_ratio",
            "comm_cumulative_bytes_per_scenario",
            "comm_average_bytes_per_frame",
            "comm_total_bytes_per_frame_after_init",
            "comm_total_normalized_ratio_after_init",
        ])
    return keys


def validate_required_metrics(summary: dict, is_receiver_request: bool, is_temporal: bool, require_ap: bool) -> List[str]:
    missing = []
    for k in required_metric_keys(is_receiver_request, is_temporal, require_ap):
        if k not in summary or summary.get(k) is None:
            missing.append(k)
    # ap30 compatibility: allow ap30 or ap_30
    if require_ap and ("ap30" not in summary and "ap_30" not in summary):
        missing.append("ap30_or_ap_30")
    return missing


def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with open(summary_path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def validate_outputs(run_dir: Path, is_receiver_request: bool, is_temporal: bool, save_debug_maps: bool, require_ap: bool) -> Tuple[bool, List[str], List[str], dict]:
    required_files = [
        run_dir / "config.yaml",
        run_dir / "smoke_inference.log",
        run_dir / "summary_eval.yaml",
    ]
    missing_files = [str(p) for p in required_files if not p.exists()]

    # Optional but expected in this repo
    optional_expected = [run_dir / "inference_summary.json"]
    frame_jsonl = run_dir / "comm_metrics_frame.jsonl"
    if frame_jsonl.exists() is False:
        missing_files.append(str(frame_jsonl))

    summary = load_summary(run_dir / "summary_eval.yaml")
    missing_metrics = validate_required_metrics(summary, is_receiver_request=is_receiver_request, is_temporal=is_temporal, require_ap=require_ap)

    if save_debug_maps:
        debug_dir = run_dir / ("temporal_receiver_request_debug" if is_temporal else "receiver_request_debug")
        npz_count = len(list(debug_dir.glob("*.npz"))) if debug_dir.exists() else 0
        if npz_count < 1:
            missing_files.append(f"{debug_dir}/*.npz")

    ok = len(missing_files) == 0 and len(missing_metrics) == 0
    # keep optional presence for report transparency
    optional_files_found = [str(p) for p in optional_expected if p.exists()]
    summary["_optional_files_found"] = optional_files_found
    return ok, missing_files, missing_metrics, summary


def prepare_run_dir(run_dir: Path, force_clean: bool):
    if run_dir.exists() and force_clean:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def resolve_python() -> str:
    return sys.executable


def run_inference_for_smoke(run_dir: Path, max_samples: int, skip_ap: bool) -> int:
    cmd = [
        resolve_python(),
        "-m",
        "src.tools.inference",
        "--model_dir",
        str(run_dir),
        "--fusion_method",
        "intermediate",
        "--max_samples",
        str(max_samples),
    ]
    if skip_ap:
        cmd.append("--skip_ap")

    LOGGER.command("Executing inference", cmd=" ".join(cmd), run_dir=str(run_dir))
    log_path = run_dir / "smoke_inference.log"
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=str(Path.cwd()), stdout=lf, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def write_report(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _infer_receiver_request_mode(cfg: dict, approach_name: str) -> bool:
    if "receiver_request" in str(approach_name):
        return True
    comm_cfg = get_comm_cfg(cfg, create=False)
    strategy = str(comm_cfg.get("strategy", "none"))
    return strategy == "receiver_request_topk"


def _infer_temporal_mode(cfg: dict, approach_name: str) -> bool:
    if "temporal_receiver_request" in str(approach_name):
        return True
    comm_cfg = get_comm_cfg(cfg, create=False)
    rr_cfg = comm_cfg.get("receiver_request", {}) if isinstance(comm_cfg.get("receiver_request", {}), dict) else {}
    temporal_cfg = rr_cfg.get("temporal", {}) if isinstance(rr_cfg.get("temporal", {}), dict) else {}
    variant = str(rr_cfg.get("strategy_variant", "")).lower()
    return bool(temporal_cfg.get("enabled", False)) or variant.startswith("temporal_")


def _is_comm_frame_json_required(cfg: dict) -> bool:
    comm_cfg = get_comm_cfg(cfg, create=False)
    logging_cfg = comm_cfg.get("logging", {}) if isinstance(comm_cfg.get("logging", {}), dict) else {}
    return bool(logging_cfg.get("save_per_frame_json", True))


def _build_run_config(args, approach_name: Optional[str], run_dir: Path) -> Path:
    if args.config:
        base_cfg = load_yaml(Path(args.config))
    else:
        base_cfg = load_yaml(Path(args.base_config))

    patch_config_for_smoke(
        base_cfg,
        split=args.split,
        root_dir_override=args.root_dir,
        validate_dir_override=args.validate_dir,
        save_debug_maps=args.save_debug_maps,
        debug_num_frames=args.debug_num_frames,
        debug_dir=run_dir / "receiver_request_debug",
        approach_name=approach_name,
    )

    cfg_path = run_dir / "config.yaml"
    save_yaml(cfg_path, base_cfg)
    return cfg_path


def run_single_smoke(args, approach_name: Optional[str]) -> dict:
    name_for_path = approach_name or Path(args.config).stem
    run_name = build_smoke_run_name(args.split, name_for_path)
    run_dir = Path(args.runs_root) / run_name
    prepare_run_dir(run_dir, args.force_clean)

    cfg_path = _build_run_config(args, approach_name, run_dir)
    cfg_data = load_yaml(cfg_path)

    if args.dry_run:
        LOGGER.info(
            "Dry run only",
            approach=name_for_path,
            split=args.split,
            config_path=str(cfg_path),
            run_dir=str(run_dir),
            copied_checkpoints=0,
        )
        return {
            "status": "dry_run",
            "approach": name_for_path,
            "split": args.split,
            "max_samples": int(args.max_samples),
            "run_dir": str(run_dir),
            "config_path": str(cfg_path),
        }

    checkpoint_src = resolve_checkpoint_source(args)
    copied_ckpts = copy_checkpoint_files(checkpoint_src, run_dir)
    copy_support_files(run_dir)

    start = time.time()
    skip_ap = bool(args.skip_ap or (not args.include_ap))
    exit_code = run_inference_for_smoke(run_dir, max_samples=int(args.max_samples), skip_ap=skip_ap)

    receiver_request = _infer_receiver_request_mode(cfg_data, name_for_path)
    temporal_request = _infer_temporal_mode(cfg_data, name_for_path)
    ok, missing_files, missing_metrics, summary = validate_outputs(
        run_dir,
        is_receiver_request=receiver_request,
        is_temporal=temporal_request,
        save_debug_maps=bool(args.save_debug_maps),
        require_ap=(not skip_ap),
    )
    if not _is_comm_frame_json_required(cfg_data):
        frame_json_path = str(run_dir / "comm_metrics_frame.jsonl")
        missing_files = [x for x in missing_files if x != frame_json_path]
        ok = len(missing_files) == 0 and len(missing_metrics) == 0

    status = "pass" if exit_code == 0 and ok else "fail"
    err = None
    if exit_code != 0:
        err = f"inference_exit_code={exit_code}"
    elif missing_files:
        err = f"missing_files={missing_files}"
    elif missing_metrics:
        err = f"missing_metrics={missing_metrics}"

    inference_summary_path = run_dir / "inference_summary.json"
    inference_summary = {}
    if inference_summary_path.exists():
        with open(inference_summary_path, "r") as f:
            inference_summary = json.load(f)

    elapsed = time.time() - start
    report = {
        "status": status,
        "approach": name_for_path,
        "split": args.split,
        "max_samples": int(args.max_samples),
        "run_dir": str(run_dir),
        "config_path": str(cfg_path),
        "summary_path": str(run_dir / "summary_eval.yaml"),
        "processed_frames": inference_summary.get("processed_frames", None),
        "elapsed_seconds": float(elapsed),
        "required_files_ok": len(missing_files) == 0,
        "required_metrics_ok": len(missing_metrics) == 0,
        "missing_files": missing_files,
        "missing_metrics": missing_metrics,
        "metrics": {
            "ap30": summary.get("ap30", summary.get("ap_30", None)),
            "ap_50": summary.get("ap_50", None),
            "ap_70": summary.get("ap_70", None),
            "comm_feature_bytes_per_frame": summary.get("comm_feature_bytes_per_frame", None),
            "comm_context_bytes_per_frame": summary.get("comm_context_bytes_per_frame", None),
            "comm_metadata_bytes_per_frame": summary.get("comm_metadata_bytes_per_frame", None),
            "comm_total_bytes_per_frame": summary.get("comm_total_bytes_per_frame", None),
            "comm_normalized_ratio": summary.get("comm_normalized_ratio", None),
            "comm_feature_normalized_ratio": summary.get("comm_feature_normalized_ratio", None),
            "comm_context_normalized_ratio": summary.get("comm_context_normalized_ratio", None),
            "comm_metadata_normalized_ratio": summary.get("comm_metadata_normalized_ratio", None),
            "comm_total_normalized_ratio": summary.get("comm_total_normalized_ratio", None),
            "receiver_request_keep_ratio": summary.get("receiver_request_keep_ratio", None),
            "temporal_novelty_mean": summary.get("temporal_novelty_mean", None),
            "temporal_cache_age_mean": summary.get("temporal_cache_age_mean", None),
            "temporal_cache_hit_ratio": summary.get("temporal_cache_hit_ratio", None),
            "temporal_init_frame_ratio": summary.get("temporal_init_frame_ratio", None),
            "comm_cumulative_bytes_per_scenario": summary.get("comm_cumulative_bytes_per_scenario", None),
            "comm_average_bytes_per_frame": summary.get("comm_average_bytes_per_frame", None),
        },
        "error_message": err,
    }
    write_report(run_dir / "smoke_test_report.json", report)

    if status == "pass":
        LOGGER.success(
            "Smoke test passed",
            approach=name_for_path,
            split=args.split,
            frames=args.max_samples,
        )
        LOGGER.metric(
            "Result",
            ap70=report["metrics"]["ap_70"],
            total_ratio=report["metrics"]["comm_total_normalized_ratio"],
            feature_ratio=report["metrics"]["comm_feature_normalized_ratio"],
            total_bytes=report["metrics"]["comm_total_bytes_per_frame"],
        )
        LOGGER.save("Outputs", summary=str(run_dir / "summary_eval.yaml"), run_dir=str(run_dir))
    else:
        LOGGER.error("Smoke test failed", approach=name_for_path, reason=err)

    return report


def write_index_csv(index_path: Path, rows: List[dict]):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "approach",
            "split",
            "status",
            "ap70",
            "comm_total_normalized_ratio",
            "comm_total_bytes_per_frame",
            "run_dir",
            "error_message",
        ])
        for r in rows:
            m = r.get("metrics", {})
            writer.writerow([
                r.get("approach"),
                r.get("split"),
                r.get("status"),
                m.get("ap_70"),
                m.get("comm_total_normalized_ratio"),
                m.get("comm_total_bytes_per_frame"),
                r.get("run_dir"),
                r.get("error_message"),
            ])


def parse_args():
    parser = argparse.ArgumentParser(description="Centralized smoke test for inference pipeline")
    parser.add_argument("--approach", type=str, default=None, help="Communication approach preset name")
    parser.add_argument("--config", type=str, default=None, help="Explicit config.yaml path")
    parser.add_argument("--all_approaches", action="store_true", help="Run smoke test for all runnable approaches")
    parser.add_argument("--split", type=str, default="carla", choices=["carla", "culver"])
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--runs_root", type=str, default=str(default_runs_root()))
    parser.add_argument("--model_dir", type=str, default=None, help="Model/checkpoint run directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory with net_epoch*.pth")
    parser.add_argument("--base_config", type=str, default="src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml")
    parser.add_argument("--root_dir", type=str, default=None, help="Override root_dir in smoke config")
    parser.add_argument("--validate_dir", type=str, default=None, help="Override validate_dir in smoke config")
    parser.add_argument("--force_clean", action="store_true")
    parser.add_argument("--save_debug_maps", action="store_true")
    parser.add_argument("--debug_num_frames", type=int, default=3)
    parser.add_argument("--include_ap", action="store_true", default=True)
    parser.add_argument("--no-include_ap", dest="include_ap", action="store_false")
    parser.add_argument("--skip_ap", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_planned", action="store_true", default=False)
    parser.add_argument("--skip_planned", action="store_true", default=True)
    parser.add_argument("--no-skip_planned", dest="skip_planned", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and args.approach:
        LOGGER.warn("Both --config and --approach provided; explicit config mode will use --config file")

    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")

    Path(args.runs_root).mkdir(parents=True, exist_ok=True)

    results = []
    if args.all_approaches:
        preset_file = Path("src/hypes_yaml/communication_approach_presets.yaml")
        preset_data = load_yaml(preset_file)
        presets = preset_data.get("communication_presets", {})
        selected, skipped = list_approaches_to_run(
            presets,
            allow_planned=bool(args.allow_planned),
            skip_planned=bool(args.skip_planned),
        )
        for name, reason in skipped:
            LOGGER.warn("Skipping approach", approach=name, reason=reason)
        if not selected:
            raise RuntimeError("No runnable approaches selected.")
        for approach_name in selected:
            try:
                res = run_single_smoke(args, approach_name=approach_name)
            except Exception as e:
                res = {
                    "status": "fail",
                    "approach": approach_name,
                    "split": args.split,
                    "max_samples": int(args.max_samples),
                    "run_dir": str(Path(args.runs_root) / build_smoke_run_name(args.split, approach_name)),
                    "metrics": {},
                    "error_message": str(e),
                }
                LOGGER.error("Smoke test crashed", approach=approach_name, reason=str(e))
            results.append(res)
        write_index_csv(Path(args.runs_root) / "smoke_test_index.csv", results)
    else:
        approach_name = None if args.config else args.approach
        if not args.config and not approach_name:
            raise ValueError("Use one of: --approach, --config, or --all_approaches")
        if approach_name and not args.config:
            preset_file = Path("src/hypes_yaml/communication_approach_presets.yaml")
            preset_data = load_yaml(preset_file)
            presets = preset_data.get("communication_presets", {})
            if approach_name not in presets:
                raise ValueError(f"Unknown approach preset: {approach_name}")
            if is_planned_placeholder(presets[approach_name]) and args.skip_planned and not args.allow_planned:
                raise ValueError(
                    f"Approach is planned placeholder and skipped by default: {approach_name}. "
                    "Use --allow_planned to force running."
                )
        result = run_single_smoke(args, approach_name=approach_name)
        results.append(result)

    total = len(results)
    failed = len([r for r in results if r.get("status") != "pass" and r.get("status") != "dry_run"])
    if failed > 0:
        raise SystemExit(1)
    LOGGER.success("All requested smoke tests completed", total=total, failed=failed)


if __name__ == "__main__":
    main()
