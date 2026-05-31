"""Notebook-ready Phase 5 run cell (prepare only; do not auto-run here)."""

# Paste this cell into Kaggle and run manually.

from pathlib import Path
import shutil
import subprocess
import re
import yaml

REPO_DIR = Path("/kaggle/working/comm-aware-v2v-perception")
ENV_PY = Path("/kaggle/working/v2v_env/bin/python")
DATA_ROOT = Path("/kaggle/input/data-all")
CHECKPOINT_INPUT = Path("/kaggle/input/best-epoch")
CARLA_TEST_DIR = DATA_ROOT / "test/test"
RUN_ROOT = Path("/kaggle/working/phase_runs")
RUN_ROOT.mkdir(exist_ok=True)

force_clean = False  # keep False by default to avoid deleting old runs


def prepare_phase_run(preset, split_name="carla", run_suffix="", for_training=False):
    run_name = f"{split_name}_{preset}{('_' + run_suffix) if run_suffix else ''}"
    run_dir = RUN_ROOT / run_name

    if run_dir.exists() and force_clean:
        shutil.rmtree(run_dir)
    if not run_dir.exists():
        shutil.copytree(CHECKPOINT_INPUT, run_dir)

    src_cfg = REPO_DIR / "src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml"
    dst_cfg = run_dir / "config.yaml"
    shutil.copy(src_cfg, dst_cfg)
    shutil.copy(REPO_DIR / "src/hypes_yaml/communication_phase_presets.yaml", run_dir / "communication_phase_presets.yaml")

    validate_dir = CARLA_TEST_DIR if not for_training else (DATA_ROOT / "validate")
    text = dst_cfg.read_text()
    text = re.sub(r"communication_preset:\s*\S+", f"communication_preset: {preset}", text)
    text = re.sub(r"root_dir:\s*[\"'].*?[\"']", f'root_dir: "{DATA_ROOT / "train"}"', text)
    text = re.sub(r"validate_dir:\s*[\"'].*?[\"']", f'validate_dir: "{validate_dir}"', text)
    dst_cfg.write_text(text)
    return run_dir


def run_inference(run_dir, log_name="inference.log"):
    cmd = (
        f"cd {REPO_DIR} && "
        f"PYTHONPATH={REPO_DIR} "
        f"{ENV_PY} -u -m src.tools.inference "
        f"--model_dir {run_dir} --fusion_method intermediate --global_sort_detections "
        f"2>&1 | tee {run_dir / log_name}"
    )
    print(cmd)
    return subprocess.call(cmd, shell=True)


def load_summary(run_dir):
    p = run_dir / "summary_eval.yaml"
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text())


# Phase 5 run (CARLA)
phase5_dir = prepare_phase_run("phase5_receiver_request_topk_10", split_name="carla", run_suffix="test", for_training=False)
code = run_inference(phase5_dir, log_name="phase5_receiver_request_topk_10.log")
print("phase5 exit:", code)

# Optional: read existing phase2 baseline if available
phase2_dir = RUN_ROOT / "carla_phase2_topk_energy_10"
phase5 = load_summary(phase5_dir)
phase2 = load_summary(phase2_dir) if phase2_dir.exists() else None

rows = []

def pick(s):
    if not s:
        return None
    return {
        "AP@0.3": s.get("ap30", s.get("ap_30")),
        "AP@0.5": s.get("ap_50"),
        "AP@0.7": s.get("ap_70"),
        "feature_bytes": s.get("comm_feature_bytes_per_frame"),
        "context_bytes": s.get("comm_context_bytes_per_frame"),
        "metadata_bytes": s.get("comm_metadata_bytes_per_frame"),
        "total_bytes": s.get("comm_total_bytes_per_frame"),
        "feature_ratio": s.get("comm_feature_normalized_ratio"),
        "total_ratio": s.get("comm_total_normalized_ratio", s.get("comm_normalized_ratio")),
    }

rows.append({"run": "phase5_receiver_request_topk_10", **(pick(phase5) or {})})
if phase2:
    rows.append({"run": "phase2_topk_energy_10", **pick(phase2)})

print("\nCompact comparison table")
for r in rows:
    print(r)

# Save compact summary + backup zip
import csv
summary_csv = RUN_ROOT / "phase5_vs_phase2_compact.csv"
with open(summary_csv, "w", newline="") as f:
    fieldnames = [
        "run", "AP@0.3", "AP@0.5", "AP@0.7",
        "feature_bytes", "context_bytes", "metadata_bytes", "total_bytes",
        "feature_ratio", "total_ratio"
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("Saved:", summary_csv)
print("Config lines:")
subprocess.call(
    f'grep -n "communication_preset\\|root_dir:\\|validate_dir:" "{phase5_dir / "config.yaml"}"',
    shell=True,
)

zip_path = Path("/kaggle/working/phase5_artifacts.zip")
subprocess.call(
    f'cd /kaggle/working && zip -r "{zip_path}" phase_runs/carla_phase5_receiver_request_topk_10_test phase_runs/phase5_vs_phase2_compact.csv',
    shell=True,
)
print("Backup zip:", zip_path)
