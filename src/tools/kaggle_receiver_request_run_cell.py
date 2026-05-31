"""Notebook-ready receiver-request run cell (prepare only; do not auto-run here)."""

from pathlib import Path
import shutil
import subprocess
import re
import yaml
import csv

REPO_DIR = Path("/kaggle/working/comm-aware-v2v-perception")
ENV_PY = Path("/kaggle/working/v2v_env/bin/python")
DATA_ROOT = Path("/kaggle/input/data-all")
CHECKPOINT_INPUT = Path("/kaggle/input/best-epoch")
CARLA_TEST_DIR = DATA_ROOT / "test/test"
RUN_ROOT = Path("/kaggle/working/approach_runs")
RUN_ROOT.mkdir(exist_ok=True)

force_clean = False
REQUEST_PRESET = "receiver_request_energy_topk_10"
REFERENCE_PRESET = "selective_topk_energy_10"


def prepare_approach_run(preset, split_name="carla", run_suffix="", for_training=False):
    run_name = f"{split_name}_{preset}{('_' + run_suffix) if run_suffix else ''}"
    run_dir = RUN_ROOT / run_name

    if run_dir.exists() and force_clean:
        shutil.rmtree(run_dir)
    if not run_dir.exists():
        shutil.copytree(CHECKPOINT_INPUT, run_dir)

    src_cfg = REPO_DIR / "src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml"
    dst_cfg = run_dir / "config.yaml"
    shutil.copy(src_cfg, dst_cfg)
    shutil.copy(REPO_DIR / "src/hypes_yaml/communication_approach_presets.yaml", run_dir / "communication_approach_presets.yaml")

    validate_dir = CARLA_TEST_DIR if not for_training else (DATA_ROOT / "validate")
    text = dst_cfg.read_text()
    text = re.sub(r"communication_preset:\s*\S+", f"communication_preset: {preset}", text)
    text = re.sub(r'root_dir:\s*["\'].*?["\']', f'root_dir: "{DATA_ROOT / "train"}"', text)
    text = re.sub(r'validate_dir:\s*["\'].*?["\']', f'validate_dir: "{validate_dir}"', text)
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


def pick(summary):
    if not summary:
        return None
    return {
        "AP@0.3": summary.get("ap30", summary.get("ap_30")),
        "AP@0.5": summary.get("ap_50"),
        "AP@0.7": summary.get("ap_70"),
        "feature_bytes": summary.get("comm_feature_bytes_per_frame"),
        "context_bytes": summary.get("comm_context_bytes_per_frame"),
        "metadata_bytes": summary.get("comm_metadata_bytes_per_frame"),
        "total_bytes": summary.get("comm_total_bytes_per_frame"),
        "feature_ratio": summary.get("comm_feature_normalized_ratio"),
        "total_ratio": summary.get("comm_total_normalized_ratio", summary.get("comm_normalized_ratio")),
    }


request_dir = prepare_approach_run(REQUEST_PRESET, split_name="carla", run_suffix="test", for_training=False)
code = run_inference(request_dir, log_name=f"{REQUEST_PRESET}.log")
print("receiver_request exit:", code)

reference_dir = RUN_ROOT / f"carla_{REFERENCE_PRESET}"
request_summary = load_summary(request_dir)
reference_summary = load_summary(reference_dir) if reference_dir.exists() else None

rows = [{"run": REQUEST_PRESET, **(pick(request_summary) or {})}]
if reference_summary:
    rows.append({"run": REFERENCE_PRESET, **pick(reference_summary)})

for row in rows:
    print(row)

summary_csv = RUN_ROOT / "receiver_request_vs_topk_compact.csv"
with open(summary_csv, "w", newline="") as f:
    fieldnames = [
        "run", "AP@0.3", "AP@0.5", "AP@0.7",
        "feature_bytes", "context_bytes", "metadata_bytes", "total_bytes",
        "feature_ratio", "total_ratio",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", summary_csv)
subprocess.call(
    f'grep -n "communication_preset\\|root_dir:\\|validate_dir:" "{request_dir / "config.yaml"}"',
    shell=True,
)

zip_path = Path("/kaggle/working/receiver_request_artifacts.zip")
subprocess.call(
    f'cd /kaggle/working && zip -r "{zip_path}" approach_runs/carla_{REQUEST_PRESET}_test approach_runs/receiver_request_vs_topk_compact.csv',
    shell=True,
)
print("Backup zip:", zip_path)
