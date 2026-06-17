# Communication-Aware V2V Cooperative Perception

Communication-aware cooperative 3D object detection with PointPillars + V2VAM intermediate fusion. The repository focuses on reducing V2V feature communication while preserving detection quality, and includes receiver-driven, temporal receiver-driven, learned experimental, and safety-oriented evaluation tools.

## What Is Included

- PointPillars-style LiDAR voxelization and BEV feature extraction.
- V2VAM intermediate feature fusion.
- Config-driven communication policies.
- Sender-side selective sharing baselines.
- Receiver-driven request-based communication.
- Temporal receiver-request with cache, novelty, age, and confidence terms.
- Trainable experimental learned temporal request head.
- Communication metrics: bytes/frame, active ratio, normalized communication ratio.
- Safety metrics: danger-zone recall, risk-weighted recall, trajectory-aware missed-risk metrics.
- Centralized logging for training, inference, communication policies, smoke tests, summaries, and evaluation tools.
- Smoke-test pipeline for reproducible public runs.

## Existing Features

This repository already includes the main pieces needed to run communication-aware cooperative perception experiments:

- Configurable approach presets in `src/hypes_yaml/communication_approach_presets.yaml`.
- Baseline full-communication evaluation.
- Sender-side selective communication baselines such as random collaborator-only masking and top-k feature-energy sharing.
- Receiver-driven request communication with `receiver_request_energy_topk_10`.
- Temporal receiver-request communication with cache, novelty, age, and confidence terms.
- Trainable experimental learned temporal request head with checkpoint-safety protection.
- Communication accounting for feature bytes, context bytes, metadata bytes, total bytes, and normalized ratios.
- Safety-oriented evaluation metrics for danger-aware and trajectory-aware missed detections.
- Centralized logging through `src/utils/logging/`.
- Smoke-test pipeline for short validation runs before launching full experiments.

## Configurable For New Environments

The repo is designed so users can move between machines without editing source code. The main values to change are paths and runtime options:

```bash
export REPO_DIR="<REPO_DIR>"
export DATA_ROOT="<DATA_ROOT>"
export CHECKPOINT_DIR="<CHECKPOINT_DIR>"
export RUNS_ROOT="<RUNS_ROOT>"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO_DIR"
```

Most experiment changes should be made through YAML presets or CLI overrides, not by editing model code. Typical configurable items are dataset paths, checkpoint path, approach name, split, `max_samples`, debug-map export, logging options, communication keep ratio, packet loss, temporal-cache settings, and learned-request training options.

## Documentation

Start here:

- Full setup and experiment guide: `docs/setup_and_experiments.md`
- Approach overview: `docs/approach_guide.md`
- Receiver-request roadmap: `docs/receiver_request_approach_guide.md`
- Kaggle/notebook workflow: `docs/notebooks/communication_approach_pipeline_testing.ipynb`

The full guide uses public placeholders such as `<REPO_DIR>`, `<DATA_ROOT>`, `<CHECKPOINT_DIR>`, and `<RUNS_ROOT>`. Replace them with paths on your own machine.

## Recommended Environment

For full inference/training:

- Linux + NVIDIA GPU.
- CUDA-compatible PyTorch.
- 16 GB GPU memory minimum; 24 GB+ recommended.
- 32 GB system RAM minimum; 64 GB recommended.
- Python 3.8-3.10 recommended for the smoothest dependency setup.

macOS is useful for editing and pure unit tests, but full experiments are best run on Linux because `spconv` is CUDA/Linux-oriented.

The detailed setup guide documents multiple supported runtime profiles, including Linux CUDA 12.1, Linux CUDA 11.8, conda, virtualenv, Kaggle current Python, Kaggle isolated virtualenv, Colab, Docker, HPC/SLURM, WSL2, macOS dev-only, CPU-only dev, offline server, and legacy Python 3.7/3.8. See `docs/setup_and_experiments.md`.

## Quick Start

```bash
git clone https://github.com/mohsenshahverdy/comm-aware-v2v-perception.git <REPO_DIR>
cd <REPO_DIR>

source <ENV_DIR>/bin/activate
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0

export DATA_ROOT="<DATA_ROOT>"
export TRAIN_DIR="$DATA_ROOT/train"
export CARLA_TEST_DIR="$DATA_ROOT/test/test"
export CHECKPOINT_DIR="<CHECKPOINT_DIR>"
export RUNS_ROOT="<RUNS_ROOT>"
```

Run a 20-frame smoke test:

```bash
python -m src.tools.testing.smoke_test_pipeline \
  --approach receiver_request_energy_topk_10 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$CARLA_TEST_DIR" \
  --split carla \
  --max_samples 20 \
  --runs_root "$RUNS_ROOT" \
  --save_box_npz \
  --force_clean
```

For installation details, dataset layout, checkpoint setup, full experiments, training, and metrics, see `docs/setup_and_experiments.md`.

## Dataset Layout

The loader expects OpenCOOD/OPV2V-style cooperative perception data:

```text
<DATA_ROOT>/
  train/<scenario_id>/<cav_id>/<frame_id>.yaml
  train/<scenario_id>/<cav_id>/<frame_id>.pcd
  validate/<scenario_id>/<cav_id>/<frame_id>.yaml
  validate/<scenario_id>/<cav_id>/<frame_id>.pcd
  test/test/<scenario_id>/<cav_id>/<frame_id>.yaml
  test/test/<scenario_id>/<cav_id>/<frame_id>.pcd
  test/test_culver_city/test_culver_city/<scenario_id>/<cav_id>/<frame_id>.yaml
  test/test_culver_city/test_culver_city/<scenario_id>/<cav_id>/<frame_id>.pcd
```

## Approach Naming

All experiments use approach-based names in:

```text
src/hypes_yaml/communication_approach_presets.yaml
```

Main approach families:

- `baseline_full_communication`
- `measurement_full_communication`
- `selective_topk_energy_*`
- `selective_random_comm_only_*`
- `robustness_packetloss_*`
- `receiver_request_energy_topk_*`
- `temporal_receiver_request_energy_topk_10`
- `learned_temporal_receiver_request_10`

## Receiver-Driven Roadmap

| Approach family | Train required | Reportable with detector checkpoint | Status |
|---|---:|---:|---|
| Snapshot receiver-request | No | Yes | Runnable |
| Temporal receiver-request | No | Yes | Runnable |
| Learned temporal receiver-request | Yes | Only after request-head training | Trainable experimental |

Runnable non-learned receiver-driven methods:

- `receiver_request_energy_topk_10`
- `temporal_receiver_request_energy_topk_10`

Trainable experimental method:

- `learned_temporal_receiver_request_10`

The learned temporal approach requires trained `comm_policy.learned_temporal_request_head.*` weights. Inference is blocked by default if these weights are missing. The debug flag `--allow_untrained_request_head` is non-reportable and should not be used for final results.

## Common Commands

Compile check:

```bash
python -m py_compile \
  src/tools/inference.py \
  src/tools/train.py \
  src/models/point_pillar_intermediate_V2VAM.py \
  src/models/fuse_modules/communication_policy.py
```

Unit tests:

```bash
python -m src.tools.testing.test_comm_policy_fake
python -m src.tools.testing.test_v2vam_correctness
python -m src.tools.testing.test_temporal_receiver_request
python -m src.tools.testing.test_danger_aware_metrics
python -m src.tools.testing.test_trajectory_danger_metrics
```

Full CARLA-style approach run:

```bash
python -m src.tools.testing.smoke_test_pipeline \
  --approach temporal_receiver_request_energy_topk_10 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$CARLA_TEST_DIR" \
  --split carla \
  --max_samples 999999 \
  --runs_root "$RUNS_ROOT" \
  --save_box_npz \
  --force_clean
```

Static danger-aware metrics:

```bash
python -m src.tools.evaluate_danger_aware_metrics \
  --run_dirs "$RUNS_ROOT/smoke_carla_receiver_request_energy_topk_10" \
  --method_names receiver_request_energy_topk_10 \
  --baseline_method receiver_request_energy_topk_10 \
  --output_path "$RUNS_ROOT/danger_aware_metrics.yaml"
```

Trajectory-aware danger metrics:

```bash
python -m src.tools.evaluate_trajectory_danger_metrics \
  --run_dirs "$RUNS_ROOT/smoke_carla_receiver_request_energy_topk_10" \
  --method_names receiver_request_energy_topk_10 \
  --baseline_method receiver_request_energy_topk_10 \
  --trajectory_source auto \
  --output_path "$RUNS_ROOT/trajectory_danger_metrics.yaml"
```

## Centralized Logging

The repository uses one logger API in `src/utils/logging/` so runs have consistent component names, message types, and key-value metrics. Important scripts now use this logger for startup config, commands, checkpoints, AP metrics, communication metrics, safety metrics, debug warnings, and saved artifacts.

```python
from src.utils.logging import get_logger

logger = get_logger("CommunicationPolicy")
logger.info("Communication approach", public_name="receiver_request_energy_topk_10")
logger.metric("Evaluation result", ap50=0.95, ap70=0.89, comm_ratio=0.10)
logger.success("Summary saved", path="<RUNS_ROOT>/summary_eval.yaml")
```

Common controls:

```bash
export LOG_LEVEL=INFO
export LOG_DEBUG=false
export LOG_TIMESTAMP=false
export LOG_COLOR=false
export LOG_TO_FILE=true
export LOG_FILE="<RUNS_ROOT>/run.log"
```

Supported message helpers include `info`, `warning`, `error`, `success`, `debug`, `metric`, `config`, `run`, `command`, and `save`. To verify the logger in a new environment:

```bash
python -m src.tools.testing.test_centralized_logger
```

## Key Files

- Main config: `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml`
- Approach presets: `src/hypes_yaml/communication_approach_presets.yaml`
- Training: `src/tools/train.py`
- Inference: `src/tools/inference.py`
- Smoke tests: `src/tools/testing/smoke_test_pipeline.py`
- Communication policy: `src/models/fuse_modules/communication_policy.py`
- Static safety metrics: `src/tools/evaluate_danger_aware_metrics.py`
- Trajectory safety metrics: `src/tools/evaluate_trajectory_danger_metrics.py`

## Outputs

Each run directory can contain:

```text
config.yaml
net_epoch*.pth
summary_eval.yaml
inference_summary.json
smoke_test_report.json
comm_metrics_epoch.csv
comm_metrics_frame.jsonl
danger_eval_boxes/frame_*.npz
receiver_request_debug/*.npz
temporal_receiver_request_debug/*.npz
```

## License

Check `setup.py` and source headers for the current license terms before redistribution or commercial use.
