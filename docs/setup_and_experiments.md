# Setup And Experiment Guide

This guide explains how to install the repository, prepare data/checkpoints, run smoke tests, run full communication-aware evaluations, train experimental approaches, and export safety-oriented metrics.

All paths below are placeholders. Replace them with paths on your own machine:

```bash
<REPO_DIR>        # cloned repository directory
<DATA_ROOT>       # dataset root containing train/validate/test splits
<CHECKPOINT_DIR>  # directory containing net_epoch*.pth or latest.pth
<RUNS_ROOT>       # output directory for experiment runs
<ENV_DIR>         # Python virtual environment or conda environment path
```

Example placeholder expansion:

```bash
export REPO_DIR="<REPO_DIR>"
export DATA_ROOT="<DATA_ROOT>"
export CHECKPOINT_DIR="<CHECKPOINT_DIR>"
export RUNS_ROOT="<RUNS_ROOT>"
```

Do not hard-code private machine paths in configs or scripts. Use CLI overrides such as `--root_dir`, `--validate_dir`, `--checkpoint_dir`, and `--runs_root`.

## 1. What This Repository Runs

This repository implements communication-aware cooperative 3D object detection with:

- PointPillars-style voxel feature extraction.
- Intermediate BEV feature fusion using V2VAM.
- Config-driven communication policies.
- Receiver-driven and temporal receiver-driven request strategies.
- Communication metrics such as bytes/frame and normalized communication ratio.
- Safety-oriented post-processing metrics such as danger-zone recall and trajectory-aware risk recall.

The current public approach names are defined in:

```text
src/hypes_yaml/communication_approach_presets.yaml
```

Important runnable approaches:

| Approach | Purpose | Train Required | Reportable With Existing Detector Checkpoint |
|---|---|---:|---:|
| `baseline_full_communication` | Full communication baseline | No | Yes |
| `measurement_full_communication` | Full communication cost measurement | No | Yes |
| `selective_topk_energy_10` | Sender-side energy top-k at 10% | No | Yes |
| `selective_random_comm_only_10` | Random collaborator-only 10% baseline | No | Yes |
| `receiver_request_energy_topk_10` | Receiver-driven snapshot request | No | Yes |
| `temporal_receiver_request_energy_topk_10` | Receiver-driven temporal cache request | No | Yes |
| `learned_temporal_receiver_request_10` | Trainable learned temporal request head | Yes | Only after trained request-head weights exist |

Planned placeholders are documented in the presets but skipped by default.

## 1.1 Existing Features

The current public codebase includes the following ready-to-use capabilities:

| Feature | Status | Main files |
|---|---|---|
| Full-communication baseline | Runnable | `src/hypes_yaml/communication_approach_presets.yaml` |
| Sender-side selective top-k energy | Runnable | `src/models/fuse_modules/communication_policy.py` |
| Random collaborator-only masking | Runnable | `src/models/fuse_modules/communication_policy.py` |
| Receiver-request energy top-k | Runnable | `src/models/fuse_modules/communication_policy.py` |
| Temporal receiver-request energy top-k | Runnable | `src/models/fuse_modules/communication_policy.py` |
| Learned temporal request head | Trainable experimental | `src/models/fuse_modules/learned_temporal_request.py` |
| Checkpoint safety for learned request head | Runnable | `src/tools/inference.py` |
| Communication byte/ratio metrics | Runnable | `src/models/fuse_modules/communication_policy.py` |
| Danger-aware safety metrics | Runnable | `src/tools/evaluate_danger_aware_metrics.py` |
| Trajectory-aware safety metrics | Runnable | `src/tools/evaluate_trajectory_danger_metrics.py` |
| Centralized logging | Runnable | `src/utils/logging/` |
| Smoke-test pipeline | Runnable | `src/tools/testing/smoke_test_pipeline.py` |

Use this as a quick mental model:

- If you only want inference comparisons, use the non-learned approaches.
- If you want trainable communication, use the learned temporal approach only after setting up training data and checkpoint safety.
- If you want safety analysis, run the normal detection experiment first, then run the danger-aware or trajectory-aware metric scripts on the saved outputs.

## 1.2 Configurable For New Environments

The repository is intended to be portable across local servers, Kaggle, Colab, HPC clusters, and Docker. Avoid hard-coding machine-specific paths in source files. Use environment variables and CLI overrides instead.

Recommended path variables:

```bash
export REPO_DIR="<REPO_DIR>"
export DATA_ROOT="<DATA_ROOT>"
export TRAIN_DIR="$DATA_ROOT/train"
export VALIDATE_DIR="$DATA_ROOT/validate"
export CARLA_TEST_DIR="$DATA_ROOT/test/test"
export CULVER_TEST_DIR="$DATA_ROOT/test/test_culver_city/test_culver_city"
export CHECKPOINT_DIR="<CHECKPOINT_DIR>"
export RUNS_ROOT="<RUNS_ROOT>"
export PYTHONPATH="$REPO_DIR"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
```

Common things you can change without touching source code:

| What to change | Preferred mechanism | Example |
|---|---|---|
| Dataset location | CLI/environment variable | `--root_dir "$TRAIN_DIR" --validate_dir "$CARLA_TEST_DIR"` |
| Checkpoint folder | CLI/environment variable | `--checkpoint_dir "$CHECKPOINT_DIR"` |
| Output folder | CLI/environment variable | `--runs_root "$RUNS_ROOT"` |
| Approach | CLI preset name | `--approach receiver_request_energy_topk_10` |
| Short/full run size | CLI | `--max_samples 20` or omit/large value for full run |
| Debug map export | CLI/config | `--save_debug_maps` |
| Logging behavior | Environment variables | `LOG_LEVEL`, `LOG_COLOR`, `LOG_TO_FILE`, `LOG_FILE` |
| Communication ratios | YAML preset | `keep_ratio: 0.10` |
| Temporal cache behavior | YAML preset | `cache_momentum`, `novelty_weight`, `age_weight` |
| Learned request training | YAML preset | `trainable`, `loss.enabled`, `request_head_lr` |
| Reproducibility | CLI/config | `--seed 42 --deterministic` |

The safest workflow for a new environment is:

1. Set the path variables above.
2. Run import and compile checks.
3. Run pure unit tests.
4. Run a 20-frame smoke test.
5. Run one full non-learned inference approach.
6. Only then start training or long sweeps.

## 2. Recommended Hardware

### Recommended For Full Experiments

- OS: Ubuntu 20.04/22.04 or another Linux CUDA environment.
- GPU: NVIDIA GPU with CUDA support.
- GPU memory: 16 GB minimum, 24 GB+ recommended for comfortable training.
- CPU RAM: 32 GB minimum, 64 GB recommended.
- Disk: 100 GB+ free for dataset, checkpoints, logs, and exported metrics.
- Python: 3.8-3.10 is the safest range for older OpenCOOD-style dependencies. Python 3.12 can work in some hosted notebook environments but may require specific `spconv/cumm` wheels.

### Smoke Tests / Debugging

Small smoke tests with `--max_samples 5` or `--max_samples 20` are much lighter, but still require the repository dependencies and the voxelization stack. CPU-only runs are not recommended for real inference.

### macOS Notes

Apple Silicon/macOS is useful for editing code and running pure unit tests, but it is not recommended for full inference/training because `spconv` is CUDA/Linux-oriented. Use Linux + NVIDIA GPU for experiments.


## 2.1 Supported Environment Recipes

There is no single environment that is best for every public user. The table below lists the main practical profiles. Use the first profile that matches your machine.

| ID | Environment | Best For | Status |
|---|---|---|---|
| E1 | Linux + CUDA 12.1 + Python 3.10/3.11 + virtualenv | Full inference and training | Recommended |
| E2 | Linux + CUDA 11.8 + Python 3.8/3.9/3.10 + virtualenv | Full inference and training on older systems | Recommended |
| E3 | Linux + CUDA 12.1 + Python 3.10/3.11 + conda | Full inference/training with conda workflows | Recommended |
| E4 | Kaggle current Python + CUDA + preinstalled PyTorch | Notebook smoke tests and medium inference | Supported |
| E5 | Kaggle isolated `v2v_env` virtualenv | Reproducible notebook environment | Supported |
| E6 | Google Colab GPU | Small smoke tests, limited debugging | Experimental |
| E7 | Docker with NVIDIA runtime | Reproducible server deployment | Recommended if you maintain your own image |
| E8 | HPC/SLURM Linux node | Long full-split inference/training | Supported |
| E9 | WSL2 Ubuntu + NVIDIA GPU | Windows users with CUDA-capable GPU | Supported with care |
| E10 | macOS / Apple Silicon | Code editing and pure unit tests only | Dev-only |
| E11 | CPU-only Linux | Logger/config/unit tests only | Dev-only |
| E12 | Offline server | Reproducible runs without internet | Supported if wheels/data/checkpoints are pre-staged |
| E13 | Legacy Python 3.7/3.8 environment | Original dependency compatibility | Legacy |

The command blocks below intentionally use placeholders. Replace `<...>` values with your own paths and versions.

### E1: Linux CUDA 12.1 Virtualenv

Use this for modern NVIDIA servers where CUDA 12.x drivers are available.

```bash
cd <REPO_DIR>
python3.10 -m venv <ENV_DIR>
source <ENV_DIR>/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  "numpy>=1.23" "scipy>=1.9" "matplotlib>=3.6" "scikit-image>=0.20" \
  "opencv-python>=4.8" "PyYAML>=6.0" "tqdm>=4.65" "easydict>=1.9" \
  "tensorboardX>=2.6" "einops>=0.7" "timm>=0.9" "shapely>=2.0" "open3d>=0.18"
python -m pip install --no-cache-dir --only-binary=:all: cumm-cu121==0.7.11 spconv-cu121==2.3.8
export PYTHONPATH="$PWD"
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import spconv; from spconv.utils import Point2VoxelCPU3d; print('spconv ok')"
```

### E2: Linux CUDA 11.8 Virtualenv

Use this if your server stack is CUDA 11.8-oriented.

```bash
cd <REPO_DIR>
python3.9 -m venv <ENV_DIR>
source <ENV_DIR>/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install \
  "numpy>=1.23" "scipy>=1.9" "matplotlib>=3.6" "scikit-image>=0.20" \
  "opencv-python>=4.8" "PyYAML>=6.0" "tqdm>=4.65" "easydict>=1.9" \
  "tensorboardX>=2.6" "einops>=0.7" "timm>=0.9" "shapely>=2.0" "open3d>=0.18"
python -m pip install --no-cache-dir --only-binary=:all: spconv-cu118
export PYTHONPATH="$PWD"
```

If `spconv-cu118` is unavailable for your exact Python version, switch to E1 or E3 on a supported Python version.

### E3: Linux CUDA Conda Environment

Use this if your team prefers conda-managed Python and CUDA libraries.

```bash
conda create -n comm-v2v python=3.10 -y
conda activate comm-v2v
cd <REPO_DIR>
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  "numpy>=1.23" "scipy>=1.9" "matplotlib>=3.6" "scikit-image>=0.20" \
  "opencv-python>=4.8" "PyYAML>=6.0" "tqdm>=4.65" "easydict>=1.9" \
  "tensorboardX>=2.6" "einops>=0.7" "timm>=0.9" "shapely>=2.0" "open3d>=0.18"
python -m pip install --no-cache-dir --only-binary=:all: cumm-cu121==0.7.11 spconv-cu121==2.3.8
export PYTHONPATH="$PWD"
```

### E4: Kaggle Current Python Environment

Use this if Kaggle already has a working PyTorch installation. Do not create a virtualenv unless you need isolation.

```bash
cd <REPO_DIR>
export PYTHONPATH="$PWD"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "easydict>=1.9" "tqdm>=4.65" "PyYAML>=6.0" "tensorboardX>=2.6" \
  "einops>=0.7" "timm>=0.9" "shapely>=2.0" "open3d>=0.18"
python -m pip install --no-cache-dir --only-binary=:all: cumm-cu121==0.7.11 spconv-cu121==2.3.8
```

Kaggle-specific verification:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import spconv; from spconv.utils import Point2VoxelCPU3d; print('spconv ok')"
python -c "import open3d; print('open3d ok')"
```

### E5: Kaggle Isolated Virtualenv

Use this when you want to avoid modifying Kaggle's system Python.

```bash
python -m pip install --upgrade virtualenv
python -m virtualenv <ENV_DIR>
<ENV_DIR>/bin/python -m pip install --upgrade pip setuptools wheel
<ENV_DIR>/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
<ENV_DIR>/bin/python -m pip install \
  "numpy>=1.23" "scipy>=1.9" "matplotlib>=3.6" "scikit-image>=0.20" \
  "opencv-python>=4.8" "PyYAML>=6.0" "tqdm>=4.65" "easydict>=1.9" \
  "tensorboardX>=2.6" "einops>=0.7" "timm>=0.9" "shapely>=2.0" "open3d>=0.18"
<ENV_DIR>/bin/python -m pip install --no-cache-dir --only-binary=:all: cumm-cu121==0.7.11 spconv-cu121==2.3.8
```

Run commands with:

```bash
<ENV_DIR>/bin/python -m src.tools.testing.smoke_test_pipeline --help
```

### E6: Google Colab GPU

Colab can be useful for small smoke tests, but full-split runs may hit runtime and storage limits.

```bash
cd <REPO_DIR>
python -m pip install --upgrade pip setuptools wheel
python -m pip install "easydict>=1.9" "PyYAML>=6.0" "tqdm>=4.65" "open3d>=0.18" "shapely>=2.0" "timm>=0.9" "einops>=0.7" "tensorboardX>=2.6"
python -m pip install --no-cache-dir --only-binary=:all: <SPCONV_WHEEL_FOR_COLAB_CUDA>
export PYTHONPATH="$PWD"
```

Before running, check Colab CUDA and Python:

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
```

### E7: Docker With NVIDIA Runtime

Use Docker when you want to freeze OS/Python/CUDA details. This repository does not require a specific Dockerfile, but a typical workflow is:

```bash
docker run --gpus all -it --rm \
  -v <REPO_DIR>:/workspace/repo \
  -v <DATA_ROOT>:/workspace/data \
  -v <CHECKPOINT_DIR>:/workspace/checkpoints \
  -v <RUNS_ROOT>:/workspace/runs \
  <CUDA_PYTORCH_IMAGE> bash

cd /workspace/repo
export PYTHONPATH="$PWD"
```

Inside the container, install dependencies using E1 or E2 depending on the CUDA image.

### E8: HPC / SLURM Node

Use this profile for long training or full-split inference on a shared cluster.

Example SLURM wrapper:

```bash
#!/bin/bash
#SBATCH --job-name=comm-v2v
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source <ENV_DIR>/bin/activate
cd <REPO_DIR>
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0

python -m src.tools.testing.smoke_test_pipeline \
  --approach receiver_request_energy_topk_10 \
  --checkpoint_dir <CHECKPOINT_DIR> \
  --root_dir <DATA_ROOT>/train \
  --validate_dir <DATA_ROOT>/test/test \
  --split carla \
  --max_samples 999999 \
  --runs_root <RUNS_ROOT> \
  --force_clean
```

### E9: WSL2 Ubuntu + NVIDIA GPU

Use Ubuntu under WSL2, install NVIDIA drivers on Windows, and verify CUDA visibility inside WSL2:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Then follow E1 or E2 inside WSL2. Keep dataset and run outputs on the Linux filesystem when possible for better I/O performance.

### E10: macOS / Apple Silicon Dev-Only

Use this only for editing, documentation, and pure Python tests that do not import `spconv`.

```bash
cd <REPO_DIR>
python3 -m venv <ENV_DIR>
source <ENV_DIR>/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.23" "PyYAML>=6.0" "tqdm>=4.65" "matplotlib>=3.6" "shapely>=2.0"
export PYTHONPATH="$PWD"
python -m src.tools.testing.test_centralized_logger
python -m src.tools.testing.test_danger_aware_metrics
python -m src.tools.testing.test_trajectory_danger_metrics
```

Do not expect full inference/training to work cleanly on macOS because `spconv` is not the target stack.

### E11: CPU-Only Linux Dev-Only

Use this for logger/config/report tests, not full model inference.

```bash
cd <REPO_DIR>
python3 -m venv <ENV_DIR>
source <ENV_DIR>/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.23" "PyYAML>=6.0" "tqdm>=4.65" "matplotlib>=3.6" "shapely>=2.0"
export PYTHONPATH="$PWD"
python -m src.tools.testing.test_centralized_logger
python -m src.tools.testing.test_danger_aware_metrics
python -m src.tools.testing.test_trajectory_danger_metrics
```

### E12: Offline Server

Use this when the server has no internet. On a connected machine, pre-download wheels:

```bash
mkdir -p <WHEELHOUSE>
python -m pip download -d <WHEELHOUSE> \
  torch torchvision torchaudio \
  numpy scipy matplotlib scikit-image opencv-python PyYAML tqdm easydict \
  tensorboardX einops timm shapely open3d cumm-cu121==0.7.11 spconv-cu121==2.3.8
```

Copy `<WHEELHOUSE>` to the offline server, then install:

```bash
python -m pip install --no-index --find-links <WHEELHOUSE> <PACKAGE_NAMES>
```

Also pre-stage:

```text
<DATA_ROOT>
<CHECKPOINT_DIR>
<REPO_DIR>
```

### E13: Legacy Python 3.7/3.8 Environment

Use this only if you need maximum compatibility with the original historical dependency pins.

```bash
conda create -n comm-v2v-legacy python=3.8 -y
conda activate comm-v2v-legacy
cd <REPO_DIR>
python -m pip install --upgrade "pip<24" setuptools wheel
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

If old pins fail to build on your platform, switch to E1/E2/E3.

## 3. Repository Setup

### 3.1 Clone

```bash
git clone https://github.com/mohsenshahverdy/comm-aware-v2v-perception.git <REPO_DIR>
cd <REPO_DIR>
```

### 3.2 Create Environment

Recommended virtualenv flow:

```bash
python3 -m venv <ENV_DIR>
source <ENV_DIR>/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Alternative conda flow:

```bash
conda create -n comm-v2v python=3.9 -y
conda activate comm-v2v
python -m pip install --upgrade pip setuptools wheel
```

### 3.3 Install PyTorch

Install PyTorch matching your CUDA runtime. Check your CUDA setup first:

```bash
nvidia-smi
```

For CUDA 12.1-style environments:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8-style environments:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you are using a managed notebook image where PyTorch is already installed, verify it instead of reinstalling:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

### 3.4 Install Core Python Packages

The historical `requirements.txt` contains older pins inherited from the original cooperative perception codebase. On modern Python versions, installing everything blindly can fail. A safer modern install is:

```bash
python -m pip install \
  "numpy>=1.23" \
  "scipy>=1.9" \
  "matplotlib>=3.6" \
  "scikit-image>=0.20" \
  "opencv-python>=4.8" \
  "PyYAML>=6.0" \
  "tqdm>=4.65" \
  "easydict>=1.9" \
  "tensorboardX>=2.6" \
  "einops>=0.7" \
  "timm>=0.9" \
  "shapely>=2.0" \
  "open3d>=0.18"
```

If you use Python 3.7/3.8 and want the original-style dependency set, you may try:

```bash
python -m pip install -r requirements.txt
```

If this fails on `numba`, `numpy`, or `spconv`, use the modern install path above and install `spconv` separately.

### 3.5 Install spconv / cumm

This repository uses `spconv` for voxelization through:

```text
src/data_utils/pre_processor/sp_voxel_preprocessor.py
```

The right wheel depends on CUDA and Python. For CUDA 12.1-compatible environments:

```bash
python -m pip uninstall -y spconv cumm spconv-cu120 cumm-cu120 spconv-cu121 cumm-cu121
python -m pip install --no-cache-dir --only-binary=:all: cumm-cu121==0.7.11 spconv-cu121==2.3.8
```

For CUDA 11.8-compatible environments, use the matching wheel if available for your Python version:

```bash
python -m pip uninstall -y spconv cumm spconv-cu118 cumm-cu118
python -m pip install --no-cache-dir --only-binary=:all: spconv-cu118
```

Verify:

```bash
python -c "import spconv; from spconv.utils import Point2VoxelCPU3d; print('spconv ok')"
python -c "import importlib; importlib.import_module('cumm.tensorview'); print('cumm.tensorview ok')"
```

If `from cumm import tensorview` fails but `import cumm.tensorview` works, the repository has a fallback for that import pattern.

### 3.6 Verify Imports

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import yaml; print('yaml ok')"
python -c "import open3d; print('open3d ok')"
python -c "import timm; print('timm ok')"
python -c "import spconv; from spconv.utils import Point2VoxelCPU3d; print('spconv ok')"
```

### 3.7 Set Runtime Environment Variables

```bash
cd <REPO_DIR>
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
```

`MPLBACKEND=Agg` is recommended for remote/headless machines so plot generation does not require a display.

## 4. Dataset Setup

The expected dataset format is OpenCOOD/OPV2V-style cooperative perception data. The loader expects scenario folders, CAV folders, and paired `.yaml` / `.pcd` files.

Expected layout:

```text
<DATA_ROOT>/
  train/
    <scenario_id>/
      <cav_id>/
        <frame_id>.yaml
        <frame_id>.pcd
  validate/
    <scenario_id>/
      <cav_id>/
        <frame_id>.yaml
        <frame_id>.pcd
  test/
    test/
      <scenario_id>/
        <cav_id>/
          <frame_id>.yaml
          <frame_id>.pcd
    test_culver_city/
      test_culver_city/
        <scenario_id>/
          <cav_id>/
            <frame_id>.yaml
            <frame_id>.pcd
```

Example frame files:

```text
<DATA_ROOT>/test/test/<scenario_id>/<cav_id>/<frame_id>.pcd
<DATA_ROOT>/test/test/<scenario_id>/<cav_id>/<frame_id>.yaml
```

Each YAML should include vehicle/object annotations and pose fields such as `lidar_pose`. The dataset loader uses the scenario folder, CAV folder, and timestamp/frame id to build metadata and pairwise transformations.

### Dataset Path Variables

```bash
export DATA_ROOT="<DATA_ROOT>"
export TRAIN_DIR="$DATA_ROOT/train"
export VAL_DIR="$DATA_ROOT/validate"
export CARLA_TEST_DIR="$DATA_ROOT/test/test"
export CULVER_TEST_DIR="$DATA_ROOT/test/test_culver_city/test_culver_city"
```

Verify:

```bash
find "$DATA_ROOT" -maxdepth 3 -type d | head -50
find "$CARLA_TEST_DIR" -name "*.yaml" | head -5
find "$CARLA_TEST_DIR" -name "*.pcd" | head -5
```

## 5. Checkpoint Setup

Inference requires a run/checkpoint directory containing at least one checkpoint:

```text
<CHECKPOINT_DIR>/
  net_epoch43.pth
  config.yaml          # optional; smoke pipeline can create config.yaml from base config
```

The smoke pipeline copies checkpoint files into each generated run directory. It looks for:

```text
net_epoch*.pth
latest.pth
```

Set:

```bash
export CHECKPOINT_DIR="<CHECKPOINT_DIR>"
```

Verify:

```bash
find "$CHECKPOINT_DIR" -maxdepth 1 -name "net_epoch*.pth" -o -name "latest.pth"
```

## 6. Configuration System

### Main Model Config

```text
src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml
```

Important fields:

```yaml
root_dir: "<TRAIN_DIR>"
validate_dir: "<VALIDATE_OR_TEST_DIR>"
communication_preset: receiver_request_energy_topk_10
model:
  core_method: point_pillar_intermediate_V2VAM
```

For public/reproducible runs, prefer CLI overrides instead of editing paths in YAML:

```bash
--root_dir "$TRAIN_DIR"
--validate_dir "$CARLA_TEST_DIR"
```

### Communication Presets

```text
src/hypes_yaml/communication_approach_presets.yaml
```

Use approach names as `communication_preset` values. The smoke pipeline applies these presets automatically via `--approach`.

### Companion Approach Configs

Some wrapper configs are available under:

```text
src/hypes_yaml/communication_approaches/
```

They are useful for direct training commands, especially baseline/learned/repair variants.

## 7. Quick Validation Before Experiments

Run compile checks:

```bash
cd <REPO_DIR>
export PYTHONPATH="$PWD"
python -m py_compile \
  src/tools/inference.py \
  src/tools/train.py \
  src/models/point_pillar_intermediate_V2VAM.py \
  src/models/fuse_modules/communication_policy.py \
  src/models/fuse_modules/V2VAM.py
```

Run lightweight unit tests:

```bash
python -m src.tools.testing.test_centralized_logger
python -m src.tools.testing.test_comm_policy_fake
python -m src.tools.testing.test_v2vam_correctness
python -m src.tools.testing.test_temporal_cache
python -m src.tools.testing.test_temporal_receiver_request
python -m src.tools.testing.test_danger_aware_metrics
python -m src.tools.testing.test_trajectory_danger_metrics
```

If a test imports CUDA-only packages, run it in the same environment used for inference.

## 8. Smoke Tests

Smoke tests are the safest way to verify that configs, dataset paths, checkpoints, model building, inference, and metric export work. They are not final AP reports unless `--max_samples` covers the full split and AP is enabled.

### 8.1 Single Approach Smoke Test

```bash
cd <REPO_DIR>
export PYTHONPATH="$PWD"
export RUNS_ROOT="<RUNS_ROOT>"

python -m src.tools.testing.smoke_test_pipeline \
  --approach receiver_request_energy_topk_10 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$CARLA_TEST_DIR" \
  --split carla \
  --max_samples 20 \
  --runs_root "$RUNS_ROOT" \
  --force_clean
```

### 8.2 Smoke Test With Debug Maps

```bash
python -m src.tools.testing.smoke_test_pipeline \
  --approach temporal_receiver_request_energy_topk_10 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$CARLA_TEST_DIR" \
  --split carla \
  --max_samples 20 \
  --runs_root "$RUNS_ROOT" \
  --save_debug_maps \
  --force_clean
```

### 8.3 Smoke Test With Danger Box Export

Use this if you want static or trajectory danger-aware metrics later:

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

This creates:

```text
<RUNS_ROOT>/smoke_carla_receiver_request_energy_topk_10/danger_eval_boxes/frame_000000.npz
```

### 8.4 Run All Default Runnable Approaches

```bash
python -m src.tools.testing.smoke_test_pipeline \
  --all_approaches \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$CARLA_TEST_DIR" \
  --split carla \
  --max_samples 20 \
  --runs_root "$RUNS_ROOT" \
  --force_clean
```

Notes:

- Planned placeholders are skipped.
- Trainable experimental learned temporal request is skipped by default.
- Use `--skip_ap` for faster forward-only checks.

## 9. Full Inference Experiments

For full AP/communication reporting, omit `--skip_ap` and set `--max_samples` to a very large value or omit the max-sample cap when using `src.tools.inference` directly.

### 9.1 Full CARLA Evaluation Matrix

```bash
cd <REPO_DIR>
source <ENV_DIR>/bin/activate
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0

export DATA_ROOT="<DATA_ROOT>"
export TRAIN_DIR="$DATA_ROOT/train"
export CARLA_TEST_DIR="$DATA_ROOT/test/test"
export CHECKPOINT_DIR="<CHECKPOINT_DIR>"
export RUNS_ROOT="<RUNS_ROOT>/carla_full"
mkdir -p "$RUNS_ROOT"

for APPROACH in \
  baseline_full_communication \
  selective_topk_energy_10 \
  receiver_request_energy_topk_10 \
  temporal_receiver_request_energy_topk_10
do
  echo "===== Running ${APPROACH} ====="
  python -m src.tools.testing.smoke_test_pipeline \
    --approach "$APPROACH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --root_dir "$TRAIN_DIR" \
    --validate_dir "$CARLA_TEST_DIR" \
    --split carla \
    --max_samples 999999 \
    --runs_root "$RUNS_ROOT" \
    --save_box_npz \
    --force_clean
 done
```

### 9.2 Full Culver Evaluation Matrix

```bash
export CULVER_TEST_DIR="$DATA_ROOT/test/test_culver_city/test_culver_city"
export RUNS_ROOT="<RUNS_ROOT>/culver_full"
mkdir -p "$RUNS_ROOT"

for APPROACH in \
  baseline_full_communication \
  selective_topk_energy_10 \
  receiver_request_energy_topk_10 \
  temporal_receiver_request_energy_topk_10
do
  echo "===== Running ${APPROACH} ====="
  python -m src.tools.testing.smoke_test_pipeline \
    --approach "$APPROACH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --root_dir "$TRAIN_DIR" \
    --validate_dir "$CULVER_TEST_DIR" \
    --split culver \
    --max_samples 999999 \
    --runs_root "$RUNS_ROOT" \
    --save_box_npz \
    --force_clean
 done
```

### 9.3 Direct Inference On An Existing Run Directory

If a run directory already contains `config.yaml` and `net_epoch*.pth`:

```bash
python -m src.tools.inference \
  --model_dir "<RUN_DIR>" \
  --fusion_method intermediate \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$CARLA_TEST_DIR" \
  --global_sort_detections \
  --save_box_npz
```

For quick debug:

```bash
python -m src.tools.inference \
  --model_dir "<RUN_DIR>" \
  --fusion_method intermediate \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$CARLA_TEST_DIR" \
  --max_samples 20 \
  --skip_ap
```

## 10. Result Files

Each run directory can contain:

```text
config.yaml
net_epoch*.pth
summary_eval.yaml
inference_summary.json
smoke_test_report.json
comm_metrics_epoch.csv
comm_metrics_frame.jsonl
receiver_request_debug/*.npz
temporal_receiver_request_debug/*.npz
danger_eval_boxes/frame_*.npz
```

Important summary fields:

- `ap_50`
- `ap_70`
- `comm_feature_bytes_per_frame`
- `comm_context_bytes_per_frame`
- `comm_metadata_bytes_per_frame`
- `comm_total_bytes_per_frame`
- `comm_normalized_ratio`
- `comm_total_normalized_ratio`
- `receiver_request_keep_ratio`
- `temporal_cache_hit_ratio`
- `temporal_novelty_mean`

## 11. Build Clean Communication Summary

If available in your version of the repository:

```bash
python -m src.tools.reporting.build_clean_comm_summary \
  --runs_root "$RUNS_ROOT" \
  --output_csv "$RUNS_ROOT/clean_summary.csv" \
  --output_yaml "$RUNS_ROOT/clean_summary.yaml"
```

Plot communication metrics:

```bash
python -m src.tools.reporting.plot_comm_metrics \
  --csv "$RUNS_ROOT/clean_summary.csv"
```

If your checkout uses older tool paths, check:

```bash
find src/tools -name "*summary*.py" -o -name "*plot*.py"
```

## 12. Safety-Oriented Metrics

### 12.1 Static Danger-Aware Metrics

Requires runs created with `--save_box_npz` or legacy `--save_npy` outputs.

```bash
python -m src.tools.evaluate_danger_aware_metrics \
  --run_dirs \
    "$RUNS_ROOT/smoke_carla_baseline_full_communication" \
    "$RUNS_ROOT/smoke_carla_selective_topk_energy_10" \
    "$RUNS_ROOT/smoke_carla_receiver_request_energy_topk_10" \
    "$RUNS_ROOT/smoke_carla_temporal_receiver_request_energy_topk_10" \
  --method_names \
    baseline_full_communication \
    selective_topk_energy_10 \
    receiver_request_energy_topk_10 \
    temporal_receiver_request_energy_topk_10 \
  --baseline_method receiver_request_energy_topk_10 \
  --output_path "$RUNS_ROOT/danger_aware_metrics.yaml" \
  --update_run_summaries
```

Default danger zone:

```text
0 < x < 40 m
|y| < 10 m
tau = 20 m
IoU thresholds = 0.5, 0.7
```

### 12.2 Trajectory-Aware Danger Metrics

Requires runs created with `--save_box_npz`. Newer exports include metadata and `ego_lidar_pose` when available.

```bash
python -m src.tools.evaluate_trajectory_danger_metrics \
  --run_dirs \
    "$RUNS_ROOT/smoke_carla_baseline_full_communication" \
    "$RUNS_ROOT/smoke_carla_selective_topk_energy_10" \
    "$RUNS_ROOT/smoke_carla_receiver_request_energy_topk_10" \
    "$RUNS_ROOT/smoke_carla_temporal_receiver_request_energy_topk_10" \
  --method_names \
    baseline_full_communication \
    selective_topk_energy_10 \
    receiver_request_energy_topk_10 \
    temporal_receiver_request_energy_topk_10 \
  --baseline_method receiver_request_energy_topk_10 \
  --trajectory_source auto \
  --output_path "$RUNS_ROOT/trajectory_danger_metrics.yaml" \
  --update_run_summaries
```

Default trajectory parameters:

```text
horizon_seconds = 3.0
dt = inferred from timestamp index, otherwise 0.1 s
d_traj_max = 5.0 m
d_critical = 3.0 m
t_critical = 3.0 s
sigma_d = 5.0 m
sigma_t = 2.0 s
```

Trajectory source priority with `--trajectory_source auto`:

1. Future ego poses in the same scenario.
2. Constant-velocity estimate from ego pose sequence.
3. Ego-forward approximation if pose metadata is unavailable.

## 13. Training

### 13.1 Baseline / Standard Training

Training should use train and validation splits, not the test split:

```bash
python -m src.tools.train \
  --hypes_yaml src/hypes_yaml/communication_approaches/baseline_full_communication.yaml \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$VAL_DIR" \
  --seed 42
```

Resume from a run directory:

```bash
python -m src.tools.train \
  --hypes_yaml src/hypes_yaml/communication_approaches/baseline_full_communication.yaml \
  --model_dir "<RUN_DIR>" \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$VAL_DIR" \
  --seed 42
```

Debug with limited batches:

```bash
python -m src.tools.train \
  --hypes_yaml src/hypes_yaml/communication_approaches/baseline_full_communication.yaml \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$VAL_DIR" \
  --max_train_batches 5 \
  --max_val_batches 5
```

### 13.2 Learned Temporal Receiver Request Training

`learned_temporal_receiver_request_10` is trainable experimental. It requires a checkpoint that includes the detector weights and will create/learn request-head weights.

Important safety rule:

- Inference with learned temporal request is blocked by default if the checkpoint does not contain trained `comm_policy.learned_temporal_request_head.*` weights.
- `--allow_untrained_request_head` is debug only and marks results as non-reportable.

A typical training config must enable:

```yaml
communication_preset: learned_temporal_receiver_request_10
model:
  args:
    communication:
      receiver_request:
        strategy_variant: learned_temporal
        temporal:
          enabled: true
        learned:
          enabled: true
          loss:
            enabled: true
```

Use a copied/custom config for training experiments:

```bash
cp src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml <RUNS_ROOT>/learned_temporal_config.yaml
# edit <RUNS_ROOT>/learned_temporal_config.yaml, or use scripts/config overrides if available
```

Training command:

```bash
python -m src.tools.train \
  --hypes_yaml "<RUNS_ROOT>/learned_temporal_config.yaml" \
  --model_dir "<INITIAL_CHECKPOINT_RUN_DIR>" \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$VAL_DIR" \
  --seed 42
```

After training, verify checkpoint keys:

```bash
python - <<'PY'
import torch
ckpt = torch.load('<TRAINED_RUN_DIR>/net_epochXX.pth', map_location='cpu')
keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
learned = [k for k in keys if 'learned_temporal_request_head' in k or 'request_head' in k]
print('learned request-head keys:', len(learned))
print('\n'.join(learned[:10]))
PY
```

Run inference only after trained request-head keys exist:

```bash
python -m src.tools.inference \
  --model_dir "<TRAINED_RUN_DIR>" \
  --fusion_method intermediate \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$CARLA_TEST_DIR" \
  --global_sort_detections
```

## 14. Reproducibility Controls

Training and inference support runtime seed overrides:

```bash
--seed 42
--deterministic
--no-benchmark
```

Recommended deterministic debug command:

```bash
python -m src.tools.testing.smoke_test_pipeline \
  --approach receiver_request_energy_topk_10 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --root_dir "$TRAIN_DIR" \
  --validate_dir "$CARLA_TEST_DIR" \
  --split carla \
  --max_samples 20 \
  --runs_root "$RUNS_ROOT" \
  --force_clean
```

For maximum speed, deterministic mode may reduce performance. For final large experiments, keep a consistent seed and record the exact environment.

## 15. Centralized Logging

The repository uses a centralized logger in `src/utils/logging/`. This keeps training, inference, smoke tests, communication policies, metric builders, and evaluation scripts readable in the same format. The goal is simple: every important message should say what component produced it, what kind of message it is, and the key values needed to debug or reproduce the run.

Example output:

```text
INFO [Inference] Dataset built | split=carla samples=2170
CONFIG [CommunicationPolicy] Communication approach | public_name=receiver_request_energy_topk_10 strategy=receiver_request_topk trainable=false
METRIC [Inference] Evaluation result | ap30=0.95 ap50=0.95 ap70=0.89 comm_total_normalized_ratio=0.10
SUCCESS [SmokeTest] Smoke test passed | approach=receiver_request_energy_topk_10 frames=20
SAVE [Inference] Summary saved | path=<RUNS_ROOT>/summary_eval.yaml
```

The logger API is:

```python
from src.utils.logging import get_logger

logger = get_logger("ComponentName")
logger.info("Dataset loaded", split="carla", samples=2170)
logger.config("Preset loaded", approach="receiver_request_energy_topk_10")
logger.metric("Communication", comm_ratio=0.10, bytes_per_frame=123456)
logger.success("Checkpoint saved", path="<RUNS_ROOT>/net_epoch46.pth")
logger.warning("Debug-only flag enabled", flag="allow_untrained_request_head")
logger.error("Invalid approach", approach="unknown")
logger.command("Executing inference", cmd="python -m src.tools.inference ...")
logger.save("Artifacts archived", path="<RUNS_ROOT>/artifacts.zip")
```

Supported helper methods:

- `info` for normal status messages.
- `warning` / `warn` for recoverable issues or debug-only behavior.
- `error` for failures.
- `success` for completed actions.
- `debug` for verbose diagnostics.
- `metric` for AP, communication, safety, and training metrics.
- `config` for loaded presets, paths, and runtime options.
- `run`, `step`, and `progress` for high-level execution stages.
- `command` for shell/Python commands launched by helper scripts.
- `save` for files, checkpoints, summaries, plots, and archives.

Runtime controls are available through environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `LOG_LEVEL` | `INFO` | Minimum level to print. Use `DEBUG` for verbose diagnostics. |
| `LOG_DEBUG` | `false` | Enables debug messages when set to `true`. |
| `LOG_TIMESTAMP` | `false` | Adds timestamps to log lines. |
| `LOG_SILENT` | `false` | Suppresses console output when set to `true`. |
| `LOG_COLOR` | `true` | Enables colored level labels. Set `false` for clean notebooks/log files. |
| `LOG_TO_FILE` | `false` | Writes logs to a file in addition to console. |
| `LOG_FILE` | `run.log` | File path used when `LOG_TO_FILE=true`. |

Recommended public-run logging setup:

```bash
export LOG_LEVEL=INFO
export LOG_DEBUG=false
export LOG_TIMESTAMP=false
export LOG_COLOR=false
export LOG_TO_FILE=true
export LOG_FILE="<RUNS_ROOT>/run.log"
```

For debugging a communication policy:

```bash
export LOG_LEVEL=DEBUG
export LOG_DEBUG=true
export LOG_COLOR=false
```

For notebook/Kaggle output, `LOG_COLOR=false` is usually easier to read and copy into reports.

Verify the logger after installation:

```bash
python -m src.tools.testing.test_centralized_logger
```

Notes:

- Training logs include config loading, checkpoint loading, epoch start/end, optimizer learning rates, losses, validation, and checkpoint saves.
- Inference logs include dataset construction, checkpoint safety, approach status, AP metrics, communication metrics, summaries, and saved debug artifacts.
- Communication-policy logs include selected strategy, keep ratio, trainable/loss status, byte accounting, receiver-request/temporal settings, and alignment warnings.
- Some upstream dataset-loading messages may still use plain `print`; these are intentionally left when changing them would risk compatibility with inherited code.

## 16. Common Problems

### `ModuleNotFoundError: No module named src`

Run from repo root and set `PYTHONPATH`:

```bash
cd <REPO_DIR>
export PYTHONPATH="$PWD"
```

### `ModuleNotFoundError: No module named open3d`

```bash
python -m pip install "open3d>=0.18"
```

### `ModuleNotFoundError: No module named spconv`

Install a matching `spconv` wheel for your CUDA/Python environment. See Section 3.5.

### `ImportError: cannot import name tensorview from cumm`

Verify the submodule import:

```bash
python -c "import importlib; importlib.import_module('cumm.tensorview'); print('ok')"
```

If it fails, reinstall a compatible `cumm` / `spconv` wheel pair.

### Python 3.12 Dependency Problems

Some old pins in `requirements.txt` are incompatible with Python 3.12. Use the modern install command in Section 3.4 and install `spconv` separately.

### CUDA Out Of Memory

Options:

- Reduce training batch size in YAML.
- Use `--max_train_batches` / `--max_val_batches` for debugging.
- Avoid visualization during inference.
- Use a GPU with more memory for full training.

### Learned Temporal Inference Fails With Missing Request-Head Weights

This is expected safety behavior. The learned request head must be trained first. Do not use `--allow_untrained_request_head` for reportable results.

## 17. Public Experiment Checklist

Before reporting numbers:

- Confirm the dataset split path is correct.
- Confirm the checkpoint is the intended checkpoint.
- Run a 20-sample smoke test.
- Run full inference without `--skip_ap`.
- Save `summary_eval.yaml` and `smoke_test_report.json`.
- Export danger-aware boxes with `--save_box_npz` if safety metrics are needed.
- Run static danger metrics.
- Run trajectory-aware danger metrics.
- Keep the exact command, git commit hash, config, checkpoint name, and environment details.

## 18. Minimal Public Command Block

A compact end-to-end inference smoke run:

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
