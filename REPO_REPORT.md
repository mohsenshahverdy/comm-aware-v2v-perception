# V2V Cooperative Perception Repository Report

## Purpose

This repository implements a cooperative 3D object detection pipeline for multi-vehicle LiDAR perception. The core idea is that one ego vehicle improves detection by using LiDAR-derived features from nearby connected vehicles instead of relying only on its own sensor.

The project is built around:

- PointPillars-style LiDAR feature extraction
- Bird's Eye View (BEV) backbone processing
- Attention-based feature fusion across vehicles
- Anchor-based 3D object detection
- Training and evaluation scripts for the OPV2V-style dataset structure


## High-Level Pipeline

At a high level, the runtime pipeline is:

1. Read configuration from YAML.
2. Build train or validation dataset.
3. Load one ego frame plus nearby cooperating vehicles.
4. Transform cooperating LiDAR into ego coordinates.
5. Voxelize each vehicle's LiDAR.
6. Encode voxels with PillarVFE.
7. Scatter pillar features into a BEV pseudo-image.
8. Run BEV CNN backbone.
9. Fuse multi-vehicle BEV features with the custom V2V attention module.
10. Predict classification and box regression maps.
11. Compute anchor-based loss during training.
12. Decode boxes, apply thresholding and NMS during inference.
13. Evaluate AP at IoU 0.3, 0.5, and 0.7.


## Main Entry Points

The main executable scripts are:

- [src/tools/train.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/tools/train.py:72)
- [src/tools/inference.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/tools/inference.py:48)

The main configuration file is:

- [src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml:1)


## Repository Structure

Important directories:

- `src/tools`: training, inference, multi-GPU utilities
- `src/data_utils`: dataset loading, preprocessing, postprocessing, augmentation
- `src/models`: main model and fusion modules
- `src/loss`: loss definitions
- `src/utils`: geometry, transforms, point cloud, evaluation, visualization helpers
- `src/hypes_yaml`: YAML configuration files
- `src/pcdet_utils`: CUDA/C++ extensions and point cloud ops


## Configuration Flow

Training starts by parsing CLI arguments in `train.py`, then loading the YAML config with `yaml_utils.load_yaml`. If `--model_dir` is passed, the loader reads `config.yaml` from that checkpoint directory instead of the original YAML path.

Relevant code:

- [src/tools/train.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/tools/train.py:75)
- [src/hypes_yaml/yaml_utils.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/hypes_yaml/yaml_utils.py:8)

The YAML also uses a custom parser:

- `yaml_parser: "load_point_pillar_params"`

That parser computes:

- voxel grid size
- scatter grid dimensions
- anchor dimensions derived from voxel size and LiDAR range

Relevant code:

- [src/hypes_yaml/yaml_utils.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/hypes_yaml/yaml_utils.py:58)


## Dataset Assumptions

The configured fusion mode is:

- `IntermediateFusionDataset`

Relevant config:

- [src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml:22)

The dataset loader expects a directory layout like:

```text
root_dir/
  scenario_x/
    cav_id/
      000001.yaml
      000001.pcd
      000002.yaml
      000002.pcd
```

Training and validation roots are configured in YAML as:

- `root_dir: "training_data\\train"`
- `validate_dir: "validating_data\\validate"`

Relevant config:

- [src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml:6)

Important note:

Those are Windows-style paths. On macOS or Linux, they should be changed to forward-slash paths such as:

```yaml
root_dir: "training_data/train"
validate_dir: "validating_data/validate"
```


## How One Sample Is Built

The dataset class is:

- [src/data_utils/datasets/intermediate_fusion_dataset.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/data_utils/datasets/intermediate_fusion_dataset.py:26)

The process for one sample is:

1. `BaseDataset.retrieve_base_data` locates the scenario and timestamp for the requested index.
2. It loads per-vehicle YAML metadata and LiDAR point clouds.
3. It computes ego-relative transformations.
4. `IntermediateFusionDataset.__getitem__` finds the ego vehicle.
5. It keeps only cooperating vehicles within communication range.
6. It projects each collaborator LiDAR into ego space.
7. It preprocesses each LiDAR into voxels.
8. It merges features from all participating vehicles.
9. It generates anchor boxes and training labels for the ego frame.

Relevant code:

- [src/data_utils/datasets/basedataset.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/data_utils/datasets/basedataset.py:153)
- [src/data_utils/datasets/basedataset.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/data_utils/datasets/basedataset.py:448)
- [src/data_utils/datasets/intermediate_fusion_dataset.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/data_utils/datasets/intermediate_fusion_dataset.py:55)


## Coordinate Transform Logic

Pose transforms are handled by:

- [src/utils/transformation_utils.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/utils/transformation_utils.py:8)

The function `x1_to_x2(x1, x2)` converts a pose from one local coordinate system into another via world coordinates.

This is used in `BaseDataset.reform_param` to compute:

- collaborator-to-ego transformation matrix
- ground-truth transformation matrix
- spatial correction matrix for delayed communication cases

Relevant code:

- [src/data_utils/datasets/basedataset.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/data_utils/datasets/basedataset.py:448)


## Preprocessing

The configured preprocessor is:

- `SpVoxelPreprocessor`

Relevant code:

- [src/data_utils/pre_processor/sp_voxel_preprocessor.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/data_utils/pre_processor/sp_voxel_preprocessor.py:13)

Its job is to convert raw LiDAR points into sparse voxels:

- `voxel_features`
- `voxel_coords`
- `voxel_num_points`

The current YAML uses:

- voxel size: `[0.4, 0.4, 4]`
- LiDAR range: `[-140.8, -40, -3, 140.8, 40, 1]`

This results in a grid size of roughly:

- `[704, 200, 1]`

The preprocessor depends on `spconv` and `cumm`.


## Model Architecture

The configured model is:

- `point_pillar_intermediate_V2VAM`

Relevant files:

- [src/models/point_pillar_intermediate_V2VAM.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/point_pillar_intermediate_V2VAM.py:21)
- [src/models/fuse_modules/V2VAM.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/fuse_modules/V2VAM.py:19)

The forward pass is:

1. `PillarVFE`
2. `PointPillarScatter`
3. `BaseBEVBackbone`
4. optional downsample head
5. `V2V_AttFusion`
6. classification head
7. regression head

Supporting modules:

- [src/models/sub_modules/pillar_vfe.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/sub_modules/pillar_vfe.py:75)
- [src/models/sub_modules/point_pillar_scatter.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/sub_modules/point_pillar_scatter.py:11)
- [src/models/sub_modules/base_bev_backbone.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/sub_modules/base_bev_backbone.py:6)
- [src/models/sub_modules/downsample_conv.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/sub_modules/downsample_conv.py:27)


## Fusion Module

The fusion network is a custom attention module called `V2V_AttFusion`.

What it does:

1. Group all vehicle features belonging to the same sample using `record_len`.
2. Treat the first feature map in each group as ego.
3. Apply criss-cross attention between ego and each vehicle feature.
4. Pool attended outputs using max and average pooling.
5. Refine the fused feature with a convolution block.

Relevant code:

- [src/models/fuse_modules/V2VAM.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/fuse_modules/V2VAM.py:31)


## Detection Heads

The model outputs:

- `psm`: classification score map
- `rm`: regression map

Relevant code:

- [src/models/point_pillar_intermediate_V2VAM.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/point_pillar_intermediate_V2VAM.py:119)

These are anchor-based predictions over the BEV grid.


## Label Generation And Postprocessing

The postprocessor is:

- `VoxelPostprocessor`

Relevant file:

- [src/data_utils/post_processor/voxel_postprocessor.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/data_utils/post_processor/voxel_postprocessor.py:18)

Responsibilities:

- generate anchor boxes
- assign positive and negative anchors using IoU
- generate regression targets
- decode predictions back into 3D boxes
- run thresholding and NMS

Anchor settings from YAML:

- length `3.9`
- width `1.6`
- height `1.56`
- rotations `[0, 90]`
- feature stride `4`

The postprocessor uses:

- Cython `bbox_overlaps`
- geometry utilities in `box_utils.py`


## Loss Function

The loss is:

- `PointPillarLoss`

Relevant file:

- [src/loss/point_pillar_loss.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/loss/point_pillar_loss.py:73)

Loss composition:

- focal classification loss
- SmoothL1 regression loss

Configured weights:

- `cls_weight: 1.0`
- `reg: 2.0`


## Training Flow

The training script:

- loads config
- builds train and validation datasets
- creates dataloaders
- creates model
- resumes checkpoint if `--model_dir` is provided
- creates optimizer and LR scheduler
- logs losses to TensorBoard
- saves `net_epoch*.pth`
- writes `loss_dict_epoch_*.json`

Relevant code:

- [src/tools/train.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/tools/train.py:88)
- [src/tools/train_utils.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/tools/train_utils.py:64)

Output directory pattern:

```text
src/logs/<model_name>_<timestamp>/
```

Saved contents typically include:

- `config.yaml`
- `net_epoch1.pth`, `net_epoch2.pth`, ...
- TensorBoard event files
- `loss_dict_epoch_*.json`


## Inference Flow

The inference script:

1. Takes `--model_dir` and `--fusion_method`.
2. Loads the saved `config.yaml` from that run folder.
3. Rebuilds the validation dataset.
4. Loads the saved checkpoint.
5. Runs prediction sample by sample.
6. Computes AP statistics.
7. Optionally saves visualizations or NumPy outputs.

Relevant code:

- [src/tools/inference.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/tools/inference.py:55)
- [src/tools/inference_utils.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/tools/inference_utils.py:49)
- [src/utils/eval_utils.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/utils/eval_utils.py:137)

Evaluation outputs:

- `eval.yaml`
- or `eval_global_sort.yaml`


## Intended Run Commands

Basic training:

```bash
python src/tools/train.py --hypes_yaml src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml
```

Resume training:

```bash
python src/tools/train.py \
  --hypes_yaml src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml \
  --model_dir src/logs/<run_dir>
```

Inference:

```bash
python src/tools/inference.py \
  --model_dir src/logs/<run_dir> \
  --fusion_method intermediate
```

Inference with saved visualizations:

```bash
python src/tools/inference.py \
  --model_dir src/logs/<run_dir> \
  --fusion_method intermediate \
  --save_vis
```


## Environment Notes

The repository contains multiple dependency files:

- [requirements.txt](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/requirements.txt:1)
- [environment.yml](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/environment.yml:1)
- [V2VAM_req.txt](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/V2VAM_req.txt:1)
- [req_colab.txt](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/req_colab.txt:1)

This shows mixed environment history. In particular:

- `environment.yml` targets Python `3.7.11`
- `req_colab.txt` looks more modern
- `V2VAM_req.txt` mentions `spconv-cu118`

Important practical point:

The current workspace where I inspected the repo does not have the required libraries installed. `torch`, `open3d`, `spconv`, `cumm`, `timm`, and `tensorboardX` were all missing at inspection time, and the active Python version was `3.13.1`, which is very likely incompatible with this code as-is.


## Native Extensions

This repo includes code that may require manual compilation:

- Cython extension for `box_overlaps.pyx`
- CUDA/C++ extensions under `src/pcdet_utils`

Relevant files:

- [src/utils/setup.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/utils/setup.py:1)
- [src/pcdet_utils/setup.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/pcdet_utils/setup.py:1)

Important note:

`pip install -e .` installs Python package dependencies from the root `requirements.txt`, but it does not automatically guarantee that these extra native modules are compiled.


## Current Gaps In This Repository

A few parts appear incomplete or fragile:

1. The dataset itself is not included in this repo.
2. The configured dataset paths are placeholders.
3. The visualization script references missing files:
   - `visualization.yaml`
   - `EarlyFusionVisDataset`
4. There are multiple environment files with inconsistent version assumptions.
5. The code contains several thesis-style comments and experimental remnants.

Example of incomplete visualization path:

- [src/visualization/vis_data_sequence.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/visualization/vis_data_sequence.py:13)


## Important Risks And Code Issues

### 1. Likely fusion bug in `V2VAM.py`

Inside `V2V_AttFusion.forward`, the list `att` is created outside the per-sample loop and is not reset inside the loop. That means attention outputs from earlier samples in the batch can leak into later samples.

Relevant code:

- [src/models/fuse_modules/V2VAM.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/fuse_modules/V2VAM.py:35)

This is likely a real bug and should be checked before relying on new experiments.

### 2. Forced CUDA usage in attention helper

`INF()` uses `.cuda()` directly, which makes the module assume CUDA is available.

Relevant code:

- [src/models/fuse_modules/V2VAM.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/models/fuse_modules/V2VAM.py:68)

This will break CPU-only runs and is not device-agnostic.

### 3. Environment compatibility risk

The repo likely will not run on the current Python `3.13.1` environment without substantial adjustment. It was clearly developed around older PyTorch/spconv combinations.

### 4. Missing dataset classes mentioned by comments

Some comments mention early and late fusion support, but the repo contents shown here only expose `IntermediateFusionDataset` through `build_dataset`.

Relevant file:

- [src/data_utils/datasets/__init__.py](/Users/mohsen/Desktop/Thesis/code/V2V_Cooperative_Perception-main/src/data_utils/datasets/__init__.py:10)


## What You Should Understand Before Continuing The Thesis

If you continue development on this repository, the most important conceptual pieces are:

1. The sample unit is not a single LiDAR frame from one car.
   It is one ego frame plus multiple neighboring CAV frames combined into one intermediate-fusion sample.

2. Fusion happens after per-vehicle feature extraction, not on raw points and not after final detections.

3. The model is anchor-based and inherits many PointPillars design decisions.

4. Training quality depends heavily on:
   - correct dataset formatting
   - correct coordinate transforms
   - working voxelization
   - stable environment and extension builds

5. Before starting new research changes, it is important to first verify that the current baseline reproduces and is not affected by implementation bugs.


## Recommended Next Steps

To continue this thesis project safely, this is the order I would recommend:

1. Create a reproducible Python and CUDA environment.
2. Fix the YAML dataset paths for your machine.
3. Confirm the dataset structure matches what `BaseDataset` expects.
4. Build any required native extensions.
5. Run a dataset sanity check by loading one batch only.
6. Run a short debug training for a few iterations.
7. Review and likely fix the fusion bug in `V2VAM.py`.
8. Only then begin architecture changes or new experiments.


## Short Summary

This repository is a research codebase for multi-vehicle LiDAR cooperative perception using PointPillars and a custom V2V attention fusion module. The overall design is understandable and modular, but the repo is not yet turnkey on a fresh machine. The main blockers are environment reproducibility, dataset path setup, extension builds, and at least one likely bug in the fusion module. As a thesis continuation base, it is workable, but it should first be stabilized before any new scientific claims or model comparisons are made.
