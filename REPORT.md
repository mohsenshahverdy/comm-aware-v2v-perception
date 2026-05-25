# V2V Cooperative Perception Project Report

## 1. Project Overview

This repository implements an end-to-end multi-agent LiDAR cooperative perception system for autonomous vehicles. The goal is to fuse perception features from multiple connected vehicles (V2V) to improve 3D object detection performance over single-vehicle perception.

Key ideas:
- Multi-agent feature fusion using attention mechanisms
- Intermediate fusion of BEV features before detection heads
- PointPillar-based LiDAR backbone for efficient voxel processing
- Support for variable numbers of cooperating vehicles


## 2. Pipeline and Architecture

### High-Level Flow

1. Multi-agent LiDAR point clouds are loaded and preprocessed.
2. Each agent's raw point cloud is voxelized using a sparse voxel preprocessor.
3. Pillar-level features are extracted from voxels using PillarVFE.
4. Pillar features are scattered to a 2D BEV feature map.
5. A BEV backbone processes the feature map with convolutional layers.
6. Features from multiple agents are fused using a V2V attention module.
7. Detection heads predict classification scores and bounding box regressions.
8. Post-processing applies NMS and anchoring to produce final 3D boxes.

### Main Files and Components

- `src/models/point_pillar_intermediate_V2VAM.py` — main model defining the full pipeline.
- `src/models/sub_modules/pillar_vfe.py` — pillar feature extraction.
- `src/models/sub_modules/point_pillar_scatter.py` — converts pillar features into BEV grid.
- `src/models/sub_modules/base_bev_backbone.py` — BEV feature extractor.
- `src/models/fuse_modules/V2VAM.py` — V2V attention fusion module.
- `src/data_utils/pre_processor/sp_voxel_preprocessor.py` — voxelization of LiDAR points.
- `src/data_utils/datasets/intermediate_fusion_dataset.py` — dataset handling for multi-agent fusion.
- `src/data_utils/post_processor/voxel_postprocessor.py` — anchor generation, target assignment, and NMS.
- `src/tools/train.py` — training entry point.
- `src/tools/inference.py` — inference and evaluation entry point.
- `src/utils/eval_utils.py` — evaluation metrics and AP calculation.


## 3. Data Processing

The data pipeline uses a PointPillar-style voxelization approach:
- Raw LiDAR points are turned into pillars by discretizing the 3D space.
- Each pillar aggregates up to a fixed number of points.
- Pillar features are extracted with a PFN block and then scattered to a BEV map.
- The BEV map is processed by a CNN backbone to produce spatial features.

Important configuration values:
- Voxel size: `[0.4, 0.4, 4]`
- Point cloud range: `[-140.8, -40, -3, 140.8, 40, 1]`
- BEV grid size: `704 x 200`
- Max points per voxel: `32`


## 4. Multi-Agent Fusion

The repository implements intermediate feature fusion using a Criss-Cross Attention (CCA) approach.

### Fusion Strategy

- Each agent's BEV features are computed independently.
- Agent features are grouped by sample using `record_len`.
- The ego vehicle feature map attends to other agent feature maps.
- Attention outputs are aggregated using max pooling and average pooling.
- A refinement convolution produces the final fused BEV feature.

This intermediate-level fusion is more powerful than late fusion because it allows spatial and semantic feature interaction before detection.


## 5. Model Components

### PillarVFE
- Converts localized pillar points into per-pillar feature vectors.
- Uses linear layers, BatchNorm, and max pooling.

### PointPillarScatter
- Maps pillar features to a dense 2D BEV representation.

### BaseBEVBackbone
- Processes BEV feature maps through downsampling and upsampling convolution blocks.
- Outputs multi-scale features concatenated into a higher-channel tensor.

### V2VAM Fusion Module
- Regroups agent-level features.
- Applies Criss-Cross Attention across feature maps.
- Pools and fuses fused attention responses.

### Detection Heads
- Classification head predicts object presence for anchors.
- Regression head predicts bounding box parameters (x, y, z, h, w, l, r).


## 6. Training and Inference

### Setup

The project can be installed in editable mode:

```bash
pip install -e .
```

Dependencies are managed in `requirements.txt` and `environment.yml`.

### Training Command

```bash
python src/tools/train.py --hypes_yaml src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml
```

Optional training flags may include:
- `--model_dir` to resume from a checkpoint
- `--half` for mixed precision
- `--dist_url env://` for distributed training

### Inference Command

```bash
python src/tools/inference.py --model_dir /path/to/trained/model --fusion_method intermediate --show_vis
```

The inference script supports different fusion methods, including `intermediate`, `early`, and `late`.


## 7. Dataset and Configuration

The project is designed for a cooperative perception dataset such as OPV2V. The dataset includes synchronized multi-agent LiDAR frames and labels.

Key dataset characteristics:
- Multiple cooperating vehicles per scene
- Ego vehicle selection and neighbor filtering by communication range
- Transformation of other-agent points to the ego frame
- Variable agent counts handled by `record_len`

The YAML config file `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml` controls training hyperparameters, model settings, data augmentation, and postprocessing.


## 8. Loss and Evaluation

### Loss
- Classification loss: focal / weighted cross-entropy style
- Regression loss: smooth L1 applied only to positive anchors
- Total loss = classification + regression_weight * regression

### Evaluation
- Computes Average Precision (AP)
- Uses IoU thresholds such as `0.3`, `0.5`, and `0.7`
- Calculates precision and recall from TP/FP matching


## 9. Dependencies and Environment

Core dependencies include:
- `torch`
- `spconv`
- `numpy`
- `opencv-python`
- `open3d`
- `tensorboardX`
- `yaml`
- `easydict`

GPU support is recommended for training and inference, with CUDA 11+ and a modern NVIDIA GPU.


## 10. Key Notes for Continuing the Thesis

- The project already uses a strong intermediate fusion architecture, making it a good foundation for further research.
- The V2V fusion module is the main research component and can be extended with newer attention or graph-based fusion techniques.
- The dataset pipeline supports realistic V2V settings, including communication range and pose transformation.
- For experimentation, focus on:
  - fusion module variants
  - communication and delay modeling
  - ablations of early vs intermediate vs late fusion
  - scalability to more agents


## 11. Repository Structure

```
src/
├── data_utils/
│   ├── augmentor/
│   ├── datasets/
│   ├── post_processor/
│   └── pre_processor/
├── hypes_yaml/
├── loss/
├── models/
│   ├── fuse_modules/
│   └── sub_modules/
├── tools/
├── utils/
└── visualization/
```


## 12. Recommended Next Steps

1. Verify dataset location and format on your machine.
2. Run a dry training session with a small subset to confirm the pipeline.
3. Inspect `src/tools/train.py` and `src/tools/inference.py` for command-line options.
4. Evaluate a checkpoint with `src/tools/inference.py` and use visualization to inspect output.
5. Consider adding documentation or comments to clarify the fusion logic for future work.
