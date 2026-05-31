# Communication-Aware V2V Cooperative Perception

This repository contains communication-aware cooperative perception experiments using PointPillars + V2VAM intermediate fusion.

## Project Links
- GitHub: `https://github.com/mohsenshahverdy/comm-aware-v2v-perception`
- Kaggle: `https://www.kaggle.com/code/mohsenshahverdi/communication-aware-v2v-perception/edit`

## Approach Naming
All experiment presets are approach-based:
- `baseline_full_communication`
- `measurement_full_communication`
- `selective_topk_energy_*`
- `selective_random_comm_only_*`
- `robustness_neighbor_packetloss_20`
- `learned_mask_*`
- `receiver_request_energy_topk_*`
- `repair_feature_reconstruction`

## Key Config Files
- `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml`
- `src/hypes_yaml/communication_approach_presets.yaml`
- `src/hypes_yaml/communication_approaches/baseline_full_communication.yaml`
- `src/hypes_yaml/communication_approaches/measurement_full_communication.yaml`
- `src/hypes_yaml/communication_approaches/selective_topk_energy_10.yaml`
- `src/hypes_yaml/communication_approaches/stress_random_drop_10.yaml`
- `src/hypes_yaml/communication_approaches/robustness_neighbor_packetloss_20.yaml`
- `src/hypes_yaml/communication_approaches/learned_mask_default.yaml`
- `src/hypes_yaml/communication_approaches/repair_feature_reconstruction.yaml`

## Quick Checks
```bash
python -m py_compile \
  src/tools/inference.py \
  src/tools/train.py \
  src/models/point_pillar_intermediate_V2VAM.py \
  src/models/fuse_modules/communication_policy.py

python -m src.tools.test_comm_policy_fake
python -m src.tools.test_v2vam_correctness
```

## Example Training
```bash
python src/tools/train.py --hypes_yaml src/hypes_yaml/communication_approaches/baseline_full_communication.yaml
```

## Example Inference
```bash
python src/tools/inference.py --model_dir <RUN_DIR> --fusion_method intermediate
```
