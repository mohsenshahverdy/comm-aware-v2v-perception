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

## Receiver-Driven Roadmap
Current runnable receiver-driven method:
- `receiver_request_energy_topk_10` (implemented and runnable)
  - need map: inverse ego feature energy
  - context map: collaborator L2 feature energy

Planned placeholders (config-only, disabled by default):
- `receiver_request_uncertainty_topk_10`
  - need map target: ego detection uncertainty/entropy
  - future work: expose ego-only objectness/uncertainty before fusion
- `receiver_request_visibility_topk`
  - need map target: point density / distance / visibility / occlusion
  - future work: BEV density/visibility map extraction
- `receiver_request_learned`
  - future work: trainable request head, separate LR, optional detector freeze
- `receiver_request_learned_budget`
  - future work: config-gated budget-aware losses
  - `L_sparse = lambda_sparse * mean(mask)`
  - `L_budget = lambda_budget * ReLU(mean(mask) - target_ratio)^2`
- `receiver_request_warped`
  - future work: `warp_context_to_ego`, `warp_mask_to_sender`
  - requires BEV warp utility and coordinate validation

## Key Config Files
- `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml`
- `src/hypes_yaml/communication_approach_presets.yaml`
- `docs/receiver_request_approach_guide.md`
- `src/hypes_yaml/communication_approaches/baseline_full_communication.yaml`
- `src/hypes_yaml/communication_approaches/measurement_full_communication.yaml`
- `src/hypes_yaml/communication_approaches/selective_topk_energy_10.yaml`
- `src/hypes_yaml/communication_approaches/stress_random_drop_10.yaml`
- `src/hypes_yaml/communication_approaches/robustness_neighbor_packetloss_20.yaml`
- `src/hypes_yaml/communication_approaches/learned_mask_default.yaml`
- `src/hypes_yaml/communication_approaches/repair_feature_reconstruction.yaml`
- `src/tools/kaggle_receiver_request_run_cell.py`

Notes:
- The config key stays `communication_preset` for runtime compatibility.
- Values are approach names (not development-phase names).

## Tools Organization
- `src/tools/testing/`: smoke tests and debug/test entry points (`smoke_test_pipeline.py`, fake-policy tests, V2VAM checks).
- `src/tools/reporting/`: summary/plot/report utilities (`build_clean_comm_summary.py`, `plot_comm_metrics.py`, label mapping).
- `src/tools/`: train/inference runtime scripts.

## Quick Checks
```bash
python -m py_compile \
  src/tools/inference.py \
  src/tools/train.py \
  src/models/point_pillar_intermediate_V2VAM.py \
  src/models/fuse_modules/communication_policy.py

python -m src.tools.testing.test_comm_policy_fake
python -m src.tools.testing.test_v2vam_correctness
python -m src.tools.testing.test_centralized_logger
python -m src.tools.testing.smoke_test_pipeline --help
```

## Example Training
```bash
python src/tools/train.py --hypes_yaml src/hypes_yaml/communication_approaches/baseline_full_communication.yaml
```

## Example Inference
```bash
python src/tools/inference.py --model_dir <RUN_DIR> --fusion_method intermediate
```
