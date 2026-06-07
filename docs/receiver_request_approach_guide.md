# Receiver-Driven Communication Approach Guide

This guide defines the receiver-driven roadmap and clearly marks which methods are runnable now vs planned placeholders.

## Status Matrix

| Approach | Preset | Status | Default |
|---|---|---|---|
| Receiver-request energy top-k | `receiver_request_energy_topk_10` | Implemented and runnable | Enabled in preset |
| Temporal receiver-request energy top-k | `temporal_receiver_request_energy_topk_10` | Implemented and runnable | Enabled in preset |
| Learned temporal receiver-request | `learned_temporal_receiver_request_10` | Trainable experimental, non-reportable until trained | Skipped from default all-approach runs |
| Receiver-request uncertainty top-k | `receiver_request_uncertainty_topk_10` | Planned placeholder | Disabled |
| Receiver-request visibility top-k | `receiver_request_visibility_topk` | Planned placeholder | Disabled |
| Receiver-request learned | `receiver_request_learned` | Planned placeholder | Disabled |
| Receiver-request learned + budget | `receiver_request_learned_budget` | Planned placeholder | Disabled |
| Receiver-request warped alignment | `receiver_request_warped` | Planned placeholder | Disabled |

## Three Receiver-Request Levels

| Level | Train required | Test-data evaluation with old checkpoint | Reportable status |
|---|---:|---:|---|
| Snapshot receiver-request | No | Yes | Reportable |
| Temporal receiver-request | No | Yes | Reportable |
| Learned temporal receiver-request | Yes | No | Not reportable until trained request-head weights exist |

Non-learned receiver-request methods need only test data for evaluation. Learned temporal receiver-request requires train data because the request head must learn useful request probabilities. Stage 4 checkpoint safety blocks random-head inference by default. The `--allow_untrained_request_head` flag is debug-only and forces `reportable_result=false`; do not use it for report numbers.

## Method Definitions

### 1) `receiver_request_energy_topk`
- Strategy: `receiver_request_topk`
- Need map: inverse ego feature energy
- Context map: collaborator L2 feature energy
- Selection: top-k by score with configurable keep ratio
- Current reference run: `receiver_request_energy_topk_10`

### 2) `temporal_receiver_request_energy_topk`
- Strategy: `receiver_request_topk`
- Need map: inverse ego feature energy
- Context map: collaborator L2 feature energy
- Temporal additions: cache, novelty, age, and confidence terms
- Current reference run: `temporal_receiver_request_energy_topk_10`
- Training: not required

### 3) `learned_temporal_receiver_request`
- Strategy: `receiver_request_topk`
- Variant: `learned_temporal`
- Request map: learned request probability from temporal maps
- Current preset: `learned_temporal_receiver_request_10`
- Training: required
- Reportability: not reportable unless checkpoint contains trained `comm_policy.learned_temporal_request_head.*` weights
- Default automation: skipped by `--all_approaches`

### 4) `receiver_request_uncertainty_topk` (planned)
- Need map target: ego detection uncertainty/entropy
- Required future work: expose ego-only objectness/uncertainty before fusion

### 5) `receiver_request_visibility_topk` (planned)
- Need map target: point density, distance, visibility, occlusion
- Required future work: BEV density/visibility map extraction

### 6) `receiver_request_learned` (planned)
- Need/request map target: learned request network
- Required future work: trainable request head, separate LR, optional detector freeze

### 7) `receiver_request_learned_budget` (planned)
- Loss target:
  - `L_sparse = lambda_sparse * mean(mask)`
  - `L_budget = lambda_budget * ReLU(mean(mask) - target_ratio)^2`
- Required future work: config-gated training path

### 8) `receiver_request_warped` (planned)
- Alignment target:
  - `warp_context_to_ego`
  - `warp_mask_to_sender`
- Required future work: BEV warp utility and coordinate convention validation

## Config Structure

Main receiver-request section (in `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml`):

- `implementation_status`: `implemented`, `implemented_runnable`, `planned`, `planned_placeholder`, or `trainable_experimental`
- `strategy_variant`: `energy_topk | temporal_energy_topk | learned_temporal | uncertainty_topk | visibility_topk | learned | learned_budget | warped`
- `trainable`: `false` for non-learned methods and `true` for learned temporal request training
- `loss.*` and `optimizer.*`: present as placeholders for future learned modes

## Important Rule

Heavy planned methods are intentionally not implemented here. They are placeholders for a clean roadmap while keeping the current runnable method stable.

## Quick Smoke Tests

Smoke tests validate pipeline health (config + dataset + checkpoint + limited-sample inference + metrics/files).
They are **not** for final AP reporting.

Single approach:

```bash
python -m src.tools.testing.smoke_test_pipeline \
  --approach receiver_request_energy_topk_10 \
  --split carla \
  --max_samples 20 \
  --model_dir /path/to/checkpoint_or_run_dir \
  --save_debug_maps
```

Baseline comparison:

```bash
python -m src.tools.testing.smoke_test_pipeline \
  --approach selective_topk_energy_10 \
  --split carla \
  --max_samples 20 \
  --model_dir /path/to/checkpoint_or_run_dir
```

All implemented approaches:

```bash
python -m src.tools.testing.smoke_test_pipeline \
  --all_approaches \
  --split carla \
  --max_samples 5 \
  --model_dir /path/to/checkpoint_or_run_dir
```

Notes:
- Planned placeholders and trainable experimental approaches are skipped by default.
- `learned_temporal_receiver_request_10` is explicitly selectable by name, but Stage 4 checkpoint safety blocks reportable inference without trained request-head weights.
- Runs are written as `approach_runs/smoke_<split>_<approach>/`.
- Each run writes `smoke_test_report.json` and `summary_eval.yaml`.
