# Receiver-Driven Communication Approach Guide

This guide defines the receiver-driven roadmap and clearly marks which methods are runnable now vs planned placeholders.

## Status Matrix

| Approach | Preset | Status | Default |
|---|---|---|---|
| Receiver-request energy top-k | `receiver_request_energy_topk_10` | Implemented and runnable | Enabled in preset |
| Receiver-request uncertainty top-k | `receiver_request_uncertainty_topk_10` | Planned placeholder | Disabled |
| Receiver-request visibility top-k | `receiver_request_visibility_topk` | Planned placeholder | Disabled |
| Receiver-request learned | `receiver_request_learned` | Planned placeholder | Disabled |
| Receiver-request learned + budget | `receiver_request_learned_budget` | Planned placeholder | Disabled |
| Receiver-request warped alignment | `receiver_request_warped` | Planned placeholder | Disabled |

## Method Definitions

### 1) `receiver_request_energy_topk`
- Strategy: `receiver_request_topk`
- Need map: inverse ego feature energy
- Context map: collaborator L2 feature energy
- Selection: top-k by score with configurable keep ratio
- Current reference run: `receiver_request_energy_topk_10`

### 2) `receiver_request_uncertainty_topk` (planned)
- Need map target: ego detection uncertainty/entropy
- Required future work: expose ego-only objectness/uncertainty before fusion

### 3) `receiver_request_visibility_topk` (planned)
- Need map target: point density, distance, visibility, occlusion
- Required future work: BEV density/visibility map extraction

### 4) `receiver_request_learned` (planned)
- Need/request map target: learned request network
- Required future work: trainable request head, separate LR, optional detector freeze

### 5) `receiver_request_learned_budget` (planned)
- Loss target:
  - `L_sparse = lambda_sparse * mean(mask)`
  - `L_budget = lambda_budget * ReLU(mean(mask) - target_ratio)^2`
- Required future work: config-gated training path

### 6) `receiver_request_warped` (planned)
- Alignment target:
  - `warp_context_to_ego`
  - `warp_mask_to_sender`
- Required future work: BEV warp utility and coordinate convention validation

## Config Structure

Main receiver-request section (in `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml`):

- `implementation_status`: `implemented` or `planned`
- `strategy_variant`: `energy_topk | uncertainty_topk | visibility_topk | learned | learned_budget | warped`
- `trainable`: `false` for current runnable method
- `loss.*` and `optimizer.*`: present as placeholders for future learned modes

## Important Rule

Heavy planned methods are intentionally not implemented here. They are placeholders for a clean roadmap while keeping the current runnable method stable.
