# Receiver-Driven Selective Communication Implementation Plan (Updated)

## Scope
This plan maps `agent_receiver_driven_selective_communication_plan.md` to concrete repository changes.

**Important update:** Phase 5 is split into two stages.
- **Stage 1 (now):** non-learned receiver-request top-k, inference-first, no training/loss coupling.
- **Stage 2 (later):** learned receiver-request with config-gated losses and optimizer/freeze controls.

---

## A. Current Architecture Understanding

### 1) Current communication policy behavior
- Main policy class: `src/models/fuse_modules/communication_policy.py` (`CommunicationPolicy`).
- Policy runs before fusion and returns:
  - masked/processed features
  - communication stats (`comm_stats`)
  - auxiliary tensors (`comm_aux`)
- Current strategies:
  - `none`
  - `random_drop_all_features` (stress mode)
  - `random_drop_comm_only`
  - `topk_energy`
  - `learnable_mask`
  - optional `packet_loss` + optional `repair_network`

### 2) Where masking happens
- In `CommunicationPolicy.forward(...)` after neighbor selection and collaborator mask creation (`tx_mask`).
- Ego/collaborator split is already explicit.
- Collaborator-only masking is already used for realistic strategies.

### 3) `record_len`, ego/collab grouping, and metrics
- Grouping helpers:
  - `_split_by_record_len(...)`
  - `_ego_indices(...)`
- Neighbor filtering:
  - `_select_neighbors(...)` returns updated features + active-neighbor ratio + selected mask.
- Existing communication metrics include:
  - `active_ratio`, `active_neighbors_ratio`
  - `feature_bytes_per_frame`, `metadata_bytes_per_frame`, `total_bytes_per_frame`, `normalized_ratio`
  - compatibility alias: `bytes_per_frame`

### 4) YAML presets loading and apply path
- `src/hypes_yaml/yaml_utils.py::load_yaml` loads `config.yaml`, then merges preset from:
  - `src/hypes_yaml/communication_phase_presets.yaml`
- Merge target:
  - `model.args.communication`
- Implication:
  - preset values can override base config; test/runtime patches must account for this.

---

## B. Required Code Changes (by file)

## `src/models/fuse_modules/communication_policy.py`
- Add new strategy dispatch:
  - `strategy == "receiver_request_topk"` (Stage 1)
  - placeholder dispatch for `receiver_request_learned` (Stage 2)
- Add helper functions:
  - `_normalize_map(...)`
  - `_feature_energy(...)`
  - `_ego_need_map(...)` (first: `inverse_energy`)
  - `_collaborator_context_map(...)` (first: `l2`)
  - `_topk_mask(...)`
  - `_estimate_context_bytes(...)`
  - `_estimate_mask_metadata_bytes(...)`
  - `compute_receiver_request_mask(...)` (core Stage 1 logic)
- Add Stage 1 metrics:
  - `context_bytes_per_frame`
  - `receiver_request_keep_ratio`
  - `receiver_request_context_ratio`
  - `receiver_request_mask_metadata_ratio`
- Why:
  - this is the core functional extension.
- Risk:
  - **High** (correctness + accounting + grouping).

## `src/models/point_pillar_intermediate_V2VAM.py`
- No structural change expected for Stage 1.
- Confirm current forward already passes `record_len` + `pairwise_t_matrix` to policy (it does).
- Optional Stage 2 hook only if request head state must be surfaced.
- Risk:
  - **Low**.

## `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml`
- Add `communication.receiver_request` schema with defaults (Stage 1 + Stage 2 placeholders).
- Keep defaults disabled/backward-compatible.
- Risk:
  - **Low**.

## `src/hypes_yaml/communication_phase_presets.yaml`
- Add Stage 1 presets:
  - `phase5_receiver_request_topk_05`
  - `phase5_receiver_request_topk_10`
  - `phase5_receiver_request_topk_25`
  - `phase5_receiver_request_topk_50`
- Add future placeholder preset:
  - `phase5_receiver_request_uncertainty_10`
- Add Stage 2 placeholder preset (disabled by default):
  - `phase5_receiver_request_learned_*` (optional)
- Risk:
  - **Low**.

## `src/tools/inference.py`
- Extend CSV/frame/summary export for new metrics.
- Preserve old keys for compatibility.
- Risk:
  - **Medium**.

## `src/tools/train.py`
- **Stage 1:** no required behavior change for policy execution.
- Keep metric logging compatible with new fields if present.
- **Stage 2 (future):** add config-gated comm loss path only when enabled.
- Risk:
  - Stage 1: **Low**
  - Stage 2: **Medium**.

## `src/tools/test_comm_policy_fake.py`
- Add Stage 1 tests for receiver-request top-k behavior.
- Keep existing tests unchanged.
- Risk:
  - **Medium**.

## `src/tools/build_clean_phase2_summary.py`
- Extend to include phase5 and new metrics; optionally rename to generic builder later.
- Risk:
  - **Low/Medium**.

## `src/tools/plot_comm_metrics.py`
- Add plots using feature ratio and total ratio and context overhead.
- Risk:
  - **Low**.

---

## C. Receiver-Driven Strategy Design (Stage 1)

New strategy:
- `strategy: receiver_request_topk`

Decision rule:
- V1 (ego/receiver) computes request.
- V2 (collaborator) provides context signal; does not decide request.

Per `record_len` group:
- `ego_idx = first index`
- `collab_indices = others`
- ego unchanged
- collaborator masked by receiver-request mask

Baseline scoring:
- `ego_need_type = inverse_energy`
- `collaborator_context_type = l2`
- `score = normalize(ego_need) * normalize(collab_context)`
- `mask = topk(score, keep_ratio)`

Stage 1 alignment:
- start with `alignment_mode: ego_aligned` only.

---

## D. Configuration Design

Add under `communication.receiver_request`:

```yaml
receiver_request:
  enabled: false
  strategy_variant: "energy_topk"       # energy_topk | uncertainty_topk | learned
  trainable: false

  keep_ratio: 0.10
  score_type: "multiplicative"          # multiplicative | weighted_sum | max_gate
  normalize_scores: true
  drop_ego: false

  ego_need_type: "inverse_energy"       # inverse_energy | uncertainty | learned
  ego_need_eps: 1.0e-6

  collaborator_context_type: "l2"       # l2 | objectness | learned
  context_resolution: "full"            # full | half | quarter
  context_quantization_bits: 32
  count_context_overhead: true

  alignment_mode: "ego_aligned"
  use_pairwise_transform: true

  count_mask_metadata: true
  metadata_encoding: "dense_binary"     # dense_binary | sparse_indices | none

  loss:
    enabled: false
    sparse_lambda: 0.0
    budget_enabled: false
    budget_lambda: 0.0
    target_ratio: 0.10
    apply_to: "collaborators_only"

  optimizer:
    separate_lr: false
    request_head_lr: 1.0e-4
    freeze_detector: false
    freeze_backbone: false
```

Presets to add:
- `phase5_receiver_request_topk_05`
- `phase5_receiver_request_topk_10`
- `phase5_receiver_request_topk_25`
- `phase5_receiver_request_topk_50`
- `phase5_receiver_request_uncertainty_10` (future placeholder)

---

## E. Metrics and Logging

Add/confirm exports:
- `comm_feature_bytes_per_frame`
- `comm_context_bytes_per_frame`
- `comm_metadata_bytes_per_frame`
- `comm_total_bytes_per_frame`
- `comm_normalized_ratio`
- `receiver_request_keep_ratio`
- `receiver_request_context_ratio`
- `receiver_request_mask_metadata_ratio`

Accounting rules:
- **feature bytes:** collaborator selected cells only
- **context bytes:** collaborator context payload (resolution + quantization aware)
- **metadata bytes:** mask signaling payload
- **total bytes:** feature + context + metadata
- ego excluded from communication accounting

Stage 1 logging target:
- inference summary + CSV/frame JSONL compatibility first.

---

## F. Unit Test Plan

Extend `src/tools/test_comm_policy_fake.py` with `record_len=[3,2]`:
- Ego unchanged (`0`, `3`).
- Collaborator masked for receiver-request top-k.
- Keep-ratio approximately correct on collaborators.
- Required metrics present (feature/context/metadata/total/normalized/active).
- Receiver-request differs from sender-only top-k on crafted tensors.
- Grouping independence across two scenes.

---

## G. Experiment Plan

### Stage 1 (non-learned, inference-first)
1. Run CARLA:
   - `phase5_receiver_request_topk_10`
2. Compare against:
   - `phase2_topk_energy_10`
3. If successful, run sweep:
   - `phase5_receiver_request_topk_05/10/25/50`
4. Then Culver validation with same sweep.

Reference target:
- CARLA `phase2_topk_energy_10`: AP@0.7 `0.8703`, ratio `0.0953`.

Success criterion:
- Phase 5 improves AP@0.7 at similar total ratio, or provides meaningful receiver-conditioned baseline with explicit overhead decomposition.

---

## H. Risk Analysis

- Inverse-energy may over-select background.
- Context overhead can erase gains.
- Alignment mismatch may distort mask utility.
- Top-k baseline already strong.
- Receiver need proxy may be weak without uncertainty/learned cues.

Mitigations:
- Stage 1 simple baseline + strict metric decomposition + visual debug maps later.

---

## I. Step-by-Step Implementation Order (Two-Stage)

### Stage 1 (implement now)
1. Add config schema + Phase 5 presets.
2. Add policy helper functions.
3. Implement `receiver_request_topk` (`strategy_variant=energy_topk`).
4. Add context/metadata metrics.
5. Add/extend tests.
6. Run fake policy test.
7. Run one CARLA inference (`phase5_receiver_request_topk_10`).
8. Build summary + plots.
9. Run sweep if stable.
10. Validate on Culver.

### Stage 2 (future)
1. Add `receiver_request_learned` strategy path.
2. Add config-gated communication losses in train path.
3. Add optional budget loss + separate LR + freeze controls.
4. Add learned-policy tests.
5. Re-run CARLA/Culver comparisons.

---

## J. Deliverables

### Checklist
- [ ] Stage 1 config schema + presets added
- [ ] `receiver_request_topk` implemented
- [ ] ego unchanged, collaborator-only masking
- [ ] context + metadata overhead counted
- [ ] inference/train metrics compatible
- [ ] Stage 1 tests pass
- [ ] CARLA 10% comparison complete
- [ ] sweep + Culver validation complete

### Files to modify
- `src/models/fuse_modules/communication_policy.py`
- `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml`
- `src/hypes_yaml/communication_phase_presets.yaml`
- `src/tools/inference.py`
- `src/tools/test_comm_policy_fake.py`
- `src/tools/build_clean_phase2_summary.py` (or generic replacement)
- `src/tools/plot_comm_metrics.py`
- `src/tools/train.py` (Stage 2-required; Stage 1 mostly compatibility only)

### Estimated difficulty
- High: `communication_policy.py`
- Medium: `inference.py`, `test_comm_policy_fake.py`, summary builder
- Low/Medium: plots, YAML presets/schema
- Stage 2 Medium/High: train/loss/optimizer/freeze controls

### Recommended first commit breakdown
1. `feat(config): add receiver_request schema + phase5 presets`
2. `feat(policy): add receiver_request_topk (energy variant) + metrics`
3. `test(policy): add receiver-request fake tests`
4. `feat(eval): add phase5 summary fields + plotting support`
5. `chore(exp): add run notes for carla topk10 comparison`
6. `feat(stage2-placeholder): config keys for learned/loss/optimizer (disabled)`

---

## Configurable Training and Loss Modes (New Section)

Phase 5 must support both deterministic and learned variants via config, without hard-coded phase behavior.

### Non-learned variant (Stage 1)
- `strategy: receiver_request_topk`
- `receiver_request.strategy_variant: energy_topk`
- `receiver_request.trainable: false`
- `receiver_request.loss.enabled: false`
- No trainable request network.
- No optimizer parameter group changes.
- No budget loss.
- Inference-first implementation target.
- Train path should remain unaffected except metric compatibility.

### Learned variant (Stage 2, future)
- `strategy: receiver_request_learned`
- `receiver_request.strategy_variant: learned`
- `receiver_request.trainable: true`
- `receiver_request.loss.enabled: true` (config-gated)
- Optional:
  - `sparse loss`
  - `budget-aware loss`
  - separate LR for request head
  - freeze detector/backbone

### Config-gated behavior requirements
- If `trainable=false`:
  - no communication loss added
  - no request-network params added
  - no optimizer-group changes required
- If `trainable=true`:
  - request params active
  - optional freeze/separate-lr logic active only when configured
  - losses computed as configured:
    - `L_sparse = sparse_lambda * mean(mask)`
    - `L_budget = budget_lambda * ReLU(mean(mask)-target_ratio)^2`
    - `L_total = L_det + L_sparse + L_budget`

### Acceptance criteria for this mode split
- Existing Phase 0/1/2/3 behavior unchanged.
- Stage 1 non-learned Phase 5 runs without train/loss dependency.
- Stage 2 learned mode activates losses only when `loss.enabled=true`.
- No hard-coded behavior based only on phase name.
