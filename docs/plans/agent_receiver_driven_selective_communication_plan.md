# Agent Implementation Plan: Receiver-Driven Selective Communication for V2V Cooperative Perception

## 0. Goal

Implement a new configurable communication strategy for the existing PointPillars + V2VAM cooperative perception pipeline.

The new idea is **receiver-driven selective communication**:

```text
V1 = ego / receiver / decision-maker
V2 = collaborator / sender / data provider
```

V2 should **not** decide what V1 needs. V2 only exposes compact context/availability information. V1 decides which parts of V2’s information are useful for V1 and requests/selects only those parts.

Core logic:

```text
V1 need map × V2 availability/context map → V1 request mask → sparse V2 message → V2VAM fusion
```

This should be implemented as a new configurable communication policy, not as a hard-coded experiment. The current project already has a communication policy layer, top-k selection, metrics, YAML presets, CARLA/Culver evaluation, and Phase 2/3 pipelines, so this should be an incremental extension.

---

## 1. Key Design Principle

The request mask must be computed from the **receiver/ego perspective**.

For an ego receiver `e` and collaborator `i`:

```text
M_e<-i = RequestPolicy(ego_need_e, collaborator_context_i, geometry_i_to_e)
```

Then collaborator `i` sends only:

```text
F_i_masked = M_e<-i ⊙ F_i
```

Important rules:

```text
1. Ego feature must remain unchanged.
2. Collaborator features may be masked.
3. The decision mask is logically generated for the ego/receiver.
4. V2 context/availability can be used, but V2 must not decide what V1 needs.
5. Context overhead must be measured separately from feature bytes.
```

---

## 2. Mathematical Formulation

Let:

```text
F_e ∈ R^{C×H×W} = ego/receiver BEV feature map
F_i ∈ R^{C×H×W} = collaborator BEV feature map
N_e ∈ R^{H×W} = ego need map
A_i ∈ R^{H×W} = collaborator availability/context map
T_i→e = transformation from collaborator frame to ego frame
```

The receiver-driven score is:

```text
A_i→e = Warp(A_i, T_i→e)

S_e←i = Normalize(N_e) ⊙ Normalize(A_i→e)

M_e←i = TopK(S_e←i, keep_ratio = r)

F_i_masked = M_e←i ⊙ F_i
```

Communication cost should be:

```text
total_bytes = feature_bytes + context_bytes + metadata_bytes
```

The method can be formulated as a receiver-utility problem:

```text
M* = argmax_M ΔU_e(M ⊙ F_i), subject to C(M) ≤ B
```

where `ΔU_e` is the expected utility improvement for the ego/receiver. Since true AP gain is unknown before transmission, practical proxy maps are used, such as ego need and collaborator availability.

---

## 3. New Config Structure

Add a new config subtree under:

```yaml
model:
  args:
    communication:
```

Add:

```yaml
receiver_request:
  enabled: false

  # Main policy
  keep_ratio: 0.10
  score_type: "multiplicative"        # multiplicative | weighted_sum | max_gate
  normalize_scores: true
  drop_ego: false

  # Ego/receiver need map
  ego_need_type: "inverse_energy"      # inverse_energy | uncertainty | point_density | occlusion | learned
  ego_need_eps: 1.0e-6
  ego_need_power: 1.0
  ego_need_clip_min: 0.0
  ego_need_clip_max: 1.0

  # Collaborator context/availability
  collaborator_context_type: "l2"      # l2 | objectness | point_density | visibility | learned
  context_resolution: "full"           # full | half | quarter
  context_quantization_bits: 32        # 32 | 16 | 8 | 1
  count_context_overhead: true

  # Geometry/alignment
  alignment_mode: "ego_aligned"        # ego_aligned | warp_context_to_ego | warp_mask_to_sender
  use_pairwise_transform: true

  # Metadata accounting
  count_mask_metadata: true
  metadata_encoding: "dense_binary"    # dense_binary | sparse_indices | none

  # Optional filters
  min_score_threshold: null
  active_neighbor_mode: "all"          # all | topk_neighbors | distance
  max_active_neighbors: null

  # Debugging and visualization
  save_request_maps: false
  save_debug_npz: false
  debug_num_frames: 5
```

---

## 4. New YAML Presets

Add these presets to:

```text
src/hypes_yaml/communication_phase_presets.yaml
```

### 4.1 Phase 5 Receiver-Request Top-k 10%

```yaml
phase5_receiver_request_topk_10:
  enabled: true
  phase: "phase5"
  strategy: "receiver_request_topk"
  drop_ego: false
  receiver_request:
    enabled: true
    keep_ratio: 0.10
    score_type: "multiplicative"
    ego_need_type: "inverse_energy"
    collaborator_context_type: "l2"
    normalize_scores: true
    alignment_mode: "ego_aligned"
    count_context_overhead: true
    context_resolution: "full"
    context_quantization_bits: 32
    count_mask_metadata: true
    metadata_encoding: "dense_binary"
```

### 4.2 Sweep Presets

```yaml
phase5_receiver_request_topk_05:
  enabled: true
  phase: "phase5"
  strategy: "receiver_request_topk"
  drop_ego: false
  receiver_request:
    enabled: true
    keep_ratio: 0.05
    ego_need_type: "inverse_energy"
    collaborator_context_type: "l2"
    normalize_scores: true
    alignment_mode: "ego_aligned"
    count_context_overhead: true

phase5_receiver_request_topk_25:
  enabled: true
  phase: "phase5"
  strategy: "receiver_request_topk"
  drop_ego: false
  receiver_request:
    enabled: true
    keep_ratio: 0.25
    ego_need_type: "inverse_energy"
    collaborator_context_type: "l2"
    normalize_scores: true
    alignment_mode: "ego_aligned"
    count_context_overhead: true

phase5_receiver_request_topk_50:
  enabled: true
  phase: "phase5"
  strategy: "receiver_request_topk"
  drop_ego: false
  receiver_request:
    enabled: true
    keep_ratio: 0.50
    ego_need_type: "inverse_energy"
    collaborator_context_type: "l2"
    normalize_scores: true
    alignment_mode: "ego_aligned"
    count_context_overhead: true
```

### 4.3 Future Uncertainty-Based Preset

Add this preset, but it does not need to be run in the first implementation round:

```yaml
phase5_receiver_request_uncertainty_10:
  enabled: true
  phase: "phase5"
  strategy: "receiver_request_topk"
  drop_ego: false
  receiver_request:
    enabled: true
    keep_ratio: 0.10
    ego_need_type: "uncertainty"
    collaborator_context_type: "l2"
    normalize_scores: true
    alignment_mode: "ego_aligned"
    count_context_overhead: true
```

---

## 5. Files to Modify

Modify these files:

```text
src/models/fuse_modules/communication_policy.py
src/hypes_yaml/communication_phase_presets.yaml
src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml
src/tools/inference.py
src/tools/train.py
src/tools/build_clean_phase2_summary.py
src/tools/plot_comm_metrics.py
src/tools/test_comm_policy_fake.py
```

Optional, only if needed:

```text
src/models/point_pillar_intermediate_V2VAM.py
```

The minimal energy-based version should mostly require changes in `communication_policy.py` and YAML presets. The V2VAM fusion module should remain unchanged unless the policy does not currently receive ego/collaborator feature grouping or pairwise transformations.

---

## 6. Implementation Details

### 6.1 Add New Strategy Dispatch

In `communication_policy.py`, add support for:

```python
strategy == "receiver_request_topk"
```

The dispatch should call:

```python
apply_receiver_request_topk(features, record_len, pairwise_t_matrix=None, config=None)
```

Expected inputs may differ depending on the current code. Adapt to the existing function signatures.

---

### 6.2 Required Helper Functions

Implement these helper functions:

```python
def _normalize_map(x, eps=1e-6):
    """
    Normalize per sample/map to [0, 1].
    x shape: (N, 1, H, W) or (N, H, W)
    """
```

```python
def _feature_energy(feat, score_type="l2", eps=1e-6):
    """
    feat shape: (N, C, H, W)
    return shape: (N, 1, H, W)
    score_type:
      - l2: sqrt(sum_c feat^2)
      - l1: mean(abs(feat))
      - max_abs: max(abs(feat))
    """
```

```python
def _ego_need_map(ego_feat, cfg):
    """
    Compute ego/receiver need.
    First version:
      inverse_energy
    Future:
      uncertainty, point_density, occlusion, learned
    return: (B_or_ego_count, 1, H, W)
    """
```

```python
def _collaborator_availability_map(collab_feat, cfg):
    """
    Compute V2 availability/context map.
    First version:
      l2 feature energy
    return: (num_collabs, 1, H, W)
    """
```

```python
def _topk_mask(score, keep_ratio):
    """
    score shape: (N, 1, H, W)
    return mask shape: (N, 1, H, W)
    """
```

```python
def _estimate_context_bytes(context_map, cfg):
    """
    Count compact context overhead.
    context_map shape: (N, 1, H, W)
    Use quantization_bits and resolution.
    """
```

```python
def _estimate_mask_metadata_bytes(mask, cfg):
    """
    Count mask/index overhead.
    dense_binary: H*W bits per collaborator
    sparse_indices: num_active * index_bytes
    none: 0
    """
```

---

### 6.3 Ego/Collaborator Grouping

The function must respect `record_len`.

Example:

```text
record_len = [3, 2]
batch 0: cav indices [0,1,2], ego index 0
batch 1: cav indices [3,4], ego index 3
```

For each scenario/group:

```python
start = 0
for b, n in enumerate(record_len):
    ego_idx = start
    collab_indices = range(start + 1, start + n)
    ...
    start += n
```

Rules:

```text
ego_idx is always kept unchanged.
Only collaborator features are masked.
Metrics must be collaborator-only for communication reduction.
```

This follows the previous correction where ego masking was separated from realistic communication-only masking.

---

### 6.4 First Version Scoring

For each ego/collaborator pair:

```python
ego_energy = _feature_energy(ego_feat)
ego_energy_norm = _normalize_map(ego_energy)

ego_need = 1.0 / (cfg.ego_need_eps + ego_energy_norm)
ego_need = _normalize_map(ego_need)

collab_context = _feature_energy(collab_feat)
collab_context = _normalize_map(collab_context)

score = ego_need * collab_context
score = _normalize_map(score)

mask = _topk_mask(score, keep_ratio)
masked_collab_feat = collab_feat * mask
```

Important:

```text
This first version assumes features are already in a comparable/aligned BEV representation at the point of masking.
If not, add alignment support using pairwise transform in a second step.
```

---

## 7. Alignment Modes

Implement as configurable, but start with `ego_aligned`.

### 7.1 `alignment_mode: ego_aligned`

Assumption:

```text
The communication policy receives features already prepared in a common BEV/alignment convention or the existing model later handles alignment.
```

Use direct score:

```text
score = ego_need × collab_context
```

This is the easiest first version.

### 7.2 `alignment_mode: warp_context_to_ego`

If pairwise transformations are available:

```text
A_i→e = warp(A_i, T_i→e)
score = ego_need × A_i→e
```

This is more correct but needs careful testing.

### 7.3 `alignment_mode: warp_mask_to_sender`

Advanced version:

```text
1. Compute score in ego frame.
2. Select mask in ego frame.
3. Warp mask back to collaborator frame.
4. Apply mask to sender feature map.
```

This better simulates real sender-side application, but it is more complex.

---

## 8. Metrics to Add or Confirm

Every run should export these fields in `summary_eval.yaml`:

```yaml
comm_active_ratio:
comm_active_neighbors_ratio:
comm_feature_bytes_per_frame:
comm_context_bytes_per_frame:
comm_metadata_bytes_per_frame:
comm_total_bytes_per_frame:
comm_normalized_ratio:
comm_packet_loss_rate:
receiver_request_keep_ratio:
receiver_request_context_ratio:
receiver_request_mask_metadata_ratio:
```

CSV fields in `comm_metrics_epoch.csv` should include:

```text
run
ap30
ap_50
ap_70
comm_active_ratio
comm_active_neighbors_ratio
comm_feature_bytes_per_frame
comm_context_bytes_per_frame
comm_metadata_bytes_per_frame
comm_total_bytes_per_frame
comm_normalized_ratio
comm_packet_loss_rate
receiver_request_keep_ratio
```

Context overhead matters because the receiver needs some V2 context/availability before requesting features. If V2 context is too large, it reduces the communication benefit.

---

## 9. Unit Tests

Add or extend `src/tools/test_comm_policy_fake.py`.

Required tests:

### 9.1 Ego Unchanged

Input:

```text
record_len = [3, 2]
ego indices = 0, 3
```

Assert:

```python
output[0] == input[0]
output[3] == input[3]
```

### 9.2 Collaborator Masked

Assert at least one collaborator has fewer active cells than full.

### 9.3 Keep Ratio Approximate

For collaborators:

```python
active_ratio ≈ keep_ratio
```

Allow small tolerance due to top-k rounding.

### 9.4 Metrics Present

Assert stats contain:

```text
feature_bytes_per_frame
context_bytes_per_frame
metadata_bytes_per_frame
total_bytes_per_frame
normalized_ratio
active_ratio
```

### 9.5 Receiver-Request Differs from Sender-Only Top-k

Construct fake tensors where:

```text
ego_need is high in region A
collab_energy is high in region B
```

Then confirm receiver-request selects the intersection/high combined score, not only collaborator high energy.

### 9.6 Batch Grouping Works

For `record_len=[3,2]`, ensure both ego vehicles are handled independently.

---

## 10. Experiment Plan

### 10.1 CARLA First

Run:

```text
phase0_baseline
phase1_measurement
phase2_topk_energy_10
phase2_random_comm_only_10
phase5_receiver_request_topk_10
```

Reference current best low-budget CARLA result:

```text
phase2_topk_energy_10:
AP@0.7 = 0.8703
comm_ratio = 0.0953
```

Success criterion:

```text
phase5_receiver_request_topk_10:
AP@0.7 > 0.8703
comm_total_ratio ≈ 0.10–0.12 including context overhead
```

If context overhead makes total ratio higher, report both:

```text
feature-only ratio
total ratio including context
```

---

### 10.2 CARLA Sweep

Run:

```text
phase5_receiver_request_topk_05
phase5_receiver_request_topk_10
phase5_receiver_request_topk_25
phase5_receiver_request_topk_50
```

Compare against:

```text
phase2_topk_energy_05
phase2_topk_energy_10
phase2_topk_energy_25
phase2_topk_energy_50
```

Generate AP-vs-communication curve.

---

### 10.3 Culver Validation

After CARLA works, run the same Phase 5 sweep on Culver.

Reference current Culver result:

```text
phase2_topk_energy_10:
AP@0.7 = 0.7356
comm_ratio = 0.0876
```

Success criterion:

```text
receiver-request topk 10% improves AP@0.7 or gives similar AP with better explanation/robustness.
```

The current reports show that CARLA and Culver both support the same conclusion: top-k energy is much stronger than random masking, so Phase 5 must be compared against top-k, not only random.

---

## 11. Plotting Requirements

Update plotting tool to support:

```text
AP@0.7 vs feature comm ratio
AP@0.7 vs total comm ratio
AP@0.7 vs total bytes/frame
Context bytes/frame by method
```

Suggested output files:

```text
ap70_vs_feature_comm_ratio.png
ap70_vs_total_comm_ratio.png
ap70_vs_total_bytes.png
context_overhead_by_method.png
```

---

## 12. Clean Summary Builder

Update `build_clean_phase2_summary.py` or create a new generic script:

```text
src/tools/build_clean_comm_summary.py
```

It should scan all `phase_runs/*/summary_eval.yaml` and output:

```text
clean_comm_summary.csv
clean_comm_summary.yaml
```

Columns:

```text
run
split
phase
strategy
keep_ratio
AP@0.3
AP@0.5
AP@0.7
comm_active_ratio
comm_feature_bytes_per_frame
comm_context_bytes_per_frame
comm_metadata_bytes_per_frame
comm_total_bytes_per_frame
comm_normalized_ratio
packet_loss_rate
notes
```

---

## 13. Notebook Updates

Update the Kaggle notebook with sections:

```text
1. Environment setup
2. Clone repo / install env
3. Verify runtime
4. Prepare run helpers
5. Phase 0/1 baseline checks
6. Phase 2 comparison references
7. Phase 5 receiver-request top-k core run
8. Phase 5 sweep
9. Culver Phase 5 validation
10. Build clean summary
11. Plot AP-vs-communication
12. Zip/download backups
```

Add one cell to run core Phase 5:

```python
phase5_core = [
    "phase5_receiver_request_topk_10",
]

for preset in phase5_core:
    run_dir = prepare_phase_run(preset, split_name="carla")
    code = run_inference(run_dir, log_name=f"{preset}.log")
    if code != 0:
        raise RuntimeError(f"Failed at {preset}")
```

Add one cell for sweep:

```python
phase5_sweep = [
    "phase5_receiver_request_topk_05",
    "phase5_receiver_request_topk_10",
    "phase5_receiver_request_topk_25",
    "phase5_receiver_request_topk_50",
]
```

Add backup cell:

```python
!cd /kaggle/working && zip -r phase5_receiver_request_results.zip phase_runs clean_comm_summary.csv *.png
```

---

## 14. Debug Output Requirements

Every Phase 5 run must print:

```text
Preset name
Split name
Config check lines
Root dir
Validate dir
Strategy
Keep ratio
Ego need type
Collaborator context type
Alignment mode
Context overhead enabled/disabled
Feature bytes/frame
Context bytes/frame
Metadata bytes/frame
Total bytes/frame
AP@0.3/AP@0.5/AP@0.7
```

This is important because Kaggle sessions can restart. The output should be sufficient to recover the result even if files are lost.

---

## 15. Visualization and Debug Maps

Add optional debug output for a small number of frames:

```yaml
receiver_request:
  save_request_maps: true
  debug_num_frames: 5
```

Save:

```text
ego_need_map.npy
collab_context_map.npy
request_score_map.npy
request_mask.npy
```

Optional PNGs:

```text
debug_receiver_request_frame_000_need.png
debug_receiver_request_frame_000_context.png
debug_receiver_request_frame_000_score.png
debug_receiver_request_frame_000_mask.png
```

This is useful for verifying that the method does not select random background.

---

## 16. Risks to Handle

### 16.1 V1 May Not Know What It Is Missing

The inverse-energy need map can mistake empty background for missing evidence.

Mitigation:

```text
Start with inverse energy as baseline.
Then add uncertainty-based need map if results are weak.
```

### 16.2 Context Overhead May Reduce Savings

V2 must expose a compact context map.

Mitigation:

```text
Count context bytes separately.
Try low-resolution or quantized context later.
```

### 16.3 Receiver-Request May Not Beat Top-k

Current top-k is already very strong.

Mitigation:

```text
Compare directly against phase2_topk_energy_10.
Do not claim success unless AP/communication frontier improves.
```

### 16.4 Alignment Errors

Sparse masks are sensitive to coordinate mismatch.

Mitigation:

```text
Start with ego_aligned mode if consistent with current model.
Add visual debugging.
Only then implement warp_context_to_ego.
```

---

## 17. Acceptance Criteria

Code is acceptable only if:

```text
1. Existing Phase 0/1/2 runs still work.
2. New Phase 5 presets run without code edits.
3. Ego features are never masked.
4. Receiver-request strategy produces communication metrics.
5. Context overhead is counted.
6. Clean summary includes Phase 5.
7. Unit tests pass.
8. CARLA phase5_receiver_request_topk_10 can be compared against phase2_topk_energy_10.
```

---

## 18. Expected Result Interpretation

### Outcome A: Phase 5 Beats Top-k at Same Communication

This supports the hypothesis:

```text
receiver-conditioned request is better than sender-only importance.
```

### Outcome B: Phase 5 Equals Top-k

This still validates the framework, but inverse-energy need is probably too simple.

Next step:

```text
detection_uncertainty need map
```

### Outcome C: Phase 5 Is Worse Than Top-k

This suggests:

```text
V1 need proxy is weak
or alignment/context overhead hurts
```

Next step:

```text
replace inverse_energy with uncertainty or occlusion-aware need.
```

---

## 19. Final Instruction to Agent

Implement the extension incrementally in this order:

```text
Step 1: Add config schema and presets.
Step 2: Add receiver_request_topk strategy using inverse-energy need and L2 collaborator context.
Step 3: Preserve ego features and mask only collaborators.
Step 4: Add context/metadata byte accounting.
Step 5: Add unit tests.
Step 6: Run CARLA phase5_receiver_request_topk_10.
Step 7: Compare with phase2_topk_energy_10.
Step 8: If successful, run 05/10/25/50 sweep.
Step 9: Validate on Culver.
Step 10: Add plots and clean summary.
```

Do not implement learned receiver request first. Do not modify V2VAM fusion unless necessary. Start with the non-learned energy-based receiver-request baseline because it is closest to the existing top-k implementation and easiest to debug.
