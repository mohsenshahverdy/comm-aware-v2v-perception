# Communication-Aware V2VAM Experiment Summary

**Project:** Communication-aware V2V cooperative perception  
**Repo:** `mohsenshahverdy/comm-aware-v2v-perception`  
**Dataset split used in most reported runs:** CARLA test split, 2,170 samples  
**Base checkpoint:** `net_epoch43.pth` from `/kaggle/input/best-epoch`  
**Main model:** PointPillar intermediate fusion with V2VAM-style fusion (`point_pillar_intermediate_V2VAM`)  
**Main objective:** reduce communication cost while preserving 3D detection AP, especially AP@0.7.

---

## 1. Executive conclusion

The work is successful up to Phase 2 and technically successful but not communication-optimal in Phase 3.

### Strongest result overall

**Phase 2 top-k energy** is still the strongest communication-efficient method.

- Top-k 10% kept only about **9.5% communication**.
- AP@0.7 stayed around **0.87**, compared with full baseline around **0.891**.
- This is the cleanest result for the thesis story: **large communication reduction with moderate AP loss**.

### Best Phase 3 accuracy result

The final tuned Phase 3 run reached the best AP:

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Communication ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|
| Phase 3, lambda=0.1, temp=0.5, batch=4, 2 epochs, LR=1e-5 | 0.9553 | 0.9511 | **0.8925** | **0.4854** | 7.409M |

This proves the learnable mask can preserve or even slightly improve detection accuracy, but it still sends too many feature cells.

### Main research conclusion

Phase 3 learned masking currently behaves more like an **accuracy-preserving soft attention mask** than a strict communication budget controller. It needs a **budget-aware sparsity loss** to become communication-efficient.

Recommended final thesis wording:

> Phase 2 energy-based top-k feature sharing is the strongest communication-efficient result. Phase 3 learnable masking can recover nearly full baseline AP, but its current mean-mask sparsity penalty does not enforce a sufficiently low communication budget. Future work should replace or augment the penalty with a target-budget loss.

---

## 2. Implementation work completed

The Codex agent implemented a config-driven communication-aware extension around the existing V2VAM/PointPillar pipeline.

### 2.1 Master communication configuration

A single `communication` tree was added to the YAML config. It controls:

- `enabled`
- `phase`
- `strategy`
- `measurement`
- `drop_random`
- `topk_energy`
- `neighbor_selection`
- `packet_loss`
- `learnable_mask`
- `repair_network`
- logging and visualization settings

A `communication_preset` selector was added so phases can be switched by preset instead of editing model code.

### 2.2 Phase presets

Presets were created for:

- `phase0_baseline`
- `phase1_measurement`
- `phase2_random_drop`
- `phase2_random_drop_all_features`
- `phase2_random_drop_comm_only`
- `phase2_topk_energy`
- `phase2_topk_energy_05`, `_10`, `_25`, `_50`
- `phase2_neighbor_packetloss`
- `phase2_packetloss_10`, `_20`, `_30`, `_50`
- `phase3_learnable_mask`
- `phase3_lam005_temp05_soft`
- `phase3_lam01_temp05_soft`
- `phase3_lam02_temp05_soft`
- `phase4_repair`

### 2.3 Communication policy module

A new communication policy module was inserted before V2V fusion. It supports:

- random spatial feature dropping
- top-k energy spatial selection
- collaborator-only masking
- neighbor selection
- packet-loss simulation
- learnable masks
- optional repair network

Important correction:

- The actual path was `src/models/fuse_modules/communication_policy.py`, not `src/models/communication/communication_policy.py`.

### 2.4 Loss changes

For Phase 3 and Phase 4, auxiliary losses were added:

```text
L_total = L_det + L_comm (+ L_repair)
```

Where:

```text
L_comm = lambda * mean(mask)
```

This is effectively an L1-style sparsity penalty on the mask.

### 2.5 Logging and metrics

The pipeline was extended to export:

- `summary_eval.yaml`
- `comm_metrics_epoch.csv`
- `comm_metrics_frame.jsonl`
- TensorBoard communication scalars
- AP + communication merged summaries
- plotting utilities for AP-vs-communication curves

Later, metrics were improved to separate:

- `feature_bytes_per_frame`
- `metadata_bytes_per_frame`
- `total_bytes_per_frame`
- `normalized_ratio`

---

## 3. Environment and execution setup

### 3.1 Local Mac status

Local Mac checks were partially successful:

- Syntax checks passed with `py_compile`.
- Communication policy import passed.
- Fake/unit policy tests passed.

But full training/inference was blocked locally because of Linux/CUDA dependencies such as `spconv` and GPU compatibility.

### 3.2 Kaggle setup

Kaggle became the main execution environment.

Verified runtime:

- CUDA available
- Torch working on GPU
- `spconv` import OK
- `open3d` import OK
- `yaml` import OK

The repo was cloned into:

```text
/kaggle/working/comm-aware-v2v-perception
```

The main checkpoint came from:

```text
/kaggle/input/best-epoch/net_epoch43.pth
```

The dataset paths were patched as:

```text
root_dir: /kaggle/input/data-all/train
validate_dir for training: /kaggle/input/data-all/validate
validate_dir for inference: /kaggle/input/data-all/test/test
```

The CARLA test split contained **2,170 samples**.

---

## 4. Important debugging and corrections

### 4.1 Ego masking problem

Initially, `random_drop` masked all CAVs including ego. That caused extreme AP collapse.

This led to two separate random-drop modes:

- `random_drop_all_features`: stress test; masks ego + collaborators.
- `random_drop_comm_only`: realistic communication setting; keeps ego unchanged and masks only collaborators.

The same collaborator-only principle was later applied to top-k and packet-loss policies.

### 4.2 Bytes/frame accounting changed

Earlier runs used legacy `bytes_per_frame`. Later runs added:

- feature bytes
- metadata bytes
- total bytes
- normalized ratio

Because of this, some earlier byte values and later byte values are not directly identical. The normalized ratio is the safest cross-run comparison.

### 4.3 YAML batch-size anchor bug

The config had YAML anchors such as:

```yaml
batch_size: &batch_size 4
```

The first regex replacement removed the anchor and sometimes failed to patch correctly. It was fixed with:

```python
txt = re.sub(
    r"batch_size:\s*(?:&[A-Za-z0-9_]+\s*)?\d+",
    f"batch_size: &batch_size {BATCH_SIZE}",
    txt
)
```

### 4.4 Learning-rate issue in Phase 3

When resuming from `net_epoch43.pth`, the scheduler had already decayed LR to:

```text
1e-6
```

This was safe for the pretrained detector but too weak for the newly added learnable mask.

Final improved Phase 3 run therefore used:

```text
LR = 1e-5
scheduler step_size = [1000]
```

This prevented immediate decay and let the learnable mask update more strongly.

### 4.5 Hard-mask caveat

Hard-mask inference is not fully trusted unless `hard_mask` is patched both in:

- `config.yaml`
- `communication_phase_presets.yaml`

Therefore, the trusted reported Phase 3 results are **soft-mask** results.

---

## 5. Phase-by-phase results

## 5.1 Phase 0: baseline

Purpose:

- Verify original model/checkpoint still works after code changes.
- Establish full-communication AP and cost.

Result:

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Communication ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|
| `carla_phase0_baseline` | 0.95185 | 0.94775 | 0.89119 | 1.000 | 9.011M |

Interpretation:

- Baseline is stable.
- Full communication reference AP@0.7 is about **0.891**.

---

## 5.2 Phase 1: measurement only

Purpose:

- Verify that enabling communication measurement does not affect AP.

Result:

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Communication ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|
| `carla_phase1_measurement` | 0.95185 | 0.94776 | 0.89121 | 1.000 | 9.011M |

Interpretation:

- Measurement layer has no meaningful AP effect.
- Phase 1 validates backward compatibility.

---

## 5.3 Phase 2: fixed communication policies

Phase 2 tested non-learned communication policies before V2V fusion.

### 5.3.1 Initial Phase 2 core runs

| Run | AP@0.5 | AP@0.7 | Communication ratio | Packet loss | Interpretation |
|---|---:|---:|---:|---:|---|
| `random_drop_all_features` | ~0.011 | ~0.005 | ~0.098 | 0.0 | Stress test; bad because ego was also masked. |
| `random_drop_comm_only` | ~0.744 | ~0.671 | ~0.092 | 0.0 | Realistic random communication drop; much better than all-feature drop but much worse than top-k. |
| `topk_energy` | ~0.834 | ~0.766 | ~0.101 | 0.0 | Strong fixed sparse policy. |
| `neighbor_packetloss` | ~0.711 | ~0.620 | ~0.129 | 0.2 | Packet loss hurts AP substantially. |

### 5.3.2 Top-k energy sweep

Reported sweep values:

| Run | Approx. communication ratio | Approx. AP@0.7 | Interpretation |
|---|---:|---:|---|
| `topk_energy_05` | ~0.047 | ~0.859 | Very strong compression, moderate AP loss. |
| `topk_energy_10` | ~0.095 | ~0.870 | Best communication-efficient result. |
| `topk_energy_25` | ~0.24–0.25 | ~0.87 | Similar AP, more communication. |
| `topk_energy_50` | ~0.473 | ~0.883 | High AP, but higher communication. |

Key Phase 2 conclusion:

> Top-k energy at about 10% communication is the best tradeoff found so far. It preserves AP@0.7 around 0.87 while using only about 9.5% of the communication.

---

## 5.4 Phase 3: learnable mask experiments

Phase 3 introduced a learnable spatial mask with sparsity loss.

General Phase 3 configuration:

```text
strategy = learnable_mask
hard_mask = false for trusted results
mask_channels = 16
temperature = usually 0.5
L_comm = sparsity_lambda * mean(mask)
```

### Phase 3 summary table

| Version | Training setup | LR | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Early Phase 3 run | batch=4, several epochs, loaded epoch 48 | unknown/decayed | ~0.95 | ~0.95 | ~0.89 | missing | missing | Accuracy worked, but communication metrics lost after kernel restart. |
| Phase 3 base rerun | lambda likely default/low, batch=8, +1 epoch | 1e-6 | 0.9416 | 0.9370 | 0.8657 | 0.4125 | 6.276M | Valid, but lower AP and still high communication. |
| Phase 3 lambda=0.05 | lambda=0.05, temp=0.5, soft mask, trained to epoch45 | likely 1e-6/decayed | 0.9550 | 0.9503 | 0.8898 | 0.4344 | 6.659M | Strong AP, less communication than final LR run, but still high. |
| Phase 3 lambda=0.1, batch=8 | lambda=0.1, temp=0.5, batch=8, +1 epoch | 1e-6 | 0.9483 | 0.9443 | 0.8798 | 0.5670 | 8.614M | Worse than lambda=0.05; too much communication. |
| Phase 3 lambda=0.1, batch=4, LR=1e-5 | lambda=0.1, temp=0.5, batch=4, +2 epochs, scheduler step=[1000] | 1e-5 | 0.9553 | 0.9511 | **0.8925** | 0.4854 | 7.409M | Best AP, but not sparse enough. |

### 5.4.1 Early Phase 3 run

The first Phase 3 training/inference completed successfully and achieved:

```text
AP@0.3 ≈ 0.95
AP@0.5 ≈ 0.95
AP@0.7 ≈ 0.89
```

However, because Kaggle restarted before saving files, the communication metrics were lost. This run is useful only as evidence that Phase 3 training/inference can run, not as a final communication-efficiency result.

### 5.4.2 Phase 3 batch-size-8 rerun

This run used batch size 8 and one fine-tuning epoch.

Result:

```text
AP@0.3 = 0.9415903896
AP@0.5 = 0.9369729619
AP@0.7 = 0.8657030327
comm_active_ratio = 0.4124608367
comm_normalized_ratio = 0.4124608367
comm_total_bytes_per_frame = 6276242.367
```

Interpretation:

- Technically valid.
- Communication was lower than full baseline but still high.
- AP was worse than top-k 10% despite much higher communication.

### 5.4.3 Phase 3 lambda=0.05

This was the first useful tuned Phase 3 result.

Result:

```text
AP@0.3 = 0.9549858812
AP@0.5 = 0.9502968634
AP@0.7 = 0.8898407877
comm_active_ratio = 0.4343978293
comm_normalized_ratio = 0.4343978293
comm_total_bytes_per_frame = 6659242.572
```

Interpretation:

- Very good accuracy.
- Slightly lower communication than top-k 50% style communication.
- But still far above top-k 10% communication.

This became the best Phase 3 tradeoff before the LR experiment.

### 5.4.4 Phase 3 lambda=0.1, batch=8, LR=1e-6

This run used lambda=0.1 but still resumed at decayed LR=1e-6.

Result:

```text
AP@0.3 = 0.9483391845
AP@0.5 = 0.9443419194
AP@0.7 = 0.8797855615
comm_active_ratio = 0.5669793492
comm_normalized_ratio = 0.5669793492
comm_total_bytes_per_frame = 8614079.546
```

Interpretation:

- Worse AP than lambda=0.05.
- Worse communication than lambda=0.05.
- Not useful as a final result.
- Likely issue: LR=1e-6 was too weak for the new mask.

### 5.4.5 Final Phase 3 lambda=0.1, batch=4, LR=1e-5, 2 epochs

This was the cleanest improved Phase 3 run.

Configuration:

```text
communication_preset = phase3_lam01_temp05_soft
sparsity_lambda = 0.1
temperature = 0.5
hard_mask = false
batch_size = 4
fine_tune_epochs = 2
optimizer LR = 1e-5
scheduler step_size = [1000]
start checkpoint = net_epoch43.pth
final checkpoint = net_epoch45.pth
```

Training:

```text
Epoch 43 mean train loss = 0.295872
Epoch 43 mean validation loss = 0.333936
Epoch 44 mean train loss ≈ 0.2925
Epoch 44 mean validation loss ≈ 0.3355
```

Inference result:

```text
AP@0.3 = 0.9553067189
AP@0.5 = 0.9510739760
AP@0.7 = 0.8925443281
comm_active_ratio = 0.4854356017
comm_normalized_ratio = 0.4854356017
comm_total_bytes_per_frame = 7409061.654
packet_loss_rate = 0.0
```

Interpretation:

- Best Phase 3 AP.
- AP@0.7 is slightly above the full baseline reference.
- Communication ratio is still around 48.5%, so it is not sparse enough.
- This supports the conclusion that the learnable mask can preserve accuracy, but the current sparsity loss is not enough to enforce a strict communication budget.

---

## 6. Phase 4 discussion

Phase 4 repair was implemented/configured conceptually, but it was not the best next run after the current Phase 3 result.

Reason:

```text
Current Phase 3 still sends ~48.5% of features.
Repair is more meaningful when communication is aggressively reduced or corrupted.
```

Best future Phase 4 design:

```text
Phase 4 = repair network on top of fixed sparse communication
Recommended base: phase2_topk_energy_10
Goal: keep communication around 0.095 and improve AP@0.7 above top-k 10%.
```

Do not run repair on top of the current high-communication Phase 3 mask unless the goal is only to test code execution.

---

## 7. Key technical lessons

### 7.1 Top-k energy is surprisingly strong

Energy-based selection is simple but effective. It selects spatial BEV cells with high feature energy and preserves most useful signal.

It is currently stronger than learnable masks for communication efficiency.

### 7.2 Learnable mask needs budget-aware control

Current Phase 3 loss:

```text
L_comm = lambda * mean(mask)
```

This is a soft L1-like penalty. It encourages smaller masks but does not enforce a target budget.

The recommended next loss is:

```text
L_budget = lambda_budget * ReLU(mask_mean - target_ratio)^2
```

Then:

```text
L_total = L_det + lambda_sparse * mean(mask) + L_budget
```

Recommended values:

```text
target_ratio = 0.10 or 0.15
lambda_sparse = 0.05 or 0.1
lambda_budget = 1.0 initially
```

### 7.3 Train only the communication module next

The detector is already good. For better Phase 3 efficiency, freeze most of the detector and train only the mask module.

Recommended future setup:

```text
freeze_detector = true
mask_lr = 1e-4
backbone_lr = 0 or 1e-6
target_ratio = 0.10 or 0.15
```

### 7.4 Do not evaluate only AP

For this project, a result is good only if it has both:

```text
high AP@0.7
low comm_normalized_ratio
```

Phase 3 has high AP but still high communication.

Phase 2 top-k 10 has slightly lower AP but much lower communication.

---

## 8. Final result ranking

### Best accuracy

| Rank | Method | AP@0.7 | Comm ratio |
|---:|---|---:|---:|
| 1 | Phase 3 lambda=0.1, batch=4, LR=1e-5 | 0.8925 | 0.4854 |
| 2 | Phase 0/1 full baseline | ~0.891 | 1.000 |
| 3 | Phase 3 lambda=0.05 | 0.8898 | 0.4344 |

### Best communication efficiency

| Rank | Method | AP@0.7 | Comm ratio |
|---:|---|---:|---:|
| 1 | Phase 2 top-k energy 10% | ~0.870 | ~0.095 |
| 2 | Phase 2 top-k energy 5% | ~0.859 | ~0.047 |
| 3 | Phase 3 lambda=0.05 | 0.8898 | 0.4344 |
| 4 | Phase 3 lambda=0.1, batch=4, LR=1e-5 | 0.8925 | 0.4854 |

### Best thesis story

Use:

1. **Phase 0/1**: baseline and measurement parity.
2. **Phase 2**: main successful communication-efficient method.
3. **Phase 3**: learnable mask prototype, high AP but not sparse enough.
4. **Phase 4**: proposed future repair experiment over top-k 10% or packet-loss settings.

---

## 9. Recommended next actions

### Immediate

1. Download final backups:
   - `phase3_lam01_lr1e5_b4_e2_full_backup.zip`
   - `all_phase_results_final_backup.zip` if created
   - compact result backup if created
2. Save final result tables:
   - `summary_eval.yaml` files
   - `comm_metrics_epoch.csv` files
   - `comm_metrics_frame.jsonl` for final runs
3. Write report around Phase 2 + Phase 3 analysis.

### Do not do next

Do not simply train Phase 3 longer with the same loss. It will likely keep AP high but not enforce low communication.

### Best future experiment

Implement budget-aware sparsity:

```text
mask_mean = mean(mask over collaborator cells only)
L_sparse = lambda_sparse * mask_mean
L_budget = lambda_budget * ReLU(mask_mean - target_ratio)^2
L_total = L_det + L_sparse + L_budget
```

Run one clean future experiment:

```text
phase3_budget_target015
batch_size = 4
fine_tune_epochs = 1 or 2
mask_lr = 1e-4
target_ratio = 0.15
hard_mask = false first
```

Success:

```text
AP@0.7 >= 0.86
comm_ratio <= 0.20
```

Excellent:

```text
AP@0.7 around 0.87
comm_ratio around 0.10–0.15
```

---

## 10. Short final conclusion paragraph for thesis/report

This work added a configurable communication-control layer to the V2VAM intermediate-fusion pipeline and evaluated multiple communication policies under the CARLA test split. Baseline and measurement-only phases showed that the added logging infrastructure preserves the original full-communication performance, with AP@0.7 around 0.891. Fixed sparse communication policies in Phase 2 demonstrated the strongest communication-efficiency result: energy-based top-k feature sharing preserved AP@0.7 around 0.87 while reducing the communication ratio to about 9.5%. Phase 3 learnable masks were technically successful and achieved near-baseline or slightly higher AP, with the best AP@0.7 reaching 0.8925. However, the learned mask still transmitted 48.5% of features, showing that the current mean-mask sparsity loss is insufficient for strict bandwidth control. Therefore, Phase 2 is the main successful contribution, while Phase 3 motivates future work with explicit budget-aware sparsity losses and mask-specific optimization.

