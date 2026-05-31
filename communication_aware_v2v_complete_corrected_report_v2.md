# Communication-Aware V2V Cooperative Perception — Complete Corrected Experiment Report

**Project:** Communication-aware V2V cooperative perception based on a PointPillar + V2VAM intermediate-fusion pipeline  
**Main datasets evaluated:** CARLA test split and Culver City test split  
**Main checkpoint:** `net_epoch43.pth` from the original / first-version repository setup  
**Main conclusion:** Phase 2 top-k energy is currently the strongest communication-efficient result. Phase 3 learnable masking works technically and can recover very high AP, but the learned mask is not sparse enough with the current loss.

---

## 1. Executive summary

We implemented a config-driven communication-aware extension around the existing V2VAM intermediate-fusion pipeline. The final code supports multiple communication policies through YAML presets, logs communication metrics, exports per-run summaries, and supports trainable Phase 3 modules.

The strongest scientific result is from **Phase 2 top-k energy feature selection**. On CARLA, top-k 10% reaches **AP@0.7 = 0.8703** with only **9.53% communication**. On Culver City, top-k 10% reaches **AP@0.7 = 0.7356** with only **8.76% communication**. This confirms that importance-aware feature selection works on both splits.

Phase 3 learnable masking was successfully trained and evaluated on CARLA. The best Phase 3 accuracy result achieved **AP@0.7 = 0.8925**, slightly above the full baseline, but required **48.54% communication**. Therefore, Phase 3 is promising as an accuracy-preserving learned policy, but it does not yet beat top-k energy on communication efficiency.

Phase 4 repair was implemented conceptually / in configuration, but not used as a final evaluated result. It should be treated as future work, preferably as repair over fixed top-k 10% or packet-loss settings.

---

## 2. What was implemented

### 2.1 Config-first communication architecture

A master `communication` config tree was added to the main YAML. It includes:

- `enabled`, `phase`, `strategy`, `seed`
- measurement toggles: `track_bytes`, `track_active_cells`, `track_active_neighbors`, `track_latency`
- random dropping config
- top-k energy config
- neighbor selection config
- packet-loss simulation config
- learnable mask config
- repair network config
- logging and visualization config

The main idea was to make every phase switchable through config/presets instead of code edits.

### 2.2 Communication policy module

A communication policy module was inserted before V2V fusion. It supports:

- random spatial dropping
- top-k energy spatial feature selection
- neighbor selection
- packet-loss simulation
- learnable mask
- optional repair network
- communication statistics output

### 2.3 Training and inference extensions

Training was extended to support auxiliary communication loss:

\[
L_{total} = L_{det} + L_{comm} + L_{repair}
\]

where relevant. In the current Phase 3 version, `L_comm` is based on `lambda * mean(mask)`, which is effectively an L1-style sparsity pressure on the mask.

Inference was extended to export:

- `summary_eval.yaml`
- `comm_metrics_epoch.csv`
- `comm_metrics_frame.jsonl`
- AP + communication averages
- communication ratios and bytes/frame

### 2.4 Plotting and summary tools

Tools were added or updated for:

- AP vs communication ratio
- AP vs total bytes/frame
- packet-loss analysis
- clean summary CSV/YAML creation

---

## 3. Important implementation fixes and lessons

### 3.1 Ego masking bug / interpretation fix

Early `random_drop` masked all CAV features, including ego. This produced a near-collapse in AP. We separated:

- `random_drop_all_features`: stress-test mode, masks ego + collaborators
- `random_drop_comm_only`: realistic communication baseline, keeps ego unchanged

This distinction is important because communication reduction should normally affect collaborator features, not the ego vehicle's own features.

### 3.2 Collaborator-only cost accounting

The metrics were refined to track:

- feature bytes/frame
- metadata bytes/frame
- total bytes/frame
- normalized communication ratio

This fixed ambiguity in older `bytes_per_frame` reporting.

### 3.3 Batch-size YAML anchor bug

The YAML used:

```yaml
batch_size: &batch_size 4
```

A simple regex like `batch_size:\s*\d+` did not correctly modify this line. The fixed version preserved the anchor:

```python
txt = re.sub(
    r"batch_size:\s*(?:&[A-Za-z0-9_]+\s*)?\d+",
    f"batch_size: &batch_size {BATCH_SIZE}",
    txt
)
```

### 3.4 Scheduler / learning-rate issue for Phase 3

When resuming from epoch 43, the original multistep scheduler reduced the learning rate to about `1e-6`. This is safe for the pretrained detector but probably too small for the new learnable mask. The final improved run used:

```text
LR = 1e-5
step_size = [1000]
```

This prevented immediate decay and allowed stronger Phase 3 learning.

### 3.5 Kernel / quota / backup lessons

Kaggle GPU quota and kernel restarts interrupted some experiments. The final workflow added immediate zipping after train and after inference, plus richer print logs so partial cell output can still be used if the kernel restarts.

---

## 4. CARLA evaluation setup

CARLA test evaluation used:

- validation/test path: `/kaggle/input/data-all/test/test`
- number of test scenarios: 16
- samples: 2170 frames
- checkpoint: `net_epoch43.pth` for non-trained phases
- fusion method: `intermediate`
- global sort detections enabled when supported

---

## 5. CARLA Phase 0 and Phase 1

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame | Meaning |
|---|---:|---:|---:|---:|---:|---|
| `carla_phase0_baseline` | 0.951850 | 0.947751 | 0.891187 | 1.000000 | 9,011,200 | full baseline |
| `carla_phase1_measurement` | 0.951854 | 0.947756 | 0.891214 | 1.000000 | 9,011,200 | measurement only |

**Interpretation:** Phase 1 matches Phase 0 almost exactly. This confirms that the measurement/logging layer does not affect detection performance.

---

## 6. CARLA Phase 2: core results

### 6.1 CARLA top-k energy sweep

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| `carla_phase2_topk_energy_05` | 0.05 | 0.916078 | 0.913815 | 0.858936 | 0.047315 | 716,063 |
| `carla_phase2_topk_energy_10` | 0.10 | 0.928485 | 0.926102 | 0.870266 | 0.095298 | 1,440,565 |
| `carla_phase2_topk_energy_25` | 0.25 | 0.934281 | 0.931609 | 0.874245 | 0.236862 | 3,582,756 |
| `carla_phase2_topk_energy_50` | 0.50 | 0.942616 | 0.939535 | 0.883088 | 0.472661 | 7,153,774 |

**Interpretation:** Top-k 10% is the strongest low-budget CARLA result. It keeps AP@0.7 around 0.870 with only about 9.5% communication.

### 6.2 CARLA random collaborator-only sweep

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| `carla_phase2_random_comm_only_05` | 0.05 | 0.728008 | 0.725193 | 0.657054 | 0.047997 | 722,130 |
| `carla_phase2_random_comm_only_10` | 0.10 | 0.747475 | 0.744328 | 0.671131 | 0.092186 | 1,396,812 |
| `carla_phase2_random_comm_only_25` | 0.25 | 0.824361 | 0.820522 | 0.739566 | 0.230801 | 3,509,978 |
| `carla_phase2_random_comm_only_50` | 0.50 | 0.907733 | 0.903791 | 0.820450 | 0.471122 | 7,140,199 |

**Interpretation:** Random communication-only masking is much weaker than top-k energy at the same budget. This proves that the content selected for transmission matters strongly.

### 6.3 CARLA stress test: random all-features masking

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Comm / active ratio | Meaning |
|---|---:|---:|---:|---:|---|
| `carla_phase2_random_drop` | 0.011451 | 0.011181 | 0.005362 | 0.097721 | old/random all-feature behavior |
| `carla_phase2_random_drop_all_features` | 0.011434 | 0.011165 | 0.005461 | 0.097721 | stress-test: masks ego + collaborators |

**Interpretation:** Masking ego features destroys detection. This is expected and supports keeping ego unchanged for realistic communication experiments.

### 6.4 CARLA neighbor + packet-loss experiment

| Run | Strategy | AP@0.3 | AP@0.5 | AP@0.7 | Active ratio | Packet loss |
|---|---|---:|---:|---:|---:|---:|
| `carla_phase2_neighbor_packetloss` | top-k + neighbor / packet loss | ~0.714 | ~0.711 | ~0.620 | ~0.129 | 0.20 |

**Interpretation:** Packet loss and neighbor constraints significantly reduce AP. This motivates Phase 4-style repair, but Phase 4 was not finalized in the current report.

---

## 7. Culver City evaluation setup

The Culver City test split used:

- split name: `culver`
- validation/test path: `/kaggle/input/data-all/test/test_culver_city/test_culver_city`
- scenarios: 4 folders
- samples: 550 frames
- checkpoint: `net_epoch43.pth`

Culver is harder than CARLA: full baseline AP@0.7 is around 0.772 instead of CARLA's 0.891.

---

## 8. Culver City results

### 8.1 Culver Phase 0 and Phase 1

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|
| `culver_phase0_baseline` | 0.878567 | 0.868824 | 0.772075 | 1.000000 | 9,846,784 |
| `culver_phase1_measurement` | 0.878609 | 0.868866 | 0.772197 | 1.000000 | 9,846,784 |

**Interpretation:** Measurement parity also holds on Culver.

### 8.2 Culver top-k energy sweep

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| `culver_phase2_topk_energy_05` | 0.05 | 0.7972 | 0.7917 | 0.7077 | 0.0435 | 492,806 |
| `culver_phase2_topk_energy_10` | 0.10 | 0.8315 | 0.8250 | 0.7356 | 0.0876 | 991,677 |
| `culver_phase2_topk_energy_25` | 0.25 | 0.8381 | 0.8312 | 0.7408 | 0.2178 | 2,466,352 |
| `culver_phase2_topk_energy_50` | 0.50 | 0.8558 | 0.8476 | 0.7543 | 0.4346 | 4,923,725 |

### 8.3 Culver random collaborator-only sweep

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| `culver_phase2_random_comm_only_05` | 0.05 | 0.6353 | 0.6287 | 0.5464 | 0.0444 | 502,332 |
| `culver_phase2_random_comm_only_10` | 0.10 | 0.6501 | 0.6433 | 0.5563 | 0.0847 | 959,630 |
| `culver_phase2_random_comm_only_25` | 0.25 | 0.7020 | 0.6934 | 0.5950 | 0.2109 | 2,397,063 |
| `culver_phase2_random_comm_only_50` | 0.50 | 0.7865 | 0.7768 | 0.6662 | 0.4324 | 4,904,506 |

### 8.4 Culver interpretation

Culver confirms the same pattern as CARLA:

1. Top-k energy is much better than random masking.
2. Top-k 10% is a strong low-budget operating point.
3. The absolute AP is lower than CARLA, showing domain difficulty, but the communication-efficiency pattern remains.
4. Phase 3 was not run fully on Culver; it was trained and evaluated only on CARLA.

---

## 9. Phase 3 learnable-mask experiments on CARLA

Phase 3 was tested only on CARLA. It was not fully evaluated on Culver.

### 9.1 Phase 3 version A: first long default run

| Setting | Value |
|---|---|
| Preset | `phase3_learnable_mask` |
| sparsity lambda | 0.01 |
| temperature | 1.0 |
| batch size | 4 |
| fine-tune | ~5 epochs from epoch 43 |
| LR after scheduler | ~1e-6 |
| inference AP | AP@0.3 ≈ 0.95, AP@0.5 ≈ 0.95, AP@0.7 ≈ 0.89 |
| comm metrics | not preserved / not enough for final claim |

**Interpretation:** This proved Phase 3 code could train and infer, but it was not enough scientifically because the communication metrics were not preserved after kernel interruption.

### 9.2 Phase 3 version B: default learnable mask, batch 8, one epoch

| Metric | Value |
|---|---:|
| AP@0.3 | 0.941590 |
| AP@0.5 | 0.936973 |
| AP@0.7 | 0.865703 |
| comm_active_ratio | 0.412461 |
| comm_normalized_ratio | 0.412461 |
| bytes/frame | 6,276,242 |

**Interpretation:** Communication dropped to about 41%, but AP@0.7 dropped below top-k 10%. Not competitive.

### 9.3 Phase 3 version C: lambda 0.05, temp 0.5, soft mask

| Metric | Value |
|---|---:|
| AP@0.3 | 0.954986 |
| AP@0.5 | 0.950297 |
| AP@0.7 | 0.889841 |
| comm_active_ratio | 0.434398 |
| comm_normalized_ratio | 0.434398 |
| bytes/frame | 6,659,243 |

**Interpretation:** This is a strong accuracy result and is close to full baseline, but it still uses about 43.4% communication. It is better than top-k 50% in AP at slightly lower communication, but it is not as efficient as top-k 10%.

### 9.4 Phase 3 version D: lambda 0.1, temp 0.5, batch 8, LR 1e-6

| Metric | Value |
|---|---:|
| AP@0.3 | 0.948339 |
| AP@0.5 | 0.944342 |
| AP@0.7 | 0.879786 |
| comm_active_ratio | 0.566979 |
| comm_normalized_ratio | 0.566979 |
| bytes/frame | 8,614,080 |

**Interpretation:** This was not an improvement. Communication increased to 56.7% and AP decreased compared with lambda 0.05. The likely cause is low LR and weak / poorly targeted sparsity learning.

### 9.5 Phase 3 version E: lambda 0.1, temp 0.5, batch 4, 2 epochs, LR 1e-5

This was the final improved Phase 3 run.

| Setting | Value |
|---|---|
| Preset | `phase3_lam01_temp05_soft` |
| sparsity lambda | 0.1 |
| temperature | 0.5 |
| hard mask | false |
| batch size | 4 |
| fine-tune epochs | 2 |
| optimizer LR | 1e-5 |
| scheduler step size | [1000] |
| expected checkpoint | `net_epoch45.pth` |

Final metrics:

| Metric | Value |
|---|---:|
| AP@0.3 | 0.955307 |
| AP@0.5 | 0.951074 |
| AP@0.7 | 0.892544 |
| comm_active_ratio | 0.485436 |
| active_neighbors_ratio | 0.945161 |
| feature bytes/frame | 7,409,062 |
| total bytes/frame | 7,409,062 |
| comm_normalized_ratio | 0.485436 |
| packet loss | 0.0 |

**Interpretation:** This is the best Phase 3 accuracy result. It slightly exceeds the full baseline AP@0.7, but communication is still high at 48.5%. Therefore, Phase 3 is accuracy-preserving but not communication-efficient enough.

---

## 10. Consolidated CARLA comparison

| Method | AP@0.7 | Comm ratio | Meaning |
|---|---:|---:|---|
| Full baseline | 0.8912 | 1.0000 | reference full communication |
| Phase 1 measurement | 0.8912 | 1.0000 | measurement does not change output |
| Top-k 5% | 0.8589 | 0.0473 | strong ultra-low communication |
| Top-k 10% | 0.8703 | 0.0953 | best low-budget point |
| Top-k 25% | 0.8742 | 0.2369 | slight AP gain, more communication |
| Top-k 50% | 0.8831 | 0.4727 | close to full baseline |
| Random comm-only 10% | 0.6711 | 0.0922 | same budget as top-k 10%, much worse |
| Random comm-only 50% | 0.8205 | 0.4711 | same budget as top-k 50%, worse |
| Phase 3 lambda 0.05 | 0.8898 | 0.4344 | high AP, medium-high communication |
| Phase 3 lambda 0.1, b8, LR 1e-6 | 0.8798 | 0.5670 | not improved |
| Phase 3 lambda 0.1, b4, LR 1e-5 | 0.8925 | 0.4854 | best AP, but high communication |

---

## 11. What we should not overclaim

1. **Do not claim Phase 3 beats Phase 2 in communication efficiency.** It does not.
2. **Do not claim learnable masking is sparse enough.** It still uses 43–49% communication in the best AP runs.
3. **Do not use the first long Phase 3 run as a final communication result**, because its communication metrics were not preserved.
4. **Do not use hard-mask inference as final unless the preset-level override is verified.** Some earlier hard-mask experiments may have been affected by config/preset override ambiguity.
5. **Do not claim Phase 4 final results.** Phase 4 is a future extension in this report.

---

## 12. What was missed in the earlier report and is now added

The previous report was missing or under-emphasized these items:

1. **Culver City results.** The other dataset was Culver City, with Phase 0/1/2 results. This is now fully included.
2. **Culver random-vs-top-k comparison.** This is important because it confirms top-k energy generalizes beyond CARLA.
3. **Final Phase 3 LR=1e-5 result.** The latest run reached AP@0.7 = 0.8925 but comm ratio = 0.4854.
4. **Phase 3 batch-8 lambda 0.1 result.** This run was not good: AP@0.7 = 0.8798 and comm ratio = 0.5670.
5. **Initial Phase 3 long run caveat.** It had high AP but missing communication metrics, so it should not be used as final evidence.
6. **Ego-masking correction.** Random all-feature dropping is a stress test, not a realistic communication policy.
7. **LR scheduler issue.** Resuming at epoch 43 caused LR ≈ 1e-6 unless scheduler was changed.
8. **Phase 4 status.** Phase 4 was not finalized and should be future work.

---

## 13. Recommended final thesis story

The final story should be:

> We implemented a communication-aware extension of the V2VAM intermediate-fusion pipeline with config-controlled communication policies, measurement logging, and trainable masking. Phase 2 top-k energy feature selection is the strongest current method: it reduces communication to around 9–10% while preserving much of the full-communication AP on both CARLA and Culver City. Phase 3 learnable masking can recover near-full or even slightly above-full AP on CARLA, but the current sparsity loss is not strong or targeted enough, so it still uses around 43–49% communication. Therefore, Phase 3 is a promising prototype, while Phase 2 is the main successful result.

---

## 14. Recommended next technical step

Do not keep training Phase 3 longer with the same loss. The next improvement should be a **budget-aware loss**:

\[
L_{budget} = \lambda_{budget} \cdot \max(0, \text{mask\_mean} - \text{target\_ratio})^2
\]

Recommended configuration:

```yaml
learnable_mask:
  target_ratio: 0.10
  budget_lambda: 1.0
  sparsity_lambda: 0.05
```

The objective should be:

\[
L_{total} = L_{det} + \lambda_{sparse} \cdot mean(mask) + \lambda_{budget} \cdot ReLU(mean(mask)-target)^2
\]

Also consider:

- higher LR only for mask head
- frozen detector/backbone for mask-only training
- hard-mask inference only after soft-mask budget behavior is good
- Phase 4 repair over fixed top-k 10% or packet-loss settings

---

## 15. Phase 4 recommendation

Phase 4 should not be run on top of the current Phase 3 mask because Phase 3 still transmits too much. A better Phase 4 experiment is:

```text
Phase 4 = repair network over fixed top-k 10% communication
```

Goal:

```text
same communication ratio ≈ 0.095
higher AP@0.7 than top-k 10%
```

This would directly test whether repair can recover accuracy under a strict communication bottleneck.

---

## 16. Final conclusion

**Best overall result:** Phase 2 top-k energy.  
**Best CARLA low-budget point:** top-k 10%, AP@0.7 = 0.8703, comm ratio = 0.0953.  
**Best Culver low-budget point:** top-k 10%, AP@0.7 = 0.7356, comm ratio = 0.0876.  
**Best Phase 3 AP:** lambda 0.1, batch 4, LR 1e-5, AP@0.7 = 0.8925, comm ratio = 0.4854.  
**Main limitation:** learned mask is not sparse enough.  
**Next method:** budget-aware sparsity loss + mask-specific optimization.

