# Professor Meeting Report — Communication-Aware V2V Cooperative Perception

**Project:** Communication-aware cooperative perception based on a PointPillars + V2VAM intermediate-fusion pipeline  
**Meeting goal:** Explain the research motivation, the implementation pipeline, the experiment phases, the results on CARLA and Culver City, and the recommended next research direction.  
**Recommended meeting structure:** Two parts:  
1. **Research foundation:** Why communication-aware cooperative perception matters and how the literature approaches the problem.  
2. **Implementation and results:** What was implemented, how the experiments were organized, what the results show, and what should be done next.

---

## 0. One-Minute Executive Summary

The starting point is a V2VAM-style intermediate-fusion cooperative perception pipeline. Dense intermediate fusion can give high 3D detection accuracy because vehicles share rich BEV features, but it is communication-heavy and assumes relatively ideal V2X links. The research problem is therefore not only “how to maximize AP,” but “how to preserve AP under limited, lossy, and non-ideal communication.”

To study this, I implemented a configurable communication layer before V2VAM fusion. This layer can measure communication cost and apply different communication policies: random dropping, top-k energy selection, neighbor/packet-loss settings, learnable masks, and future repair. The strongest result so far is Phase 2 top-k energy. On CARLA, top-k 10% reaches **AP@0.7 = 0.8703** using only **9.53% communication**. On Culver City, top-k 10% reaches **AP@0.7 = 0.7356** using only **8.76% communication**. This shows that simple importance-based feature selection is a strong and robust baseline.

Phase 3 learnable masking also works technically and can preserve excellent accuracy. The best Phase 3 run reached **AP@0.7 = 0.8925** on CARLA, slightly above the full baseline, but it still used **48.54% communication**. So Phase 3 is promising but not sparse enough yet. The next research step should be budget-aware sparsity loss or repair under a strict top-k 10% bottleneck.

---

# Part I — Research Foundation

## 1. Core Problem

Cooperative perception allows autonomous vehicles or infrastructure agents to share information so the ego vehicle can perceive beyond its own sensor range. This helps especially with occlusion, long-range detection, and partial visibility.

However, most strong cooperative-perception methods become difficult to deploy if they assume that communication is cheap, dense, instantaneous, and reliable. In real V2X settings, the communication link has constraints:

- **Bandwidth limits:** full BEV feature maps are large.
- **Latency:** messages may arrive too late for the current frame.
- **Packet loss/interruption:** transmitted features may be missing or corrupted.
- **Neighbor scalability:** more collaborators means higher communication cost.
- **Real-time requirements:** autonomous driving requires stable low-latency operation.

The central trade-off is:

\[
\text{Detection Accuracy} \quad \leftrightarrow \quad \text{Communication Cost}
\]

So the main research question becomes:

> How much cooperative detection performance can we preserve while transmitting only a small fraction of the intermediate features?

This leads to the main project framing:

> Communication should be treated as a model component, not just as an external constraint.

Instead of only learning how to fuse all features, the system should also decide:

- who should communicate,
- what should be communicated,
- how much should be communicated,
- how to handle missing/corrupted/delayed communication.

---

## 2. Why Intermediate Fusion Is the Right Starting Point

Cooperative perception methods are usually grouped by fusion level.

### 2.1 Early Fusion

Early fusion shares raw sensor data such as point clouds.

**Strength:** maximum geometric detail.  
**Weakness:** very high communication cost and expensive alignment.

### 2.2 Late Fusion

Late fusion shares final detections such as boxes and confidence scores.

**Strength:** very low communication cost.  
**Weakness:** if a local detector misses an object, late fusion cannot recover it; it also loses dense scene context.

### 2.3 Intermediate Fusion

Intermediate fusion shares learned feature maps, often BEV features, after local encoding and before final detection.

**Strength:** strong accuracy–information balance.  
**Weakness:** dense feature maps are still large.

The current repository is built around **PointPillars + V2VAM intermediate fusion**, which gives a clean insertion point:

\[
\text{Local BEV features} \rightarrow \textbf{Communication policy} \rightarrow \text{V2VAM fusion} \rightarrow \text{Detection head}
\]

This means we can add a communication module without rewriting the whole detector.

---

## 3. Research Approach Categories

The literature can be organized into several communication-efficient cooperative perception strategies.

| Approach | Main idea | Typical benefit | Fit to this project |
|---|---|---|---|
| Dense intermediate fusion | Send full BEV/intermediate features | High AP | Baseline / current model family |
| Importance-based spatial selection | Send only important BEV cells or regions | Large bandwidth reduction | Very high |
| Spatial-channel selection | Select important positions and channels | More precise pruning | Future extension |
| Compression / codebook / quantization | Send compact codes or low-bit features | Lower bits per transmitted feature | Future extension after selection |
| Neighbor scheduling | Choose which collaborators should transmit | Reduces agent-level traffic | Medium/high |
| Loss / delay / interruption robustness | Repair or align missing/stale/corrupted messages | Robustness under real V2X | High, especially Phase 4 |
| Alternative message units | Send boxes, objects, clusters, tokens | Extreme efficiency | More invasive |
| Joint communication-control | Optimize perception + resources together | System-level realism | Future policy layer |

A short way to explain this in the meeting:

> There are four big ways to reduce communication: send less often, send fewer regions, send fewer bits, or recover from bad/missing messages. My implementation mainly starts with sending fewer regions, because that fits naturally before V2VAM fusion.

---

## 4. Literature Map and What Each Family Contributes

### 4.1 Dense Intermediate Fusion

Examples: **V2VNet, DiscoNet, V2X-ViT, V2VAM**.

These methods focus on strong feature fusion. They show that exchanging intermediate features can improve detection, but they do not fully solve bandwidth-aware communication.

**Connection to this project:** V2VAM is the base architecture. We keep the fusion backbone and add a communication policy before fusion.

### 4.2 Importance-Aware Selective Sharing

Examples: **Where2comm, How2comm, COOPERTRIM, Reason-to-Transmit**.

These methods ask: where, what, and when should agents communicate? The key insight is that not all spatial cells or feature channels are equally useful.

**Connection to this project:** Phase 2 top-k energy and Phase 3 learnable masks are direct implementations of this idea.

### 4.3 Compression / Codebook / Quantization

Examples: **CodeFilling, QuantV2X, V2X-DSC, WaveComm**.

These methods reduce payload size by representing transmitted information more compactly.

**Connection to this project:** After selecting useful cells, a future version could compress the selected feature values using quantization or codebooks.

### 4.4 Loss, Delay, and Interruption Robustness

Examples: **V2VAM + LCRN, SyncNet / latency-aware collaborative perception, V2X-INCOP, Fresh2comm, CoDynTrust, QPoint2Comm**.

These methods address real-world non-ideal links: packet loss, delayed messages, stale features, or interrupted communication.

**Connection to this project:** Phase 4 repair is motivated by this family, but it should be run under a strict communication bottleneck such as top-k 10% or packet loss.

### 4.5 Scheduling and Resource Allocation

These methods decide which agents should communicate and how communication resources should be assigned.

**Connection to this project:** Neighbor selection and active-neighbor metrics are a first step, but full radio/resource scheduling is future work.

---

## 5. Model Definitions and Operations

Let each agent produce a BEV feature tensor:

\[
F_i \in \mathbb{R}^{C \times H \times W}
\]

where \(C\) is the number of channels and \(H, W\) are BEV spatial dimensions.

The ego feature is \(F_e\). Collaborator features are aligned into the ego frame before fusion.

### 5.1 Dense Communication

In the original intermediate-fusion setup, every collaborator sends the full feature map:

\[
M_i = F_i
\]

This gives strong accuracy but high communication.

### 5.2 Random Drop

A random spatial mask is sampled:

\[
R_i(h,w) \sim \text{Bernoulli}(p)
\]

and applied to the feature map:

\[
M_i = R_i \odot F_i
\]

This is a lower-bound baseline, not a smart communication policy.

### 5.3 Top-k Energy Selection

For each spatial cell, compute feature energy:

\[
s_i(h,w) = \|F_i[:,h,w]\|_2
\]

Then keep the top \(K\) cells:

\[
R_i(h,w) = 1 \quad \text{if } s_i(h,w) \text{ is in TopK}
\]

\[
M_i = R_i \odot F_i
\]

This is simple, interpretable, and turned out to be the strongest low-budget method.

### 5.4 Learnable Mask

A small neural policy predicts a mask:

\[
R_i = \sigma(g_\theta(F_i) / \tau)
\]

where \(g_\theta\) is a mask head and \(\tau\) is temperature.

The training loss includes detection loss and sparsity pressure:

\[
L_{total} = L_{det} + \lambda_{comm} \cdot \text{mean}(R)
\]

This is similar to L1 regularization on the mask. In the current experiments, it preserved accuracy but did not force enough sparsity.

### 5.5 Budget-Aware Future Loss

The next improvement should explicitly target a communication budget:

\[
L_{budget} = \lambda_{budget} \cdot \max(0, \text{mean}(R) - r_{target})^2
\]

with target ratio such as 0.10 or 0.15.

Then:

\[
L_{total} = L_{det} + \lambda_{sparse}\text{mean}(R) + L_{budget}
\]

This is more aligned with the objective: stay near a target communication budget.

### 5.6 Repair Network

If features are dropped or lost, a repair network can reconstruct missing feature content:

\[
\tilde{F}_i = \text{RepairNet}(M_i, R_i)
\]

with optional reconstruction loss:

\[
L_{repair} = \|\tilde{F}_i - F_i\|_2^2
\]

This motivates Phase 4, but only under a real bottleneck.

---

# Part II — Implementation Pipeline

## 6. Implementation Strategy

The implementation kept the existing PointPillars + V2VAM pipeline intact and inserted a communication policy before fusion.

The final code supports:

- config-driven phase presets,
- communication measurement,
- random drop,
- top-k energy selection,
- neighbor selection,
- packet-loss simulation,
- learnable mask,
- optional repair network,
- AP + communication logging,
- per-frame and per-epoch metric export.

The philosophy was:

> Do not modify the detector more than necessary. Add a communication-control layer around the message interface, then compare AP–communication trade-offs.

---

## 7. Phase Plan

| Phase | Goal | What it answers |
|---|---|---|
| Phase 0 | Baseline reproduction | Can we reproduce original full-communication AP? |
| Phase 1 | Measurement only | Does logging/measurement change AP? |
| Phase 2 | Non-learned policies | How good are simple policies like top-k and random drop? |
| Phase 3 | Learnable mask | Can a learned mask beat fixed top-k? |
| Phase 4 | Repair network | Can repair recover AP under strict loss/bottleneck? |

This phase design is important because it avoids mixing too many variables at once.

---

## 8. Implementation Details

### 8.1 Config-First Design

A master `communication` tree was added to the YAML config. It includes:

- `enabled`, `phase`, `strategy`, `seed`
- measurement toggles
- random drop settings
- top-k energy settings
- neighbor selection settings
- packet-loss settings
- learnable mask settings
- repair network settings
- logging and visualization settings

This makes experiments reproducible and switchable by changing presets.

### 8.2 Communication Policy Module

The communication policy was inserted before V2V fusion. It outputs:

- modified feature tensors,
- communication statistics,
- auxiliary tensors for losses.

### 8.3 Training Changes

Training was extended with optional auxiliary losses:

\[
L_{total} = L_{det} + L_{comm} + L_{repair}
\]

For Phase 3, the communication loss is based on mask mean.

### 8.4 Inference and Logging Changes

Inference now exports:

- `summary_eval.yaml`
- `comm_metrics_epoch.csv`
- `comm_metrics_frame.jsonl`
- AP values
- communication ratios
- bytes/frame
- packet-loss metrics

### 8.5 Important Debugging Fixes

#### Ego masking correction

Random drop originally masked all CAV features, including ego. This collapsed AP. We separated:

- `random_drop_all_features`: stress test, masks ego + collaborators.
- `random_drop_comm_only`: realistic communication baseline, keeps ego unchanged.

#### Communication accounting correction

Metrics were refined to separate:

- feature bytes/frame,
- metadata bytes/frame,
- total bytes/frame,
- normalized communication ratio.

#### YAML anchor bug

The YAML used:

```yaml
batch_size: &batch_size 4
```

The config patch needed to preserve the anchor.

#### Learning rate issue

Resuming from epoch 43 made the scheduler reduce LR to around `1e-6`. This was safe for the pretrained detector but probably too weak for the new mask. The final Phase 3 run used:

```text
LR = 1e-5
scheduler step_size = [1000]
```

---

# Part III — Experimental Setup

## 9. Datasets and Checkpoint

### CARLA

- test path: `/kaggle/input/data-all/test/test`
- scenarios: 16
- samples: 2170 frames
- checkpoint: `net_epoch43.pth`

### Culver City

- test path: `/kaggle/input/data-all/test/test_culver_city/test_culver_city`
- scenarios: 4
- samples: 550 frames
- checkpoint: `net_epoch43.pth`

Culver City is harder: full baseline AP@0.7 is around 0.772, compared with about 0.891 on CARLA.

---

# Part IV — Results

## 10. CARLA Phase 0 and Phase 1

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame | Meaning |
|---|---:|---:|---:|---:|---:|---|
| Phase 0 baseline | 0.951850 | 0.947751 | 0.891187 | 1.000000 | 9,011,200 | full communication baseline |
| Phase 1 measurement | 0.951854 | 0.947756 | 0.891214 | 1.000000 | 9,011,200 | measurement only |

**Interpretation:** Phase 1 matches Phase 0 almost exactly. Measurement/logging does not disturb the model.

---

## 11. CARLA Phase 2 — Top-k Energy Sweep

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| Top-k 5% | 0.05 | 0.916078 | 0.913815 | 0.858936 | 0.047315 | 716,063 |
| Top-k 10% | 0.10 | 0.928485 | 0.926102 | 0.870266 | 0.095298 | 1,440,565 |
| Top-k 25% | 0.25 | 0.934281 | 0.931609 | 0.874245 | 0.236862 | 3,582,756 |
| Top-k 50% | 0.50 | 0.942616 | 0.939535 | 0.883088 | 0.472661 | 7,153,774 |

**Key result:** Top-k 10% reaches AP@0.7 = **0.8703** with only **9.53% communication**.

---

## 12. CARLA Phase 2 — Random Collaborator-Only Sweep

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| Random 5% | 0.05 | 0.728008 | 0.725193 | 0.657054 | 0.047997 | 722,130 |
| Random 10% | 0.10 | 0.747475 | 0.744328 | 0.671131 | 0.092186 | 1,396,812 |
| Random 25% | 0.25 | 0.824361 | 0.820522 | 0.739566 | 0.230801 | 3,509,978 |
| Random 50% | 0.50 | 0.907733 | 0.903791 | 0.820450 | 0.471122 | 7,140,199 |

**Interpretation:** Random communication is much worse than top-k at the same communication ratio. This proves that what we transmit matters.

---

## 13. CARLA Phase 2 — Stress and Packet-Loss Results

### Random all-features stress test

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Active ratio | Meaning |
|---|---:|---:|---:|---:|---|
| Random all-features | 0.011434 | 0.011165 | 0.005461 | 0.097721 | masks ego + collaborators |

**Interpretation:** Masking ego features destroys detection, so realistic policies should preserve ego features.

### Neighbor + packet loss

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Active ratio | Packet loss |
|---|---:|---:|---:|---:|---:|
| Neighbor + packet loss | ~0.714 | ~0.711 | ~0.620 | ~0.129 | 0.20 |

**Interpretation:** Packet loss and neighbor constraints significantly degrade AP, motivating future repair.

---

## 14. Culver City Results

### Phase 0 and Phase 1

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|
| Culver baseline | 0.878567 | 0.868824 | 0.772075 | 1.000000 | 9,846,784 |
| Culver measurement | 0.878609 | 0.868866 | 0.772197 | 1.000000 | 9,846,784 |

### Culver top-k energy

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| Top-k 5% | 0.05 | 0.7972 | 0.7917 | 0.7077 | 0.0435 | 492,806 |
| Top-k 10% | 0.10 | 0.8315 | 0.8250 | 0.7356 | 0.0876 | 991,677 |
| Top-k 25% | 0.25 | 0.8381 | 0.8312 | 0.7408 | 0.2178 | 2,466,352 |
| Top-k 50% | 0.50 | 0.8558 | 0.8476 | 0.7543 | 0.4346 | 4,923,725 |

### Culver random collaborator-only

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| Random 5% | 0.05 | 0.6353 | 0.6287 | 0.5464 | 0.0444 | 502,332 |
| Random 10% | 0.10 | 0.6501 | 0.6433 | 0.5563 | 0.0847 | 959,630 |
| Random 25% | 0.25 | 0.7020 | 0.6934 | 0.5950 | 0.2109 | 2,397,063 |
| Random 50% | 0.50 | 0.7865 | 0.7768 | 0.6662 | 0.4324 | 4,904,506 |

**Interpretation:** Culver confirms the CARLA pattern. Top-k is much stronger than random at the same budget. Phase 3 was not run fully on Culver.

---

## 15. Phase 3 Learnable-Mask Results on CARLA

Phase 3 was evaluated only on CARLA.

### Version A — Initial long run

| Setting | Value |
|---|---|
| Preset | `phase3_learnable_mask` |
| lambda | 0.01 |
| temperature | 1.0 |
| batch size | 4 |
| fine-tune | ~5 epochs |
| LR | ~1e-6 |
| AP | AP@0.7 ≈ 0.89 |
| caveat | communication metrics not preserved |

**Interpretation:** This proved that Phase 3 could train and infer, but it is not enough for a final communication-efficiency claim.

### Version B — Default mask, batch 8, one epoch

| Metric | Value |
|---|---:|
| AP@0.3 | 0.941590 |
| AP@0.5 | 0.936973 |
| AP@0.7 | 0.865703 |
| comm ratio | 0.412461 |
| bytes/frame | 6,276,242 |

**Interpretation:** Communication dropped to 41%, but AP dropped below top-k 10%.

### Version C — Lambda 0.05, temperature 0.5

| Metric | Value |
|---|---:|
| AP@0.3 | 0.954986 |
| AP@0.5 | 0.950297 |
| AP@0.7 | 0.889841 |
| comm ratio | 0.434398 |
| bytes/frame | 6,659,243 |

**Interpretation:** Very strong accuracy, close to full baseline, but still 43.4% communication.

### Version D — Lambda 0.1, batch 8, LR 1e-6

| Metric | Value |
|---|---:|
| AP@0.3 | 0.948339 |
| AP@0.5 | 0.944342 |
| AP@0.7 | 0.879786 |
| comm ratio | 0.566979 |
| bytes/frame | 8,614,080 |

**Interpretation:** Not an improvement. Communication increased and AP decreased compared with lambda 0.05.

### Version E — Final improved Phase 3: lambda 0.1, batch 4, 2 epochs, LR 1e-5

| Setting | Value |
|---|---|
| Preset | `phase3_lam01_temp05_soft` |
| lambda | 0.1 |
| temperature | 0.5 |
| hard mask | false |
| batch size | 4 |
| fine-tune epochs | 2 |
| LR | 1e-5 |
| scheduler step size | [1000] |
| checkpoint | `net_epoch45.pth` |

| Metric | Value |
|---|---:|
| AP@0.3 | 0.955307 |
| AP@0.5 | 0.951074 |
| AP@0.7 | 0.892544 |
| comm ratio | 0.485436 |
| active neighbors ratio | 0.945161 |
| bytes/frame | 7,409,062 |
| packet loss | 0.0 |

**Interpretation:** This is the best Phase 3 accuracy result. It slightly exceeds the full baseline AP@0.7, but still uses 48.5% communication. Therefore, Phase 3 is accuracy-preserving but not communication-efficient enough.

---

## 16. Consolidated CARLA Comparison

| Method | AP@0.7 | Comm ratio | Main meaning |
|---|---:|---:|---|
| Full baseline | 0.8912 | 1.0000 | reference full communication |
| Phase 1 measurement | 0.8912 | 1.0000 | logging does not change AP |
| Top-k 5% | 0.8589 | 0.0473 | strong ultra-low communication |
| Top-k 10% | 0.8703 | 0.0953 | best low-budget point |
| Top-k 25% | 0.8742 | 0.2369 | modest AP gain, higher cost |
| Top-k 50% | 0.8831 | 0.4727 | close to full baseline |
| Random 10% | 0.6711 | 0.0922 | same budget as top-k 10%, much worse |
| Random 50% | 0.8205 | 0.4711 | same budget as top-k 50%, worse |
| Phase 3 lambda 0.05 | 0.8898 | 0.4344 | high AP, medium-high communication |
| Phase 3 lambda 0.1, b8, LR 1e-6 | 0.8798 | 0.5670 | not improved |
| Phase 3 lambda 0.1, b4, LR 1e-5 | 0.8925 | 0.4854 | best AP, but high communication |

---

# Part V — Interpretation and Meeting Story

## 17. Main Scientific Findings

### Finding 1 — Measurement is safe

Phase 0 and Phase 1 match almost exactly. This validates the logging/measurement layer.

### Finding 2 — Importance matters

Top-k energy strongly outperforms random masking at the same communication ratio on both CARLA and Culver City.

This is the strongest evidence in the project.

### Finding 3 — Top-k 10% is the best low-budget operating point

On CARLA:

\[
\text{AP@0.7} = 0.8703, \quad \text{comm ratio} = 0.0953
\]

On Culver:

\[
\text{AP@0.7} = 0.7356, \quad \text{comm ratio} = 0.0876
\]

### Finding 4 — Learnable mask preserves AP but is not sparse enough

Phase 3 reached excellent AP, but communication stayed around 43–49% in the best runs.

### Finding 5 — Phase 4 should be future work under a stricter bottleneck

Repair makes sense only when many features are truly missing or corrupted. It should be evaluated over fixed top-k 10% or packet-loss settings, not over a high-communication learned mask.

---

## 18. What Not to Overclaim

Do not claim:

1. Phase 3 beats Phase 2 in communication efficiency.
2. The learned mask is already sparse enough.
3. The first long Phase 3 run is final evidence, because communication metrics were not preserved.
4. Hard-mask inference is final unless preset override is fully verified.
5. Phase 4 has final results.

The honest claim is stronger:

> Phase 2 gives a strong communication-efficient result. Phase 3 is promising but needs better budget-aware training.

---

## 19. Recommended Next Technical Step

Do **not** keep training Phase 3 longer with the same loss. The next step should be budget-aware sparsity:

```yaml
learnable_mask:
  target_ratio: 0.10
  budget_lambda: 1.0
  sparsity_lambda: 0.05
```

Objective:

\[
L_{total} = L_{det} + \lambda_{sparse}\text{mean}(R) + \lambda_{budget}\max(0, \text{mean}(R)-r_{target})^2
\]

Other useful improvements:

- use higher LR only for the mask head,
- optionally freeze detector/backbone and train mask only,
- evaluate hard mask only after soft mask reaches the target budget,
- add compression/quantization after top-k selection.

---

## 20. Phase 4 Recommendation

Do not run Phase 4 on top of the current Phase 3 mask because Phase 3 still sends too much.

A cleaner Phase 4 experiment:

```text
Phase 4 = repair network over fixed top-k 10% communication
```

Goal:

```text
same communication ratio ≈ 0.095
higher AP@0.7 than top-k 10%
```

This directly tests whether repair can recover accuracy under a strict communication bottleneck.

---

# Part VI — Meeting Delivery

## 21. Suggested Meeting Flow

### Part A — Research Foundation

1. Cooperative perception improves detection but creates communication cost.
2. Intermediate fusion is accurate but still bandwidth-heavy.
3. Literature categories: dense fusion, selective sharing, compression, robustness, scheduling.
4. Research gap: current V2VAM-style pipeline needs explicit communication policy and AP–communication measurement.
5. Proposed idea: insert a communication policy before V2VAM fusion.

### Part B — Implementation and Results

1. Config-driven communication module implemented.
2. Phase 0/1 validated baseline and measurement.
3. Phase 2 showed top-k energy is a strong communication-efficient baseline.
4. Phase 3 learned masks achieved excellent AP but were not sparse enough.
5. Next: budget-aware loss or Phase 4 repair over top-k 10%.

---

## 22. Verbal Script for the Meeting

> I started from the observation that cooperative perception improves detection, but dense intermediate feature sharing is too communication-heavy for realistic V2X. So I framed the problem as an AP–communication trade-off rather than only an AP maximization problem.
>
> From the literature, there are several directions: compress the message, select important regions, choose which neighbors should communicate, or repair missing/delayed messages. Since the current repository already has a PointPillars + V2VAM intermediate-fusion interface, the most natural first step was to add a communication policy before fusion.
>
> I implemented this as a config-driven layer. Phase 0 reproduced the baseline. Phase 1 added measurement and showed no AP change. Phase 2 tested non-learned communication baselines. The strongest result is top-k energy: on CARLA, top-k 10% keeps AP@0.7 around 0.870 with only 9.5% communication. On Culver, top-k 10% also works, reaching AP@0.7 around 0.736 with 8.8% communication. Random selection at the same budget is much worse, so the result shows that feature importance matters.
>
> Then I trained Phase 3 learnable masks. The learned mask can preserve very high AP — the best run reached AP@0.7 = 0.8925 — but it still used around 48.5% communication. So I would not claim Phase 3 is better than top-k yet. The main lesson is that the current sparsity loss is not targeted enough; the next step should be a budget-aware loss that forces the mask toward a target ratio like 10% or 15%.
>
> For Phase 4, I think repair is meaningful, but only after a strict bottleneck. So instead of repairing after the current high-communication Phase 3 mask, I would test repair over fixed top-k 10% or packet-loss settings.

---

## 23. Questions to Ask the Professor

1. Should the main thesis contribution focus on **top-k / importance-based communication** as the successful result, with learnable masking as an extension?
2. Is it better to next improve Phase 3 with **budget-aware loss**, or to move toward **repair over top-k 10%**?
3. Should the final thesis emphasize **AP–communication frontier curves** rather than individual best results?
4. Should we include **Culver City** as a generalization dataset in the main thesis results?
5. For Phase 3, would he prefer **mask-only training with frozen detector** or full fine-tuning?

---

## 24. Final Message to Communicate

The project is already in a strong state because it has:

- a working config-driven communication-aware V2VAM pipeline,
- validated baseline and measurement parity,
- strong Phase 2 results on CARLA and Culver,
- multiple Phase 3 trained variants with clear conclusions,
- a well-defined next step.

The most defensible final conclusion is:

> Importance-based top-k BEV feature selection is currently the strongest communication-efficient method. It preserves much of the full cooperative perception benefit using only about 9–10% communication. Learnable masks can recover full AP, but need budget-aware training to become truly sparse.

---

## 25. Appendix — Short Tables for Quick Discussion

### Best result per category

| Category | Best run | AP@0.7 | Comm ratio |
|---|---|---:|---:|
| Full CARLA baseline | Phase 0 | 0.8912 | 1.0000 |
| Best low-budget CARLA | Top-k 10% | 0.8703 | 0.0953 |
| Best Phase 3 AP | Lambda 0.1, b4, LR 1e-5 | 0.8925 | 0.4854 |
| Full Culver baseline | Phase 0 | 0.7721 | 1.0000 |
| Best low-budget Culver | Top-k 10% | 0.7356 | 0.0876 |

### Main research conclusion

| Question | Answer |
|---|---|
| Does communication measurement affect AP? | No, Phase 1 matches baseline. |
| Is random feature dropping enough? | No, it is much worse than top-k. |
| Does top-k energy work? | Yes, strongly, on both CARLA and Culver. |
| Does learnable masking work? | Technically yes; AP is excellent. |
| Is learnable masking sparse enough? | Not yet. |
| Should Phase 4 be run now? | Only under strict top-k/packet-loss settings. |
