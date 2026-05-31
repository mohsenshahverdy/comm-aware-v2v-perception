# Research Foundations Report: Communication-Aware V2V Cooperative Perception

**Purpose:** This report prepares the research-foundation part of the meeting with the professor. It explains the problem, the literature landscape, the main method families, the model definitions, and how these ideas motivate the implementation phases in the repository.

**Project framing:** Improving a PointPillars + V2VAM-style intermediate-fusion cooperative perception pipeline under limited, lossy, and non-ideal V2X communication.

---

## 1. Executive Summary

The central research problem is that cooperative perception improves 3D detection by allowing multiple connected/autonomous vehicles to share observations, but most strong cooperative-perception models assume communication is either ideal or cheap. In practice, V2X links have bandwidth limits, latency, packet loss, interruptions, and changing neighbor availability. Therefore, a model that simply exchanges dense BEV feature maps from all collaborators can be accurate in simulation but impractical or fragile under real communication constraints.

The most important research conclusion from the literature is:

> Communication should be treated as part of the perception model, not as an external nuisance.

This leads to a practical design principle:

> Instead of asking only “how do we fuse all collaborator features?”, the model should also ask “who should communicate, what should be communicated, how much should be communicated, and how should corrupted/missing messages be handled?”

The literature can be organized into five major method families:

1. **Dense intermediate fusion**: share rich BEV/intermediate features and learn strong fusion. Examples: V2VNet, DiscoNet, V2X-ViT, V2VAM.
2. **Spatial/importance-aware selective sharing**: transmit only important spatial regions or feature cells. Examples: Where2comm, How2comm, COOPERTRIM, Reason-to-Transmit.
3. **Compression/codebook/quantization methods**: reduce payload size by transmitting compact codes instead of full FP32 feature tensors. Examples: CodeFilling, QuantV2X, V2X-DSC, WaveComm.
4. **Loss/delay/interruption robustness**: handle non-ideal links with repair, synchronization, temporal cache, freshness, or interruption-aware prediction. Examples: V2VAM + LCRN, Latency-Aware Collaborative Perception, V2X-INCOP, Fresh2comm, CoDynTrust, QPoint2Comm.
5. **Scheduling/resource allocation/message-unit redesign**: decide which agents transmit and what communication resources/message units should be used. Examples: action-branching/federated RL scheduling, SchedCP, Point Cluster, Which2comm.

For the current repository, the most natural first direction is **importance-guided sparse BEV feature sharing before V2VAM fusion**, because it preserves the existing PointPillars/V2VAM architecture while adding a communication policy at the message interface. This is why the experimental plan was organized into Phase 0–4:

- **Phase 0:** baseline stabilization and parity check.
- **Phase 1:** measurement-only communication accounting.
- **Phase 2:** non-learned communication baselines: random drop, top-k energy, neighbor selection, packet loss.
- **Phase 3:** learnable communication mask with sparsity loss.
- **Phase 4:** repair network / reconstruction under missing or corrupted messages.

---

## 2. The Core Research Problem

### 2.1 What is cooperative perception?

In autonomous driving, each vehicle has limited field of view because of occlusion, sensor range, road geometry, and object density. Cooperative perception allows vehicles or infrastructure agents to exchange perception information through V2X communication so that the ego vehicle can perceive beyond its own sensors.

The ego vehicle receives messages from other connected autonomous vehicles (CAVs) or infrastructure agents and fuses them into its own perception pipeline. This can improve object detection, especially for occluded or distant objects.

However, the benefit is limited by communication constraints:

- **Bandwidth:** dense BEV feature maps are large.
- **Latency:** messages may arrive too late to match the current frame.
- **Packet loss/interruption:** some feature packets may be missing or corrupted.
- **Agent selection:** not every neighbor is equally useful.
- **Scalability:** communication cost grows with number of collaborators.
- **Real-time constraints:** autonomous driving needs low latency and stable throughput.

The central trade-off is:

\[
\text{Perception accuracy} \quad \leftrightarrow \quad \text{Communication cost}
\]

A good system should preserve most of the cooperative detection benefit while using only a small fraction of the communication.

---

## 3. Why Intermediate Fusion Is Powerful but Expensive

Cooperative perception can be implemented at different fusion levels.

### 3.1 Early fusion

Early fusion shares raw sensor data, such as point clouds, before feature extraction.

**Advantage:** It preserves geometric detail.

**Problem:** Raw LiDAR data is very large, and aligning raw observations from different agents can be expensive.

### 3.2 Late fusion

Late fusion shares final detections, such as bounding boxes and class scores.

**Advantage:** Very low communication cost.

**Problem:** It loses rich scene context. If an object is missed by a collaborator’s detector, late fusion cannot recover it. It also has weaker ability to reason about occlusion and ambiguous regions.

### 3.3 Intermediate fusion

Intermediate fusion shares learned features, usually BEV feature maps, after local encoding but before final detection.

**Advantage:** Strong accuracy–information balance. It preserves more context than late fusion while being more compact than raw point clouds.

**Problem:** Dense feature maps are still large. If every collaborator sends full BEV tensors to the ego vehicle, the payload can exceed practical V2X bandwidth budgets.

### 3.4 Why this matters for PointPillars + V2VAM

Your current codebase is based on a PointPillars-style encoder and V2VAM-style intermediate fusion. This makes the project tractable because the communication boundary is clear:

1. Each agent produces a BEV/intermediate feature tensor.
2. Collaborator features are spatially aligned to the ego frame.
3. V2VAM fuses ego and collaborator features.
4. A detection head predicts 3D bounding boxes.

Therefore, the clean insertion point for a communication policy is:

> after BEV feature extraction and before V2VAM fusion.

This allows communication masking, top-k selection, packet loss simulation, or repair to be added without rewriting the full detector.

---

## 4. Baseline Architecture and Definitions

### 4.1 Ego vehicle and collaborators

Let the ego vehicle be indexed by \(e\), and collaborators by \(i \in \mathcal{N}_e\). Each agent has a local LiDAR observation and produces an intermediate feature map:

\[
F_i \in \mathbb{R}^{C \times H \times W}
\]

where:

- \(C\): feature channels,
- \(H, W\): BEV spatial dimensions.

The ego feature is \(F_e\). Collaborator features \(F_i\) are transformed into the ego coordinate frame using pairwise transformation matrices.

### 4.2 Dense intermediate fusion

In dense intermediate fusion, every collaborator sends the full feature map:

\[
M_i = F_i
\]

Then the fusion module computes:

\[
\hat{F}_e = \text{Fuse}(F_e, \{T_{i \rightarrow e}(M_i)\}_{i \in \mathcal{N}_e})
\]

where \(T_{i \rightarrow e}\) aligns collaborator features into the ego frame.

The detection head then predicts:

\[
\hat{Y}_e = \text{DetHead}(\hat{F}_e)
\]

### 4.3 Communication-aware fusion

In a communication-aware system, the message is no longer the full feature map. Instead:

\[
M_i = \mathcal{P}_{\theta}(F_i, B, q_i, s_i)
\]

where:

- \(\mathcal{P}_{\theta}\): communication policy,
- \(B\): bandwidth budget,
- \(q_i\): link/channel quality or packet loss state,
- \(s_i\): scene/neighbor utility score.

The policy can select spatial cells, channels, neighbors, or compressed tokens.

### 4.4 Communication metrics

The project needs metrics beyond AP. The most important are:

- **AP@0.3, AP@0.5, AP@0.7:** detection performance.
- **active ratio / communication ratio:** fraction of transmitted feature cells.
- **bytes per frame:** estimated communication payload per frame.
- **active neighbors ratio:** fraction of collaborators kept.
- **packet loss rate:** simulated channel loss.
- **AP-per-bit / AP-vs-communication curve:** rate–accuracy trade-off.

The key scientific question is not only whether AP is high, but:

> How much AP can be preserved per unit communication?

---

## 5. Literature Category 1: Dense Intermediate Fusion

Dense intermediate fusion methods share rich intermediate features and mainly focus on how to fuse them effectively.

### 5.1 V2VNet

V2VNet introduced a multi-agent feature-sharing framework where agents exchange compressed intermediate representations and perform iterative message passing over BEV features. It uses learned compression and graph/message-passing ideas to improve collaboration between agents.

**Main contribution:** demonstrate that intermediate feature sharing is a strong compromise between raw-data sharing and late detection sharing.

**Limitation for this project:** still not communication-budget-aware enough for strict bandwidth constraints.

### 5.2 DiscoNet

DiscoNet improves communication efficiency through teacher–student learning and graph-based collaboration. It uses compressed feature maps and distillation to reduce communication while preserving performance.

**Main contribution:** better accuracy–communication tradeoff than dense methods, but still within the family of intermediate feature exchange.

### 5.3 V2X-ViT

V2X-ViT uses transformer-style fusion for V2X cooperative perception and shows that attention mechanisms can improve multi-agent feature fusion.

**Main contribution:** stronger fusion under multi-agent, multi-view conditions.

**Limitation:** transformers improve fusion but do not automatically solve communication cost.

### 5.4 V2VAM

V2VAM is directly relevant to the current codebase. It addresses lossy V2V feature communication and introduces:

1. **LC-aware Repair Network (LCRN):** repairs corrupted features.
2. **V2V Attention Module (V2VAM):** uses intra-vehicle attention and uncertainty-aware inter-vehicle attention.

The V2VAM paper is important because it moves from ideal communication toward lossy communication. However, its lossy simulation is still more like feature corruption than a full packetized bandwidth-limited communication stack.

**Relevance to this project:** V2VAM is a strong base model for robust fusion, but we can extend it by adding explicit communication selection, measurement, and bandwidth-aware policies before fusion.

---

## 6. Literature Category 2: Importance-Aware Selective Sharing

This is the most relevant family for the current implementation.

The key idea is:

> Do not transmit every feature. Transmit only the spatial regions, channels, agents, or tokens that are likely to improve ego perception.

### 6.1 Where2comm

Where2comm introduces a spatial confidence map that identifies which BEV locations are useful to communicate. The model shares spatially sparse but perceptually critical features.

**Core idea:** “where to communicate” can be learned from perception confidence.

**Why it matters:** it directly motivates top-k energy selection and learnable spatial masks in this project.

### 6.2 How2comm

How2comm generalizes communication selection from only spatial regions to spatial-channel selection. Instead of asking only which BEV cells matter, it asks which spatial-channel feature components should be transmitted.

**Core idea:** communication utility is not uniform over space or feature channels.

**Relevance:** future improvement of Phase 3 could select not only cells but also feature channels.

### 6.3 COOPERTRIM

COOPERTRIM uses temporal uncertainty to reduce repetitive transmission of static or low-value features. It adapts the sharing quantity according to scene dynamics.

**Core idea:** if a region is static and already known, retransmitting it every frame is wasteful. **not real scenario**

**Relevance:** a future extension could include temporal cache and transmit only changes or uncertain regions.

### 6.4 Reason-to-Transmit

Reason-to-Transmit frames communication as a deliberative decision: transmit when the expected information gain justifies the communication cost.

**Core idea:** communication is an action, not a fixed pipeline step.

**Relevance:** this supports the thesis framing that communication should be optimized jointly with perception.

---

## 7. Literature Category 3: Compression, Codebooks, and Quantization

Selection answers **what to send**. Compression answers **how compactly to send it**.

### 7.1 CodeFilling

CodeFilling replaces high-dimensional feature maps with codebook indices. Instead of transmitting FP32 BEV tensors, agents transmit compact integer codes. It also uses information-demand-driven selection so collaborators fill what the ego lacks.

**Core idea:** represent messages as compact symbols rather than dense feature arrays.

**Relevance:** after top-k or mask selection, the next step could be compressing the selected features using vector quantization/codebooks.

### 7.2 QuantV2X

QuantV2X quantizes both the neural network and transmitted messages, targeting real-time deployment. It treats communication and computation efficiency jointly.

**Core idea:** full-precision models/messages are not deployment-friendly; low-bit representations can preserve accuracy with lower latency and bandwidth.

**Relevance:** future work can quantize selected BEV features after Phase 2/3 selection.

### 7.3 V2X-DSC

V2X-DSC uses distributed source coding principles. Since different agents observe the same world, their features are correlated. The ego already has side information, so collaborators should transmit only the innovation or residual that the ego cannot infer locally.

**Core idea:** do not transmit redundant information already predictable from ego features.

**Relevance:** this is a strong future research direction for conditional/residual feature transmission.

### 7.4 WaveComm

WaveComm uses frequency-domain representation, sending low-frequency components and reconstructing missing details.

**Core idea:** lower-frequency information may preserve most of the scene structure with less communication.

**Relevance:** alternative to top-k masking; could be combined with repair networks.

---

## 8. Literature Category 4: Loss, Delay, and Interruption Robustness

Real V2X communication is not only bandwidth-limited. It is also unreliable.

### 8.1 V2VAM + LCRN

V2VAM studies lossy communication by simulating corrupted transmitted feature maps. Its repair network reconstructs missing/corrupted features, and attention fusion decides how much to trust each source.

**Relevance:** this motivates Phase 4 repair.

### 8.2 Latency-Aware Collaborative Perception / SyncNet

Latency-aware collaborative perception explicitly handles delayed messages. SyncNet synchronizes delayed feature-level messages to the current timestamp.

**Core idea:** a stale feature is not necessarily useless, but it must be temporally aligned or downweighted.

**Relevance:** the current dataset/model already has delay-related metadata, so a future extension could add freshness-weighted fusion.

### 8.3 V2X-INCOP

V2X-INCOP handles communication interruption by predicting missing collaboration information from historical spatiotemporal context.

**Core idea:** when a message is missing, recover it from past information and motion context.

**Relevance:** similar to Phase 4 repair, but more temporal and interruption-aware.

### 8.4 Fresh2comm and CoDynTrust

Freshness-aware and trust-aware methods model the age and reliability of collaborator information. They help decide whether old or uncertain messages should influence fusion.

**Relevance:** future work can combine communication ratio, packet loss, delay, and trust into a single fusion confidence.

### 8.5 QPoint2Comm

QPoint2Comm combines quantized point-cloud index transmission with masked training for random packet loss tolerance.

**Core idea:** robustness and compact communication should be designed together.

**Relevance:** supports combining Phase 2/3 communication selection with Phase 4 repair or masked training.

---

## 9. Literature Category 5: Scheduling and Resource Allocation

Some papers focus less on the feature-fusion network and more on communication control.

### 9.1 Action-branching and federated RL paper

The shared communication-control paper studies which neighbor should send, which radio resources should be used, and what content/resolution should be transmitted. It uses quadtree-compressed sensory messages, action-branching reinforcement learning, and federated reinforcement learning.

This is not a direct replacement for V2VAM. It is better interpreted as a policy layer.

**Main lesson for the project:** communication decisions can be learned and optimized jointly with perception utility.

### 9.2 SchedCP-style scheduling

Scheduling methods choose which users/agents should transmit based on channel state, semantic utility, or perception value.

**Relevance:** future extension of neighbor selection in Phase 2.

---

## 10. Alternative Message Units

Not all methods transmit BEV feature maps.

### 10.1 Object-level messages

Object-level communication sends detections, boxes, embeddings, or object-level semantic representations.

**Advantage:** very small payload.

**Problem:** loses dense scene context and may fail when objects are missed locally.

### 10.2 Point clusters / sparse raw messages

Some methods transmit sparse point clusters or foreground points.

**Advantage:** preserves geometry better than boxes.

**Problem:** more invasive to integrate into a PointPillars + V2VAM intermediate-fusion repo.

### 10.3 Why we did not start there

The current repository already has a clean intermediate feature interface, so sparse BEV feature sharing is the lowest-risk path. Object-level or point-cluster communication may be promising, but would require larger architectural changes.

---

## 11. Practical Taxonomy

| Family | What is communicated? | Main goal | Best examples | Fit to current repo |
|---|---|---|---|---|
| Dense intermediate fusion | Full BEV/intermediate features | Maximize AP | V2VNet, DiscoNet, V2X-ViT, V2VAM | Already close to current baseline |
| Spatial selective sharing | Important BEV cells/regions | Reduce bytes while preserving AP | Where2comm, COOPERTRIM | Very high |
| Spatial-channel selection | Important cells and channels | More precise feature pruning | How2comm | High, future extension |
| Codebook/quantization | Compact feature codes | Lower payload per selected feature | CodeFilling, QuantV2X | High, after selection |
| Conditional/residual coding | Innovation beyond ego feature | Avoid redundant transmission | V2X-DSC | High but more complex |
| Loss/delay repair | Corrupted/missing/stale features | Robustness under non-ideal links | V2VAM-LCRN, V2X-INCOP, SyncNet | High |
| Scheduling/resource allocation | Which agent/resource/content | System-level communication control | FRL/action branching, SchedCP | Medium, future policy layer |
| Object/point sparse messages | Boxes, clusters, sparse points | Extreme efficiency | Point Cluster, Which2comm, CoLC | More invasive |

---

## 12. Model Operations Used in Our Implementation Plan

The implementation plan follows the pipeline:

\[
\text{select} \rightarrow \text{compress/account} \rightarrow \text{simulate channel} \rightarrow \text{repair} \rightarrow \text{fuse}
\]

### 12.1 Baseline dense transmission

\[
M_i = F_i
\]

All collaborator features are sent.

### 12.2 Random drop

A random spatial mask \(R_i\) is sampled:

\[
R_i(h,w) \sim \text{Bernoulli}(p)
\]

and applied to the feature map:

\[
M_i = R_i \odot F_i
\]

This is not a smart policy. It is a stress test and lower-bound baseline.

### 12.3 Top-k energy selection

For each spatial cell, compute an energy score:

\[
s_i(h,w) = \|F_i[:,h,w]\|_2
\]

Then keep the top \(K\) cells:

\[
R_i(h,w) = \mathbf{1}\{s_i(h,w) \in \text{TopK}\}
\]

\[
M_i = R_i \odot F_i
\]

This is simple but powerful because high-energy BEV cells often correspond to informative scene regions.

### 12.4 Neighbor selection

Instead of sending from all collaborators:

\[
\mathcal{N}_e = \{1,2,3,4\}
\]

select a subset:

\[
\mathcal{S}_e \subset \mathcal{N}_e
\]

based on distance, importance, or top-k score.

### 12.5 Packet-loss simulation

Packet loss applies stochastic corruption after selection:

\[
Z_i(h,w) \sim \text{Bernoulli}(1 - p_{loss})
\]

\[
M_i = Z_i \odot R_i \odot F_i
\]

This simulates missing transmitted units.

### 12.6 Learnable mask

A small neural policy predicts a mask:

\[
R_i = \sigma(g_\theta(F_i) / \tau)
\]

where:

- \(g_\theta\): mask head,
- \(\tau\): temperature,
- \(\sigma\): sigmoid or relaxed gate.

The message becomes:

\[
M_i = R_i \odot F_i
\]

The training objective adds communication sparsity:

\[
L_{total} = L_{det} + \lambda_{comm} \cdot \text{mean}(R)
\]

In practice, this is already similar to an L1 penalty on the mask.

### 12.7 Budget-aware future loss

A stronger future loss would explicitly target a communication budget:

\[
L_{budget} = \lambda_{budget} \cdot \max(0, \text{mean}(R) - r_{target})^2
\]

Then:

\[
L_{total} = L_{det} + \lambda_{sparse}\text{mean}(R) + L_{budget}
\]

This is better aligned with the research goal because it asks the model to stay near a target budget, such as 10% or 15% communication.

### 12.8 Repair network

If features are dropped or corrupted, a repair network predicts missing content:

\[
\tilde{F}_i = \text{RepairNet}(M_i, R_i)
\]

with an optional reconstruction loss:

\[
L_{repair} = \|\tilde{F}_i - F_i\|_2^2
\]

Then:

\[
L_{total} = L_{det} + L_{comm} + L_{repair}
\]

This motivates Phase 4, but Phase 4 is most meaningful when communication is already aggressively reduced.

---

## 13. Why the Phased Experimental Design Makes Sense

The phase design avoids mixing too many variables at once.

### Phase 0: Baseline reproduction

Goal: verify that the checkpoint, dataset, model, and inference pipeline reproduce expected AP.

Why needed: without baseline parity, communication experiments cannot be trusted.

### Phase 1: Measurement-only mode

Goal: add communication accounting without changing the model output.

Why needed: proves that metrics/logging do not disturb AP.

### Phase 2: Non-learned communication baselines

Goal: evaluate simple communication policies:

- random drop,
- top-k energy,
- neighbor selection,
- packet loss.

Why important: simple baselines are interpretable and establish whether the feature space contains easy-to-exploit importance structure.

### Phase 3: Learnable mask

Goal: train a mask to learn communication importance.

Why important: tests whether a learned policy can outperform fixed heuristics.

Observed finding from our experiments: learnable masking preserved high AP, but did not become sparse enough with the current loss.

### Phase 4: Repair network

Goal: recover missing/corrupted features under strict communication bottlenecks or packet loss.

Why it should be future work: repair is most meaningful after we force a lower communication regime, for example top-k 10% or packet-loss 20–30%.

---

## 14. Research Hypotheses for the Project

### Hypothesis 1: Dense feature exchange is overkill

Many BEV cells are background or redundant. A smaller subset of feature cells can preserve most detection AP.

### Hypothesis 2: Importance-aware selection beats random selection

If the feature energy or learned importance correlates with object/scene relevance, top-k or learned masks should outperform random drop at the same communication ratio.

### Hypothesis 3: Simple top-k energy is a strong baseline

Before training complex communication policies, top-k feature energy should be tested because it is cheap, stable, and interpretable.

### Hypothesis 4: Learnable masks need explicit budget control

A learned mask trained only with detection loss plus weak sparsity may preserve AP by keeping too much communication. A target-budget loss may be necessary.

### Hypothesis 5: Repair is useful only under a real bottleneck

A repair network is meaningful when the message is sparse or corrupted. If too much communication is still sent, repair may not prove a strong benefit.

---

## 15. How to Explain the Research Gap to the Professor

A concise explanation:

> Existing cooperative perception methods show that intermediate fusion improves detection accuracy, but many assume ideal or high-bandwidth communication. Recent work shows that communication should be selective, compressed, and robust to delay/loss. My project starts from a V2VAM-style intermediate-fusion pipeline and adds a configurable communication policy before fusion. This lets us measure and control the AP–communication trade-off. The goal is not only to maximize AP, but to understand how much communication is actually necessary.

Then explain the exact gap:

1. V2VAM handles lossy communication, but not full bandwidth-aware communication policy.
2. Where2comm/How2comm show selective sharing is powerful, but the current repo does not have this policy layer.
3. Codebook/quantization methods show further compression is possible, but first we need to identify what to send.
4. Our implementation creates an experimental framework to test these ideas phase by phase.

---

## 16. Research-to-Implementation Mapping

| Research concept | Implementation decision |
|---|---|
| Communication should be part of model | Added communication policy module before fusion |
| Need reproducible switches | Added YAML communication config and phase presets |
| Need AP vs communication trade-off | Added active ratio, bytes/frame, AP logging |
| Need simple baselines | Added random drop, top-k energy, neighbor/packet loss |
| Need learned policy | Added learnable mask and sparsity loss |
| Need robustness to missing features | Prepared repair network path for Phase 4 |
| Need comparability | Kept PointPillars + V2VAM backbone unchanged |

---

## 17. What This Means for the Presentation

A clear meeting structure can be:

### Part A — Research Foundations

1. Cooperative perception improves perception but creates communication bottleneck.
2. Intermediate fusion is strong but still too bandwidth-heavy.
3. Literature moved from dense fusion to selective communication, compression, and robustness.
4. The best-fit direction for our repo is sparse BEV sharing before V2VAM fusion.
5. Communication should be measured and optimized explicitly.

### Part B — Implementation and Results

1. We implemented a config-driven communication policy layer.
2. We validated baseline and measurement mode.
3. Phase 2 showed top-k energy is very strong.
4. Phase 3 learnable mask reached excellent AP but still used too much communication.
5. Next research step: budget-aware loss or repair over strict top-k communication.

---

## 18. Recommended Slide/Report Outline

1. **Motivation:** why cooperative perception and V2X matter.
2. **Problem:** dense feature sharing is too expensive and assumes ideal links.
3. **Base architecture:** PointPillars + intermediate BEV + V2VAM fusion.
4. **Literature map:** dense fusion, selective sharing, compression, delay/loss robustness, scheduling.
5. **Research gap:** V2VAM robust fusion needs explicit communication policy and measurement.
6. **Proposed framework:** communication policy before fusion.
7. **Phase plan:** Phase 0–4.
8. **Key equations:** mask, top-k, communication loss, budget loss.
9. **Expected contribution:** AP–communication frontier and robust communication-aware V2VAM.
10. **Transition to results:** Phase 2/3 findings.

---

## 19. Key Takeaways

1. **The fundamental issue is not only perception accuracy, but perception under constrained communication.**
2. **Intermediate fusion is powerful but communication-heavy.**
3. **Selective BEV sharing is the most natural first research direction for the current repo.**
4. **Top-k energy is a surprisingly strong and interpretable baseline.**
5. **Learnable masks are promising, but need stronger budget-aware training to become truly sparse.**
6. **Phase 4 repair should be used under strict top-k/packet-loss settings, not after a high-communication learned mask.**
7. **The clean research story is: measure → select → learn → repair/compress.**

---

## 20. Short Verbal Summary for the Meeting

> I started by studying communication-aware cooperative perception. The main issue is that collaborative perception improves detection by sharing information between vehicles, but dense intermediate features are too expensive and real V2X links are lossy and delayed. The literature has moved from dense fusion methods like V2VNet, DiscoNet, V2X-ViT, and V2VAM toward selective communication methods like Where2comm and How2comm, compact message methods like CodeFilling and QuantV2X, and robustness methods like V2VAM-LCRN, SyncNet, and V2X-INCOP. Based on that, I treated communication as a module in the model. I inserted a configurable communication policy before V2VAM fusion, so we can test random drop, top-k energy, neighbor selection, packet loss, learnable masks, and later repair. This gives us a controlled AP-versus-communication framework. The research conclusion is that top-k/importance-based feature sharing is the strongest immediate direction, while learnable masks require stronger budget-aware loss to become truly sparse.

---

## 21. Source Notes

This report is based on the earlier deep-research analysis and implementation-plan notes from the project conversation, plus the recent literature reviewed there. Key papers/methods referenced include:

- V2VNet: intermediate feature sharing and message passing.
- DiscoNet: teacher–student/graph-based communication-efficient cooperative perception.
- V2X-ViT: transformer-based V2X fusion.
- V2VAM / LCRN: lossy communication repair and uncertainty-aware attention.
- Where2comm: spatial confidence map for sparse communication.
- How2comm: spatial-channel communication selection.
- COOPERTRIM: adaptive temporal/uncertainty-aware selection.
- CodeFilling: codebook-based compact messages.
- QuantV2X: quantized model and message representation.
- V2X-DSC: distributed-source-coding-inspired conditional communication.
- Latency-Aware Collaborative Perception / SyncNet: delayed feature synchronization.
- V2X-INCOP: interruption-aware recovery using historical context.
- Fresh2comm, CoDynTrust, QPoint2Comm: freshness, trust, and packet-loss robust communication.

