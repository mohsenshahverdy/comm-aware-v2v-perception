# Receiver-Driven Selective Communication for V2V Cooperative Perception

## 1. Purpose of the Extension

The current project studies communication-aware V2V cooperative perception in a PointPillars + V2VAM-style intermediate-fusion pipeline. The existing implementation already supports communication policies such as full feature sharing, top-k energy selection, random dropping, packet-loss simulation, and learnable masks. The strongest current result is an importance-based top-k feature selection policy, which significantly reduces transmitted BEV feature cells while preserving much of the full-communication detection performance.

The proposed extension changes the communication decision logic:

> Instead of a sender deciding what to transmit, the receiver decides what information it needs from collaborators.

For two vehicles:

- **V1**: ego vehicle / receiver / decision-maker
- **V2**: collaborator / data provider / sender

The core requirement is:

> V2 should not decide what V1 needs. V2 exposes available information or compact context, and V1 decides which parts of V2’s information are relevant for V1’s own perception task.

This leads to a receiver-driven communication framework:

\[
\text{V1 need} \times \text{V2 availability} \rightarrow \text{V1 request mask} \rightarrow \text{V2 sparse message}
\]

The extension is theoretically sound, aligns well with the current project, and can be implemented incrementally as a new communication policy before the existing V2VAM fusion module.

---

## 2. Motivation and Problem Formulation

### 2.1 Why the Current Sender-Only Logic Is Limited

The current top-k energy baseline selects feature cells based on the collaborator’s feature strength:

\[
s_2(h,w) = \|F_2[:,h,w]\|_2
\]

Then V2 transmits the top-k cells with highest energy.

This is simple and effective, but it answers only:

> Which regions are important in V2’s own feature map?

It does not directly answer:

> Which regions of V2 are useful for V1?

A high-energy region in V2 may be irrelevant to V1 if:

- V1 already sees that region clearly,
- the region is outside V1’s useful fusion area,
- the region does not reduce V1’s uncertainty,
- the information is redundant with V1’s own observation.

Therefore, a receiver-driven policy is more aligned with the cooperative perception goal.

### 2.2 Receiver-Driven Communication Objective

The desired policy should answer:

\[
\text{What information from V2 reduces V1's perception uncertainty most under a communication budget?}
\]

Formally, for a communication budget \(B\), V1 should choose a request mask \(M_{1 \leftarrow 2}\) over V2’s feature map such that:

\[
M_{1 \leftarrow 2}^{*}
=
\arg\max_{M: C(M) \leq B}
\Delta \mathcal{U}_{1}(M \odot F_2)
\]

where:

- \(F_2 \in \mathbb{R}^{C \times H \times W}\) is V2’s BEV feature map,
- \(M_{1 \leftarrow 2} \in \{0,1\}^{H \times W}\) is the request mask computed by V1,
- \(C(M)\) is the communication cost,
- \(\Delta \mathcal{U}_{1}\) is the expected utility gain for V1, such as AP improvement, uncertainty reduction, or detection-confidence improvement.

Since the true AP gain is not directly known before transmission, the implementation needs practical proxies for V1’s need and V2’s availability.

---

## 3. V1–V2 Communication Logic

### 3.1 High-Level Protocol

For one ego vehicle V1 and one collaborator V2:

1. V1 computes its own BEV feature map \(F_1\).
2. V1 estimates a **need map** \(N_1\), indicating where its own perception is weak or uncertain.
3. V2 exposes a compact **availability/context map** \(A_2\), not its full feature tensor.
4. V1 aligns V2’s context map into V1’s coordinate frame.
5. V1 computes a request score map:

\[
S_{1 \leftarrow 2} = N_1 \odot A_{2 \rightarrow 1}
\]

6. V1 selects the top-k cells under the communication budget:

\[
M_{1 \leftarrow 2} = \text{TopK}(S_{1 \leftarrow 2}, k)
\]

7. V2 sends only the requested feature cells:

\[
\text{Msg}_{2 \rightarrow 1} = M_{1 \leftarrow 2} \odot F_2
\]

8. V1 fuses the received sparse message using the existing V2VAM/intermediate-fusion pipeline.

### 3.2 Key Design Principle

The request mask \(M_{1 \leftarrow 2}\) is computed by V1.

This preserves the design principle that:

> The receiver decides what it needs. The sender only provides the requested data.

V2 does not decide what V1 should receive; it only makes its available data/context accessible.

---

## 4. Mathematical Components

### 4.1 Feature Maps

Let the BEV feature maps of V1 and V2 be:

\[
F_1, F_2 \in \mathbb{R}^{C \times H \times W}
\]

where:

- \(C\): feature channels,
- \(H, W\): BEV grid height and width.

The ego/receiver feature map is \(F_1\). The collaborator feature map is \(F_2\).

### 4.2 V1 Need Map

The need map indicates where V1 requires additional information:

\[
N_1 \in \mathbb{R}^{H \times W}
\]

Large \(N_1(h,w)\) means that cell \((h,w)\) is important for V1 to improve or disambiguate its perception.

---

## 5. How V1 Can Estimate What It Needs

### 5.1 Option A: Inverse Feature Energy

This is the simplest version and can be implemented with minimal changes.

First compute ego feature energy:

\[
E_1(h,w) = \|F_1[:,h,w]\|_2
\]

Then define need as inverse energy:

\[
N_1(h,w) = \frac{1}{\epsilon + \tilde{E}_1(h,w)}
\]

where \(\tilde{E}_1\) is a normalized feature-energy map.

Interpretation:

- high ego feature energy \(\Rightarrow\) V1 likely has strong local evidence,
- low ego feature energy \(\Rightarrow\) V1 may need help.

Advantages:

- easy to implement,
- no detector-head modification,
- directly compatible with the existing top-k energy code.

Limitations:

- low feature energy may also correspond to background,
- it is not explicitly detection-aware.

This is the best first prototype because it is simple and establishes whether receiver-conditioned selection improves over sender-only top-k.

---

### 5.2 Option B: Detection-Uncertainty Need Map

A stronger version uses the ego detector’s uncertainty.

Let the detection head produce objectness/class probabilities \(p_1(h,w)\). A need score can be computed from uncertainty:

\[
N_1(h,w) = H(p_1(h,w))
\]

where \(H\) is entropy:

\[
H(p) = -\sum_c p_c \log p_c
\]

For binary/objectness confidence, another possible proxy is:

\[
N_1(h,w) = 1 - |2p_1(h,w)-1|
\]

This is high when the model is uncertain, for example around \(p_1 \approx 0.5\).

Interpretation:

- high confidence object/background \(\Rightarrow\) low need,
- ambiguous objectness \(\Rightarrow\) high need,
- V1 asks for help where its detector is uncertain.

Advantages:

- task-aware,
- more meaningful than inverse feature energy,
- easier to justify scientifically.

Limitations:

- may require exposing pre-fusion or ego-only detector outputs,
- may require an extra forward pass or auxiliary detection head before fusion,
- more code changes than energy-based need.

This is a good second-stage improvement after the energy-based prototype.

---

### 5.3 Option C: Occlusion or Visibility-Based Need Map

V1 can estimate need from physical visibility:

\[
N_1(h,w) = f(\text{point density}, \text{distance}, \text{occlusion}, \text{visibility})
\]

Possible signals:

- low LiDAR point density,
- far-range regions,
- shadow regions behind foreground objects,
- known blind zones,
- low occupancy evidence.

A simple point-density version:

\[
D_1(h,w) = \text{number of LiDAR points in BEV cell }(h,w)
\]

\[
N_1(h,w) = \frac{1}{\epsilon + D_1(h,w)}
\]

Advantages:

- physically intuitive,
- directly matches cooperative perception motivation,
- helpful for occluded or far-away objects.

Limitations:

- requires access to point-density or visibility information,
- may require additional preprocessing,
- more engineering effort.

This is valuable for a later, more interpretable version.

---

### 5.4 Option D: Learned Need Predictor

A small network can learn the need map:

\[
N_1 = g_{\theta}(F_1)
\]

or with more context:

\[
N_1 = g_{\theta}(F_1, \text{ego metadata}, \text{past frames})
\]

This can be trained using detection loss and a communication-budget loss.

Advantages:

- most flexible,
- can learn task-specific request behavior,
- can combine feature, uncertainty, and geometry cues.

Limitations:

- requires careful loss design,
- can easily keep too much communication without budget control,
- higher risk of instability.

Given the Phase 3 experience, a learned request policy should use explicit budget-aware training from the beginning.

---

## 6. How V1 Can Know Whether V2 Has Useful Information

V1 should not receive the full V2 feature map before making the decision, because that would defeat the communication-saving purpose. Instead, V2 exposes a compact context map:

\[
A_2 \in \mathbb{R}^{H \times W}
\]

Possible V2 context maps include:

### 6.1 V2 Feature-Energy Availability

\[
A_2(h,w) = \|F_2[:,h,w]\|_2
\]

This is the simplest availability signal.

Interpretation:

- high \(A_2\) means V2 has strong feature evidence in that cell,
- low \(A_2\) means V2 likely has little useful information.

### 6.2 V2 Objectness / Confidence Map

If V2 has a local detection/objectness head:

\[
A_2(h,w) = p_2^{obj}(h,w)
\]

This indicates where V2 likely observes objects.

### 6.3 V2 Visibility or Point-Density Map

\[
A_2(h,w) = D_2(h,w)
\]

where \(D_2\) is V2’s LiDAR point density or visibility strength.

### 6.4 Compact Metadata Size

The context map is much smaller than the full BEV tensor. If the full feature map has size \(C \times H \times W\), the context map has size \(1 \times H \times W\). With \(C=256\), this context map is roughly \(1/256\) of the dense feature tensor before additional compression.

This overhead must still be counted in communication metrics, but it is small compared to dense feature exchange.

---

## 7. Coordinate Alignment

V1 and V2 operate in different coordinate frames. Therefore, request scoring requires coordinate transformation.

Let:

\[
T_{2 \rightarrow 1}
\]

be the transformation from V2’s coordinate frame to V1’s coordinate frame.

### 7.1 Align V2 Context to V1 Frame

\[
A_{2 \rightarrow 1} = \text{Warp}(A_2, T_{2 \rightarrow 1})
\]

Then V1 computes:

\[
S_{1 \leftarrow 2} = N_1 \odot A_{2 \rightarrow 1}
\]

This is conceptually simple because the request score is computed in V1’s frame.

### 7.2 Mapping the Mask Back to V2 Frame

If V2 needs to apply the mask before sending its own features, V1’s request mask must be mapped back:

\[
M_{2}^{req} = \text{Warp}(M_{1 \leftarrow 2}, T_{1 \rightarrow 2})
\]

Then V2 sends:

\[
\text{Msg}_{2 \rightarrow 1} = M_{2}^{req} \odot F_2
\]

### 7.3 Practical Implementation Choice

For a first implementation, it may be simpler to use the same alignment conventions already used by the V2VAM/intermediate-fusion pipeline. If the current model already stacks and aligns features into the ego frame before fusion, then the policy can compute masks in the ego-aligned representation. This reduces the risk of coordinate-frame errors.

---

## 8. Receiver-Driven Scoring Functions

### 8.1 Basic Multiplicative Score

\[
S_{1 \leftarrow 2}(h,w) =
N_1(h,w)
\cdot
A_{2 \rightarrow 1}(h,w)
\]

This keeps cells where:

- V1 needs help, and
- V2 has useful evidence.

### 8.2 Normalized Score

To avoid scale problems:

\[
\bar{N}_1 = \frac{N_1 - \min(N_1)}{\max(N_1)-\min(N_1)+\epsilon}
\]

\[
\bar{A}_{2 \rightarrow 1} =
\frac{A_{2 \rightarrow 1} - \min(A_{2 \rightarrow 1})}
{\max(A_{2 \rightarrow 1})-\min(A_{2 \rightarrow 1})+\epsilon}
\]

\[
S_{1 \leftarrow 2} =
\bar{N}_1 \odot \bar{A}_{2 \rightarrow 1}
\]

This is recommended for the first prototype.

### 8.3 Weighted Score with Distance and Freshness

A more complete version can include additional factors:

\[
S_{1 \leftarrow 2}(h,w)
=
\bar{N}_1(h,w)
\cdot
\bar{A}_{2 \rightarrow 1}(h,w)
\cdot
G_{2 \rightarrow 1}(h,w)
\cdot
Q_{2}(h,w)
\]

where:

- \(G_{2 \rightarrow 1}\): geometric relevance, e.g. overlapping field of view,
- \(Q_2\): link quality, freshness, trust, or packet reliability.

This is useful for future work, but not necessary in the first version.

---

## 9. Request Mask and Communication Budget

Given a score map \(S_{1 \leftarrow 2}\), V1 selects a mask:

\[
M_{1 \leftarrow 2} = \text{TopK}(S_{1 \leftarrow 2}, r)
\]

where \(r\) is the keep ratio, for example:

\[
r = 0.10
\]

The mask satisfies approximately:

\[
\frac{1}{HW}\sum_{h,w} M_{1 \leftarrow 2}(h,w) = r
\]

The sparse transmitted message is:

\[
\text{Msg}_{2 \rightarrow 1} =
M_{1 \leftarrow 2} \odot F_{2 \rightarrow 1}
\]

The communication cost can be estimated as:

\[
C =
r \cdot C_{feat} \cdot H \cdot W \cdot b
+
C_{context}
+
C_{metadata}
\]

where:

- \(C_{feat}\): number of feature channels,
- \(b\): bytes per feature value,
- \(C_{context}\): cost of V2’s context map,
- \(C_{metadata}\): mask/index overhead.

For fair comparison with Phase 2 top-k, context overhead should be included in the total bytes/frame.

---

## 10. Proposed Method Variants

### 10.1 Variant 1: Receiver-Request Energy Top-k

This is the recommended first version.

\[
N_1(h,w) = \frac{1}{\epsilon + \|F_1[:,h,w]\|_2}
\]

\[
A_2(h,w) = \|F_2[:,h,w]\|_2
\]

\[
S_{1 \leftarrow 2} = \text{Norm}(N_1) \odot \text{Norm}(A_{2 \rightarrow 1})
\]

\[
M_{1 \leftarrow 2} = \text{TopK}(S_{1 \leftarrow 2}, r)
\]

Pros:

- very close to current Phase 2 top-k implementation,
- low engineering effort,
- explainable,
- good first baseline.

Expected difficulty: **low to medium**.

---

### 10.2 Variant 2: Receiver-Request Uncertainty Top-k

\[
N_1(h,w) = H(p_1(h,w))
\]

\[
A_2(h,w) = \|F_2[:,h,w]\|_2
\quad \text{or} \quad p_2^{obj}(h,w)
\]

\[
S_{1 \leftarrow 2} = \text{Norm}(N_1) \odot \text{Norm}(A_{2 \rightarrow 1})
\]

Pros:

- stronger research motivation,
- directly tied to detection uncertainty,
- better explanation: V1 requests help where it is unsure.

Expected difficulty: **medium**.

---

### 10.3 Variant 3: Receiver-Request Occlusion-Aware Top-k

\[
N_1(h,w) =
\alpha \cdot \text{LowPointDensity}_1(h,w)
+
\beta \cdot \text{Distance}_1(h,w)
+
\gamma \cdot \text{OcclusionShadow}_1(h,w)
\]

\[
S_{1 \leftarrow 2} = \text{Norm}(N_1) \odot \text{Norm}(A_{2 \rightarrow 1})
\]

Pros:

- physically meaningful,
- directly connected to cooperative driving use cases.

Expected difficulty: **medium to high**.

---

### 10.4 Variant 4: Learned Receiver Request Network

\[
M_{1 \leftarrow 2} =
g_{\theta}
(
F_1,
A_{2 \rightarrow 1},
T_{2 \rightarrow 1}
)
\]

Training objective:

\[
L_{total}
=
L_{det}
+
\lambda_{budget}
\max(0, \text{mean}(M)-r_{target})^2
\]

Pros:

- most flexible,
- can learn nonlinear request patterns.

Expected difficulty: **high**.

This should be attempted only after the non-learned receiver-request baselines are implemented.

---

## 11. Alignment With the Current Project

### 11.1 Current State

The current project already has the following components:

- config-driven communication policy layer,
- top-k energy selection,
- random communication baselines,
- packet-loss simulation,
- learnable mask,
- communication metrics,
- YAML phase presets,
- CARLA and Culver evaluation workflow,
- Phase 2 AP-vs-communication results,
- Phase 3 trainable mask experiments.

This creates a strong foundation for receiver-driven selection.

### 11.2 What Can Be Reused

| Existing component | Reuse |
|---|---|
| `communication_policy.py` | add new strategy |
| top-k selection logic | reuse for request mask |
| energy score computation | reuse for V2 availability |
| communication metrics | reuse and extend for context overhead |
| YAML presets | add receiver-request presets |
| V2VAM fusion | unchanged |
| inference/training scripts | mostly unchanged |
| CARLA/Culver evaluation pipeline | unchanged |

### 11.3 What Needs to Be Added

The new method requires:

1. ego need map computation,
2. collaborator context/availability map computation,
3. coordinate alignment for context maps,
4. receiver-side request score computation,
5. request mask generation,
6. optional mask mapping back to sender frame,
7. context overhead accounting,
8. new YAML presets and result logging.

---

## 12. Implementation Difficulty Assessment

### 12.1 Overall Difficulty

The first version is an **incremental extension**, not a major rewrite.

Reason:

- the current pipeline already modifies features before fusion,
- top-k masking already exists,
- communication metrics already exist,
- YAML presets already exist,
- V2VAM fusion can remain unchanged.

### 12.2 Difficulty by Variant

| Variant | Difficulty | Reason |
|---|---|---|
| Receiver-request energy top-k | Low–Medium | reuse top-k + energy logic, add ego need map |
| Detection-uncertainty request | Medium | needs access to ego uncertainty/objectness map |
| Occlusion-aware request | Medium–High | needs point-density/visibility computation |
| Learned receiver request | High | needs training objective, budget loss, stability work |

### 12.3 Estimated Effort

| Task | Effort |
|---|---|
| Add YAML presets | Low |
| Add `receiver_request_topk` strategy | Medium |
| Compute inverse-energy ego need map | Low |
| Compute V2 availability map | Low |
| Normalize and combine maps | Low |
| Apply top-k mask | Low, existing logic reusable |
| Count context overhead | Medium |
| Coordinate alignment verification | Medium |
| Run CARLA Phase 2 comparison | Low–Medium |
| Run Culver validation | Low–Medium |
| Detection-uncertainty version | Medium |
| Learned version | High |

A minimal energy-based prototype could likely be implemented with moderate effort because it extends existing Phase 2 logic rather than replacing the architecture.

---

## 13. Concrete Implementation Plan

### Step 1: Add New Strategy Name

Add a new communication strategy:

```yaml
strategy: receiver_request_topk
```

Possible preset:

```yaml
phase5_receiver_request_topk_10:
  enabled: true
  phase: "phase5"
  strategy: "receiver_request_topk"
  drop_ego: false
  receiver_request:
    keep_ratio: 0.1
    ego_need_type: "inverse_energy"
    collaborator_score_type: "l2"
    normalize_scores: true
    count_context_overhead: true
```

### Step 2: Add Ego Need Computation

In the communication policy module:

\[
E_1 = \|F_1\|_2
\]

\[
N_1 = \frac{1}{\epsilon + \text{Norm}(E_1)}
\]

Then normalize \(N_1\) to \([0,1]\).

### Step 3: Add Collaborator Availability Computation

For each collaborator:

\[
A_i = \|F_i\|_2
\]

Normalize \(A_i\) to \([0,1]\).

### Step 4: Align Maps

If features are already ego-aligned at the policy point, compute directly. Otherwise:

\[
A_{i \rightarrow e} = \text{Warp}(A_i, T_{i \rightarrow e})
\]

Use the existing pairwise transformation information from the model input.

### Step 5: Compute Receiver Request Score

\[
S_{e \leftarrow i} = N_e \odot A_{i \rightarrow e}
\]

Then select top-k:

\[
M_{e \leftarrow i} = \text{TopK}(S_{e \leftarrow i}, r)
\]

### Step 6: Apply Mask to Collaborator Features

\[
F_i^{masked} = M_{e \leftarrow i} \odot F_i
\]

Keep ego unchanged:

\[
F_e^{masked} = F_e
\]

### Step 7: Log Metrics

Add or reuse:

- `comm_active_ratio`,
- `comm_normalized_ratio`,
- `comm_feature_bytes_per_frame`,
- `comm_context_bytes_per_frame`,
- `comm_total_bytes_per_frame`,
- `comm_active_neighbors_ratio`.

### Step 8: Evaluation

Evaluate first on CARLA:

- baseline: `phase2_topk_energy_10`
- new: `phase5_receiver_request_topk_10`

Then validate on Culver.

Success criterion:

\[
AP@0.7_{\text{receiver-request}}
>
AP@0.7_{\text{top-k 10}}
\]

with:

\[
\text{comm ratio} \approx 0.10 \text{ to } 0.12
\]

including context overhead.

---

## 14. Repository Change Map

The likely files to modify are:

| File | Required change |
|---|---|
| `src/models/fuse_modules/communication_policy.py` | add receiver-request strategy |
| `src/models/point_pillar_intermediate_V2VAM.py` | pass ego/collaborator context if not already available |
| `src/hypes_yaml/communication_phase_presets.yaml` | add new phase5 presets |
| `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml` | add `receiver_request` config tree |
| `src/tools/inference.py` | ensure new metrics are exported |
| `src/tools/train.py` | only needed if learned request policy is trained later |
| `src/tools/build_clean_phase2_summary.py` | extend summary for context bytes |
| `src/tools/plot_comm_metrics.py` | optionally add plots including context overhead |

For the non-learned energy-based version, most changes should be concentrated in `communication_policy.py` and YAML presets.

---

## 15. Evaluation Plan

### 15.1 Baselines

Compare against:

1. Full baseline,
2. Phase 1 measurement,
3. Phase 2 top-k energy 10%,
4. Phase 2 random comm-only 10%,
5. Receiver-request top-k 10%.

### 15.2 Metrics

Use:

- AP@0.3,
- AP@0.5,
- AP@0.7,
- communication ratio,
- feature bytes/frame,
- context bytes/frame,
- total bytes/frame,
- active neighbors ratio.

### 15.3 Expected Outcomes

Possible outcomes:

#### Outcome A: Receiver-request improves AP at same budget

This would support the hypothesis that receiver-conditioned selection is better than sender-only importance.

#### Outcome B: Receiver-request equals top-k

This still validates the framework but suggests inverse-energy need is too simple. Next step would be uncertainty-based need.

#### Outcome C: Receiver-request is worse

This would suggest that inverse feature energy is not a reliable need proxy. Then the method should move to detection uncertainty or occlusion-aware need.

---

## 16. Key Risks and Mitigations

### Risk 1: Context overhead reduces the benefit

If V2’s context map is too large, communication savings shrink.

Mitigation:

- compress context map,
- use low-resolution context,
- use binary or quantized availability maps,
- count context bytes explicitly.

### Risk 2: Inverse feature energy selects background

Low ego feature energy may correspond to empty background rather than missing objects.

Mitigation:

- combine inverse energy with objectness or uncertainty,
- restrict request to regions near candidate objects,
- use point-density or occlusion maps.

### Risk 3: Coordinate alignment errors

Misalignment between V1 need and V2 availability can produce poor masks.

Mitigation:

- first implement using ego-aligned features if available,
- visualize request masks,
- test with a few frames before full inference.

### Risk 4: Legal/design interpretation ambiguity

The method should clearly state that V1 computes the request mask. V2 only exposes context or responds to requests.

Mitigation:

- name the method receiver-driven,
- log request masks as V1-generated,
- avoid wording that V2 decides what V1 receives.

---

## 17. Recommended First Experiment

The first experiment should be:

```text
phase5_receiver_request_topk_10
```

with:

```yaml
receiver_request:
  keep_ratio: 0.1
  ego_need_type: inverse_energy
  collaborator_score_type: l2
  normalize_scores: true
  count_context_overhead: true
```

Compare against:

```text
phase2_topk_energy_10
```

Current CARLA reference:

\[
AP@0.7 = 0.8703, \quad \text{comm ratio} = 0.0953
\]

Success:

\[
AP@0.7 > 0.8703
\]

at similar total communication cost.

---

## 18. Final Assessment

The receiver-driven selective communication idea is feasible and well aligned with the current project.

It is theoretically sound because it optimizes communication from the receiver’s utility perspective:

\[
\text{send what reduces V1's uncertainty, not simply what is strong in V2}
\]

It is also practically suitable for the current codebase because the repository already contains most of the infrastructure required:

- communication policy before fusion,
- top-k selection,
- metrics,
- YAML presets,
- CARLA/Culver evaluation workflow.

The minimal energy-based version is an incremental extension. The more advanced uncertainty-based and learned versions are larger research contributions but should be built after validating the simple receiver-request top-k baseline.

The recommended direction is therefore:

1. implement receiver-request energy top-k,
2. compare against top-k energy 10%,
3. add context-overhead accounting,
4. validate on CARLA and Culver,
5. then extend to detection uncertainty or learned request policy if the first version is promising.
