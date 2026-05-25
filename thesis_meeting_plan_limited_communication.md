# Meeting Report: project Improvement Plan Under Limited V2V Communication

## 1. Very Simple Summary

The current project/repository already has a working cooperative perception pipeline.

In simple words:

> Several vehicles observe the road with LiDAR. Each vehicle creates a top-down feature map. The ego vehicle receives feature maps from nearby vehicles and fuses them to detect objects better.

The current system mainly assumes that vehicles can share a lot of information. The next research direction is to make this more realistic:

> The communication channel is limited, so vehicles cannot send everything. They must decide what is important enough to share.

So the new project direction can be:

> **Communication-aware cooperative perception: deciding what, when, and from whom to receive information under limited or lossy V2V communication.**

---

## 2. Current State of the project and Repo

### What the current pipeline does

The current pipeline is roughly:

```text
LiDAR from each vehicle
   ↓
Point cloud preprocessing / voxelization
   ↓
PillarVFE feature extraction
   ↓
Scatter to BEV feature map
   ↓
BEV CNN backbone
   ↓
V2V attention fusion
   ↓
Detection heads
   ↓
3D object boxes
```

### Simple meaning of the main concepts

| Term | Simple meaning |
|---|---|
| LiDAR | A sensor that creates 3D points around the car. |
| Point cloud | The 3D points captured by LiDAR. |
| BEV | Bird's-Eye View, like looking at the road from above. |
| Feature map | A learned representation that helps the model detect objects. |
| Fusion | Combining information from multiple vehicles. |
| Ego vehicle | The main vehicle that receives information and makes detections. |
| Collaborator / CAV | Nearby vehicle that shares information with the ego vehicle. |
| Attention | A mechanism that helps the model focus on useful information. |
| AP | Average Precision, the main detection accuracy metric. |

### Current limitation

The current pipeline is powerful, but it is close to this assumption:

```text
All nearby vehicles can share their BEV features.
```

In real V2X communication, this is not realistic because:

```text
bandwidth is limited
messages can be delayed
packets can be lost
some vehicles may be less useful than others
```

So the research problem is:

> How can we keep high 3D detection accuracy while sending much less data?

---

## 3. What the Deep Research Suggests

The deep research report grouped the state-of-the-art approaches into several families. For our project, the most useful families are below.

---

## 4. Approach Family A: Importance-Aware Spatial Selection

### Core idea

Not all parts of the BEV map are useful.

Some cells are empty road or background. Some cells contain vehicles, object boundaries, occluded regions, or uncertain areas.

So instead of sending the whole BEV feature map, a vehicle sends only the important regions.

### Simple example

Imagine a top-down map of the road.

```text
Empty road       → not very important
Area near car    → important
Occluded region  → important
Object boundary  → important
Uncertain region → important
```

### How importance can be computed

Simple non-learned options:

```text
importance = feature magnitude
importance = objectness score
importance = uncertainty
importance = change compared to previous frame
```

A simple feature-energy score:

```text
importance(x, y) = mean(abs(F[:, x, y]))
```

or:

```text
importance(x, y) = ||F[:, x, y]||₂
```

Where:

```text
F = BEV feature map
x, y = spatial BEV cell location
```

### Learnable option

Add a small neural network that predicts an importance mask:

```text
M = sigmoid(Conv1x1(F))
F_sent = F × M
```

Where:

```text
M = importance mask
F_sent = transmitted feature map
```

To prevent the model from sending everything:

```text
Loss = Detection Loss + λ × Communication Cost
```

A simple communication cost can be:

```text
Communication Cost = mean(M)
```

### Why this is a good first direction

This is the best fit for the current repo because the repo already uses BEV feature maps before V2VAM fusion. We can insert the selection module before the fusion module without changing the full architecture.

### Related research family

This direction is connected to methods such as:

```text
Where2comm
How2comm
COOPERTRIM
Reason-to-Transmit-style approaches
```

### project value

This directly answers the professor's question:

> Which data should be shared and which data should not?

---

## 5. Approach Family B: Compression / Codebook Communication

### Core idea

Even if we select only important regions, the selected features may still be large.

So the next step is compression.

Instead of sending full floating-point feature values, vehicles can send:

```text
low-bit values
integer codes
codebook indices
compressed feature tokens
```

### Simple example

Instead of sending:

```text
[0.23482, -1.88312, 0.98123, ...]
```

send:

```text
[code_17, code_03, code_88]
```

The receiver then reconstructs an approximate version of the feature.

### Related research family

This direction is connected to:

```text
CodeFilling
QuantV2X
V2X-DSC
WaveComm
```

### Why not start here first?

Compression is useful, but it is more complex than selection. A safer project path is:

```text
first decide what to send
then decide how to compress it
```

### project value

This can become the second contribution after importance-aware selection.

---

## 6. Approach Family C: Lossy-Channel Robustness

### Core idea

In real V2V communication, messages may not arrive perfectly.

Possible problems:

```text
packet loss
delay
jitter
corrupted features
missing collaborator messages
stale information
```

So the model must be robust when communication is imperfect.

### Simple experiment

Simulate packet loss by dropping part of the transmitted feature map.

Test:

```text
0% packet loss
10% packet loss
20% packet loss
30% packet loss
50% packet loss
```

Then measure:

```text
AP@0.5
AP@0.7
```

### Simple output graph

```text
AP@0.7 vs Packet Loss Rate
```

### Possible improvement

Train with communication dropout:

```text
During training, randomly drop some shared features.
```

This teaches the model not to collapse when communication is imperfect.

### Related research family

This direction is connected to:

```text
V2VAM with LC-aware Repair Network
V2X-INCOP
Fresh2comm
Latency-aware collaborative perception
CoDynTrust
```

### project value

This helps answer:

> Can the model still work when communication is unreliable?

---

## 7. Approach Family D: Neighbor Selection / Scheduling

### Core idea

If many vehicles are nearby, the ego vehicle may not need all of them.

Maybe one nearby vehicle is very useful, while another vehicle sees mostly the same thing or has bad communication quality.

So the ego vehicle should decide:

```text
Which vehicle should send information?
```

### Simple strategies

```text
Use all vehicles
Use nearest 1 vehicle
Use nearest 2 vehicles
Use vehicle with highest feature importance
Use vehicle with best channel quality
Use vehicle with most different viewpoint
```

### Why this fits the repo

The repo groups vehicle features using `record_len`. That makes it possible to choose a subset of collaborators before fusion.

### project value

This answers:

> Which neighbor is worth communicating with?

---

## 8. Approach Family E: The Shared Paper's Idea

The paper shared by the professor is more about communication control than about a new 3D detection backbone.

Its main question is:

```text
Under limited communication resources, which vehicle should send, what content should be sent, and which radio resources should be used?
```

It uses:

```text
quadtree-compressed sensory messages
action-branching reinforcement learning
federated reinforcement learning
```

### How to interpret it for our project

It is not a direct replacement for V2VAM.

It is better understood as a policy layer:

```text
communication policy → decides what/who/when to send
V2VAM fusion → fuses what arrives
```

### Practical lesson

The paper suggests that communication decisions should be part of the system, not an afterthought.

For our repo, this means we can add a communication policy before V2VAM fusion.

---

## 9. Recommended Direction for This project

The best first direction is:

```text
Importance-aware sparse BEV feature sharing before V2VAM fusion.
```

Simple version:

```text
Instead of sending the full BEV feature map, each vehicle sends only the most useful BEV regions.
```

Why this is the best first choice:

```text
It is directly connected to the professor's question.
It fits the current repo.
It does not require replacing the full model.
It can be tested quickly.
It gives clear AP-vs-bandwidth plots.
```

---

## 10. Concrete Experimental Plan

## Phase 0: Stabilize the Baseline

### Goal

Make sure the current repo is trustworthy before adding research changes.

### Tasks

```text
1. Fix environment.
2. Fix dataset paths.
3. Run one-batch sanity check.
4. Run baseline inference.
5. Run short debug training.
6. Check/fix the V2VAM fusion logic.
7. Save baseline AP and runtime.
```

### Output

| Method | Communication | AP@0.5 | AP@0.7 | Notes |
|---|---:|---:|---:|---|
| Current V2VAM | 100% | TBD | TBD | Full communication baseline |

---

## Phase 1: Add Communication Measurement

### Goal

Measure how much data is being transmitted.

### What to measure

```text
number of active vehicles
number of transmitted BEV cells
percentage of transmitted features
bytes per frame
AP@0.5
AP@0.7
FPS / inference time
```

### Simple formula

If the feature tensor is:

```text
C × H × W
```

and values are FP32:

```text
bytes = C × H × W × 4
```

If only 10% of spatial cells are sent:

```text
bytes = 0.10 × C × H × W × 4
```

### Main graph

```text
AP@0.7 vs Communication Cost
```

---

## Phase 2: Fast Baselines

### Experiment 1: Random Feature Dropping

Keep only a random subset of feature cells.

Test:

```text
100%, 50%, 25%, 10%, 5%, 1%
```

Purpose:

```text
Show how badly the model degrades when communication is limited.
```

---

### Experiment 2: Feature-Energy Top-K Selection

Compute feature importance using feature magnitude.

```text
importance(x, y) = mean(abs(F[:, x, y]))
```

Send only top-K cells.

Test:

```text
Top 50%, 25%, 10%, 5%, 1%
```

Purpose:

```text
Show that importance-aware selection is better than random dropping.
```

---

### Experiment 3: Neighbor Selection

Use only selected vehicles.

Test:

```text
all vehicles
nearest 1 vehicle
nearest 2 vehicles
top-1 vehicle by importance
top-2 vehicles by importance
```

Purpose:

```text
Show whether we can save bandwidth by selecting fewer collaborators.
```

---

### Experiment 4: Packet Loss Simulation

Randomly drop transmitted packets/features.

Test:

```text
0%, 10%, 20%, 30%, 50% packet loss
```

Purpose:

```text
Measure robustness under lossy communication.
```

---

## Phase 3: Main Proposed Improvement

## Learnable Importance Mask

### Goal

Make the model learn which BEV regions are important to transmit.

### Method

Add a small importance head:

```text
M = sigmoid(Conv1x1(F))
F_sent = F × M
```

Add a communication penalty:

```text
Loss = Detection Loss + λ × mean(M)
```

Where:

```text
Detection Loss = object detection loss
mean(M) = communication cost
λ = controls how strongly we penalize communication
```

### Experiments

Train with different λ values:

```text
λ = 0.001
λ = 0.01
λ = 0.05
λ = 0.1
```

Compare against:

```text
full communication
random dropping
feature-energy top-K
neighbor selection
learnable importance mask
```

### Main expected result

The learnable mask should keep good AP while sending fewer features.

---

## Phase 4: Optional Extension

After the learnable importance mask, choose one stronger extension.

### Option A: Quantization / Codebook

Compress selected features into low-bit codes.

### Option B: Repair Network

Recover missing/corrupted features before fusion.

### Option C: Delay-Aware Fusion

Use old messages but downweight them based on age.

### Option D: Learned Neighbor Scheduler

Learn which vehicles should communicate under a fixed bandwidth budget.

---

## 11. Recommended Timeline

### Week 1: Baseline + Measurement

```text
fix environment
fix dataset paths
run one batch
run baseline inference
check V2VAM fusion logic
add communication logger
```

Output:

```text
baseline AP + baseline communication cost
```

---

### Week 2: Fast Limited-Communication Experiments

```text
random dropping
feature-energy top-K
neighbor selection
packet loss simulation
```

Output:

```text
AP-vs-bandwidth plots
AP-vs-packet-loss plots
```

---

### Week 3–4: Learnable Importance Mask

```text
add importance head
add communication budget loss
train with different λ values
compare against baselines
```

Output:

```text
main project comparison table
main AP-vs-bandwidth curve
```

---

### Week 5+: Optional Extension

Choose one:

```text
quantization/codebook
repair network
delay-aware fusion
neighbor scheduling
```

Output:

```text
second contribution or future-work section
```

---

## 12. Main Evaluation Metrics

For every method, report:

```text
AP@0.5
AP@0.7
bytes per frame
percentage of transmitted features
number of active collaborators
FPS / inference time
robustness under packet loss
```

Main plots:

```text
AP@0.7 vs Communication Cost
AP@0.5 vs Communication Cost
AP@0.7 vs Packet Loss Rate
Communication Cost vs Number of Vehicles
Inference Time vs Communication Strategy
```

Most important plot:

```text
AP@0.7 vs Communication Cost
```

---

## 13. Meeting Talking Points

### What I understand about the current system

```text
The current repository implements a PointPillars-based intermediate fusion pipeline.
Each vehicle extracts BEV features from LiDAR, and the ego vehicle fuses collaborator features using V2VAM attention before object detection.
```

### What the limitation is

```text
The current setup is strong but assumes too much communication. In realistic V2V/V2X, the channel is limited, delayed, and sometimes lossy.
```

### What I propose

```text
I propose to first add communication measurement and channel simulation, then test simple baselines such as random dropping, feature-energy top-K selection, neighbor selection, and packet loss simulation. After that, I propose a learnable importance mask with a communication budget loss.
```

### Main research question

```text
Can we maintain high 3D object detection accuracy while transmitting only the most important BEV features under limited V2V communication?
```

### Main expected contribution

```text
A communication-aware extension of V2VAM using importance-aware sparse BEV feature sharing.
```

---

## 14. Questions to Ask the Professor

1. Should the main project focus be **what to send**, **when to send**, or **which vehicle to select**?
2. Should OPV2V be enough for the first experiments?
3. Should packet loss and delay be main experiments or future work?
4. What bandwidth budget should we consider realistic?
5. Should the first method be simple and interpretable, or should we directly attempt a learnable communication policy?
6. Does he prefer focusing on importance selection, compression, or lossy-channel repair?

---

## 15. Final Recommended Plan

Start simple and measurable.

Recommended order:

```text
1. Stabilize current baseline.
2. Add communication cost measurement.
3. Add random dropping baseline.
4. Add feature-energy top-K selection.
5. Add neighbor selection.
6. Add packet loss simulation.
7. Add learnable importance mask.
8. Optionally add compression or repair network.
```

The strongest first project contribution is:

```text
Learned importance-aware sparse BEV sharing under limited V2V communication.
```

Why this is the best first contribution:

```text
It is realistic.
It is explainable.
It fits the current repo.
It directly answers the professor's question.
It can produce clear AP-vs-bandwidth results.
```

---

## 16. Final Short Version for the Meeting

> The current pipeline already performs cooperative perception using PointPillars, BEV features, and V2VAM fusion. The main limitation is that it assumes rich communication. My proposed project direction is to make it communication-aware. I will first measure communication cost and simulate limited bandwidth. Then I will compare random feature dropping, feature-energy-based top-K selection, neighbor selection, and packet loss simulation. After that, I will implement a learnable importance mask with a communication budget loss. The main evaluation will be AP@0.7 versus transmitted bytes per frame.

