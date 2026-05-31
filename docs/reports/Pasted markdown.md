## Communication-Aware V2VAM: Full-Phase Config-Driven Implementation Plan

### Summary
Implement all thesis phases in one integrated pipeline with a single master `communication` config tree, so every feature can be turned on/off without code edits.  
The implementation will keep the current PointPillar + V2VAM backbone and add a modular communication policy layer before fusion, with standardized metrics/logging/plots for AP-vs-communication and robustness analysis.

### Key Implementation Changes
1. **Config-first architecture (single switch tree)**
- Extend the main YAML schema with a top-level `communication` section containing:
  - `enabled`, `phase`, `strategy`, `seed`
  - `measurement` toggles (`track_bytes`, `track_active_cells`, `track_active_neighbors`, `track_latency`)
  - `drop_random` params (`keep_ratio`)
  - `topk_energy` params (`keep_ratio`, `score_type`)
  - `neighbor_selection` params (`mode`, `k`, `distance_metric`)
  - `packet_loss` params (`enabled`, `loss_rate`, `unit`)
  - `learnable_mask` params (`enabled`, `mask_channels`, `sparsity_lambda`, `temperature`, `hard_mask`)
  - `repair_network` params (`enabled`, `type`, `hidden_dim`, `loss_weight`)
  - `logging` + `visualization` params (`save_csv`, `save_per_frame_json`, `plot_curves`)
- Add phase presets (Phase0–Phase4) as config blocks or companion override YAMLs generated from the master schema.

2. **Communication policy module inserted before V2V fusion**
- In the model forward path (before `V2V_AttFusion`), add a policy pipeline:
  - compute candidate transmit features
  - apply spatial selection (random or energy top-k or learnable mask)
  - apply neighbor selection (all / nearest-k / importance-k)
  - apply packet-loss channel corruption simulation
  - optionally run repair network at receiver side
- Output both:
  - fused features for detection
  - communication stats dict for logging and loss calculation.
- Ensure policy respects `record_len` grouping per sample and supports batched CAV groups correctly.

3. **Loss extension for learnable mask and repair training**
- Keep existing detection loss unchanged as base.
- Add optional auxiliary terms:
  - `L_comm = lambda * mean(mask)` (or active ratio proxy)
  - `L_repair` (feature reconstruction on dropped/corrupted regions)
- Total loss when enabled:
  - `L_total = L_det + L_comm (+ L_repair)`
- Keep backward compatibility: when communication modules are disabled, loss and behavior match baseline.

4. **Measurement, logging, and visualization upgrades**
- Standardize per-step/per-epoch metrics:
  - AP@0.5, AP@0.7
  - bytes/frame, active cell ratio, active neighbors, packet-loss rate, inference time/FPS
- Persist outputs to run directory:
  - `comm_metrics_epoch.csv`
  - `comm_metrics_frame.jsonl`
  - `summary_eval.yaml` (merge of AP + communication metrics)
- Add plotting utility scripts:
  - AP@0.7 vs communication cost
  - AP@0.5 vs communication cost
  - AP@0.7 vs packet loss
  - communication cost vs number of neighbors
- Add TensorBoard scalar groups:
  - `Comm/bytes_per_frame`, `Comm/active_ratio`, `Comm/active_neighbors`, `Comm/loss_rate`, `Comm/fps`.

5. **Phase-by-phase execution plan (all implemented now, run by config)**
- **Phase 0**: baseline stabilization and parity checks.
- **Phase 1**: communication measurement only.
- **Phase 2**: fast baselines (random drop, energy top-k, neighbor selection, packet loss).
- **Phase 3**: learnable importance mask + sparsity penalty sweep.
- **Phase 4**: repair network extension (selected default).

### Public Interfaces / Config Additions
- **New YAML interface:** `communication` tree (master control plane).
- **Model forward contract change:** return communication stats alongside detector outputs (or attach under output dict keys).
- **Training loop interface:** consume optional `comm_loss_terms` and log `comm_stats`.
- **Inference interface:** support communication strategy selection and loss-rate sweeps from config/CLI override.

### Test Plan
1. **Baseline parity tests**
- With `communication.enabled=false`, confirm identical AP and near-identical runtime to current baseline.
- Confirm no additional loss terms are active.

2. **Unit tests for policy operators**
- Random drop keeps expected ratio within tolerance.
- Energy top-k selects exactly configured ratio.
- Neighbor selection honors `record_len` and configured `k`.
- Packet-loss simulator drops expected fraction by seed-controlled randomness.

3. **Integration tests**
- One mini-epoch per phase configuration runs end-to-end without crash.
- `comm_metrics_epoch.csv` and `summary_eval.yaml` are produced for each run.
- Phase 3 with nonzero lambda reduces active ratio vs lambda=0 baseline.

4. **Robustness sweep checks**
- Packet-loss sweep (0, 10, 20, 30, 50%) shows monotonic communication perturbation handling and valid AP outputs.
- Repair network enabled improves AP over no-repair at moderate/high loss rates (target check, not hard assert).

### Assumptions and Defaults
- Use **single master switch tree** in YAML as requested.
- Use **core experiment matrix** first (not exhaustive) for manageable runtime and stable debugging.
- Phase 4 default extension is **Repair Network**.
- Existing model/data pipeline stays intact; communication policy is a pre-fusion addon to minimize regression risk.
- Any expensive sweeps are driven by config presets so you can scale to exhaustive runs later without code changes.
