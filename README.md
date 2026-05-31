# Communication-Aware V2V Cooperative Perception (project Project)

This repository is my project project on **communication-aware cooperative perception** for V2V systems.

The baseline model is PointPillars + V2VAM intermediate fusion, extended with configurable communication policies to study:
- what to transmit,
- which neighbors to use,
- and how robust perception is under limited/lossy communication.

## Project Links

- GitHub repository: `https://github.com/mohsenshahverdy/comm-aware-v2v-perception`
- Kaggle notebook/project: `https://www.kaggle.com/code/mohsenshahverdi/communication-aware-v2v-perception/edit`

## project Scope

This project implements a full phase-based communication pipeline:

1. `Phase 0` baseline (full communication)
2. `Phase 1` communication measurement
3. `Phase 2` fast baselines
   - random dropping
   - energy top-k selection
   - neighbor selection
   - packet-loss simulation
4. `Phase 3` learnable importance mask + communication regularization
5. `Phase 4` repair network under lossy communication

All phases are controlled via config with no code edits needed for switching experiments.

## Repository Highlights

- Communication policy module before V2V fusion
- Config-driven toggles for all communication features
- Auxiliary communication losses (`L_comm`, `L_repair`)
- Communication-aware logging for training and inference
- Output files for AP-vs-communication analysis

## Configuration

Main config:
- `src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml`

Dataset path defaults in config are POSIX-style placeholders (`training_data/train`, `validating_data/validate`).
You can override them at runtime with `--root_dir` and `--validate_dir` in both training and inference scripts.

Phase presets:
- `src/hypes_yaml/communication_phase_presets.yaml`
- `src/hypes_yaml/communication_phases/phase0_baseline.yaml`
- `src/hypes_yaml/communication_phases/phase1_measurement.yaml`
- `src/hypes_yaml/communication_phases/phase2_random_drop.yaml`
- `src/hypes_yaml/communication_phases/phase2_topk_energy.yaml`
- `src/hypes_yaml/communication_phases/phase2_neighbor_packetloss.yaml`
- `src/hypes_yaml/communication_phases/phase3_learnable_mask.yaml`
- `src/hypes_yaml/communication_phases/phase4_repair.yaml`

## Environment Notes

## Documentation

- `docs/reports/` for experiment and technical reports
- `docs/plans/` for implementation plans
- `docs/notebooks/` for notebook artifacts


- On macOS (especially Apple Silicon), `spconv` is typically unavailable.
- Full end-to-end training/inference is recommended on **Linux + NVIDIA CUDA**.
- You can still run communication policy unit tests locally without `spconv`.

## Local Sanity Checks (No `spconv` Required)

Compile checks:

```bash
python -m py_compile \
  src/tools/inference.py \
  src/models/point_pillar_intermediate_V2VAM.py \
  src/models/fuse_modules/communication_policy.py
```

Communication policy fake-tensor test:

```bash
python -m src.tools.test_comm_policy_fake
```

## Training

Example with a phase config:

```bash
python src/tools/train.py --hypes_yaml src/hypes_yaml/communication_phases/phase0_baseline.yaml
```

## Inference

```bash
python src/tools/inference.py --model_dir <RUN_DIR> --fusion_method intermediate
```

Notes:
- `open3d` is optional unless you use `--show_sequence`.
- Inference writes communication metrics and merged summary files.

## Outputs

Per run directory (under `src/logs/...`):
- `comm_metrics_epoch.csv`
- `comm_metrics_frame.jsonl`
- `summary_eval.yaml`
- `eval.yaml` or `eval_global_sort.yaml`

## Plotting

Generate communication-performance plots:

```bash
python src/tools/plot_comm_metrics.py --csv <RUN_DIR>/comm_metrics_epoch.csv
```

Generated plots:
- `ap70_vs_comm_cost.png`
- `ap50_vs_comm_cost.png`
- `ap70_vs_packet_loss.png`
- `comm_cost_vs_neighbors.png`

## Recommended project Evaluation Order

1. Run `phase0_baseline`
2. Run `phase1_measurement`
3. Run phase-2 baselines
4. Run `phase3_learnable_mask`
5. Run `phase4_repair`

Primary analysis target:
- **AP@0.7 vs communication cost**

## License

Code headers and packaging metadata use `TDG-Attribution-NonCommercial-NoDistrib` (non-commercial / no redistribution).
Follow original repository and dataset licenses before any reuse.
