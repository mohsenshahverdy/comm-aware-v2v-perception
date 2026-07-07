# Publication Server Run Guide

Do not launch the budget campaign before environment preflight, one smoke run,
and one full single-method run succeed.

## 1. Configure Inputs

```bash
export PUBLICATION_CHECKPOINT_DIR="<base-checkpoint-directory>"
export LEARNED_PUBLICATION_CHECKPOINT_DIR="<trained-learned-checkpoint-directory>"
export CARLA_TRAIN_ROOT="<training-data-directory>"
export CARLA_2021_VALIDATE_ROOT="<carla-validation-directory>"
export CULVER_VALIDATE_ROOT="<culver-validation-directory>"
```

## 2. Preflight

```bash
./env/bin/python tools/publication/check_publication_environment.py \
  --dataset carla_2021
```

Resolve every required `exists=no` row before proceeding.

## 3. Smoke Validation

```bash
bash results/publication/commands_smoke_carla_topk10.sh
```

Inspect the smoke directory for:

```text
publication_run.log
publication_inference.log
run_metadata.json
command.txt
config.yaml
config_resolved.yaml
publication_result.json
summary_eval.yaml
inference_summary.json
```

`publication_run.log` records wrapper decisions. `publication_inference.log`
contains native inference output. Continue only when metadata and the publication
result report `smoke_completed`.

## 4. Single Full Validation

```bash
bash results/publication/commands_full_carla_topk10.sh
```

Continue only when the publication result reports `completed` and contains
AP@0.7, measured total communication ratio, and bytes per frame.

For a run completed before automatic post-evaluation was enabled, update it in
place without repeating inference:

```bash
./env/bin/python tools/publication/run_publication_experiments.py \
  --config experiments/publication/publication_sweep_config.yaml \
  --dataset carla_2021 --method selective_topk --budget 10 \
  --post-evaluate-only
```

This reuses `danger_eval_boxes`, updates `summary_eval.yaml`, and refreshes
`publication_result.json`. It also writes `publication_post_evaluation.log` and
`post_evaluation_commands.json`. Absolute static and trajectory metrics are
available for a single run. Missed-risk reduction versus the snapshot receiver
baseline is populated only after the matching receiver run is available and
co-evaluated.

## 5. Aggregate and Plot

```bash
./env/bin/python tools/publication/aggregate_publication_results.py
./env/bin/python tools/publication/plot_publication_curves.py
```

## 6. Campaign Commands

The CARLA, Culver, and full-sweep command files are generated under
`results/publication/`. The 122-job script contains a warning and must only be
used after the preceding validation stages. Every command uses `--resume` and
`--overwrite false`; existing outputs are not silently replaced.

For additional diagnostics, append `--debug` to an individual runner command.
