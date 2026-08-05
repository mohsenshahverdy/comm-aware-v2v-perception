# Publication Experiment Framework

This directory defines the reproducible experiment-management layer for the
publication extension of the cooperative V2V perception work. It does not
change model behavior or existing thesis result directories.

## Scope

`publication_sweep_config.yaml` describes:

- CARLA 2021-only and Culver City evaluation splits;
- full communication, sender-side Top-K, snapshot receiver-request, temporal
  receiver-request, and learned temporal receiver-request;
- nominal budgets of 1--10 (one-point increments), 20, 25, 50, 75, and
  100 percent for sparse methods;
- deterministic seeds and future Monte Carlo loss scenarios;
- environment-variable paths rather than machine-specific paths.

The external baseline and lossy Monte Carlo campaign are placeholders and are
disabled until their adapters and protocol assumptions are validated.

## Dry Run

From the repository root:

```bash
python tools/publication/run_publication_experiments.py \
  --config experiments/publication/publication_sweep_config.yaml \
  --dry-run
```

Dry-run is also the default when `--execute` is omitted. It does not create run
directories or copy checkpoints. With the checked-in datasets, one seed, and
loss simulation disabled, the complete plan contains 122 jobs: two dense
references and 120 sparse operating points.

## Environment Preflight

Before execution on a new machine, check the selected dataset without loading
PyTorch or model code:

```bash
./env/bin/python tools/publication/check_publication_environment.py --dataset carla_2021
./env/bin/python tools/publication/check_publication_environment.py --dataset culver_city
```

The command prints each required variable, resolved value, existence status,
and checkpoint notes. Missing variables produce a concise non-zero result, not
a traceback. The learned checkpoint is reported separately because it is not
required for non-learned smoke validation.

## Export Server Commands

`--export-commands` writes an executable shell script but never runs inference:

```bash
./env/bin/python tools/publication/run_publication_experiments.py \
  --config experiments/publication/publication_sweep_config.yaml \
  --dataset carla_2021 \
  --export-commands results/publication/commands_carla_budget_sweep.sh
```

Each exported line selects one exact dataset/method/budget/seed/loss realization
and invokes the runner with `--execute --resume --overwrite false`. The full
full-sweep export contains a prominent warning and should not be run before smoke
and single-full validation succeed.

Filter the plan with, for example:

```bash
python tools/publication/run_publication_experiments.py \
  --config experiments/publication/publication_sweep_config.yaml \
  --dataset carla_2021 \
  --methods where2comm_style_confidence_topk \
  --budgets 10 \
  --dry-run
```

The method `where2comm_style_confidence_topk` is a Where2Comm-style confidence-map
sparse proxy baseline. It uses a spatial confidence map derived from collaborator
BEV feature activations when detector objectness logits are not available at the
communication-policy stage. It must be reported as a proxy baseline, not as a
faithful reproduction of Where2Comm.

## Execute One Experiment Later

Set portable path variables first:

```bash
export PUBLICATION_CHECKPOINT_DIR="<path-to-base-checkpoints>"
export LEARNED_PUBLICATION_CHECKPOINT_DIR="<path-to-trained-learned-checkpoints>"
export CARLA_TRAIN_ROOT="<path-to-train-data>"
export CARLA_2021_VALIDATE_ROOT="<path-to-carla-2021-validation>"
export CULVER_VALIDATE_ROOT="<path-to-culver-validation>"
```

Then select exactly one job:

```bash
python tools/publication/run_publication_experiments.py \
  --config experiments/publication/publication_sweep_config.yaml \
  --dataset carla_2021 \
  --method selective_topk \
  --budget 10 \
  --execute \
  --resume \
  --overwrite false
```

Each job is staged under
`results/publication/runs/<dataset>/<method>/budget_<percent>/seed_<seed>/<loss>/mc_<run>/<full|smoke>/`.
The
runner merges the existing repository preset into a copied base configuration,
applies declared budget overrides only to that staged config, copies checkpoint
files, invokes `src.tools.inference`, and writes `publication_result.json`.
Existing directories are rejected unless `--resume` or `--overwrite true` is
explicitly provided. Learned temporal inference remains protected by the
repository's trained request-head checkpoint safety.

Every run stores `config.yaml` for the inference entry point and an identical
`config_resolved.yaml` publication artifact, plus `command.txt`,
`run_metadata.json`, the inference log, native inference outputs, and
`publication_result.json`. Only the latest resolved checkpoint is copied into
each run directory.

For a 20-frame validation before full evaluation, add `--smoke`:

```bash
python tools/publication/run_publication_experiments.py \
  --config experiments/publication/publication_sweep_config.yaml \
  --dataset carla_2021 --method selective_topk --budget 10 \
  --execute --smoke
```

Smoke and full outputs use separate directories. Smoke rows remain in the raw
CSV but are excluded from grouped publication summaries. Execution refuses a
multi-job selection unless `--allow-multiple` is explicitly supplied, which
guards against accidentally launching the full campaign.

## Unified Results

Every publication row follows the columns in
`tools/publication/result_schema.py`. Missing measurements remain empty/NaN;
columns are never dropped. The schema includes experiment identity, seed and
loss realization, AP, feature and total communication, bytes per frame, static
and trajectory-aware metrics, paths, timestamp, status, and notes.

Aggregate available result records with:

```bash
python tools/publication/aggregate_publication_results.py
```

Outputs:

- `results/publication/all_experiments_raw.csv`
- `results/publication/all_experiments_summary.csv`

The summary groups by dataset, method, budget, loss type, and loss probability,
then reports mean and standard deviation for numeric fields across seeds and
Monte Carlo runs.

## Plots

```bash
python tools/publication/plot_publication_curves.py
```

When compatible data exist, dataset-specific PDF and PNG plots are written to
`results/publication/figures/` for AP@0.7, trajectory-time risk recall, missed
trajectory risk, measured total communication ratio, and trajectory efficiency
(`TTRR@0.7 / total_comm_ratio`) versus nominal budget. Missing data produce warnings rather than
empty or misleading figures.

The learned temporal sweep changes `receiver_request.keep_ratio` at inference
and uses hard Top-K over the trained request probability map. The same head,
trained around the 10 percent target, is reused at every operating point. These
curves must therefore be described as an inference-time budget sweep; a final
journal study may additionally retrain per budget or use a budget-conditioned
head.

## Planned Extensions

Future steps will add a validated external-method adapter, learned request-head
stability across seeds, cache-state diagnostics and ablations, and Monte Carlo
feature-cell/packet-loss simulation. Those campaigns are intentionally not run
by this infrastructure-only step.

## Local Tests Without Pytest

The same mock tests can run directly in environments where `pytest` is absent:

```bash
./env/bin/python tools/publication/run_local_mock_tests.py
```

They validate the 122-job grid, budget overrides, command counts, graceful
missing-environment behavior, resolved artifacts, failed-output handling,
aggregation, and plotting using temporary files only.

## Logging and Config Conventions

Publication scripts use the repository logger from `src/utils/logging`. Console
messages therefore share the same component, level, emoji, and key/value format
as training, inference, communication policy, and metric evaluators. Supported
options are `--log-level DEBUG|INFO|WARN|ERROR` and `--debug`; debug mode also
allows tracebacks for expected command/configuration failures.

Real and smoke run directories contain two distinct logs:

- `publication_run.log`: publication wrapper events, resolved paths, command,
  and final status;
- `publication_inference.log`: unmodified stdout/stderr from
  `src.tools.inference`.

The publication orchestration YAML is loaded with `yaml.safe_load` rather than
the model-oriented `yaml_utils.load_yaml`. This is intentional: publication
configuration contains unresolved environment placeholders and a dataset ×
method × budget grid, while `yaml_utils.load_yaml` is responsible for loading a
single model configuration and applying its `communication_preset`. For each
run, the publication tool safely loads the base model YAML and checked-in preset,
deep-merges them in memory, applies explicit budget overrides, and writes only
`config.yaml` and `config_resolved.yaml` under the run directory. Checked-in
presets are never rewritten.

`run_metadata.json` records the experiment identity, nominal ratio, override
keys, exact command, checkpoint source, config path, timestamps, run mode, and
status. Expected errors are concise by default; add `--debug` when a traceback
is needed during development.

## Static and Trajectory Post-Evaluation

`post_evaluation` in `publication_sweep_config.yaml` controls reuse of the
existing thesis evaluators:

- `src.tools.evaluate_danger_aware_metrics`;
- `src.tools.evaluate_trajectory_danger_metrics`.

After successful inference, the runner evaluates exported
`danger_eval_boxes/frame_*.npz`, asks the evaluators to update
`summary_eval.yaml`, and refreshes `publication_result.json`. Commands and
subprocess output are stored in `post_evaluation_commands.json` and
`publication_post_evaluation.log`.

When completed runs at the same dataset, budget, seed, loss setting, and run
mode exist, they are evaluated together. This allows missed-risk reduction
against `snapshot_receiver_request` once that baseline is available. Before the
baseline exists, absolute danger and trajectory metrics are still populated,
while baseline-dependent reduction remains NaN rather than being fabricated.

To update a completed run without repeating inference:

```bash
./env/bin/python tools/publication/run_publication_experiments.py \
  --config experiments/publication/publication_sweep_config.yaml \
  --dataset carla_2021 --method selective_topk --budget 10 \
  --post-evaluate-only
```

Use `--smoke` as well when targeting the isolated smoke directory.

See `LOCAL_VALIDATION.md` and `SERVER_RUN_GUIDE.md` for the complete safe
validation and server sequence.
