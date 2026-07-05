# Local Publication Validation

Local validation does not require CARLA data, checkpoints, CUDA, or inference.
Run commands from the repository root using the repository environment.

## Environment Check

```bash
./env/bin/python tools/publication/check_publication_environment.py \
  --dataset carla_2021
```

On a machine without data, this prints a table of missing variables and exits
with status 1 without a traceback. Use `--dataset culver_city` for the Culver
validation path.

## Grid Validation

```bash
./env/bin/python tools/publication/run_publication_experiments.py \
  --config experiments/publication/publication_sweep_config.yaml \
  --dry-run --log-level INFO
```

Expected job count: 122.

Focused debug output:

```bash
./env/bin/python tools/publication/run_publication_experiments.py \
  --config experiments/publication/publication_sweep_config.yaml \
  --dataset carla_2021 --method selective_topk --budget 10 \
  --dry-run --debug
```

## Mock Validation

```bash
./env/bin/python tools/publication/run_local_mock_tests.py
```

The direct test entry point is provided because `pytest` is not guaranteed to
be installed. It validates config resolution, command exports, run artifacts,
failed-output handling, aggregation, and plotting in temporary directories.

## Logging

All publication scripts accept `--log-level` and `--debug` where applicable.
Default output is concise `INFO`; `--debug` enables detailed fields and
tracebacks. No real run directory is created by preflight, dry-run, export, or
mock validation.

