#!/usr/bin/env bash
set -e

# Generated publication experiment commands. Run from the repository root.
# Required for CARLA/non-learned runs:
#   export PUBLICATION_CHECKPOINT_DIR="<checkpoint-directory>"
#   export CARLA_TRAIN_ROOT="<training-data-directory>"
#   export CARLA_2021_VALIDATE_ROOT="<carla-validation-directory>"
# Required for Culver runs:
#   export CULVER_VALIDATE_ROOT="<culver-validation-directory>"
# Required for learned temporal runs:
#   export LEARNED_PUBLICATION_CHECKPOINT_DIR="<learned-checkpoint-directory>"

# Selected publication jobs: 1

# Job 1/1: carla_2021__selective_topk__b010__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 10 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false --smoke
