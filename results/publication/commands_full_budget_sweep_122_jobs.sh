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

# WARNING: This launches the full publication sweep. Run only after smoke and single full validation succeed.

# Job 1/122: carla_2021__full_communication__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method full_communication --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 2/122: carla_2021__selective_topk__b001__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 1 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 3/122: carla_2021__selective_topk__b002__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 2 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 4/122: carla_2021__selective_topk__b003__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 3 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 5/122: carla_2021__selective_topk__b004__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 4 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 6/122: carla_2021__selective_topk__b005__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 5 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 7/122: carla_2021__selective_topk__b006__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 6 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 8/122: carla_2021__selective_topk__b007__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 7 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 9/122: carla_2021__selective_topk__b008__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 8 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 10/122: carla_2021__selective_topk__b009__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 9 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 11/122: carla_2021__selective_topk__b010__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 10 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 12/122: carla_2021__selective_topk__b020__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 20 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 13/122: carla_2021__selective_topk__b025__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 25 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 14/122: carla_2021__selective_topk__b050__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 50 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 15/122: carla_2021__selective_topk__b075__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 75 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 16/122: carla_2021__selective_topk__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method selective_topk --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 17/122: carla_2021__snapshot_receiver_request__b001__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 1 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 18/122: carla_2021__snapshot_receiver_request__b002__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 2 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 19/122: carla_2021__snapshot_receiver_request__b003__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 3 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 20/122: carla_2021__snapshot_receiver_request__b004__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 4 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 21/122: carla_2021__snapshot_receiver_request__b005__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 5 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 22/122: carla_2021__snapshot_receiver_request__b006__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 6 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 23/122: carla_2021__snapshot_receiver_request__b007__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 7 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 24/122: carla_2021__snapshot_receiver_request__b008__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 8 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 25/122: carla_2021__snapshot_receiver_request__b009__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 9 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 26/122: carla_2021__snapshot_receiver_request__b010__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 10 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 27/122: carla_2021__snapshot_receiver_request__b020__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 20 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 28/122: carla_2021__snapshot_receiver_request__b025__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 25 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 29/122: carla_2021__snapshot_receiver_request__b050__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 50 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 30/122: carla_2021__snapshot_receiver_request__b075__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 75 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 31/122: carla_2021__snapshot_receiver_request__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method snapshot_receiver_request --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 32/122: carla_2021__temporal_receiver_request__b001__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 1 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 33/122: carla_2021__temporal_receiver_request__b002__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 2 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 34/122: carla_2021__temporal_receiver_request__b003__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 3 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 35/122: carla_2021__temporal_receiver_request__b004__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 4 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 36/122: carla_2021__temporal_receiver_request__b005__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 5 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 37/122: carla_2021__temporal_receiver_request__b006__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 6 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 38/122: carla_2021__temporal_receiver_request__b007__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 7 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 39/122: carla_2021__temporal_receiver_request__b008__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 8 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 40/122: carla_2021__temporal_receiver_request__b009__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 9 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 41/122: carla_2021__temporal_receiver_request__b010__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 10 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 42/122: carla_2021__temporal_receiver_request__b020__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 20 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 43/122: carla_2021__temporal_receiver_request__b025__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 25 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 44/122: carla_2021__temporal_receiver_request__b050__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 50 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 45/122: carla_2021__temporal_receiver_request__b075__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 75 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 46/122: carla_2021__temporal_receiver_request__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method temporal_receiver_request --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 47/122: carla_2021__learned_temporal_receiver_request__b001__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 1 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 48/122: carla_2021__learned_temporal_receiver_request__b002__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 2 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 49/122: carla_2021__learned_temporal_receiver_request__b003__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 3 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 50/122: carla_2021__learned_temporal_receiver_request__b004__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 4 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 51/122: carla_2021__learned_temporal_receiver_request__b005__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 5 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 52/122: carla_2021__learned_temporal_receiver_request__b006__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 6 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 53/122: carla_2021__learned_temporal_receiver_request__b007__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 7 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 54/122: carla_2021__learned_temporal_receiver_request__b008__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 8 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 55/122: carla_2021__learned_temporal_receiver_request__b009__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 9 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 56/122: carla_2021__learned_temporal_receiver_request__b010__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 10 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 57/122: carla_2021__learned_temporal_receiver_request__b020__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 20 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 58/122: carla_2021__learned_temporal_receiver_request__b025__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 25 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 59/122: carla_2021__learned_temporal_receiver_request__b050__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 50 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 60/122: carla_2021__learned_temporal_receiver_request__b075__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 75 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 61/122: carla_2021__learned_temporal_receiver_request__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset carla_2021 --method learned_temporal_receiver_request --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 62/122: culver_city__full_communication__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method full_communication --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 63/122: culver_city__selective_topk__b001__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 1 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 64/122: culver_city__selective_topk__b002__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 2 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 65/122: culver_city__selective_topk__b003__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 3 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 66/122: culver_city__selective_topk__b004__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 4 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 67/122: culver_city__selective_topk__b005__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 5 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 68/122: culver_city__selective_topk__b006__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 6 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 69/122: culver_city__selective_topk__b007__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 7 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 70/122: culver_city__selective_topk__b008__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 8 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 71/122: culver_city__selective_topk__b009__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 9 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 72/122: culver_city__selective_topk__b010__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 10 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 73/122: culver_city__selective_topk__b020__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 20 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 74/122: culver_city__selective_topk__b025__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 25 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 75/122: culver_city__selective_topk__b050__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 50 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 76/122: culver_city__selective_topk__b075__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 75 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 77/122: culver_city__selective_topk__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method selective_topk --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 78/122: culver_city__snapshot_receiver_request__b001__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 1 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 79/122: culver_city__snapshot_receiver_request__b002__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 2 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 80/122: culver_city__snapshot_receiver_request__b003__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 3 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 81/122: culver_city__snapshot_receiver_request__b004__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 4 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 82/122: culver_city__snapshot_receiver_request__b005__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 5 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 83/122: culver_city__snapshot_receiver_request__b006__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 6 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 84/122: culver_city__snapshot_receiver_request__b007__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 7 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 85/122: culver_city__snapshot_receiver_request__b008__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 8 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 86/122: culver_city__snapshot_receiver_request__b009__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 9 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 87/122: culver_city__snapshot_receiver_request__b010__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 10 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 88/122: culver_city__snapshot_receiver_request__b020__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 20 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 89/122: culver_city__snapshot_receiver_request__b025__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 25 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 90/122: culver_city__snapshot_receiver_request__b050__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 50 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 91/122: culver_city__snapshot_receiver_request__b075__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 75 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 92/122: culver_city__snapshot_receiver_request__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method snapshot_receiver_request --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 93/122: culver_city__temporal_receiver_request__b001__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 1 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 94/122: culver_city__temporal_receiver_request__b002__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 2 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 95/122: culver_city__temporal_receiver_request__b003__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 3 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 96/122: culver_city__temporal_receiver_request__b004__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 4 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 97/122: culver_city__temporal_receiver_request__b005__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 5 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 98/122: culver_city__temporal_receiver_request__b006__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 6 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 99/122: culver_city__temporal_receiver_request__b007__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 7 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 100/122: culver_city__temporal_receiver_request__b008__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 8 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 101/122: culver_city__temporal_receiver_request__b009__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 9 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 102/122: culver_city__temporal_receiver_request__b010__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 10 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 103/122: culver_city__temporal_receiver_request__b020__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 20 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 104/122: culver_city__temporal_receiver_request__b025__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 25 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 105/122: culver_city__temporal_receiver_request__b050__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 50 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 106/122: culver_city__temporal_receiver_request__b075__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 75 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 107/122: culver_city__temporal_receiver_request__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method temporal_receiver_request --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 108/122: culver_city__learned_temporal_receiver_request__b001__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 1 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 109/122: culver_city__learned_temporal_receiver_request__b002__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 2 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 110/122: culver_city__learned_temporal_receiver_request__b003__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 3 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 111/122: culver_city__learned_temporal_receiver_request__b004__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 4 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 112/122: culver_city__learned_temporal_receiver_request__b005__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 5 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 113/122: culver_city__learned_temporal_receiver_request__b006__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 6 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 114/122: culver_city__learned_temporal_receiver_request__b007__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 7 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 115/122: culver_city__learned_temporal_receiver_request__b008__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 8 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 116/122: culver_city__learned_temporal_receiver_request__b009__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 9 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 117/122: culver_city__learned_temporal_receiver_request__b010__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 10 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 118/122: culver_city__learned_temporal_receiver_request__b020__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 20 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 119/122: culver_city__learned_temporal_receiver_request__b025__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 25 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 120/122: culver_city__learned_temporal_receiver_request__b050__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 50 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 121/122: culver_city__learned_temporal_receiver_request__b075__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 75 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false

# Job 122/122: culver_city__learned_temporal_receiver_request__b100__s0__none_p0.000__mc000
./env/bin/python tools/publication/run_publication_experiments.py --config experiments/publication/publication_sweep_config.yaml --dataset culver_city --method learned_temporal_receiver_request --budget 100 --seed 0 --loss-type none --loss-probability 0 --monte-carlo-run 0 --execute --resume --overwrite false
