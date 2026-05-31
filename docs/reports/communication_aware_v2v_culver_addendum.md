# Addendum: Culver City Dataset Results

This addendum fills the missing "other dataset" part from the previous experiment summary. The other dataset was the **Culver City test split** (`test_culver_city`), not CARLA.

## Why it was missing from the first report

The previous summary focused mainly on:

- CARLA Phase 0/1/2/3 results
- Phase 3 training variants on CARLA
- Final Phase 3 learnable-mask conclusions

However, we also ran a Culver City Phase-2 sweep. Those results should be included because they show whether the communication policies generalize to another test split.

## Culver City evaluation setup

The Culver City test split used:

- split name: `culver`
- validation/test path: `/kaggle/input/data-all/test/test_culver_city/test_culver_city`
- scenarios: 4 folders
- samples: 550 frames
- checkpoint: `net_epoch43.pth`

The Culver evaluation is useful because it is harder than the CARLA test split: the full-communication baseline AP@0.7 is lower on Culver than on CARLA.

## Culver City Phase 0/1 baseline

| Run | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio |
|---|---:|---:|---:|---:|
| `culver_phase0_baseline` | 0.8786 | 0.8688 | 0.7721 | 1.0000 |
| `culver_phase1_measurement` | 0.8786 | 0.8689 | 0.7722 | 1.0000 |

Interpretation:

- Phase 1 matches Phase 0 almost exactly.
- This confirms that the measurement/logging layer does not change the model output.
- Culver baseline AP@0.7 is about 0.772, much lower than CARLA baseline AP@0.7 around 0.891, so Culver is a harder/domain-shifted evaluation.

## Culver City Phase 2: top-k energy sweep

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| `culver_phase2_topk_energy_05` | 0.05 | 0.7972 | 0.7917 | 0.7077 | 0.0435 | 492,806 |
| `culver_phase2_topk_energy_10` | 0.10 | 0.8315 | 0.8250 | 0.7356 | 0.0876 | 991,677 |
| `culver_phase2_topk_energy_25` | 0.25 | 0.8381 | 0.8312 | 0.7408 | 0.2178 | 2,466,352 |
| `culver_phase2_topk_energy_50` | 0.50 | 0.8558 | 0.8476 | 0.7543 | 0.4346 | 4,923,725 |

Interpretation:

- Top-k energy remains the strongest communication-efficient strategy on Culver.
- At about 8.8% communication, top-k 10% reaches AP@0.7 = 0.7356.
- Compared with the Culver full baseline AP@0.7 = 0.7721, this is only about 0.0365 AP drop while using less than 10% communication.
- At 43.5% communication, top-k 50% reaches AP@0.7 = 0.7543, very close to full baseline.

## Culver City Phase 2: random communication-only sweep

| Run | Keep ratio | AP@0.3 | AP@0.5 | AP@0.7 | Comm ratio | Bytes/frame |
|---|---:|---:|---:|---:|---:|---:|
| `culver_phase2_random_comm_only_05` | 0.05 | 0.6353 | 0.6287 | 0.5464 | 0.0444 | 502,332 |
| `culver_phase2_random_comm_only_10` | 0.10 | 0.6501 | 0.6433 | 0.5563 | 0.0847 | 959,630 |
| `culver_phase2_random_comm_only_25` | 0.25 | 0.7020 | 0.6934 | 0.5950 | 0.2109 | 2,397,063 |
| `culver_phase2_random_comm_only_50` | 0.50 | 0.7865 | 0.7768 | 0.6662 | 0.4324 | 4,904,506 |

Interpretation:

- Random communication-only masking is much worse than top-k energy at the same communication ratio.
- At around 8.5% communication, random masking gives AP@0.7 = 0.5563, while top-k 10% gives AP@0.7 = 0.7356.
- This strongly supports the main conclusion: **what is transmitted matters more than only how much is transmitted**.

## Culver City comparison summary

| Method | Comm ratio | AP@0.7 | Main meaning |
|---|---:|---:|---|
| Full baseline | 1.0000 | 0.7721 | Reference full communication |
| Top-k 5% | 0.0435 | 0.7077 | Very low communication, still useful |
| Top-k 10% | 0.0876 | 0.7356 | Best low-budget tradeoff |
| Top-k 25% | 0.2178 | 0.7408 | Slight AP gain over 10%, but much more communication |
| Top-k 50% | 0.4346 | 0.7543 | Close to full baseline |
| Random 10% | 0.0847 | 0.5563 | Same budget as top-k 10%, much worse AP |
| Random 50% | 0.4324 | 0.6662 | Same budget as top-k 50%, much worse AP |

## Culver City conclusion

Culver confirms the same pattern as CARLA:

1. Full communication gives the highest or near-highest AP.
2. Top-k energy is the strongest Phase-2 communication policy.
3. Random masking is a weak baseline.
4. Importance-aware feature selection is much better than random feature selection.
5. The 10% top-k setting is especially strong because it keeps communication below 10% while preserving much of the baseline AP.

## Relation to CARLA

On CARLA, top-k 10% reached AP@0.7 around 0.87 with about 9.5% communication.  
On Culver, top-k 10% reached AP@0.7 = 0.7356 with about 8.8% communication.

The absolute AP is lower on Culver, but the pattern is the same:

- top-k energy beats random masking
- top-k 10% is a strong low-communication point
- top-k 50% approaches full communication performance

## Phase 3 on Culver

We did not run the full Phase-3 learnable-mask variants on Culver. The Phase-3 training and tuning experiments were performed on CARLA.

Therefore, the report should clearly separate:

- **Culver City:** Phase 0/1/2 results only
- **CARLA:** Phase 0/1/2 plus Phase 3 learnable-mask experiments

## Final corrected thesis story

The corrected overall story is:

1. **Phase 2 is the strongest contribution.**  
   It works on both CARLA and Culver. Top-k energy gives large communication reduction with controlled AP loss.

2. **Culver validates generality.**  
   The same communication-efficiency pattern appears on the harder Culver split.

3. **Phase 3 is a CARLA prototype.**  
   Learnable masking reached very high AP on CARLA, but communication remained too high.

4. **Phase 4 should be future work.**  
   It should be tested as repair over fixed top-k 10% or packet-loss settings, not over the current high-communication Phase-3 mask.
