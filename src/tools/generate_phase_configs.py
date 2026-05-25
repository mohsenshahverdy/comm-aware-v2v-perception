import copy
import os

from src.hypes_yaml import yaml_utils


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def main():
    base_path = "src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml"
    out_dir = "src/hypes_yaml/communication_phases"
    os.makedirs(out_dir, exist_ok=True)
    base_cfg = yaml_utils.load_yaml(base_path)

    presets = {
        "phase0_baseline.yaml": {
            "model": {"args": {"communication": {"enabled": False, "phase": "phase0", "strategy": "none"}}}
        },
        "phase1_measurement.yaml": {
            "model": {"args": {"communication": {"enabled": True, "phase": "phase1", "strategy": "none"}}}
        },
        "phase2_random_drop.yaml": {
            "model": {"args": {"communication": {
                "enabled": True, "phase": "phase2", "strategy": "random_drop",
                "drop_random": {"keep_ratio": 0.1}
            }}}
        },
        "phase2_topk_energy.yaml": {
            "model": {"args": {"communication": {
                "enabled": True, "phase": "phase2", "strategy": "topk_energy",
                "topk_energy": {"keep_ratio": 0.1, "score_type": "l2"}
            }}}
        },
        "phase2_neighbor_packetloss.yaml": {
            "model": {"args": {"communication": {
                "enabled": True, "phase": "phase2", "strategy": "topk_energy",
                "neighbor_selection": {"mode": "nearest", "k": 2, "distance_metric": "euclidean"},
                "packet_loss": {"enabled": True, "loss_rate": 0.2, "unit": "cell"}
            }}}
        },
        "phase3_learnable_mask.yaml": {
            "model": {"args": {"communication": {
                "enabled": True, "phase": "phase3", "strategy": "learnable_mask",
                "learnable_mask": {"enabled": True, "mask_channels": 16, "sparsity_lambda": 0.01, "temperature": 1.0, "hard_mask": False}
            }}}
        },
        "phase4_repair.yaml": {
            "model": {"args": {"communication": {
                "enabled": True, "phase": "phase4", "strategy": "learnable_mask",
                "learnable_mask": {"enabled": True, "mask_channels": 16, "sparsity_lambda": 0.01, "temperature": 1.0, "hard_mask": False},
                "packet_loss": {"enabled": True, "loss_rate": 0.2, "unit": "cell"},
                "repair_network": {"enabled": True, "type": "conv", "hidden_dim": 128, "loss_weight": 0.1}
            }}}
        },
    }

    for fname, patch in presets.items():
        cfg = copy.deepcopy(base_cfg)
        cfg = deep_update(cfg, patch)
        yaml_utils.save_yaml(cfg, os.path.join(out_dir, fname))
        print("generated:", os.path.join(out_dir, fname))


if __name__ == "__main__":
    main()

