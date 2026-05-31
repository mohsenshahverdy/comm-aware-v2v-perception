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
    out_dir = "src/hypes_yaml/communication_approaches"
    os.makedirs(out_dir, exist_ok=True)
    base_cfg = yaml_utils.load_yaml(base_path)

    presets = {
        "baseline_full_communication.yaml": {"model": {"args": {"communication": {"enabled": False, "strategy": "none"}}}},
        "measurement_full_communication.yaml": {"model": {"args": {"communication": {"enabled": True, "strategy": "none"}}}},
        "stress_random_drop_10.yaml": {"model": {"args": {"communication": {"enabled": True, "strategy": "random_drop", "drop_random": {"keep_ratio": 0.1}}}}},
        "selective_topk_energy_10.yaml": {"model": {"args": {"communication": {"enabled": True, "strategy": "topk_energy", "topk_energy": {"keep_ratio": 0.1, "score_type": "l2"}}}}},
        "robustness_neighbor_packetloss_20.yaml": {"model": {"args": {"communication": {"enabled": True, "strategy": "topk_energy", "topk_energy": {"keep_ratio": 0.1, "score_type": "l2"}, "neighbor_selection": {"mode": "nearest", "k": 2, "distance_metric": "euclidean"}, "packet_loss": {"enabled": True, "loss_rate": 0.2, "unit": "cell"}}}}},
        "learned_mask_default.yaml": {"model": {"args": {"communication": {"enabled": True, "strategy": "learnable_mask", "learnable_mask": {"enabled": True, "mask_channels": 16, "sparsity_lambda": 0.01, "temperature": 1.0, "hard_mask": False}}}}},
        "repair_feature_reconstruction.yaml": {"model": {"args": {"communication": {"enabled": True, "strategy": "learnable_mask", "learnable_mask": {"enabled": True, "mask_channels": 16, "sparsity_lambda": 0.01, "temperature": 1.0, "hard_mask": False}, "packet_loss": {"enabled": True, "loss_rate": 0.2, "unit": "cell"}, "repair_network": {"enabled": True, "type": "conv", "hidden_dim": 128, "loss_weight": 0.1}}}}},
    }

    for fname, patch in presets.items():
        cfg = copy.deepcopy(base_cfg)
        cfg = deep_update(cfg, patch)
        cfg["communication_preset"] = os.path.splitext(fname)[0]
        cfg["root_dir"] = str(cfg.get("root_dir", "training_data/train")).replace("\\", "/")
        cfg["validate_dir"] = str(cfg.get("validate_dir", "validating_data/validate")).replace("\\", "/")
        yaml_utils.save_yaml(cfg, os.path.join(out_dir, fname))
        print("generated:", os.path.join(out_dir, fname))


if __name__ == "__main__":
    main()
