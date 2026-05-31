import re


PUBLIC_LABELS = {
    "baseline_full_communication": "Full communication",
    "measurement_full_communication": "Measurement-only full communication",
    "stress_random_drop_10": "Stress random drop 10%",
    "stress_random_all_features": "Stress random all-features 10%",
    "selective_topk_energy_05": "Top-k energy 5%",
    "selective_topk_energy_10": "Top-k energy 10%",
    "selective_topk_energy_25": "Top-k energy 25%",
    "selective_topk_energy_50": "Top-k energy 50%",
    "selective_random_comm_only_05": "Random comm-only 5%",
    "selective_random_comm_only_10": "Random comm-only 10%",
    "selective_random_comm_only_25": "Random comm-only 25%",
    "selective_random_comm_only_50": "Random comm-only 50%",
    "robustness_neighbor_packetloss_20": "Neighbor+packet loss 20%",
    "robustness_packetloss_10": "Packet loss 10%",
    "robustness_packetloss_20": "Packet loss 20%",
    "robustness_packetloss_30": "Packet loss 30%",
    "robustness_packetloss_50": "Packet loss 50%",
    "learned_mask_default": "Learned mask default",
    "learned_mask_lam005_temp05_soft": "Learned mask λ=0.05 T=0.5 soft",
    "learned_mask_lam01_temp05_soft": "Learned mask λ=0.1 T=0.5 soft",
    "receiver_request_energy_topk_05": "Receiver-request energy top-k 5%",
    "receiver_request_energy_topk_10": "Receiver-request energy top-k 10%",
    "receiver_request_energy_topk_25": "Receiver-request energy top-k 25%",
    "receiver_request_energy_topk_50": "Receiver-request energy top-k 50%",
    "receiver_request_uncertainty_topk_10": "Receiver-request uncertainty top-k 10%",
    "receiver_request_visibility_topk": "Receiver-request visibility top-k (planned)",
    "receiver_request_learned": "Receiver-request learned (planned)",
    "receiver_request_learned_budget": "Receiver-request learned+budget (planned)",
    "receiver_request_warped": "Receiver-request warped alignment (planned)",
    "repair_feature_reconstruction": "Repair feature reconstruction",
}


def canonical_public_name(name):
    return name


def public_label(name):
    return PUBLIC_LABELS.get(name, name)


def parse_public_name(public_name):
    if not public_name:
        return None, None, None

    patterns = [
        (r"^(baseline)_(.+)$", lambda m: (m.group(1), m.group(2), "default")),
        (r"^(measurement)_(.+)$", lambda m: (m.group(1), m.group(2), "default")),
        (r"^(stress)_([^_]+(?:_[^_]+)*)_(\d{2})$", lambda m: (m.group(1), m.group(2), str(int(m.group(3))))),
        (r"^(stress)_([^_]+(?:_[^_]+)*)$", lambda m: (m.group(1), m.group(2), "default")),
        (r"^(selective)_(topk_energy|random_comm_only)_(\d{2})$", lambda m: (m.group(1), m.group(2), str(int(m.group(3))))),
        (r"^(robustness)_([^_]+(?:_[^_]+)*)_(\d{2})$", lambda m: (m.group(1), m.group(2), str(int(m.group(3))))),
        (r"^(learned)_(mask)_(.+)$", lambda m: (m.group(1), m.group(2), m.group(3))),
        (r"^(receiver_request)_(energy_topk|uncertainty_topk)_(\d{2})$", lambda m: (m.group(1), m.group(2), str(int(m.group(3))))),
        (r"^(repair)_(.+)$", lambda m: (m.group(1), m.group(2), "default")),
    ]

    for pattern, mapper in patterns:
        m = re.match(pattern, public_name)
        if m:
            return mapper(m)
    return None, None, None


def infer_public_name_from_run(run_name):
    if not run_name:
        return run_name

    stripped = re.sub(r"^(carla|culver)_", "", run_name)
    stripped = re.sub(r"_(train|test)$", "", stripped)
    return stripped
