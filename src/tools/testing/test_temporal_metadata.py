import torch

from src.models.fuse_modules.communication_policy import CommunicationPolicy
from src.utils.logging import get_logger


logger = get_logger("TestTemporalMetadata")


def main():
    features = torch.randn(5, 8, 4, 4)
    record_len = torch.tensor([3, 2])
    metadata = [
        {
            "sample_idx": 0,
            "scenario_index": 7,
            "scenario_id": "scenario_007",
            "timestamp": "000010",
            "frame_id": "000010",
            "ego_id": "ego_a",
            "cav_ids": ["ego_a", "cav_b", "cav_c"],
            "record_len": 3,
        },
        {
            "sample_idx": 1,
            "scenario_index": 8,
            "scenario_id": "scenario_008",
            "timestamp": "000001",
            "frame_id": "000001",
            "ego_id": "ego_d",
            "cav_ids": ["ego_d", "cav_e"],
            "record_len": 2,
        },
    ]

    policy = CommunicationPolicy(
        in_channels=features.shape[1],
        comm_cfg={
            "enabled": False,
            "strategy": "none",
            "drop_ego": False,
        },
    )
    out = policy(features, record_len, metadata=metadata)

    assert torch.equal(out.features, features)
    preview = policy._last_metadata_key_preview
    assert len(preview) == 3
    assert preview[0] == {
        "scenario_id": "scenario_007",
        "ego_id": "ego_a",
        "collaborator_id": "cav_b",
        "timestamp": "000010",
    }
    assert preview[-1] == {
        "scenario_id": "scenario_008",
        "ego_id": "ego_d",
        "collaborator_id": "cav_e",
        "timestamp": "000001",
    }

    logger.success("Temporal metadata policy handoff test passed", key_count=len(preview))


if __name__ == "__main__":
    main()
