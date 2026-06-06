import torch

from src.models.fuse_modules.temporal_comm_cache import ReceiverRequestTemporalCache
from src.utils.logging import get_logger


logger = get_logger("TestTemporalCache")


def main():
    cache = ReceiverRequestTemporalCache(momentum=0.5, confidence_decay=0.8)

    ctx0 = torch.zeros(1, 1, 2, 2)
    ctx1 = torch.ones(1, 1, 2, 2)

    entry = cache.update("scenario_a", "ego_1", "cav_2", ctx0, timestamp="000001")
    assert torch.equal(entry.context, ctx0)
    assert entry.age == 0
    assert entry.confidence == 1.0
    assert entry.update_count == 1

    entry = cache.update("scenario_a", "ego_1", "cav_2", ctx1, timestamp="000002")
    expected = torch.full_like(ctx1, 0.5)
    assert torch.allclose(entry.context, expected)
    assert entry.age == 0
    assert entry.update_count == 2

    cache.increment_age("scenario_a")
    aged = cache.get("scenario_a", "ego_1", "cav_2")
    assert aged.age == 1
    assert abs(aged.confidence - 0.8) < 1e-6

    cache.update("scenario_b", "ego_1", "cav_2", ctx1, timestamp="000001")
    assert len(cache) == 2
    cache.reset_scenario("scenario_a")
    assert cache.get("scenario_a", "ego_1", "cav_2") is None
    assert cache.get("scenario_b", "ego_1", "cav_2") is not None

    novelty_static = cache.compute_novelty(ctx1, ctx1).mean()
    novelty_changed = cache.compute_novelty(ctx1 * 3.0, ctx1).mean()
    assert novelty_changed > novelty_static

    cache.reset_pair("scenario_b", "ego_1", "cav_2")
    assert len(cache) == 0

    metrics = cache.export_metrics()
    assert metrics["temporal_cache_entries"] == 0

    logger.success("Temporal cache tests passed")


if __name__ == "__main__":
    main()
