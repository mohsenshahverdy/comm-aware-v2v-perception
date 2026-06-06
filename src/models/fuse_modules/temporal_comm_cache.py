from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


CacheKey = Tuple[str, str, str]


@dataclass
class TemporalCacheEntry:
    context: torch.Tensor
    age: int = 0
    confidence: float = 1.0
    last_timestamp: Optional[str] = None
    update_count: int = 0


class ReceiverRequestTemporalCache:
    """
    Receiver-side temporal cache for future temporal receiver-request policies.

    The first intended cache payload is a collaborator context-energy map. This
    class is deliberately policy-agnostic so learned/object-aware cache types can
    reuse the same lifecycle and accounting logic later.
    """

    def __init__(self, momentum: float = 0.9, confidence_decay: float = 0.95):
        self.momentum = float(max(min(momentum, 1.0), 0.0))
        self.confidence_decay = float(max(min(confidence_decay, 1.0), 0.0))
        self._entries: Dict[CacheKey, TemporalCacheEntry] = {}

    @staticmethod
    def make_key(scenario_id, ego_id, collaborator_id) -> CacheKey:
        return (str(scenario_id), str(ego_id), str(collaborator_id))

    def reset_all(self):
        self._entries.clear()

    def reset_scenario(self, scenario_id):
        scenario_id = str(scenario_id)
        for key in list(self._entries.keys()):
            if key[0] == scenario_id:
                del self._entries[key]

    def reset_pair(self, scenario_id, ego_id, collaborator_id):
        self._entries.pop(self.make_key(scenario_id, ego_id, collaborator_id), None)

    def get(self, scenario_id, ego_id, collaborator_id) -> Optional[TemporalCacheEntry]:
        return self._entries.get(self.make_key(scenario_id, ego_id, collaborator_id))

    def update(
        self,
        scenario_id,
        ego_id,
        collaborator_id,
        context: torch.Tensor,
        timestamp: Optional[str] = None,
        refreshed: bool = True,
    ) -> TemporalCacheEntry:
        key = self.make_key(scenario_id, ego_id, collaborator_id)
        context = context.detach().clone()
        entry = self._entries.get(key)
        if entry is None:
            entry = TemporalCacheEntry(
                context=context,
                age=0,
                confidence=1.0 if refreshed else self.confidence_decay,
                last_timestamp=None if timestamp is None else str(timestamp),
                update_count=1,
            )
            self._entries[key] = entry
            return entry

        old_context = entry.context.to(device=context.device, dtype=context.dtype)
        entry.context = (self.momentum * old_context + (1.0 - self.momentum) * context).detach().clone()
        entry.age = 0 if refreshed else entry.age + 1
        entry.confidence = 1.0 if refreshed else entry.confidence * self.confidence_decay
        entry.last_timestamp = entry.last_timestamp if timestamp is None else str(timestamp)
        entry.update_count += 1
        return entry

    def increment_age(self, scenario_id=None):
        for key, entry in self._entries.items():
            if scenario_id is not None and key[0] != str(scenario_id):
                continue
            entry.age += 1
            entry.confidence *= self.confidence_decay

    @staticmethod
    def compute_novelty(
        current_context: torch.Tensor,
        cached_context: torch.Tensor,
        novelty_type: str = "absolute_diff",
        eps: float = 1e-6,
    ) -> torch.Tensor:
        cached_context = cached_context.to(device=current_context.device, dtype=current_context.dtype)
        diff = current_context - cached_context
        if novelty_type == "squared_diff":
            novelty = diff.pow(2)
        elif novelty_type == "normalized_diff":
            novelty = diff.abs() / (cached_context.abs() + eps)
        else:
            novelty = diff.abs()
        return novelty

    def export_metrics(self) -> dict:
        entries = list(self._entries.values())
        if not entries:
            return {
                "temporal_cache_entries": 0,
                "temporal_cache_age_mean": 0.0,
                "temporal_cache_confidence_mean": 0.0,
                "temporal_cache_update_count_mean": 0.0,
            }
        return {
            "temporal_cache_entries": len(entries),
            "temporal_cache_age_mean": float(sum(e.age for e in entries) / len(entries)),
            "temporal_cache_confidence_mean": float(sum(e.confidence for e in entries) / len(entries)),
            "temporal_cache_update_count_mean": float(sum(e.update_count for e in entries) / len(entries)),
        }

    def keys(self):
        return list(self._entries.keys())

    def __len__(self):
        return len(self._entries)
