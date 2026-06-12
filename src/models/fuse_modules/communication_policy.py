from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.fuse_modules.learned_temporal_request import (
    DEFAULT_INPUT_MAPS,
    LearnedTemporalRequestHead,
    build_request_input,
    request_mask_entropy,
    request_prob_from_logits,
    topk_mask_from_prob,
)
from src.models.fuse_modules.temporal_comm_cache import ReceiverRequestTemporalCache
from src.utils.logging import get_logger


@dataclass
class CommOutput:
    features: torch.Tensor
    stats: Dict[str, float]
    aux: Dict[str, torch.Tensor]


class LearnableMaskHead(nn.Module):
    def __init__(self, in_channels: int, mask_channels: int = 16, temperature: float = 1.0):
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mask_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mask_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.sigmoid(logits / self.temperature)


class RepairNet(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CommunicationPolicy(nn.Module):
    def __init__(self, in_channels: int, comm_cfg: Optional[dict]):
        super().__init__()
        self.cfg = comm_cfg or {}
        self.logger = get_logger("CommunicationPolicy")
        self.enabled = bool(self.cfg.get("enabled", False))
        self.strategy = self.cfg.get("strategy", "none")
        self.seed = int(self.cfg.get("seed", 0))
        self.drop_ego = bool(self.cfg.get("drop_ego", False))

        learn_cfg = self.cfg.get("learnable_mask", {})
        self.learnable_enabled = bool(learn_cfg.get("enabled", False))
        self.hard_mask = bool(learn_cfg.get("hard_mask", False))
        self.sparsity_lambda = float(learn_cfg.get("sparsity_lambda", 0.0))
        self.mask_head = LearnableMaskHead(
            in_channels=in_channels,
            mask_channels=int(learn_cfg.get("mask_channels", 16)),
            temperature=float(learn_cfg.get("temperature", 1.0)),
        )

        repair_cfg = self.cfg.get("repair_network", {})
        self.repair_enabled = bool(repair_cfg.get("enabled", False))
        self.repair_weight = float(repair_cfg.get("loss_weight", 0.0))
        self.repair_net = RepairNet(in_channels=in_channels, hidden_dim=int(repair_cfg.get("hidden_dim", 128)))
        self.receiver_cfg = self.cfg.get("receiver_request", {})
        self._logged_once = False
        self._warned_alignment = False
        self._checked_alignment = False
        self._debug_saved_count = 0
        self._temporal_debug_saved_count = 0
        self._logged_metadata_once = False
        self._last_metadata_key_preview = []
        temporal_cfg = self.receiver_cfg.get("temporal", {}) if isinstance(self.receiver_cfg, dict) else {}
        self.temporal_cache = ReceiverRequestTemporalCache(
            momentum=float(temporal_cfg.get("cache_momentum", 0.9)),
            confidence_decay=float(temporal_cfg.get("cache_confidence_decay", 0.95)),
        )
        self._temporal_warned_missing_metadata = False
        self._temporal_scenario_bytes: Dict[str, float] = {}
        self._temporal_scenario_frames: Dict[str, int] = {}
        self._temporal_after_init_bytes: Dict[str, float] = {}
        self._temporal_after_init_frames: Dict[str, int] = {}
        self.learned_temporal_cfg = self.receiver_cfg.get("learned", {}) if isinstance(self.receiver_cfg, dict) else {}
        self.learned_temporal_input_maps = tuple(self.learned_temporal_cfg.get("input_maps", DEFAULT_INPUT_MAPS))
        self.learned_temporal_request_head = None
        if self._is_learned_temporal_receiver_request(self.receiver_cfg):
            self.learned_temporal_request_head = LearnedTemporalRequestHead(
                in_channels=len(self.learned_temporal_input_maps),
                hidden_channels=int(self.learned_temporal_cfg.get("hidden_channels", 16)),
            )
        self._learned_temporal_debug_saved_count = 0

        self._log_runtime_approach_status()
        self._validate_runtime_approach_status()
        self.logger.config(
            "Policy initialized",
            enabled=self.enabled,
            strategy=self.strategy,
            drop_ego=self.drop_ego,
            approach=self.cfg.get("metadata", {}).get("public_name", "unknown"),
        )

    def _runtime_approach_fields(self) -> Dict[str, object]:
        metadata = self.cfg.get("metadata", {}) if isinstance(self.cfg.get("metadata", {}), dict) else {}
        rr_cfg = self.cfg.get("receiver_request", {}) if isinstance(self.cfg.get("receiver_request", {}), dict) else {}
        rr_loss = rr_cfg.get("loss", {}) if isinstance(rr_cfg.get("loss", {}), dict) else {}
        implementation_status = metadata.get("implementation_status", rr_cfg.get("implementation_status", "implemented"))
        return {
            "public_name": metadata.get("public_name", "unknown"),
            "approach_family": metadata.get("approach_family", "default"),
            "approach_name": metadata.get("approach_name", "default"),
            "approach_setting": metadata.get("approach_setting", "default"),
            "implementation_status": implementation_status,
            "strategy": self.strategy,
            "trainable": bool(rr_cfg.get("trainable", False)),
            "loss_enabled": bool(rr_loss.get("enabled", False)),
            "receiver_request_enabled": bool(rr_cfg.get("enabled", False)),
            "communication_enabled": bool(self.enabled),
        }

    def _log_runtime_approach_status(self):
        status = self._runtime_approach_fields()
        self.logger.info(
            "Communication approach",
            public_name=status["public_name"],
            approach_family=status["approach_family"],
            approach_name=status["approach_name"],
            approach_setting=status["approach_setting"],
            implementation_status=status["implementation_status"],
            strategy=status["strategy"],
            trainable=status["trainable"],
            **{"loss.enabled": status["loss_enabled"]},
        )

    def _validate_runtime_approach_status(self):
        status = self._runtime_approach_fields()
        implementation_status = str(status["implementation_status"]).lower()
        is_planned = implementation_status in {"planned_placeholder", "planned"}
        strategy = str(status["strategy"]).lower()
        requested_policy = (
            strategy != "none"
            or bool(status["receiver_request_enabled"])
            or bool(status["trainable"])
            or bool(status["loss_enabled"])
        )
        if is_planned and requested_policy:
            raise ValueError(
                "Communication approach is planned_placeholder and cannot run yet: "
                f"public_name={status['public_name']} strategy={status['strategy']}"
            )
        if not bool(status["communication_enabled"]) and strategy != "none":
            raise ValueError(
                "Communication approach is disabled but a strategy was requested: "
                f"public_name={status['public_name']} strategy={status['strategy']} enabled={status['communication_enabled']}"
            )

    def update_debug_dir(self, debug_dir: Optional[str]):
        if not isinstance(self.receiver_cfg, dict):
            self.receiver_cfg = {}
        self.receiver_cfg["debug_dir"] = debug_dir
        temporal_cfg = self.receiver_cfg.setdefault("temporal", {})
        if isinstance(temporal_cfg, dict) and debug_dir:
            temporal_cfg["debug_dir"] = os.path.join(
                os.path.dirname(debug_dir),
                "temporal_receiver_request_debug",
            )
        learned_cfg = self.receiver_cfg.setdefault("learned", {})
        if isinstance(learned_cfg, dict) and debug_dir:
            learned_debug_cfg = learned_cfg.setdefault("debug", {})
            if isinstance(learned_debug_cfg, dict):
                learned_debug_cfg["debug_dir"] = os.path.join(
                    os.path.dirname(debug_dir),
                    "learned_temporal_receiver_request_debug",
                )

    def _check_pairwise_identity(self, pairwise_t_matrix: Optional[torch.Tensor], record_len: torch.Tensor, atol: float = 1e-3):
        if pairwise_t_matrix is None:
            return None, None
        try:
            bsz = pairwise_t_matrix.shape[0]
            max_dev = 0.0
            for b in range(bsz):
                n = int(record_len[b].item())
                if n <= 0:
                    continue
                mat = pairwise_t_matrix[b, :n, :n]
                eye = torch.eye(4, device=mat.device, dtype=mat.dtype).view(1, 1, 4, 4)
                dev = torch.max(torch.abs(mat - eye)).item()
                max_dev = max(max_dev, float(dev))
            return bool(max_dev <= atol), max_dev
        except Exception:
            return None, None

    def _maybe_save_request_maps(
        self,
        rr_cfg: dict,
        frame_idx: int,
        group_idx: int,
        collab_local_idx: int,
        ego_need: torch.Tensor,
        collab_context: torch.Tensor,
        score: torch.Tensor,
        mask: torch.Tensor,
    ):
        if not bool(rr_cfg.get("save_request_maps", False)):
            return
        debug_num = int(rr_cfg.get("debug_num_frames", 5))
        if self._debug_saved_count >= max(debug_num, 0):
            return
        debug_dir = rr_cfg.get("debug_dir", None)
        if not debug_dir:
            debug_dir = os.path.join(os.getcwd(), "receiver_request_debug")
        os.makedirs(debug_dir, exist_ok=True)
        save_name = os.path.join(
            debug_dir,
            f"rr_frame{frame_idx:05d}_group{group_idx:02d}_cav{collab_local_idx:02d}.npz",
        )
        np.savez_compressed(
            save_name,
            ego_need_map=ego_need.detach().cpu().numpy(),
            collaborator_context_map=collab_context.detach().cpu().numpy(),
            request_score_map=score.detach().cpu().numpy(),
            request_mask=mask.detach().cpu().numpy(),
        )
        self._debug_saved_count += 1
        self.logger.save("Receiver-request debug map saved", path=save_name)

    def _maybe_save_temporal_maps(
        self,
        temporal_cfg: dict,
        frame_idx: int,
        scenario_id: str,
        group_idx: int,
        collab_local_idx: int,
        ego_need: torch.Tensor,
        collab_context: torch.Tensor,
        previous_cache: torch.Tensor,
        novelty: torch.Tensor,
        temporal_factor: torch.Tensor,
        score: torch.Tensor,
        mask: torch.Tensor,
        cache_age: torch.Tensor,
        cache_confidence: torch.Tensor,
    ):
        if not bool(temporal_cfg.get("save_temporal_maps", False)):
            return
        debug_num = int(temporal_cfg.get("debug_num_frames", 5))
        if self._temporal_debug_saved_count >= max(debug_num, 0):
            return
        debug_dir = temporal_cfg.get("debug_dir", None)
        if not debug_dir:
            debug_dir = os.path.join(os.getcwd(), "temporal_receiver_request_debug")
        os.makedirs(debug_dir, exist_ok=True)
        save_name = os.path.join(
            debug_dir,
            f"trr_frame{frame_idx:05d}_scenario{scenario_id}_group{group_idx:02d}_cav{collab_local_idx:02d}.npz",
        )
        np.savez_compressed(
            save_name,
            ego_need_map=ego_need.detach().cpu().numpy(),
            collaborator_context_map=collab_context.detach().cpu().numpy(),
            previous_cache_map=previous_cache.detach().cpu().numpy(),
            novelty_map=novelty.detach().cpu().numpy(),
            temporal_factor_map=temporal_factor.detach().cpu().numpy(),
            request_score_map=score.detach().cpu().numpy(),
            request_mask=mask.detach().cpu().numpy(),
            cache_age_map=cache_age.detach().cpu().numpy(),
            cache_confidence_map=cache_confidence.detach().cpu().numpy(),
        )
        self._temporal_debug_saved_count += 1
        self.logger.save("Temporal receiver-request debug map saved", path=save_name)

    def _maybe_save_learned_temporal_maps(
        self,
        learned_cfg: dict,
        frame_idx: int,
        scenario_id: str,
        group_idx: int,
        collab_local_idx: int,
        logits: torch.Tensor,
        prob: torch.Tensor,
        mask: torch.Tensor,
        novelty: torch.Tensor,
        previous_cache: torch.Tensor,
        cache_age: torch.Tensor,
        cache_confidence: torch.Tensor,
    ):
        debug_cfg = learned_cfg.get("debug", {}) if isinstance(learned_cfg.get("debug", {}), dict) else {}
        if not bool(debug_cfg.get("save_learned_maps", False)):
            return
        debug_num = int(debug_cfg.get("debug_num_frames", learned_cfg.get("debug_num_frames", 5)))
        if self._learned_temporal_debug_saved_count >= max(debug_num, 0):
            return
        debug_dir = debug_cfg.get("debug_dir", learned_cfg.get("debug_dir", None))
        if not debug_dir:
            debug_dir = os.path.join(os.getcwd(), "learned_temporal_receiver_request_debug")
        os.makedirs(debug_dir, exist_ok=True)
        save_name = os.path.join(
            debug_dir,
            f"ltrr_frame{frame_idx:05d}_scenario{scenario_id}_group{group_idx:02d}_cav{collab_local_idx:02d}.npz",
        )
        np.savez_compressed(
            save_name,
            learned_request_logits=logits.detach().cpu().numpy(),
            learned_request_prob=prob.detach().cpu().numpy(),
            final_request_mask=mask.detach().cpu().numpy(),
            novelty_map=novelty.detach().cpu().numpy(),
            previous_cache_map=previous_cache.detach().cpu().numpy(),
            cache_age_map=cache_age.detach().cpu().numpy(),
            cache_confidence_map=cache_confidence.detach().cpu().numpy(),
        )
        self._learned_temporal_debug_saved_count += 1
        self.logger.save("Learned temporal receiver-request debug map saved", path=save_name)

    def _split_by_record_len(self, x: torch.Tensor, record_len: torch.Tensor):
        cum = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum[:-1].cpu())

    def _ego_indices(self, record_len: torch.Tensor, device: torch.device):
        idx = []
        start = 0
        for n in record_len.tolist():
            if n > 0:
                idx.append(start)
            start += int(n)
        if len(idx) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(idx, dtype=torch.long, device=device)

    def _energy_mask(self, x: torch.Tensor, keep_ratio: float, score_type: str = "l2") -> torch.Tensor:
        keep_ratio = float(max(min(keep_ratio, 1.0), 0.0))
        if x.shape[0] == 0:
            return torch.ones((0, 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        if keep_ratio >= 1.0:
            return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        if score_type == "l1_mean":
            score = x.abs().mean(dim=1, keepdim=True)
        else:
            score = torch.norm(x, p=2, dim=1, keepdim=True)
        flat = score.view(score.shape[0], -1)
        k = max(1, int(flat.shape[1] * keep_ratio))
        topk_vals, _ = torch.topk(flat, k=k, dim=1)
        thr = topk_vals[:, -1].view(-1, 1, 1, 1)
        return (score >= thr).to(x.dtype)

    def _random_mask(self, x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        keep_ratio = float(max(min(keep_ratio, 1.0), 0.0))
        if x.shape[0] == 0:
            return torch.ones((0, 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        if keep_ratio >= 1.0:
            return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        g = torch.Generator(device=x.device)
        g.manual_seed(self.seed)
        rnd = torch.rand((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, generator=g)
        return (rnd < keep_ratio).to(x.dtype)

    def _normalize_map(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if x.numel() == 0:
            return x
        flat = x.view(x.shape[0], -1)
        x_min = flat.min(dim=1)[0].view(-1, 1, 1, 1)
        x_max = flat.max(dim=1)[0].view(-1, 1, 1, 1)
        return (x - x_min) / (x_max - x_min + eps)

    def _feature_energy(self, feat: torch.Tensor, score_type: str = "l2", eps: float = 1e-6) -> torch.Tensor:
        if feat.shape[0] == 0:
            return torch.zeros((0, 1, feat.shape[2], feat.shape[3]), device=feat.device, dtype=feat.dtype)
        if score_type == "l1":
            return feat.abs().mean(dim=1, keepdim=True)
        if score_type == "max_abs":
            return feat.abs().amax(dim=1, keepdim=True)
        return torch.sqrt((feat ** 2).sum(dim=1, keepdim=True) + eps)

    def _ego_need_map(self, ego_feat: torch.Tensor, cfg: dict) -> torch.Tensor:
        need_type = cfg.get("ego_need_type", "inverse_energy")
        eps = float(cfg.get("ego_need_eps", 1e-6))
        score_type = cfg.get("collaborator_context_type", "l2")
        if need_type == "inverse_energy":
            ego_energy = self._feature_energy(ego_feat, score_type=score_type, eps=eps)
            ego_energy = self._normalize_map(ego_energy, eps=eps)
            need = 1.0 / (eps + ego_energy)
            return self._normalize_map(need, eps=eps)
        # TODO: implement uncertainty / learned need maps in Stage 2.
        fallback = self._feature_energy(ego_feat, score_type=score_type, eps=eps)
        return self._normalize_map(fallback, eps=eps)

    def _resize_context(self, context_map: torch.Tensor, resolution: str) -> torch.Tensor:
        if resolution == "half":
            return F.avg_pool2d(context_map, kernel_size=2, stride=2, ceil_mode=False)
        if resolution == "quarter":
            return F.avg_pool2d(context_map, kernel_size=4, stride=4, ceil_mode=False)
        return context_map

    def _collaborator_context_map(self, collab_feat: torch.Tensor, cfg: dict) -> torch.Tensor:
        ctx_type = cfg.get("collaborator_context_type", "l2")
        eps = float(cfg.get("ego_need_eps", 1e-6))
        context = self._feature_energy(collab_feat, score_type=ctx_type, eps=eps)
        return self._normalize_map(context, eps=eps)

    def _topk_mask(self, score: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        keep_ratio = float(max(min(keep_ratio, 1.0), 0.0))
        if score.shape[0] == 0:
            return torch.ones_like(score)
        if keep_ratio >= 1.0:
            return torch.ones_like(score)
        flat = score.view(score.shape[0], -1)
        if keep_ratio <= 0.0:
            k = 1
        else:
            k = max(1, int(flat.shape[1] * keep_ratio))
        topk_vals, _ = torch.topk(flat, k=k, dim=1)
        thr = topk_vals[:, -1].view(-1, 1, 1, 1)
        return (score >= thr).to(score.dtype)

    def _estimate_context_bytes(self, context_map: torch.Tensor, cfg: dict) -> float:
        if context_map.shape[0] == 0:
            return 0.0
        if not bool(cfg.get("count_context_overhead", True)):
            return 0.0
        bits = int(cfg.get("context_quantization_bits", 32))
        bits = max(bits, 1)
        values = int(context_map.shape[0] * context_map.shape[2] * context_map.shape[3])
        return float(values * bits / 8.0)

    def _estimate_mask_metadata_bytes(self, mask: torch.Tensor, cfg: dict) -> float:
        if mask.shape[0] == 0:
            return 0.0
        if not bool(cfg.get("count_mask_metadata", True)):
            return 0.0
        enc = cfg.get("metadata_encoding", "dense_binary")
        if enc == "none":
            return 0.0
        if enc == "sparse_indices":
            active = int((mask > 0).sum().item())
            # uint32 indices
            return float(active * 4)
        # dense_binary
        bits = int(mask.shape[0] * mask.shape[2] * mask.shape[3])
        return float(bits / 8.0)

    def _metadata_key_preview(self, metadata, record_len: torch.Tensor):
        if metadata is None:
            return []
        samples = metadata if isinstance(metadata, list) else [metadata]
        preview = []
        for sample_idx, sample_meta in enumerate(samples):
            if not isinstance(sample_meta, dict):
                continue
            cav_ids = sample_meta.get("cav_ids", [])
            if not isinstance(cav_ids, (list, tuple)):
                cav_ids = []
            if record_len is not None and sample_idx < len(record_len):
                n = int(record_len[sample_idx].detach().cpu().item()) if isinstance(record_len[sample_idx], torch.Tensor) else int(record_len[sample_idx])
            else:
                n = len(cav_ids)
            ego_id = str(sample_meta.get("ego_id", cav_ids[0] if cav_ids else "ego"))
            for local_i in range(1, min(n, len(cav_ids))):
                preview.append({
                    "scenario_id": str(sample_meta.get("scenario_id", sample_meta.get("scenario_index", "unknown"))),
                    "ego_id": ego_id,
                    "collaborator_id": str(cav_ids[local_i]),
                    "timestamp": str(sample_meta.get("timestamp", sample_meta.get("frame_id", "unknown"))),
                })
        return preview

    def _sample_metadata(self, metadata, sample_idx: int):
        if isinstance(metadata, list) and sample_idx < len(metadata):
            return metadata[sample_idx] if isinstance(metadata[sample_idx], dict) else {}
        if isinstance(metadata, dict) and sample_idx == 0:
            return metadata
        return {}

    def _temporal_cache_key(self, metadata, sample_idx: int, group_idx: int, local_i: int):
        sample_meta = self._sample_metadata(metadata, sample_idx)
        cav_ids = sample_meta.get("cav_ids", [])
        if not isinstance(cav_ids, (list, tuple)):
            cav_ids = []
        scenario_id = str(sample_meta.get("scenario_id", sample_meta.get("scenario_index", f"sample_{sample_idx}")))
        ego_id = str(sample_meta.get("ego_id", cav_ids[0] if cav_ids else f"ego_{group_idx}"))
        if local_i < len(cav_ids):
            collaborator_id = str(cav_ids[local_i])
        else:
            collaborator_id = f"collab_{group_idx}_{local_i}"
        timestamp = str(sample_meta.get("timestamp", sample_meta.get("frame_id", sample_meta.get("sample_idx", "unknown"))))
        missing = not bool(sample_meta) or not cav_ids
        if missing and not self._temporal_warned_missing_metadata:
            self.logger.warn(
                "Temporal metadata missing; using local fallback cache keys",
                scenario_id=scenario_id,
                ego_id=ego_id,
                collaborator_id=collaborator_id,
            )
            self._temporal_warned_missing_metadata = True
        return scenario_id, ego_id, collaborator_id, timestamp

    def _metadata_scenarios(self, metadata):
        if metadata is None:
            return ["unknown"]
        samples = metadata if isinstance(metadata, list) else [metadata]
        scenarios = []
        for i, sample_meta in enumerate(samples):
            if isinstance(sample_meta, dict):
                scenarios.append(str(sample_meta.get("scenario_id", sample_meta.get("scenario_index", f"sample_{i}"))))
        return scenarios or ["unknown"]

    def _log_metadata_preview(self, metadata, record_len: torch.Tensor):
        preview = self._metadata_key_preview(metadata, record_len)
        self._last_metadata_key_preview = preview
        if preview and not self._logged_metadata_once:
            first = preview[0]
            self.logger.info(
                "Temporal metadata received",
                scenario_id=first["scenario_id"],
                ego_id=first["ego_id"],
                collaborator_id=first["collaborator_id"],
                timestamp=first["timestamp"],
                key_count=len(preview),
            )
            self._logged_metadata_once = True

    def _select_neighbors(
        self,
        x: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        cfg = self.cfg.get("neighbor_selection", {})
        mode = cfg.get("mode", "all")
        k = int(cfg.get("k", 0))
        selected_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        if mode == "all" or k <= 0:
            return x, 1.0, selected_mask

        groups = self._split_by_record_len(x, record_len)
        new_groups = []
        active_collab = 0
        total_collab = 0
        start = 0
        for b, grp in enumerate(groups):
            n = grp.shape[0]
            total_collab += max(n - 1, 0)
            if n <= 1:
                new_groups.append(grp)
                if n == 1:
                    selected_mask[start] = True
                start += n
                continue

            keep = min(k + 1, n)  # include ego
            idx = [0]
            if mode == "nearest" and pairwise_t_matrix is not None:
                t = pairwise_t_matrix[b, 0, :n, :2, 3]
                dist = torch.norm(t, dim=1)
                _, order = torch.sort(dist)
                idx = order[:keep].tolist()
            elif mode == "topk_importance":
                score = torch.norm(grp, p=2, dim=1).mean(dim=(1, 2))
                _, order = torch.sort(score, descending=True)
                idx = order[:keep].tolist()
            else:
                idx = list(range(keep))
            if 0 not in idx:
                idx[0] = 0

            collab_kept = len([i for i in idx if i != 0])
            active_collab += collab_kept

            mask = torch.zeros((n, 1, 1, 1), device=grp.device, dtype=grp.dtype)
            mask[idx] = 1.0
            new_groups.append(grp * mask)
            group_selected = torch.zeros(n, dtype=torch.bool, device=x.device)
            group_selected[idx] = True
            selected_mask[start:start + n] = group_selected
            start += n

        ratio = float(active_collab) / float(max(total_collab, 1))
        return torch.cat(new_groups, dim=0), ratio, selected_mask

    def _is_temporal_receiver_request(self, rr_cfg: dict) -> bool:
        temporal_cfg = rr_cfg.get("temporal", {}) if isinstance(rr_cfg.get("temporal", {}), dict) else {}
        variant = str(rr_cfg.get("strategy_variant", "")).lower()
        return bool(temporal_cfg.get("enabled", False)) or variant.startswith("temporal_")

    def _is_learned_temporal_receiver_request(self, rr_cfg: dict) -> bool:
        if not isinstance(rr_cfg, dict):
            return False
        temporal_cfg = rr_cfg.get("temporal", {}) if isinstance(rr_cfg.get("temporal", {}), dict) else {}
        learned_cfg = rr_cfg.get("learned", {}) if isinstance(rr_cfg.get("learned", {}), dict) else {}
        variant = str(rr_cfg.get("strategy_variant", "")).lower()
        return (
            variant in {"learned_temporal", "learned_temporal_topk"}
            or variant.startswith("learned_temporal_")
            or (bool(temporal_cfg.get("enabled", False)) and bool(learned_cfg.get("enabled", False)))
        )

    def _apply_learned_temporal_receiver_request(
        self,
        x_mod: torch.Tensor,
        record_len: torch.Tensor,
        tx_mask: torch.Tensor,
        rr_cfg: dict,
        metadata,
        bytes_per_value: float,
    ):
        temporal_cfg = rr_cfg.get("temporal", {}) if isinstance(rr_cfg.get("temporal", {}), dict) else {}
        learned_cfg = rr_cfg.get("learned", {}) if isinstance(rr_cfg.get("learned", {}), dict) else {}
        if self.learned_temporal_request_head is None:
            raise RuntimeError("learned temporal receiver-request is enabled but the request head was not initialized")

        keep_ratio = float(max(min(float(rr_cfg.get("keep_ratio", learned_cfg.get("keep_ratio", 1.0))), 1.0), 0.0))
        target_budget = float(learned_cfg.get("target_budget", learned_cfg.get("loss", {}).get("target_budget", keep_ratio)))
        use_soft_mask_train = bool(learned_cfg.get("use_soft_mask_train", True))
        straight_through = bool(learned_cfg.get("straight_through", False))
        use_hard_topk_inference = bool(learned_cfg.get("use_hard_topk_inference", True))

        for scenario_id in set(self._metadata_scenarios(metadata)):
            self.temporal_cache.increment_age(scenario_id)

        groups = self._split_by_record_len(x_mod, record_len)
        out_groups = []
        start = 0

        global_context_bytes = 0.0
        global_metadata_bytes = 0.0
        global_selected_cells = 0.0
        global_total_cells = 0.0
        after_init_feature_bytes = 0.0
        after_init_context_bytes = 0.0
        after_init_metadata_bytes = 0.0
        after_init_denominator = 0.0
        novelty_vals = []
        age_vals = []
        prob_vals = []
        prob_std_vals = []
        entropy_vals = []
        cache_hits = 0
        processed = 0
        init_count = 0

        aux_probs = []
        aux_logits = []
        aux_masks = []

        for group_idx, grp in enumerate(groups):
            n = grp.shape[0]
            if n <= 1:
                out_groups.append(grp)
                start += n
                continue

            # Keep the learned path autograd-safe: do not assign into views of
            # grp/grp_out because the request head backpropagates through
            # collaborator slices used to build the request maps.
            group_out_parts = [grp[0:1]]
            ego_local = grp[0:1]
            ego_need = self._ego_need_map(ego_local, rr_cfg)
            group_tx = tx_mask[start:start + n]

            for local_i in range(1, n):
                collab = grp[local_i:local_i + 1]
                if not bool(group_tx[local_i].item()):
                    group_out_parts.append(collab)
                    continue

                scenario_id, ego_id, collaborator_id, timestamp = self._temporal_cache_key(
                    metadata,
                    sample_idx=group_idx,
                    group_idx=group_idx,
                    local_i=local_i,
                )
                context_full = self._collaborator_context_map(collab, rr_cfg)
                context_small = self._resize_context(context_full, rr_cfg.get("context_resolution", "full"))
                context_for_score = context_small
                if context_for_score.shape[2:] != ego_need.shape[2:]:
                    context_for_score = F.interpolate(context_for_score, size=ego_need.shape[2:], mode="nearest")
                context_for_score = self._normalize_map(context_for_score)
                need_map = self._normalize_map(ego_need)

                entry = self.temporal_cache.get(scenario_id, ego_id, collaborator_id)
                cache_hit = entry is not None
                cache_hits += int(cache_hit)
                processed += 1

                if entry is None:
                    previous_cache = torch.zeros_like(context_for_score)
                    age = 0.0
                    confidence = 0.0
                    update_count = 0
                else:
                    previous_cache = entry.context.to(device=context_for_score.device, dtype=context_for_score.dtype)
                    if previous_cache.shape[2:] != context_for_score.shape[2:]:
                        previous_cache = F.interpolate(previous_cache, size=context_for_score.shape[2:], mode="nearest")
                    age = float(entry.age)
                    confidence = float(entry.confidence)
                    update_count = int(entry.update_count)

                novelty = self.temporal_cache.compute_novelty(
                    context_for_score,
                    previous_cache,
                    novelty_type=temporal_cfg.get("novelty_type", "absolute_diff"),
                )
                novelty = self._normalize_map(novelty)
                age_map = torch.full_like(context_for_score, age)
                confidence_map = torch.full_like(context_for_score, confidence)
                learned_maps = {
                    "ego_need": need_map,
                    "collaborator_context": context_for_score,
                    "previous_cache": previous_cache,
                    "novelty": novelty,
                    "cache_age": age_map,
                    "cache_confidence": confidence_map,
                }
                request_input = build_request_input(learned_maps, self.learned_temporal_input_maps)
                logits = self.learned_temporal_request_head(request_input)
                prob = request_prob_from_logits(logits)
                hard_mask = topk_mask_from_prob(prob, keep_ratio)

                if self.training and use_soft_mask_train:
                    if straight_through:
                        mask = hard_mask + (prob - prob.detach())
                    else:
                        mask = prob
                elif use_hard_topk_inference:
                    mask = hard_mask
                else:
                    mask = prob

                group_out_parts.append(collab * mask)
                self._maybe_save_learned_temporal_maps(
                    learned_cfg=learned_cfg,
                    frame_idx=int(self._learned_temporal_debug_saved_count),
                    scenario_id=scenario_id,
                    group_idx=group_idx,
                    collab_local_idx=local_i,
                    logits=logits,
                    prob=prob,
                    mask=mask,
                    novelty=novelty,
                    previous_cache=previous_cache,
                    cache_age=age_map,
                    cache_confidence=confidence_map,
                )

                selected_cells = float(mask.detach().sum().cpu().item())
                total_cells = float(mask.numel())
                context_b = self._estimate_context_bytes(context_small, rr_cfg)
                metadata_b = self._estimate_mask_metadata_bytes(hard_mask if self.training and use_soft_mask_train else mask, rr_cfg)
                feature_b = selected_cells * float(x_mod.shape[1]) * float(bytes_per_value)

                global_selected_cells += selected_cells
                global_total_cells += total_cells
                global_context_bytes += context_b
                global_metadata_bytes += metadata_b
                novelty_vals.append(float(novelty.mean().detach().cpu().item()))
                age_vals.append(age)
                prob_vals.append(float(prob.mean().detach().cpu().item()))
                prob_std_vals.append(float(prob.std(unbiased=False).detach().cpu().item()))
                entropy_vals.append(float(request_mask_entropy(prob).detach().cpu().item()))

                if update_count > 0:
                    after_init_feature_bytes += feature_b
                    after_init_context_bytes += context_b
                    after_init_metadata_bytes += metadata_b
                    after_init_denominator += total_cells * float(x_mod.shape[1]) * float(bytes_per_value)
                else:
                    init_count += 1

                aux_probs.append(prob)
                aux_logits.append(logits)
                aux_masks.append(mask)

                self.temporal_cache.update(
                    scenario_id,
                    ego_id,
                    collaborator_id,
                    context_for_score,
                    timestamp=timestamp,
                    refreshed=True,
                )

            out_groups.append(torch.cat(group_out_parts, dim=0))
            start += n

        x_out = torch.cat(out_groups, dim=0)
        after_init_total = after_init_feature_bytes + after_init_context_bytes + after_init_metadata_bytes
        if aux_probs:
            prob_cat = torch.cat(aux_probs, dim=0)
            logits_cat = torch.cat(aux_logits, dim=0)
            mask_cat = torch.cat(aux_masks, dim=0)
            prob_mean_tensor = prob_cat.mean()
            effective_keep_tensor = mask_cat.mean()
            budget_target_tensor = torch.as_tensor(target_budget, device=prob_cat.device, dtype=prob_cat.dtype)
            budget_error_tensor = effective_keep_tensor - budget_target_tensor
            entropy_tensor = request_mask_entropy(prob_cat)
        else:
            prob_cat = x_mod.new_empty((0, 1, x_mod.shape[2], x_mod.shape[3]))
            logits_cat = x_mod.new_empty((0, 1, x_mod.shape[2], x_mod.shape[3]))
            mask_cat = x_mod.new_empty((0, 1, x_mod.shape[2], x_mod.shape[3]))
            prob_mean_tensor = x_mod.new_tensor(0.0)
            effective_keep_tensor = x_mod.new_tensor(0.0)
            budget_target_tensor = x_mod.new_tensor(target_budget)
            budget_error_tensor = effective_keep_tensor - budget_target_tensor
            entropy_tensor = x_mod.new_tensor(0.0)

        learned_stats = {
            "active_ratio": float(global_selected_cells / global_total_cells) if global_total_cells > 0 else 0.0,
            "context_bytes": float(global_context_bytes),
            "metadata_bytes": float(global_metadata_bytes),
            "temporal_novelty_mean": float(sum(novelty_vals) / max(len(novelty_vals), 1)),
            "temporal_cache_age_mean": float(sum(age_vals) / max(len(age_vals), 1)),
            "temporal_cache_hit_ratio": float(cache_hits / max(processed, 1)),
            "temporal_refresh_ratio": 0.0,
            "temporal_init_frame_ratio": float(init_count / max(processed, 1)),
            "total_bytes_per_frame_after_init": float(after_init_total),
            "total_normalized_ratio_after_init": float(after_init_total / after_init_denominator) if after_init_denominator > 0 else 0.0,
            "learned_request_prob_mean": float(sum(prob_vals) / max(len(prob_vals), 1)),
            "learned_request_prob_std": float(sum(prob_std_vals) / max(len(prob_std_vals), 1)),
            "learned_effective_keep_ratio": float(global_selected_cells / global_total_cells) if global_total_cells > 0 else 0.0,
            "learned_budget_target": float(target_budget),
            "learned_budget_error": float((global_selected_cells / global_total_cells) - target_budget) if global_total_cells > 0 else float(-target_budget),
            "learned_mask_entropy": float(sum(entropy_vals) / max(len(entropy_vals), 1)),
        }
        learned_stats.update(self.temporal_cache.export_metrics())
        learned_aux = {
            "learned_request_prob": prob_cat,
            "learned_request_logits": logits_cat,
            "learned_request_mask": mask_cat,
            "learned_request_prob_mean_tensor": prob_mean_tensor,
            "learned_effective_keep_ratio_tensor": effective_keep_tensor,
            "learned_budget_target_tensor": budget_target_tensor,
            "learned_budget_error_tensor": budget_error_tensor,
            "learned_mask_entropy_tensor": entropy_tensor,
        }
        return x_out, learned_stats, learned_aux

    def _apply_temporal_receiver_request(
        self,
        x_mod: torch.Tensor,
        record_len: torch.Tensor,
        tx_mask: torch.Tensor,
        rr_cfg: dict,
        metadata,
        bytes_per_value: float,
    ):
        temporal_cfg = rr_cfg.get("temporal", {}) if isinstance(rr_cfg.get("temporal", {}), dict) else {}
        keep_ratio = float(max(min(float(rr_cfg.get("keep_ratio", 1.0)), 1.0), 0.0))
        init_mode = str(temporal_cfg.get("init_mode", "full_request"))
        init_frames = max(int(temporal_cfg.get("init_frames", 1)), 0)
        init_keep_ratio = float(max(min(float(temporal_cfg.get("init_keep_ratio", 1.0)), 1.0), 0.0))
        novelty_weight = float(temporal_cfg.get("novelty_weight", 1.0))
        age_weight = float(temporal_cfg.get("age_weight", 0.05))
        confidence_weight = float(temporal_cfg.get("cache_confidence_weight", 0.5))
        min_factor = float(temporal_cfg.get("min_temporal_factor", 0.25))
        max_factor = float(temporal_cfg.get("max_temporal_factor", 3.0))
        periodic_refresh = bool(temporal_cfg.get("periodic_refresh_enabled", False))
        refresh_interval = max(int(temporal_cfg.get("periodic_refresh_interval", 10)), 1)
        refresh_keep_ratio = float(max(min(float(temporal_cfg.get("periodic_refresh_keep_ratio", 0.25)), 1.0), 0.0))

        for scenario_id in set(self._metadata_scenarios(metadata)):
            self.temporal_cache.increment_age(scenario_id)

        groups = self._split_by_record_len(x_mod, record_len)
        out_groups = []
        start = 0

        global_context_bytes = 0.0
        global_metadata_bytes = 0.0
        global_selected_cells = 0.0
        global_total_cells = 0.0
        after_init_feature_bytes = 0.0
        after_init_context_bytes = 0.0
        after_init_metadata_bytes = 0.0
        after_init_denominator = 0.0
        novelty_vals = []
        age_vals = []
        cache_hits = 0
        processed = 0
        refresh_count = 0
        init_count = 0

        for group_idx, grp in enumerate(groups):
            n = grp.shape[0]
            grp_out = grp.clone()
            if n <= 1:
                out_groups.append(grp_out)
                start += n
                continue

            ego_local = grp_out[0:1]
            ego_need = self._ego_need_map(ego_local, rr_cfg)
            group_tx = tx_mask[start:start + n]

            for local_i in range(1, n):
                if not bool(group_tx[local_i].item()):
                    continue

                scenario_id, ego_id, collaborator_id, timestamp = self._temporal_cache_key(
                    metadata,
                    sample_idx=group_idx,
                    group_idx=group_idx,
                    local_i=local_i,
                )
                collab = grp_out[local_i:local_i + 1]
                context_full = self._collaborator_context_map(collab, rr_cfg)
                context_small = self._resize_context(context_full, rr_cfg.get("context_resolution", "full"))
                context_for_score = context_small
                if context_for_score.shape[2:] != ego_need.shape[2:]:
                    context_for_score = F.interpolate(context_for_score, size=ego_need.shape[2:], mode="nearest")
                context_for_score = self._normalize_map(context_for_score)
                need_map = self._normalize_map(ego_need)

                entry = self.temporal_cache.get(scenario_id, ego_id, collaborator_id)
                cache_hit = entry is not None
                cache_hits += int(cache_hit)
                processed += 1

                if entry is None:
                    previous_cache = torch.zeros_like(context_for_score)
                    age = 0.0
                    confidence = 0.0
                    update_count = 0
                else:
                    previous_cache = entry.context.to(device=context_for_score.device, dtype=context_for_score.dtype)
                    if previous_cache.shape[2:] != context_for_score.shape[2:]:
                        previous_cache = F.interpolate(previous_cache, size=context_for_score.shape[2:], mode="nearest")
                    age = float(entry.age)
                    confidence = float(entry.confidence)
                    update_count = int(entry.update_count)

                novelty = self.temporal_cache.compute_novelty(
                    context_for_score,
                    previous_cache,
                    novelty_type=temporal_cfg.get("novelty_type", "absolute_diff"),
                )
                novelty = self._normalize_map(novelty)
                age_map = torch.full_like(context_for_score, age)
                confidence_map = torch.full_like(context_for_score, confidence)
                temporal_factor = 1.0 + novelty_weight * novelty + age_weight * age_map - confidence_weight * confidence_map
                temporal_factor = torch.clamp(temporal_factor, min=min_factor, max=max_factor)
                score = self._normalize_map(need_map * context_for_score * temporal_factor)

                is_init = entry is None or update_count < init_frames
                is_refresh = (not is_init) and periodic_refresh and update_count > 0 and (update_count % refresh_interval == 0)
                if is_init:
                    init_count += 1
                    if init_mode == "full_request":
                        mask = torch.ones_like(score)
                    elif init_mode == "high_ratio":
                        mask = self._topk_mask(score, init_keep_ratio)
                    else:
                        mask = self._topk_mask(score, keep_ratio)
                elif is_refresh:
                    refresh_count += 1
                    mask = self._topk_mask(score, refresh_keep_ratio)
                else:
                    mask = self._topk_mask(score, keep_ratio)

                grp_out[local_i:local_i + 1] = collab * mask
                self._maybe_save_temporal_maps(
                    temporal_cfg=temporal_cfg,
                    frame_idx=int(self._temporal_debug_saved_count),
                    scenario_id=scenario_id,
                    group_idx=group_idx,
                    collab_local_idx=local_i,
                    ego_need=need_map,
                    collab_context=context_for_score,
                    previous_cache=previous_cache,
                    novelty=novelty,
                    temporal_factor=temporal_factor,
                    score=score,
                    mask=mask,
                    cache_age=age_map,
                    cache_confidence=confidence_map,
                )

                selected_cells = float(mask.sum().detach().cpu().item())
                total_cells = float(mask.numel())
                context_b = self._estimate_context_bytes(context_small, rr_cfg)
                metadata_b = self._estimate_mask_metadata_bytes(mask, rr_cfg)
                feature_b = selected_cells * float(x_mod.shape[1]) * float(bytes_per_value)

                global_selected_cells += selected_cells
                global_total_cells += total_cells
                global_context_bytes += context_b
                global_metadata_bytes += metadata_b
                novelty_vals.append(float(novelty.mean().detach().cpu().item()))
                age_vals.append(age)

                if not is_init:
                    after_init_feature_bytes += feature_b
                    after_init_context_bytes += context_b
                    after_init_metadata_bytes += metadata_b
                    after_init_denominator += total_cells * float(x_mod.shape[1]) * float(bytes_per_value)

                self.temporal_cache.update(
                    scenario_id,
                    ego_id,
                    collaborator_id,
                    context_for_score,
                    timestamp=timestamp,
                    refreshed=True,
                )

            out_groups.append(grp_out)
            start += n

        x_out = torch.cat(out_groups, dim=0)
        after_init_total = after_init_feature_bytes + after_init_context_bytes + after_init_metadata_bytes
        temporal_stats = {
            "active_ratio": float(global_selected_cells / global_total_cells) if global_total_cells > 0 else 0.0,
            "context_bytes": float(global_context_bytes),
            "metadata_bytes": float(global_metadata_bytes),
            "temporal_novelty_mean": float(sum(novelty_vals) / max(len(novelty_vals), 1)),
            "temporal_cache_age_mean": float(sum(age_vals) / max(len(age_vals), 1)),
            "temporal_cache_hit_ratio": float(cache_hits / max(processed, 1)),
            "temporal_refresh_ratio": float(refresh_count / max(processed, 1)),
            "temporal_init_frame_ratio": float(init_count / max(processed, 1)),
            "total_bytes_per_frame_after_init": float(after_init_total),
            "total_normalized_ratio_after_init": float(after_init_total / after_init_denominator) if after_init_denominator > 0 else 0.0,
        }
        temporal_stats.update(self.temporal_cache.export_metrics())
        return x_out, temporal_stats

    def forward(
        self,
        x: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        metadata=None,
    ) -> CommOutput:
        self._log_metadata_preview(metadata, record_len)
        stats = {
            "active_ratio": 1.0,
            "active_neighbors_ratio": 1.0,
            "packet_loss_rate": 0.0,
            "bytes_per_frame": 0.0,
            "feature_bytes_per_frame": 0.0,
            "context_bytes_per_frame": 0.0,
            "metadata_bytes_per_frame": 0.0,
            "total_bytes_per_frame": 0.0,
            "normalized_ratio": 1.0,
            "feature_normalized_ratio": 1.0,
            "context_normalized_ratio": 0.0,
            "metadata_normalized_ratio": 0.0,
            "total_normalized_ratio": 1.0,
            "receiver_request_keep_ratio": 1.0,
            "receiver_request_context_ratio": 0.0,
            "receiver_request_mask_metadata_ratio": 0.0,
            "temporal_novelty_mean": 0.0,
            "temporal_cache_age_mean": 0.0,
            "temporal_cache_hit_ratio": 0.0,
            "temporal_refresh_ratio": 0.0,
            "temporal_init_frame_ratio": 0.0,
            "temporal_cache_entries": 0,
            "temporal_cache_confidence_mean": 0.0,
            "temporal_cache_update_count_mean": 0.0,
            "cumulative_bytes_per_scenario": 0.0,
            "average_bytes_per_frame": 0.0,
            "total_bytes_per_frame_after_init": 0.0,
            "total_normalized_ratio_after_init": 0.0,
        }
        aux: Dict[str, torch.Tensor] = {}

        # collaborator accounting tensors
        ego_idx = self._ego_indices(record_len, x.device)
        collab_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        if not self.drop_ego:
            collab_mask[ego_idx] = False

        collab_n = int(collab_mask.sum().item())
        bytes_per_value = float(self.cfg.get("bytes_per_value", 4.0))
        per_agent_bytes = float(x.shape[1] * x.shape[2] * x.shape[3] * bytes_per_value)
        full_comm_feature_bytes = float(collab_n) * per_agent_bytes
        metadata_bytes = 0.0
        context_bytes = 0.0

        if not self.enabled:
            stats["active_ratio"] = 1.0 if collab_n > 0 else 0.0
            stats["active_neighbors_ratio"] = 1.0 if collab_n > 0 else 0.0
            stats["feature_bytes_per_frame"] = full_comm_feature_bytes
            stats["context_bytes_per_frame"] = context_bytes
            stats["metadata_bytes_per_frame"] = metadata_bytes
            stats["total_bytes_per_frame"] = stats["feature_bytes_per_frame"] + stats["context_bytes_per_frame"] + stats["metadata_bytes_per_frame"]
            stats["normalized_ratio"] = 1.0 if full_comm_feature_bytes > 0 else 0.0
            stats["feature_normalized_ratio"] = stats["normalized_ratio"]
            stats["context_normalized_ratio"] = 0.0
            stats["metadata_normalized_ratio"] = 0.0
            stats["total_normalized_ratio"] = stats["normalized_ratio"]
            stats["bytes_per_frame"] = stats["total_bytes_per_frame"]
            return CommOutput(features=x, stats=stats, aux=aux)

        x_mod = x.clone()
        x_mod, active_neighbor_ratio, neighbor_selected_mask = self._select_neighbors(x_mod, record_len, pairwise_t_matrix)
        stats["active_neighbors_ratio"] = active_neighbor_ratio
        tx_mask = collab_mask & neighbor_selected_mask
        tx_n = int(tx_mask.sum().item())
        if tx_n <= 0:
            stats["active_neighbors_ratio"] = 0.0

        strategy = self.strategy

        # Backward compatibility: old random_drop maps to all-features stress mode.
        if strategy == "random_drop":
            strategy = "random_drop_all_features"
        temporal_active = False

        # Stress mode: mask all features (ego + collaborators).
        if strategy == "random_drop_all_features":
            keep_ratio = float(self.cfg.get("drop_random", {}).get("keep_ratio", 1.0))
            mask = self._random_mask(x_mod, keep_ratio)
            x_mod = x_mod * mask
            if tx_n > 0:
                tx_mask_f = tx_mask.view(-1, 1, 1, 1).to(mask.dtype)
                active_ratio = float((mask * tx_mask_f).sum().detach().cpu().item() / (tx_n * x.shape[2] * x.shape[3]))
            else:
                active_ratio = 0.0
            stats["active_ratio"] = active_ratio

        # Communication-only mode: preserve ego; drop collaborators only.
        elif strategy == "random_drop_comm_only":
            keep_ratio = float(self.cfg.get("drop_random", {}).get("keep_ratio", 1.0))
            tx_x = x_mod[tx_mask]
            mask_tx = self._random_mask(tx_x, keep_ratio)
            x_mod[tx_mask] = tx_x * mask_tx

            if tx_n > 0:
                active_ratio = float(mask_tx.mean().detach().cpu().item())
            else:
                active_ratio = 0.0
            stats["active_ratio"] = active_ratio

        elif self.learnable_enabled or strategy == "learnable_mask":
            if tx_n > 0:
                tx_x = x_mod[tx_mask]
                mask = self.mask_head(tx_x)
                if self.hard_mask:
                    hard = (mask > 0.5).to(mask.dtype)
                    mask = hard + (mask - mask.detach())
                x_mod[tx_mask] = tx_x * mask
                aux["mask_mean"] = mask.mean()
                stats["active_ratio"] = float((mask > 0.5).float().mean().detach().cpu().item()) if self.hard_mask else float(mask.mean().detach().cpu().item())
            else:
                stats["active_ratio"] = 0.0

        elif strategy == "topk_energy":
            topk_cfg = self.cfg.get("topk_energy", {})
            keep_ratio = float(topk_cfg.get("keep_ratio", 1.0))
            score_type = topk_cfg.get("score_type", "l2")
            tx_x = x_mod[tx_mask]
            mask = self._energy_mask(tx_x, keep_ratio, score_type)
            x_mod[tx_mask] = tx_x * mask
            stats["active_ratio"] = float(mask.mean().detach().cpu().item()) if tx_n > 0 else 0.0

        elif strategy == "receiver_request_topk" and self._is_learned_temporal_receiver_request(self.receiver_cfg):
            rr_cfg = self.receiver_cfg
            keep_ratio = float(rr_cfg.get("keep_ratio", 1.0))
            keep_ratio = float(max(min(keep_ratio, 1.0), 0.0))
            stats["receiver_request_keep_ratio"] = keep_ratio
            temporal_active = True
            x_mod, learned_stats, learned_aux = self._apply_learned_temporal_receiver_request(
                x_mod=x_mod,
                record_len=record_len,
                tx_mask=tx_mask,
                rr_cfg=rr_cfg,
                metadata=metadata,
                bytes_per_value=bytes_per_value,
            )
            stats["active_ratio"] = learned_stats.pop("active_ratio")
            context_bytes = float(learned_stats.pop("context_bytes"))
            metadata_bytes = float(learned_stats.pop("metadata_bytes"))
            stats.update(learned_stats)
            aux.update(learned_aux)

        elif strategy == "receiver_request_topk" and self._is_temporal_receiver_request(self.receiver_cfg):
            rr_cfg = self.receiver_cfg
            keep_ratio = float(rr_cfg.get("keep_ratio", 1.0))
            keep_ratio = float(max(min(keep_ratio, 1.0), 0.0))
            stats["receiver_request_keep_ratio"] = keep_ratio
            temporal_active = True
            x_mod, temporal_stats = self._apply_temporal_receiver_request(
                x_mod=x_mod,
                record_len=record_len,
                tx_mask=tx_mask,
                rr_cfg=rr_cfg,
                metadata=metadata,
                bytes_per_value=bytes_per_value,
            )
            stats["active_ratio"] = temporal_stats.pop("active_ratio")
            context_bytes = float(temporal_stats.pop("context_bytes"))
            metadata_bytes = float(temporal_stats.pop("metadata_bytes"))
            stats.update(temporal_stats)

        elif strategy == "receiver_request_topk":
            rr_cfg = self.receiver_cfg
            keep_ratio = float(rr_cfg.get("keep_ratio", 1.0))
            keep_ratio = float(max(min(keep_ratio, 1.0), 0.0))
            stats["receiver_request_keep_ratio"] = keep_ratio
            score_mode = rr_cfg.get("score_type", "multiplicative")
            normalize_scores = bool(rr_cfg.get("normalize_scores", True))
            alignment_mode = rr_cfg.get("alignment_mode", "ego_aligned")
            need_power = float(rr_cfg.get("need_power", 1.0))
            context_power = float(rr_cfg.get("context_power", 1.0))
            alpha = float(rr_cfg.get("alpha", 1.0))
            beta = float(rr_cfg.get("beta", 1.0))
            # TODO(Stage 2): implement warp_context_to_ego / warp_mask_to_sender using pairwise transforms.
            if alignment_mode not in ("ego_aligned", "warp_context_to_ego", "warp_mask_to_sender"):
                alignment_mode = "ego_aligned"
            if alignment_mode != "ego_aligned" and not self._warned_alignment:
                self.logger.warn(
                    "Alignment mode is approximate in Stage1",
                    alignment_mode=alignment_mode,
                    note="TODO warp_context_to_ego / warp_mask_to_sender",
                )
                self._warned_alignment = True
            if alignment_mode == "ego_aligned" and not self._checked_alignment:
                is_identity, max_dev = self._check_pairwise_identity(pairwise_t_matrix, record_len)
                self.logger.config(
                    "Alignment validation",
                    alignment_mode=alignment_mode,
                    pairwise_identity_like=("unknown" if is_identity is None else is_identity),
                    max_abs_deviation=max_dev,
                )
                self._checked_alignment = True

            groups = self._split_by_record_len(x_mod, record_len)
            out_groups = []
            global_context_bytes = 0.0
            global_metadata_bytes = 0.0
            global_selected_cells = 0.0
            global_total_cells = 0.0
            start = 0

            for group_idx, grp in enumerate(groups):
                n = grp.shape[0]
                grp_out = grp.clone()
                if n <= 1:
                    out_groups.append(grp_out)
                    start += n
                    continue

                ego_local = grp_out[0:1]
                ego_need = self._ego_need_map(ego_local, rr_cfg)

                # collaborator indices that survived neighbor selection
                group_tx = tx_mask[start:start + n]
                for local_i in range(1, n):
                    if not bool(group_tx[local_i].item()):
                        # neighbor-filtered collaborator => keep as zeroed state from _select_neighbors
                        continue
                    collab = grp_out[local_i:local_i + 1]
                    context_full = self._collaborator_context_map(collab, rr_cfg)
                    context_small = self._resize_context(context_full, rr_cfg.get("context_resolution", "full"))
                    context_for_score = context_small
                    if context_for_score.shape[2:] != ego_need.shape[2:]:
                        context_for_score = F.interpolate(context_for_score, size=ego_need.shape[2:], mode="nearest")

                    need_map = ego_need
                    if normalize_scores:
                        need_map = self._normalize_map(need_map)
                        context_for_score = self._normalize_map(context_for_score)
                    need_map = torch.pow(torch.clamp(need_map, min=0.0), max(need_power, 1e-6))
                    context_for_score = torch.pow(torch.clamp(context_for_score, min=0.0), max(context_power, 1e-6))

                    if score_mode == "weighted_sum":
                        score = alpha * need_map + beta * context_for_score
                    elif score_mode == "max_gate":
                        score = torch.maximum(need_map, context_for_score)
                    else:
                        score = (alpha * need_map) * (beta * context_for_score)

                    score = self._normalize_map(score)
                    mask = self._topk_mask(score, keep_ratio)
                    grp_out[local_i:local_i + 1] = collab * mask
                    self._maybe_save_request_maps(
                        rr_cfg=rr_cfg,
                        frame_idx=int(self._debug_saved_count),
                        group_idx=group_idx,
                        collab_local_idx=local_i,
                        ego_need=need_map,
                        collab_context=context_for_score,
                        score=score,
                        mask=mask,
                    )

                    global_selected_cells += float(mask.sum().detach().cpu().item())
                    global_total_cells += float(mask.numel())
                    global_context_bytes += self._estimate_context_bytes(context_small, rr_cfg)
                    global_metadata_bytes += self._estimate_mask_metadata_bytes(mask, rr_cfg)

                out_groups.append(grp_out)
                start += n

            x_mod = torch.cat(out_groups, dim=0)
            if global_total_cells > 0:
                stats["active_ratio"] = float(global_selected_cells / global_total_cells)
            else:
                stats["active_ratio"] = 0.0
            context_bytes = float(global_context_bytes)
            metadata_bytes = float(global_metadata_bytes)

        else:
            # strategy none
            stats["active_ratio"] = 1.0 if tx_n > 0 else 0.0

        pl_cfg = self.cfg.get("packet_loss", {})
        if bool(pl_cfg.get("enabled", False)):
            loss_rate = float(pl_cfg.get("loss_rate", 0.0))
            loss_rate = max(min(loss_rate, 1.0), 0.0)
            g = torch.Generator(device=x_mod.device)
            g.manual_seed(self.seed + 13)
            keep = torch.ones((x_mod.shape[0], 1, x_mod.shape[2], x_mod.shape[3]), device=x_mod.device, dtype=x_mod.dtype)
            if tx_n > 0:
                keep_tx = (torch.rand((tx_n, 1, x_mod.shape[2], x_mod.shape[3]), device=x_mod.device, generator=g) > loss_rate).to(x_mod.dtype)
                keep[tx_mask] = keep_tx
            x_corrupted = x_mod * keep
            stats["packet_loss_rate"] = loss_rate
            aux["packet_keep_mean"] = keep[tx_mask].mean() if tx_n > 0 else torch.tensor(1.0, device=x_mod.device)
            if self.repair_enabled:
                repaired_delta = self.repair_net(x_corrupted)
                x_repaired = x_corrupted + repaired_delta
                aux["repair_target"] = x_mod.detach()
                aux["repair_pred"] = x_repaired
                x_mod = x_repaired
            else:
                x_mod = x_corrupted
            if tx_n > 0:
                stats["active_ratio"] = float((stats["active_ratio"] * float(aux["packet_keep_mean"].detach().cpu().item())))
            else:
                stats["active_ratio"] = 0.0

        stats["feature_bytes_per_frame"] = float(stats["active_ratio"] * tx_n * per_agent_bytes)
        stats["context_bytes_per_frame"] = float(context_bytes)
        stats["metadata_bytes_per_frame"] = metadata_bytes
        stats["total_bytes_per_frame"] = stats["feature_bytes_per_frame"] + stats["context_bytes_per_frame"] + stats["metadata_bytes_per_frame"]
        if full_comm_feature_bytes > 0:
            stats["feature_normalized_ratio"] = float(stats["feature_bytes_per_frame"] / full_comm_feature_bytes)
            stats["context_normalized_ratio"] = float(stats["context_bytes_per_frame"] / full_comm_feature_bytes)
            stats["metadata_normalized_ratio"] = float(stats["metadata_bytes_per_frame"] / full_comm_feature_bytes)
            stats["total_normalized_ratio"] = float(stats["total_bytes_per_frame"] / full_comm_feature_bytes)
        else:
            stats["feature_normalized_ratio"] = 0.0
            stats["context_normalized_ratio"] = 0.0
            stats["metadata_normalized_ratio"] = 0.0
            stats["total_normalized_ratio"] = 0.0
        stats["normalized_ratio"] = stats["total_normalized_ratio"]
        stats["receiver_request_context_ratio"] = float(stats["context_bytes_per_frame"] / full_comm_feature_bytes) if full_comm_feature_bytes > 0 else 0.0
        stats["receiver_request_mask_metadata_ratio"] = float(stats["metadata_bytes_per_frame"] / full_comm_feature_bytes) if full_comm_feature_bytes > 0 else 0.0
        stats["bytes_per_frame"] = stats["total_bytes_per_frame"]

        if temporal_active:
            scenarios = self._metadata_scenarios(metadata)
            share = stats["total_bytes_per_frame"] / float(max(len(scenarios), 1))
            after_init_share = stats["total_bytes_per_frame_after_init"] / float(max(len(scenarios), 1))
            for scenario_id in scenarios:
                self._temporal_scenario_bytes[scenario_id] = self._temporal_scenario_bytes.get(scenario_id, 0.0) + share
                self._temporal_scenario_frames[scenario_id] = self._temporal_scenario_frames.get(scenario_id, 0) + 1
                if stats["temporal_init_frame_ratio"] < 1.0:
                    self._temporal_after_init_bytes[scenario_id] = self._temporal_after_init_bytes.get(scenario_id, 0.0) + after_init_share
                    self._temporal_after_init_frames[scenario_id] = self._temporal_after_init_frames.get(scenario_id, 0) + 1
            stats["cumulative_bytes_per_scenario"] = float(sum(self._temporal_scenario_bytes.values()) / max(len(self._temporal_scenario_bytes), 1))
            stats["average_bytes_per_frame"] = float(
                sum(self._temporal_scenario_bytes.values()) / max(sum(self._temporal_scenario_frames.values()), 1)
            )
            if self._temporal_after_init_frames:
                after_avg = float(
                    sum(self._temporal_after_init_bytes.values()) / max(sum(self._temporal_after_init_frames.values()), 1)
                )
                stats["total_bytes_per_frame_after_init"] = after_avg
                stats["total_normalized_ratio_after_init"] = float(after_avg / full_comm_feature_bytes) if full_comm_feature_bytes > 0 else 0.0

        if not self._logged_once:
            self.logger.info(
                "Strategy selected",
                strategy=strategy,
                drop_ego=self.drop_ego,
                keep_ratio=self.receiver_cfg.get("keep_ratio", None) if strategy == "receiver_request_topk" else self.cfg.get("topk_energy", {}).get("keep_ratio", None),
            )
            if strategy == "receiver_request_topk":
                self.logger.config(
                    "Receiver-request config",
                    ego_need_type=self.receiver_cfg.get("ego_need_type", "inverse_energy"),
                    collaborator_context_type=self.receiver_cfg.get("collaborator_context_type", "l2"),
                    alignment_mode=self.receiver_cfg.get("alignment_mode", "ego_aligned"),
                    count_context_overhead=self.receiver_cfg.get("count_context_overhead", True),
                )
                temporal_cfg = self.receiver_cfg.get("temporal", {}) if isinstance(self.receiver_cfg.get("temporal", {}), dict) else {}
                if temporal_active:
                    self.logger.config(
                        "Temporal receiver-request config",
                        init_mode=temporal_cfg.get("init_mode", "full_request"),
                        init_frames=temporal_cfg.get("init_frames", 1),
                        cache_momentum=temporal_cfg.get("cache_momentum", 0.9),
                        novelty_weight=temporal_cfg.get("novelty_weight", 1.0),
                    )
                if self._is_learned_temporal_receiver_request(self.receiver_cfg):
                    learned_cfg = self.receiver_cfg.get("learned", {}) if isinstance(self.receiver_cfg.get("learned", {}), dict) else {}
                    self.logger.config(
                        "Learned temporal receiver-request scaffold",
                        use_soft_mask_train=learned_cfg.get("use_soft_mask_train", True),
                        hidden_channels=learned_cfg.get("hidden_channels", 16),
                        trainable=self.receiver_cfg.get("trainable", False),
                        loss_enabled=learned_cfg.get("loss", {}).get("enabled", False)
                        if isinstance(learned_cfg.get("loss", {}), dict) else False,
                    )
            self._logged_once = True

        if self.logger.is_debug_enabled():
            self.logger.metric(
                "Communication stats",
                active_ratio=f"{stats['active_ratio']:.6f}",
                active_neighbors_ratio=f"{stats['active_neighbors_ratio']:.6f}",
                feature_bytes_per_frame=f"{stats['feature_bytes_per_frame']:.2f}",
                context_bytes_per_frame=f"{stats['context_bytes_per_frame']:.2f}",
                metadata_bytes_per_frame=f"{stats['metadata_bytes_per_frame']:.2f}",
                total_bytes_per_frame=f"{stats['total_bytes_per_frame']:.2f}",
                normalized_ratio=f"{stats['normalized_ratio']:.6f}",
            )

        return CommOutput(features=x_mod, stats=stats, aux=aux)
