from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor, record_len: torch.Tensor, pairwise_t_matrix: Optional[torch.Tensor] = None) -> CommOutput:
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
            "receiver_request_keep_ratio": 1.0,
            "receiver_request_context_ratio": 0.0,
            "receiver_request_mask_metadata_ratio": 0.0,
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

        elif strategy == "receiver_request_topk":
            rr_cfg = self.receiver_cfg
            keep_ratio = float(rr_cfg.get("keep_ratio", 1.0))
            keep_ratio = float(max(min(keep_ratio, 1.0), 0.0))
            stats["receiver_request_keep_ratio"] = keep_ratio
            score_mode = rr_cfg.get("score_type", "multiplicative")
            normalize_scores = bool(rr_cfg.get("normalize_scores", True))
            alignment_mode = rr_cfg.get("alignment_mode", "ego_aligned")
            # TODO(Stage 2): implement warp_context_to_ego / warp_mask_to_sender using pairwise transforms.
            if alignment_mode not in ("ego_aligned", "warp_context_to_ego", "warp_mask_to_sender"):
                alignment_mode = "ego_aligned"

            groups = self._split_by_record_len(x_mod, record_len)
            out_groups = []
            global_context_bytes = 0.0
            global_metadata_bytes = 0.0
            global_selected_cells = 0.0
            global_total_cells = 0.0
            start = 0

            for grp in groups:
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

                    if score_mode == "weighted_sum":
                        score = 0.5 * need_map + 0.5 * context_for_score
                    elif score_mode == "max_gate":
                        score = torch.maximum(need_map, context_for_score)
                    else:
                        score = need_map * context_for_score

                    score = self._normalize_map(score)
                    mask = self._topk_mask(score, keep_ratio)
                    grp_out[local_i:local_i + 1] = collab * mask

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
        stats["normalized_ratio"] = float(stats["total_bytes_per_frame"] / full_comm_feature_bytes) if full_comm_feature_bytes > 0 else 0.0
        stats["receiver_request_context_ratio"] = float(stats["context_bytes_per_frame"] / full_comm_feature_bytes) if full_comm_feature_bytes > 0 else 0.0
        stats["receiver_request_mask_metadata_ratio"] = float(stats["metadata_bytes_per_frame"] / full_comm_feature_bytes) if full_comm_feature_bytes > 0 else 0.0
        stats["bytes_per_frame"] = stats["total_bytes_per_frame"]

        return CommOutput(features=x_mod, stats=stats, aux=aux)
