from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


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

    def _select_neighbors(self, x: torch.Tensor, record_len: torch.Tensor, pairwise_t_matrix: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float]:
        cfg = self.cfg.get("neighbor_selection", {})
        mode = cfg.get("mode", "all")
        k = int(cfg.get("k", 0))
        if mode == "all" or k <= 0:
            return x, 1.0

        groups = self._split_by_record_len(x, record_len)
        new_groups = []
        active_collab = 0
        total_collab = 0
        for b, grp in enumerate(groups):
            n = grp.shape[0]
            total_collab += max(n - 1, 0)
            if n <= 1:
                new_groups.append(grp)
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

        ratio = float(active_collab) / float(max(total_collab, 1))
        return torch.cat(new_groups, dim=0), ratio

    def forward(self, x: torch.Tensor, record_len: torch.Tensor, pairwise_t_matrix: Optional[torch.Tensor] = None) -> CommOutput:
        stats = {
            "active_ratio": 1.0,
            "active_neighbors_ratio": 1.0,
            "packet_loss_rate": 0.0,
            "bytes_per_frame": 0.0,
        }
        aux: Dict[str, torch.Tensor] = {}

        # collaborator accounting tensors
        ego_idx = self._ego_indices(record_len, x.device)
        collab_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        if not self.drop_ego:
            collab_mask[ego_idx] = False

        collab_n = int(collab_mask.sum().item())
        per_agent_bytes = float(x.shape[1] * x.shape[2] * x.shape[3] * 4.0)

        if not self.enabled:
            stats["active_ratio"] = 1.0 if collab_n > 0 else 0.0
            stats["bytes_per_frame"] = float(collab_n) * per_agent_bytes
            return CommOutput(features=x, stats=stats, aux=aux)

        x_mod = x.clone()
        x_mod, active_neighbor_ratio = self._select_neighbors(x_mod, record_len, pairwise_t_matrix)
        stats["active_neighbors_ratio"] = active_neighbor_ratio

        strategy = self.strategy

        # Backward compatibility: old random_drop maps to all-features stress mode.
        if strategy == "random_drop":
            strategy = "random_drop_all_features"

        # Stress mode: mask all features (ego + collaborators).
        if strategy == "random_drop_all_features":
            keep_ratio = float(self.cfg.get("drop_random", {}).get("keep_ratio", 1.0))
            mask = self._random_mask(x_mod, keep_ratio)
            x_mod = x_mod * mask
            stats["active_ratio"] = float(mask.mean().detach().cpu().item())
            stats["bytes_per_frame"] = float(stats["active_ratio"] * x.shape[0] * per_agent_bytes)

        # Communication-only mode: preserve ego; drop collaborators only.
        elif strategy == "random_drop_comm_only":
            keep_ratio = float(self.cfg.get("drop_random", {}).get("keep_ratio", 1.0))
            collab_x = x_mod[collab_mask]
            mask_collab = self._random_mask(collab_x, keep_ratio)
            x_mod[collab_mask] = collab_x * mask_collab

            if collab_n > 0:
                active_ratio = float(mask_collab.mean().detach().cpu().item())
            else:
                active_ratio = 0.0
            stats["active_ratio"] = active_ratio
            stats["bytes_per_frame"] = float(active_ratio * collab_n * per_agent_bytes)

        elif self.learnable_enabled or strategy == "learnable_mask":
            mask = self.mask_head(x_mod)
            if self.hard_mask:
                hard = (mask > 0.5).to(mask.dtype)
                mask = hard + (mask - mask.detach())
            x_mod = x_mod * mask
            aux["mask_mean"] = mask.mean()
            stats["active_ratio"] = float((mask > 0.5).float().mean().detach().cpu().item()) if self.hard_mask else float(mask.mean().detach().cpu().item())
            stats["bytes_per_frame"] = float(stats["active_ratio"] * x.shape[0] * per_agent_bytes)

        elif strategy == "topk_energy":
            topk_cfg = self.cfg.get("topk_energy", {})
            keep_ratio = float(topk_cfg.get("keep_ratio", 1.0))
            score_type = topk_cfg.get("score_type", "l2")
            mask = self._energy_mask(x_mod, keep_ratio, score_type)
            x_mod = x_mod * mask
            stats["active_ratio"] = float(mask.mean().detach().cpu().item())
            stats["bytes_per_frame"] = float(stats["active_ratio"] * x.shape[0] * per_agent_bytes)

        else:
            # strategy none
            stats["active_ratio"] = 1.0 if collab_n > 0 else 0.0
            stats["bytes_per_frame"] = float(collab_n) * per_agent_bytes

        pl_cfg = self.cfg.get("packet_loss", {})
        if bool(pl_cfg.get("enabled", False)):
            loss_rate = float(pl_cfg.get("loss_rate", 0.0))
            loss_rate = max(min(loss_rate, 1.0), 0.0)
            g = torch.Generator(device=x_mod.device)
            g.manual_seed(self.seed + 13)
            keep = (torch.rand((x_mod.shape[0], 1, x_mod.shape[2], x_mod.shape[3]), device=x_mod.device, generator=g) > loss_rate).to(x_mod.dtype)
            x_corrupted = x_mod * keep
            stats["packet_loss_rate"] = loss_rate
            aux["packet_keep_mean"] = keep.mean()
            if self.repair_enabled:
                repaired_delta = self.repair_net(x_corrupted)
                x_repaired = x_corrupted + repaired_delta
                aux["repair_target"] = x_mod.detach()
                aux["repair_pred"] = x_repaired
                x_mod = x_repaired
            else:
                x_mod = x_corrupted

        return CommOutput(features=x_mod, stats=stats, aux=aux)
