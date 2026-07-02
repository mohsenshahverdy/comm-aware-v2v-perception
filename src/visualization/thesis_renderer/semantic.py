"""Semantic styles and legend handles for thesis renderer."""
from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .config import RendererConfig


def legend_handles(cfg: RendererConfig):
    return [
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=cfg.matched_gt_color, linewidth=cfg.matched_gt_linewidth, linestyle="--", label="matched GT"),
        Rectangle((0, 0), 1, 1, facecolor="#111111", edgecolor=cfg.tp_color, linewidth=cfg.tp_linewidth, label="true-positive prediction"),
        Rectangle((0, 0), 1, 1, facecolor="#231B0C", edgecolor=cfg.fp_color, linewidth=cfg.fp_linewidth, linestyle=":", label="false-positive prediction"),
        Rectangle((0, 0), 1, 1, facecolor="#2A0F0F", edgecolor=cfg.missed_color, linewidth=cfg.missed_linewidth, label="missed GT"),
        Line2D([0], [0], color=cfg.ego_color, marker="^", linestyle="None", markersize=cfg.ego_legend_marker_size, label="ego vehicle"),
    ]
