"""Panel renderer for road-background car-icon thesis figures."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .background import draw_road_background
from .config import RendererConfig
from .vehicles import draw_box_outline, draw_vehicle_icon


class RoadCarPanelRenderer:
    """Render one BEV panel with road background and car icons.

    Matching arrays and boxes are supplied by the caller. This class does not
    perform matching and therefore cannot change TP/FP/missed counts.
    """

    def __init__(self, config: Optional[RendererConfig] = None):
        self.config = config or RendererConfig()

    def draw_panel(
        self,
        ax,
        *,
        frame: Any,
        gt_boxes: np.ndarray,
        match: Dict[str, Any],
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        title: str,
        show_roi: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        show_xlabel: bool = False,
        show_ylabel: bool = False,
        anchor: str = "C",
    ) -> None:
        cfg = self.config
        gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4, 2)
        pred = np.asarray(frame.pred_boxes, dtype=np.float32).reshape(-1, 4, 2)
        gt_matched = np.asarray(match["gt_matched"], dtype=bool)
        pred_matched = np.asarray(match["pred_matched"], dtype=bool)

        draw_road_background(ax, xlim, ylim, cfg)

        if cfg.show_matched_gt and gt.shape[0] and gt_matched.shape[0] == gt.shape[0]:
            for box in gt[gt_matched]:
                draw_box_outline(ax, box, cfg.matched_gt_color, cfg.matched_gt_linewidth, linestyle="--", alpha=cfg.matched_gt_alpha, zorder=2)

        if gt.shape[0] and gt_matched.shape[0] == gt.shape[0]:
            for box in gt[~gt_matched]:
                draw_vehicle_icon(ax, box, "missed", cfg, cfg.missed_color, cfg.missed_linewidth, linestyle="-", zorder=6)

        if pred.shape[0] and pred_matched.shape[0] == pred.shape[0]:
            for box in pred[pred_matched]:
                draw_vehicle_icon(ax, box, "tp", cfg, cfg.tp_color, cfg.tp_linewidth, linestyle="-", zorder=7)
            for box in pred[~pred_matched]:
                draw_vehicle_icon(ax, box, "fp", cfg, cfg.fp_color, cfg.fp_linewidth, linestyle=":", zorder=8)
        else:
            for box in pred:
                draw_vehicle_icon(ax, box, "tp", cfg, cfg.tp_color, cfg.tp_linewidth, linestyle="-", zorder=7)

        if getattr(frame, "trajectory", None) is not None and len(frame.trajectory) > 0:
            traj = np.asarray(frame.trajectory, dtype=np.float32)
            ax.plot(traj[:, 0], traj[:, 1], color="#FFFFFF", linewidth=2.3, alpha=0.45, zorder=5)
            ax.plot(traj[:, 0], traj[:, 1], color=cfg.ego_color, linewidth=1.4, marker=".", markersize=2.4, alpha=0.75, zorder=5.5)

        if getattr(frame, "collaborator_positions", None) is not None and len(frame.collaborator_positions) > 0:
            collab = np.asarray(frame.collaborator_positions, dtype=np.float32)
            ax.scatter(collab[:, 0], collab[:, 1], marker="s", s=25, facecolor="none", edgecolor="#F5F5F5", linewidth=1.0, zorder=9)

        ax.scatter([0.0], [0.0], marker="^", s=110, color=cfg.ego_color, edgecolor="white", linewidth=0.8, zorder=10)

        if show_roi is not None:
            from matplotlib.patches import Rectangle

            rx, ry = show_roi
            ax.add_patch(Rectangle((rx[0], ry[0]), rx[1] - rx[0], ry[1] - ry[0], fill=False, edgecolor="#F2F2F2", linewidth=2.0, linestyle="--", zorder=11))

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor(anchor)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=4)
        if cfg.show_grid:
            ax.grid(True, color="#FFFFFF", linewidth=0.35, alpha=0.25)
        else:
            ax.grid(False)
        ax.tick_params(axis="both", labelsize=10, length=3.0, pad=1.2, colors="#111111")
        ax.set_xlabel("x [m]" if show_xlabel else "", fontsize=11)
        ax.set_ylabel("y [m]" if show_ylabel else "", fontsize=11)
