"""Panel renderer for road-background car-icon thesis figures."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .background import draw_road_background
from .config import RendererConfig
from .vehicles import box_center_length_width_yaw, draw_box_outline, draw_vehicle_icon


def _point_inside_or_near_box(point: np.ndarray, box: np.ndarray, margin: float) -> bool:
    center, length, width, yaw = box_center_length_width_yaw(box)
    c, s = np.cos(-yaw), np.sin(-yaw)
    rel = np.asarray(point, dtype=np.float32) - center
    local = np.asarray([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]], dtype=np.float32)
    return bool(abs(local[0]) <= 0.5 * length + margin and abs(local[1]) <= 0.5 * width + margin)


def _ego_overlaps_boxes(gt: np.ndarray, pred: np.ndarray, cfg: RendererConfig) -> bool:
    origin = np.asarray([0.0, 0.0], dtype=np.float32)
    for boxes in (gt, pred):
        for box in boxes:
            if _point_inside_or_near_box(origin, box, cfg.ego_collision_margin_m):
                return True
    return False


def _dominant_box_yaw(gt: np.ndarray, pred: np.ndarray) -> float:
    """Estimate the dominant BEV traffic orientation from visible boxes.

    The angle is axial: yaw and yaw+pi describe the same lane direction. This
    is used only for the schematic road background; detection geometry remains
    unchanged.
    """
    yaws = []
    for boxes in (gt, pred):
        for box in boxes:
            _center, length, width, yaw = box_center_length_width_yaw(box)
            if max(length, width) > 1.0:
                yaws.append(yaw)
    if not yaws:
        return 0.0
    yaws_np = np.asarray(yaws, dtype=np.float32)
    return float(0.5 * np.arctan2(np.sin(2.0 * yaws_np).mean(), np.cos(2.0 * yaws_np).mean()))


def _offset_ego_marker_position(xlim: Tuple[float, float], ylim: Tuple[float, float], gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    xmin, xmax = map(float, xlim)
    ymin, ymax = map(float, ylim)
    w = xmax - xmin
    h = ymax - ymin
    local = []
    for radius in (4.5, 7.0, 9.5):
        local.extend(
            [
                [-radius, -radius],
                [-radius, radius],
                [radius, -radius],
                [radius, radius],
                [-radius, 0.0],
                [radius, 0.0],
                [0.0, -radius],
                [0.0, radius],
            ]
        )
    corner_fallback = [
        [xmin + 0.08 * w, ymax - 0.14 * h],
        [xmin + 0.08 * w, ymin + 0.14 * h],
        [xmax - 0.08 * w, ymax - 0.14 * h],
        [xmax - 0.08 * w, ymin + 0.14 * h],
    ]
    candidates = np.asarray(local + corner_fallback, dtype=np.float32)
    inside = (
        (candidates[:, 0] >= xmin + 0.03 * w)
        & (candidates[:, 0] <= xmax - 0.03 * w)
        & (candidates[:, 1] >= ymin + 0.05 * h)
        & (candidates[:, 1] <= ymax - 0.05 * h)
    )
    candidates = candidates[inside] if inside.any() else candidates
    centers = []
    for boxes in (gt, pred):
        if boxes.shape[0]:
            centers.append(boxes.mean(axis=1))
    if not centers:
        return tuple(candidates[0])
    all_centers = np.concatenate(centers, axis=0)
    distances = np.linalg.norm(candidates[:, None, :] - all_centers[None, :, :], axis=-1)
    overlap_penalty = np.asarray(
        [
            any(_point_inside_or_near_box(candidate, box, 0.6) for boxes in (gt, pred) for box in boxes)
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    scores = distances.min(axis=1) - 0.28 * np.linalg.norm(candidates, axis=1) - 1000.0 * overlap_penalty
    return tuple(candidates[int(np.argmax(scores))])


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
        ego_marker_position: Optional[np.ndarray] = None,
    ) -> None:
        cfg = self.config
        gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4, 2)
        pred = np.asarray(frame.pred_boxes, dtype=np.float32).reshape(-1, 4, 2)
        gt_matched = np.asarray(match["gt_matched"], dtype=bool)
        pred_matched = np.asarray(match["pred_matched"], dtype=bool)

        draw_road_background(ax, xlim, ylim, cfg, road_yaw=_dominant_box_yaw(gt, pred))

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

        if ego_marker_position is not None:
            ego_xy = np.asarray(ego_marker_position, dtype=np.float32).reshape(2)
            ego_visible = xlim[0] <= float(ego_xy[0]) <= xlim[1] and ylim[0] <= float(ego_xy[1]) <= ylim[1]
            if ego_visible:
                ax.scatter([ego_xy[0]], [ego_xy[1]], marker="^", s=cfg.ego_marker_size, color=cfg.ego_color, edgecolor="white", linewidth=1.2, zorder=12)

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
