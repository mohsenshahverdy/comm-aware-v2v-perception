"""Top-down vehicle-icon rendering from exact BEV box geometry."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib.patches import Polygon

from .config import RendererConfig


def box_center_length_width_yaw(box: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Return center, length, width, yaw from a four-corner BEV box.

    The input geometry is not modified; icons are fitted inside the supplied
    box and use the first edge as the length direction. This is deterministic
    and preserves the original center/orientation semantics.
    """
    pts = np.asarray(box, dtype=np.float32)[:4, :2]
    center = pts.mean(axis=0)
    edge01 = pts[1] - pts[0]
    edge12 = pts[2] - pts[1]
    len01 = float(np.linalg.norm(edge01))
    len12 = float(np.linalg.norm(edge12))
    if len01 >= len12:
        length = max(len01, 1e-3)
        width = max(len12, 1e-3)
        yaw = float(np.arctan2(edge01[1], edge01[0]))
    else:
        length = max(len12, 1e-3)
        width = max(len01, 1e-3)
        yaw = float(np.arctan2(edge12[1], edge12[0]))
    return center, length, width, yaw


def _local_to_world(local: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + center


def draw_box_outline(ax, box: np.ndarray, color: str, linewidth: float, linestyle: str = "-", alpha: float = 1.0, zorder: int = 3):
    pts = np.asarray(box, dtype=np.float32)[:4, :2]
    closed = np.vstack([pts, pts[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=linewidth, ls=linestyle, alpha=alpha, zorder=zorder)


def draw_vehicle_icon(
    ax,
    box: np.ndarray,
    semantic: str,
    cfg: RendererConfig,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    alpha: float | None = None,
    zorder: int = 5,
):
    """Draw a vehicle icon fitted to an existing BEV box.

    The semantic color is applied as a scientific overlay/outline. The vehicle
    body remains neutral so TP/FP/missed semantics stay readable.
    """
    alpha = cfg.vehicle_alpha if alpha is None else alpha
    center, length, width, yaw = box_center_length_width_yaw(box)
    length *= cfg.vehicle_scale
    width *= cfg.vehicle_scale

    # Top-down car silhouette, not a random rectangle: tapered nose/rear and roof/window cue.
    body_local = np.asarray(
        [
            [-0.48 * length, -0.42 * width],
            [0.34 * length, -0.42 * width],
            [0.50 * length, -0.22 * width],
            [0.50 * length, 0.22 * width],
            [0.34 * length, 0.42 * width],
            [-0.48 * length, 0.42 * width],
        ],
        dtype=np.float32,
    )
    body = _local_to_world(body_local, center, yaw)

    if semantic == "missed":
        face = "#2A0F0F"
        fill_alpha = min(0.34, alpha)
    elif semantic == "fp":
        face = "#231B0C"
        fill_alpha = min(0.30, alpha)
    else:
        face = cfg.vehicle_body_color
        fill_alpha = min(0.82, alpha)

    ax.add_patch(
        Polygon(body, closed=True, facecolor=face, edgecolor=color, lw=linewidth, linestyle=linestyle, alpha=fill_alpha, zorder=zorder)
    )

    roof_local = np.asarray(
        [
            [-0.18 * length, -0.28 * width],
            [0.20 * length, -0.25 * width],
            [0.30 * length, 0.00 * width],
            [0.20 * length, 0.25 * width],
            [-0.18 * length, 0.28 * width],
            [-0.28 * length, 0.00 * width],
        ],
        dtype=np.float32,
    )
    roof = _local_to_world(roof_local, center, yaw)
    ax.add_patch(
        Polygon(roof, closed=True, facecolor=cfg.vehicle_window_color, edgecolor="#060606", lw=max(0.5, linewidth * 0.35), alpha=0.84, zorder=zorder + 0.2)
    )

    # Bright semantic outline exactly on the original box footprint.
    draw_box_outline(ax, box, color=color, linewidth=linewidth, linestyle=linestyle, alpha=1.0, zorder=zorder + 0.5)

    # Small front highlight to make yaw visible.
    front = _local_to_world(np.asarray([[0.43 * length, -0.18 * width], [0.43 * length, 0.18 * width]], dtype=np.float32), center, yaw)
    ax.plot(front[:, 0], front[:, 1], color=cfg.vehicle_highlight_color, lw=max(0.6, linewidth * 0.25), alpha=0.75, zorder=zorder + 0.8)
