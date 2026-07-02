"""Road-style BEV background renderer."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib.patches import Polygon, Rectangle

from .config import RendererConfig


def _stable_noise(nx: int, ny: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(ny, nx))
    # Simple deterministic smoothing without scipy.
    for _ in range(3):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) / 5.0
    return noise


def _uv_to_xy(points_uv: np.ndarray, road_yaw: float) -> np.ndarray:
    c, s = np.cos(road_yaw), np.sin(road_yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return np.asarray(points_uv, dtype=np.float32) @ rot.T


def _panel_uv_bounds(xlim: Tuple[float, float], ylim: Tuple[float, float], road_yaw: float) -> Tuple[float, float, float, float]:
    xmin, xmax = map(float, xlim)
    ymin, ymax = map(float, ylim)
    corners = np.asarray([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]], dtype=np.float32)
    c, s = np.cos(-road_yaw), np.sin(-road_yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    uv = corners @ rot.T
    pad_u = 0.08 * max(1.0, float(uv[:, 0].max() - uv[:, 0].min()))
    pad_v = 0.08 * max(1.0, float(uv[:, 1].max() - uv[:, 1].min()))
    return (
        float(uv[:, 0].min() - pad_u),
        float(uv[:, 0].max() + pad_u),
        float(uv[:, 1].min() - pad_v),
        float(uv[:, 1].max() + pad_v),
    )


def draw_road_background(ax, xlim: Tuple[float, float], ylim: Tuple[float, float], cfg: RendererConfig, road_yaw: float = 0.0) -> None:
    """Draw a deterministic road-like background in the panel coordinate frame.

    The road is an x-aligned local BEV surface. It intentionally does not infer
    or alter object geometry; it only fills the visible coordinate extent.
    """
    xmin, xmax = map(float, xlim)
    ymin, ymax = map(float, ylim)
    width = xmax - xmin
    height = ymax - ymin
    ax.set_facecolor("#ECEBE4")

    if cfg.background_type in {"none", "plain"}:
        return

    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            width,
            height,
            facecolor=cfg.asphalt_color,
            edgecolor="none",
            alpha=cfg.road_alpha,
            zorder=-40,
        )
    )

    # Asphalt texture fills the panel, with subtle deterministic noise. The
    # base rectangle is drawn first so that the texture appears as variation
    # in the road surface rather than as a detached image patch.
    if cfg.show_road_details:
        noise = _stable_noise(280, 120)
        ax.imshow(
            noise,
            extent=(xmin, xmax, ymin, ymax),
            cmap="gray",
            alpha=0.075,
            origin="lower",
            aspect="auto",
            vmin=-1.45,
            vmax=1.45,
            zorder=-30,
        )

    # The qualitative BEV panels often crop tightly around disagreement boxes.
    # By default the whole visible panel is treated as drivable asphalt so that
    # no valid detection appears to sit on a rendered sidewalk/curb. Optional
    # shoulders can still be enabled for schematic uses.
    shoulder_h = 0.0
    if cfg.show_shoulders:
        shoulder_h = min(3.0, max(0.8, 0.08 * height))
        ax.add_patch(Rectangle((xmin, ymax - shoulder_h), width, shoulder_h, facecolor=cfg.sidewalk_color, alpha=0.35, zorder=-28))
        ax.add_patch(Rectangle((xmin, ymin), width, shoulder_h, facecolor=cfg.sidewalk_color, alpha=0.35, zorder=-28))
        ax.plot([xmin, xmax], [ymax - shoulder_h, ymax - shoulder_h], color=cfg.asphalt_edge_color, lw=0.9, zorder=-10)
        ax.plot([xmin, xmax], [ymin + shoulder_h, ymin + shoulder_h], color=cfg.asphalt_edge_color, lw=0.9, zorder=-10)

    if not cfg.show_road_details:
        return

    # Subtle lane bands give the BEV a road-like geometry without adding bright
    # white markings that compete with detection boxes.
    lane_spacing = 5.0
    umin, umax, vmin, vmax = _panel_uv_bounds(xlim, ylim, road_yaw)
    first_lane = np.floor(vmin / lane_spacing) * lane_spacing
    if cfg.show_lane_bands:
        y_band = first_lane
        band_index = int(np.floor(first_lane / lane_spacing))
        while y_band < vmax:
            y_next = min(y_band + lane_spacing, vmax)
            if band_index % 2 == 0:
                pts = _uv_to_xy(
                    np.asarray(
                        [
                            [umin, max(y_band, vmin)],
                            [umax, max(y_band, vmin)],
                            [umax, y_next],
                            [umin, y_next],
                        ],
                        dtype=np.float32,
                    ),
                    road_yaw,
                )
                ax.add_patch(Polygon(pts, closed=True, facecolor=cfg.asphalt_lane_band_color, edgecolor="none", alpha=0.17, zorder=-27))
            y_band += lane_spacing
            band_index += 1

    # Lane markings. By default only the muted centerline and very faint lane
    # separators are shown; repeated bright white dashes made dense BEV panels
    # visually noisy and could be confused with detection boxes.
    y = first_lane
    while y <= vmax:
        pts = _uv_to_xy(np.asarray([[umin, y], [umax, y]], dtype=np.float32), road_yaw)
        if abs(y) < 0.25:
            ax.plot(pts[:, 0], pts[:, 1], color=cfg.centerline_color, lw=0.9, alpha=0.42, zorder=-8)
        elif cfg.show_lane_markings and vmin + shoulder_h < y < vmax - shoulder_h:
            ax.plot(pts[:, 0], pts[:, 1], color=cfg.lane_marking_color, lw=0.8, ls=(0, (4.2, 10.5)), alpha=0.30, zorder=-8)
        elif vmin + shoulder_h < y < vmax - shoulder_h:
            ax.plot(pts[:, 0], pts[:, 1], color=cfg.lane_marking_color, lw=0.45, alpha=0.10, zorder=-9)
        y += lane_spacing

    # Crosswalk/intersection hint only when the visible panel includes x=0-ish.
    if cfg.show_crosswalk and xmin <= 8.0 <= xmax and height > 16.0:
        stripe_w = max(0.7, min(1.15, 0.012 * width))
        x0 = xmin + 4.0
        n = 8
        for i in range(n):
            ax.add_patch(
                Rectangle(
                    (x0 + i * stripe_w * 1.85, ymin + max(shoulder_h * 0.35, 0.4)),
                    stripe_w,
                    height - max(shoulder_h * 0.7, 0.8),
                    facecolor="#F2F2F2",
                    edgecolor="none",
                    alpha=0.58,
                    zorder=-7,
                )
            )
