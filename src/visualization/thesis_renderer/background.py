"""Road-style BEV background renderer."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib.patches import Rectangle

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


def draw_road_background(ax, xlim: Tuple[float, float], ylim: Tuple[float, float], cfg: RendererConfig) -> None:
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

    # Asphalt texture fills the panel, with subtle deterministic noise.
    if cfg.show_road_details:
        noise = _stable_noise(260, 90)
        ax.imshow(
            noise,
            extent=(xmin, xmax, ymin, ymax),
            cmap="gray",
            alpha=0.17,
            origin="lower",
            aspect="auto",
            vmin=-1.8,
            vmax=1.8,
            zorder=-30,
        )

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

    # Lane markings. Keep them generic and aligned to local x-axis.
    lane_spacing = 5.0
    first_lane = np.floor(ymin / lane_spacing) * lane_spacing
    y = first_lane
    while y <= ymax:
        if abs(y) < 0.25:
            ax.plot([xmin, xmax], [y, y], color=cfg.centerline_color, lw=1.1, alpha=0.75, zorder=-8)
        elif ymin + shoulder_h < y < ymax - shoulder_h:
            ax.plot([xmin, xmax], [y, y], color=cfg.lane_marking_color, lw=1.4, ls=(0, (5.5, 9.0)), alpha=0.62, zorder=-8)
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
