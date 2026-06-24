#!/usr/bin/env python3
"""Generate schematic scene views from a qualitative BEV reference.

These figures are illustrative reconstructions for thesis explanation. They are
based on the CARLA frame_000011 qualitative BEV layout, but they are not
georeferenced map renders and do not replace the quantitative BEV figures.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.colors import to_rgb


OUT_DIR = Path("Classical_Format_Thesis/figures/generated")

BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"
GREEN = "#009E73"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
ROAD = "#D8D6CE"
LANE = "#FFFFFF"
POLIMI_BLUE = "#003A70"


# Approximate local BEV layout read from the CARLA frame_000011 qualitative
# figure. Coordinates are local BEV metres, not latitude/longitude.
OBJECTS = [
    (5.0, -7.0, 4.8, 2.0, -3),
    (8.5, 7.0, 4.6, 2.0, 0),
    (14.0, -3.4, 4.1, 2.0, 2),
    (14.2, -6.8, 5.2, 2.0, 0),
    (21.5, 3.2, 4.2, 2.0, -3),
    (22.0, -7.1, 5.2, 2.0, 0),
    (26.2, 6.6, 4.5, 2.0, 2),
    (28.3, -3.4, 4.1, 2.0, 0),
    (29.7, -7.0, 5.2, 2.0, 0),
    (31.0, 7.0, 4.3, 2.0, 0),
    (33.5, -3.5, 4.2, 2.0, 0),
    (43.5, -0.2, 4.5, 2.0, 2),
    (49.5, 3.0, 4.8, 2.0, 4),
    (50.0, -7.0, 5.0, 2.0, -4),
    (54.0, -3.5, 5.0, 2.0, -2),
]

RECOVERED = [
    (28.2, 6.8, 4.5, 2.0, 2),
    (50.0, 6.8, 4.8, 2.0, 3),
]

EGO = (12.0, 0.0, 4.7, 2.1, 0)
COLLABORATORS = [(5.5, -7.5), (32.0, 7.5), (53.0, -4.0)]


def rotated_rect(cx: float, cy: float, length: float, width: float, yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    base = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ]
    )
    rot = np.array([[c, -s], [s, c]])
    return base @ rot.T + np.array([cx, cy])


def draw_vehicle_2d(
    ax,
    vehicle: Tuple[float, float, float, float, float],
    edge: str = BLUE,
    face: str = "white",
    lw: float = 2.0,
    alpha: float = 1.0,
    linestyle: str = "-",
    zorder: int = 5,
):
    poly = rotated_rect(*vehicle)
    patch = Polygon(poly, closed=True, facecolor=face, edgecolor=edge, lw=lw, alpha=alpha, linestyle=linestyle, zorder=zorder)
    ax.add_patch(patch)
    return patch


def draw_road(ax, xlim=(-2, 60), ylim=(-13, 13)):
    ax.set_facecolor("#F7F5EF")
    ax.add_patch(Rectangle((xlim[0] - 5, -10.5), xlim[1] - xlim[0] + 10, 21.0, facecolor=ROAD, edgecolor="none", zorder=0))
    for y in [-5.0, 0.0, 5.0]:
        ax.plot(xlim, [y, y], color=LANE, lw=1.5, ls=(0, (7, 7)), zorder=1)
    ax.plot(xlim, [-10.5, -10.5], color="#B8B5AA", lw=1.1, zorder=1)
    ax.plot(xlim, [10.5, 10.5], color="#B8B5AA", lw=1.1, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#FFFFFF", lw=0.6, alpha=0.5)
    ax.set_xlabel("local x [m]")
    ax.set_ylabel("local y [m]")


def add_note(fig):
    fig.text(
        0.5,
        0.01,
        "Illustrative reconstruction from qualitative BEV output; local CARLA coordinates only, not a georeferenced map render.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=GRAY,
    )


def save(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=320)
    plt.close(fig)


def map_style():
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.16)
    draw_road(ax)
    ax.set_title("CARLA frame 000011: map-style local scene schematic", fontsize=15, fontweight="bold", color=POLIMI_BLUE)

    # Receiver trajectory / route corridor.
    ax.add_patch(Rectangle((EGO[0] - 1, -2.2), 45, 4.4, facecolor=GREEN, alpha=0.10, edgecolor="none", zorder=2))
    ax.add_patch(FancyArrowPatch((EGO[0] + 1.5, 0.0), (55.0, 0.0), arrowstyle="->", mutation_scale=18, lw=2.1, color=GREEN, alpha=0.9, zorder=7))
    ax.text(35, 2.8, "receiver future path / relevant corridor", color=GREEN, fontsize=10, ha="center")

    for obj in OBJECTS:
        draw_vehicle_2d(ax, obj, edge=BLUE, face="white", lw=1.8)
    for obj in RECOVERED:
        draw_vehicle_2d(ax, obj, edge=RED, face="#FEE2E2", lw=2.8, zorder=8)
        ax.text(obj[0], obj[1] + 2.1, "recovered", color=RED, fontsize=8.5, ha="center", fontweight="bold")

    draw_vehicle_2d(ax, EGO, edge=GREEN, face="#D1FAE5", lw=2.7, zorder=10)
    ax.scatter([EGO[0]], [EGO[1]], marker="^", s=100, color=GREEN, edgecolors="white", linewidths=1.0, zorder=11, label="ego/receiver")
    for i, (x, y) in enumerate(COLLABORATORS, start=1):
        ax.scatter([x], [y], s=80, color="#7C3AED", edgecolors="white", linewidths=1.0, zorder=9)
        ax.text(x, y - 2.0, f"CAV {i}", color="#5B21B6", fontsize=8, ha="center")

    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=BLUE, lw=2, label="detected objects"),
            Rectangle((0, 0), 1, 1, facecolor="#FEE2E2", edgecolor=RED, lw=2, label="objects recovered by learned view"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=GREEN, markeredgecolor="white", markersize=10, label="ego/receiver"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=9,
    )
    add_note(fig)
    save(fig, "carla_frame000011_map_style_reconstruction")


def shade(color: str, factor: float) -> Tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple(np.clip(rgb * factor + (1.0 - factor), 0, 1))


def iso_project(points: np.ndarray) -> np.ndarray:
    """Project local (x, y, z) coordinates to a compact oblique drawing plane."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.c_[x + 0.42 * y, 0.32 * y + 2.8 * z]


def add_iso_cuboid(ax, vehicle, color=BLUE, height=1.5, alpha=0.88, zorder=5):
    xy = rotated_rect(*vehicle)
    bottom3 = np.c_[xy, np.zeros(4)]
    top3 = np.c_[xy, np.ones(4) * height]
    bottom = iso_project(bottom3)
    top = iso_project(top3)

    # Draw visible side faces first, then the roof.
    side_faces = [
        np.array([bottom[0], bottom[1], top[1], top[0]]),
        np.array([bottom[1], bottom[2], top[2], top[1]]),
        np.array([bottom[2], bottom[3], top[3], top[2]]),
    ]
    for i, face in enumerate(side_faces):
        ax.add_patch(
            Polygon(
                face,
                closed=True,
                facecolor=shade(color, 0.58 + 0.08 * i),
                edgecolor="white",
                lw=0.65,
                alpha=alpha,
                zorder=zorder,
            )
        )
    ax.add_patch(
        Polygon(top, closed=True, facecolor=shade(color, 0.82), edgecolor=color, lw=1.1, alpha=alpha, zorder=zorder + 1)
    )


def pseudo_3d():
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.84, bottom=0.13)
    ax.set_title("CARLA frame 000011: pseudo-3D scene reconstruction", fontsize=15, fontweight="bold", color=POLIMI_BLUE)

    road3 = np.array([[-2, -11, 0], [60, -11, 0], [60, 11, 0], [-2, 11, 0]])
    road2 = iso_project(road3)
    ax.add_patch(Polygon(road2, closed=True, facecolor=ROAD, edgecolor="#B8B5AA", lw=1.3, zorder=0))
    for y in [-5, 0, 5]:
        pts = iso_project(np.array([[-2, y, 0.03], [60, y, 0.03]]))
        ax.plot(pts[:, 0], pts[:, 1], color="white", lw=2.0, ls=(0, (6, 8)), zorder=1)

    # Draw back-to-front for cleaner overlap.
    for obj in sorted(OBJECTS, key=lambda v: v[0] + 0.2 * v[1]):
        add_iso_cuboid(ax, obj, color=BLUE, alpha=0.72, height=1.4, zorder=5)
    for obj in RECOVERED:
        add_iso_cuboid(ax, obj, color=RED, alpha=0.92, height=1.65, zorder=8)
    add_iso_cuboid(ax, EGO, color=GREEN, alpha=0.95, height=1.65, zorder=10)

    path = iso_project(np.c_[np.linspace(EGO[0], 56, 40), np.zeros(40), np.ones(40) * 0.08])
    ax.plot(path[:, 0], path[:, 1], color=GREEN, lw=3.0, alpha=0.8, zorder=4)
    label_pt = iso_project(np.array([[37, 2.5, 0.25]]))[0]
    ax.text(label_pt[0], label_pt[1], "future path", color=GREEN, fontsize=10)
    for obj in RECOVERED:
        pt = iso_project(np.array([[obj[0], obj[1], 2.15]]))[0]
        ax.text(pt[0], pt[1], "recovered", color=RED, fontsize=8.5, ha="center", fontweight="bold")

    ax.text(1.0, -7.8, "local oblique view", color=GRAY, fontsize=10)
    ax.set_xlim(-5, 65)
    ax.set_ylim(-8.5, 16.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    add_note(fig)
    save(fig, "carla_frame000011_pseudo3d_reconstruction")


def receiver_story():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.17, wspace=0.08)
    titles = [
        ("Snapshot receiver-request", "missed=5", 5, RED),
        ("Temporal receiver-request", "missed=2", 2, ORANGE),
        ("Learned temporal request", "missed=0", 0, GREEN),
    ]
    for ax, (title, subtitle, missed, color) in zip(axes, titles):
        draw_road(ax, xlim=(0, 58), ylim=(-12, 12))
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", color=color)
        ax.add_patch(Rectangle((EGO[0] - 1, -2.3), 44, 4.6, facecolor=GREEN, alpha=0.08, edgecolor="none", zorder=2))
        ax.add_patch(FancyArrowPatch((EGO[0] + 1.5, 0), (55, 0), arrowstyle="->", mutation_scale=14, lw=2, color=GREEN, alpha=0.75, zorder=7))
        draw_vehicle_2d(ax, EGO, edge=GREEN, face="#D1FAE5", lw=2.4, zorder=10)
        for obj in OBJECTS:
            draw_vehicle_2d(ax, obj, edge=BLUE, face="white", lw=1.5, alpha=0.85)
        if missed >= 5:
            missed_objs = [(0.0, 0.0, 4.5, 2.0, 0), *RECOVERED, (24.0, 6.8, 4.0, 2.0, 0), (33.0, 6.8, 4.0, 2.0, 0)]
        elif missed == 2:
            missed_objs = RECOVERED
        else:
            missed_objs = []
        for obj in missed_objs:
            draw_vehicle_2d(ax, obj, edge=RED, face="#FEE2E2", lw=2.5, alpha=0.95, zorder=9)
        if title.startswith("Learned"):
            for obj in RECOVERED:
                ax.add_patch(plt.Circle((obj[0], obj[1]), 3.0, fill=False, edgecolor=GREEN, lw=2.0, ls="--", alpha=0.9, zorder=12))
                ax.text(obj[0], obj[1] + 3.6, "recovered", color=GREEN, fontsize=8.5, ha="center", fontweight="bold")
        ax.set_xlabel("local x [m]")
    axes[0].set_ylabel("local y [m]")
    fig.suptitle("Receiver-driven progression: from missed boxes to learned recovery", fontsize=15, fontweight="bold", color=POLIMI_BLUE)
    add_note(fig)
    save(fig, "carla_frame000011_receiver_story_reconstruction")


def main():
    map_style()
    pseudo_3d()
    receiver_story()
    print(f"Saved scene reconstruction figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
