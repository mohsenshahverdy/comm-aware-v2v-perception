"""Configuration for deterministic thesis BEV rendering."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RendererConfig:
    """Visual settings for road + vehicle-icon thesis figures."""

    background_type: str = "road"
    theme: str = "polimi_road"
    show_axes: bool = True
    show_grid: bool = False
    show_road_details: bool = True
    show_shoulders: bool = False
    show_crosswalk: bool = True
    show_matched_gt: bool = True
    vehicle_alpha: float = 0.96
    matched_gt_alpha: float = 0.75
    road_alpha: float = 1.0
    asphalt_color: str = "#2F3334"
    asphalt_edge_color: str = "#6D7275"
    shoulder_color: str = "#6B7072"
    sidewalk_color: str = "#9A9A94"
    lane_marking_color: str = "#E8E6DA"
    centerline_color: str = "#C9942C"
    tp_color: str = "#0072B2"
    fp_color: str = "#E69F00"
    missed_color: str = "#D62728"
    matched_gt_color: str = "#B8B8B8"
    ego_color: str = "#009E73"
    vehicle_body_color: str = "#111111"
    vehicle_window_color: str = "#3A3A3A"
    vehicle_highlight_color: str = "#F2F2F2"
    tp_linewidth: float = 2.2
    fp_linewidth: float = 2.2
    missed_linewidth: float = 2.8
    matched_gt_linewidth: float = 1.5
    vehicle_scale: float = 0.86
    dpi: int = 320
