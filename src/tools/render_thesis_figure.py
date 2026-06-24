"""Render thesis BEV figures with deterministic road/car visualization.

Example:
    python -m src.tools.render_thesis_figure \
      --input panel_spec.json \
      --output Classical_Format_Thesis/figures/generated/example_road_cars \
      --style road_cars

The JSON input is intentionally small and explicit. It should contain the boxes
and match masks already computed by the evaluation pipeline; this renderer does
not perform detection matching and therefore does not change counts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.visualization.thesis_renderer import RendererConfig, RoadCarPanelRenderer, legend_handles


@dataclass
class _FrameShim:
    method: str
    pred_boxes: np.ndarray
    trajectory: Any = None
    collaborator_positions: Any = None


def _as_boxes(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    return arr.reshape(-1, 4, 2)


def _axis_limits(gt: np.ndarray, pred: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts = []
    if gt.size:
        pts.append(gt.reshape(-1, 2))
    if pred.size:
        pts.append(pred.reshape(-1, 2))
    if not pts:
        return (-10.0, 60.0), (-15.0, 15.0)
    all_pts = np.concatenate(pts, axis=0)
    xmin, ymin = np.nanmin(all_pts, axis=0)
    xmax, ymax = np.nanmax(all_pts, axis=0)
    return (float(xmin - 8.0), float(xmax + 8.0)), (float(ymin - 6.0), float(ymax + 6.0))


def _demo_spec() -> Dict[str, Any]:
    def box(cx, cy, l=4.5, w=2.0):
        return [[cx - l / 2, cy - w / 2], [cx + l / 2, cy - w / 2], [cx + l / 2, cy + w / 2], [cx - l / 2, cy + w / 2]]

    gt = [box(12, 0), box(24, -5), box(30, 6), box(42, 0), box(50, 7)]
    pred = [box(12, 0), box(24, -5), box(30, 6), box(42, 0), box(50, 7)]
    return {
        "title": "Demo road-cars renderer",
        "subtitle": "Deterministic rendering; geometry supplied by JSON input.",
        "rows": 1,
        "cols": 1,
        "panels": [
            {
                "title": "Learned\nTP=5  FP=0  Missed=0",
                "gt_boxes": gt,
                "pred_boxes": pred,
                "gt_matched": [True, True, True, True, True],
                "pred_matched": [True, True, True, True, True],
                "xlim": [0, 60],
                "ylim": [-14, 14],
            }
        ],
    }


def _load_spec(path: str | None, demo: bool) -> Dict[str, Any]:
    if demo:
        return _demo_spec()
    if not path:
        raise ValueError("Provide --input JSON or use --demo.")
    with open(path, "r") as f:
        return json.load(f)


def _render(spec: Dict[str, Any], output: Path, style: str) -> None:
    import matplotlib.pyplot as plt

    panels: List[Dict[str, Any]] = list(spec.get("panels", []))
    if not panels:
        raise ValueError("Input spec must contain a non-empty 'panels' list.")
    rows = int(spec.get("rows", 1))
    cols = int(spec.get("cols", len(panels)))
    if rows * cols < len(panels):
        raise ValueError("rows * cols is smaller than number of panels.")

    cfg = RendererConfig()
    renderer = RoadCarPanelRenderer(cfg)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.reshape(-1)[len(panels):]:
        ax.axis("off")

    for ax, panel in zip(axes.reshape(-1), panels):
        gt = _as_boxes(panel.get("gt_boxes", []))
        pred = _as_boxes(panel.get("pred_boxes", []))
        gt_matched = np.asarray(panel.get("gt_matched", [False] * len(gt)), dtype=bool)
        pred_matched = np.asarray(panel.get("pred_matched", [False] * len(pred)), dtype=bool)
        xlim = tuple(panel.get("xlim", _axis_limits(gt, pred)[0]))
        ylim = tuple(panel.get("ylim", _axis_limits(gt, pred)[1]))
        frame = _FrameShim(method=str(panel.get("method", panel.get("title", "Panel"))), pred_boxes=pred)
        if style != "road_cars":
            raise ValueError("This standalone renderer currently supports --style road_cars.")
        renderer.draw_panel(
            ax,
            frame=frame,
            gt_boxes=gt,
            match={"gt_matched": gt_matched, "pred_matched": pred_matched, "pairs": []},
            xlim=xlim,
            ylim=ylim,
            title=str(panel.get("title", "")),
            show_xlabel=True,
            show_ylabel=True,
        )

    if spec.get("title"):
        fig.suptitle(str(spec["title"]), fontsize=15, fontweight="bold", y=0.980)
    if spec.get("subtitle"):
        fig.text(0.5, 0.905, str(spec["subtitle"]), ha="center", fontsize=11, color="#333333")
    fig.legend(handles=legend_handles(cfg), loc="lower center", ncol=5, frameon=False, fontsize=9.3, bbox_to_anchor=(0.5, 0.020))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.800, bottom=0.205, wspace=0.08, hspace=0.16)

    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved {stem.with_suffix('.pdf')}")
    print(f"saved {stem.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render deterministic thesis BEV figures with road/car styling.")
    parser.add_argument("--input", default=None, help="JSON panel specification. Use --demo for a synthetic example.")
    parser.add_argument("--output", required=True, help="Output path stem or file path. .pdf and .png are both written.")
    parser.add_argument("--style", default="road_cars", choices=["road_cars"])
    parser.add_argument("--demo", action="store_true", help="Render a small synthetic demo spec.")
    args = parser.parse_args()
    spec = _load_spec(args.input, args.demo)
    _render(spec, Path(args.output), args.style)


if __name__ == "__main__":
    main()
