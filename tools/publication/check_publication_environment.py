#!/usr/bin/env python3
"""Check publication dataset/checkpoint paths without importing model code."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger  # noqa: E402


def load_config(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not load publication config {path}: {exc}") from None


def variable_name(template: str) -> str | None:
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(template).strip())
    return match.group(1) if match else None


def _path_status(name: str, notes: str, *, checkpoint_patterns: List[str] | None = None) -> Tuple[str, str, str, str]:
    value = os.environ.get(name, "")
    if not value:
        return name, "<unset>", "no", f"{notes}; environment variable is missing"
    path = Path(os.path.expanduser(value))
    if not path.exists():
        return name, str(path), "no", f"{notes}; path does not exist"
    if not path.is_dir():
        return name, str(path), "no", f"{notes}; expected a directory"
    if checkpoint_patterns:
        matches = {candidate for pattern in checkpoint_patterns for candidate in path.glob(pattern)}
        if not matches:
            return name, str(path), "no", f"{notes}; no checkpoint matches {checkpoint_patterns}"
        return name, str(path), "yes", f"{notes}; {len(matches)} checkpoint file(s) found"
    return name, str(path), "yes", notes


def required_checks(config: Dict, dataset: str) -> Tuple[List[Tuple[str, str, List[str] | None]], List[Tuple[str, str, List[str] | None]]]:
    patterns = list(config.get("execution", {}).get("checkpoint_patterns", ["net_epoch*.pth", "latest.pth"]))
    paths = config.get("paths", {})
    dataset_cfg = paths.get("datasets", {}).get(dataset, {})
    required: List[Tuple[str, str, List[str] | None]] = []

    checkpoint_var = variable_name(paths.get("checkpoint_dir", "")) or "PUBLICATION_CHECKPOINT_DIR"
    required.append((checkpoint_var, "base detector/fusion checkpoint", patterns))

    for key, note in (("root_dir", "training/root dataset path"), ("validate_dir", f"{dataset} evaluation path")):
        name = variable_name(dataset_cfg.get(key, ""))
        if name and all(existing[0] != name for existing in required):
            required.append((name, note, None))

    learned_var = variable_name(paths.get("learned_checkpoint_dir", "")) or "LEARNED_PUBLICATION_CHECKPOINT_DIR"
    optional = [(learned_var, "required only for learned temporal jobs", patterns)]
    return required, optional


def print_table(rows: List[Tuple[str, str, str, str]]) -> None:
    headers = ("VARIABLE", "VALUE", "EXISTS", "NOTES")
    widths = []
    for index, header in enumerate(headers):
        widths.append(max(len(header), *(len(row[index]) for row in rows)))
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/publication/publication_sweep_config.yaml"),
    )
    parser.add_argument("--dataset", choices=["carla_2021", "culver_city"], default="carla_2021")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "WARN" if args.log_level == "WARNING" else args.log_level
    logger = get_logger("PublicationEnvironment", level="DEBUG" if args.debug else log_level, debug=args.debug)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    try:
        config = load_config(config_path)
        required, optional = required_checks(config, args.dataset)
    except ValueError as exc:
        logger.error("Environment check failed", error=str(exc))
        if args.debug:
            raise
        return 2

    rows = [_path_status(name, note, checkpoint_patterns=patterns) for name, note, patterns in required]
    rows.extend(
        _path_status(name, note, checkpoint_patterns=patterns)
        for name, note, patterns in optional
    )
    logger.config("Checking publication environment", dataset=args.dataset, config=config_path)
    # A plain table is intentionally retained because aligned columns are more
    # readable than repeated key/value log fields for this preflight report.
    print_table(rows)
    required_names = {item[0] for item in required}
    missing_required = [row[0] for row in rows if row[0] in required_names and row[2] != "yes"]
    if missing_required:
        logger.error(
            "Publication environment is not ready",
            missing=missing_required,
            hint="Export the listed variables before running the selected dataset",
        )
        return 1
    logger.success("Publication environment ready", dataset=args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
