#!/usr/bin/env python3
"""Run publication framework tests directly when pytest is unavailable."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests import test_publication_experiments as tests
from src.utils.logging import get_logger


class DirectMonkeyPatch:
    def __init__(self):
        self._environment = {}
        self._attributes = []

    def setenv(self, key, value):
        if key not in self._environment:
            self._environment[key] = os.environ.get(key)
        os.environ[key] = str(value)

    def delenv(self, key, raising=True):
        if key not in self._environment:
            self._environment[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
        elif raising:
            raise KeyError(key)

    def setattr(self, target, name, value):
        self._attributes.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, value in reversed(self._attributes):
            setattr(target, name, value)
        for key, value in self._environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run(logger, name, function, *args):
    function(*args)
    logger.success("Mock validation passed", test=name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "WARN" if args.log_level == "WARNING" else args.log_level
    logger = get_logger("PublicationMockTests", level="DEBUG" if args.debug else log_level, debug=args.debug)
    try:
        _run(logger, "grid", tests.test_publication_config_loads_and_grid_is_stable)
        _run(logger, "presets", tests.test_existing_preset_names_resolve)
        _run(logger, "schema", tests.test_result_schema_is_complete_and_ordered)
        _run(logger, "packet schema counters", tests.test_result_schema_reads_empirical_packet_loss_counters)
        _run(logger, "packet configured probability not actual", tests.test_result_schema_does_not_copy_configured_packet_probability_as_actual_rate)
        _run(logger, "layout", tests.test_run_layout_and_smoke_command)

        patch = DirectMonkeyPatch()
        try:
            _run(logger, "budget overrides", tests.test_budget_overrides_resolve_to_ratios, patch)
            _run(logger, "packet Monte Carlo seeds", tests.test_packet_loss_seed_uses_monte_carlo_run, patch)
        finally:
            patch.undo()

        _run(logger, "explicit packet loss grid", tests.test_explicit_packet_loss_filter_expands_disabled_scenario)
        _run(logger, "packet masks reproducible", tests.test_packet_loss_masks_are_reproducible_and_counted)
        _run(logger, "where2comm-style confidence mask", tests.test_where2comm_style_confidence_topk_mask_ratio)

        patch = DirectMonkeyPatch()
        try:
            _run(logger, "missing environment", tests.test_missing_environment_is_reported_without_exception, patch)
        finally:
            patch.undo()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _run(logger, "command exports", tests.test_command_exports_have_expected_counts, root / "commands")
            patch = DirectMonkeyPatch()
            try:
                _run(logger, "mock successful execution", tests.test_mock_execution_stages_reproducibility_artifacts, root / "success", patch)
            finally:
                patch.undo()
            patch = DirectMonkeyPatch()
            try:
                _run(logger, "missing native outputs", tests.test_missing_native_outputs_never_mark_run_completed, root / "missing", patch)
            finally:
                patch.undo()
            _run(logger, "aggregation and plotting", tests.test_mock_result_aggregation_and_plotting, root / "aggregate")
    except Exception as exc:
        logger.error("Publication mock validation failed", error=str(exc))
        if args.debug:
            raise
        return 1

    logger.success("Publication framework direct tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
