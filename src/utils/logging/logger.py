"""Centralized logger for consistent project output."""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .constants import (
    EMOJI_MAP,
    LEVEL_COMMAND,
    LEVEL_CONFIG,
    LEVEL_DEBUG,
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_METRIC,
    LEVEL_ORDER,
    LEVEL_RUN,
    LEVEL_SAVE,
    LEVEL_SUCCESS,
    LEVEL_WARN,
)

_DEFAULT_CONFIG = {
    "level": os.getenv("LOG_LEVEL", LEVEL_INFO).upper(),
    "debug": os.getenv("LOG_DEBUG", "false").lower() == "true",
    "timestamp": os.getenv("LOG_TIMESTAMP", "false").lower() == "true",
    "silent": os.getenv("LOG_SILENT", "false").lower() == "true",
    "log_to_file": os.getenv("LOG_TO_FILE", "false").lower() == "true",
    "file_path": os.getenv("LOG_FILE", ""),
    "color": os.getenv("LOG_COLOR", "true").lower() == "true",
}

_COLOR_MAP = {
    LEVEL_INFO: "[36m",
    LEVEL_WARN: "[33m",
    LEVEL_ERROR: "[31m",
    LEVEL_SUCCESS: "[32m",
    LEVEL_DEBUG: "[35m",
    LEVEL_METRIC: "[34m",
    LEVEL_CONFIG: "[96m",
    LEVEL_RUN: "[95m",
    LEVEL_COMMAND: "[90m",
    LEVEL_SAVE: "[92m",
}
_COLOR_END = "[0m"


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class LoggerConfig:
    level: str = LEVEL_INFO
    debug: bool = False
    timestamp: bool = False
    silent: bool = False
    log_to_file: bool = False
    file_path: str = ""
    color: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "LoggerConfig":
        level = str(cfg.get("level", LEVEL_INFO)).upper()
        if level not in LEVEL_ORDER:
            level = LEVEL_INFO
        return cls(
            level=level,
            debug=_coerce_bool(cfg.get("debug"), False),
            timestamp=_coerce_bool(cfg.get("timestamp"), False),
            silent=_coerce_bool(cfg.get("silent"), False),
            log_to_file=_coerce_bool(cfg.get("log_to_file"), False),
            file_path=str(cfg.get("file_path") or ""),
            color=_coerce_bool(cfg.get("color"), True),
        )


_GLOBAL_CONFIG = LoggerConfig.from_dict(_DEFAULT_CONFIG)
_LOGGER_CACHE: Dict[str, "CentralLogger"] = {}
_LOCK = threading.Lock()


def set_logging_config(**kwargs) -> None:
    global _GLOBAL_CONFIG
    merged = {
        "level": kwargs.get("level", _GLOBAL_CONFIG.level),
        "debug": kwargs.get("debug", _GLOBAL_CONFIG.debug),
        "timestamp": kwargs.get("timestamp", _GLOBAL_CONFIG.timestamp),
        "silent": kwargs.get("silent", _GLOBAL_CONFIG.silent),
        "log_to_file": kwargs.get("log_to_file", _GLOBAL_CONFIG.log_to_file),
        "file_path": kwargs.get("file_path", _GLOBAL_CONFIG.file_path),
        "color": kwargs.get("color", _GLOBAL_CONFIG.color),
    }
    _GLOBAL_CONFIG = LoggerConfig.from_dict(merged)


def get_logger(component: str, **overrides) -> "CentralLogger":
    key = f"{component}|{sorted(overrides.items())}"
    with _LOCK:
        if key in _LOGGER_CACHE:
            return _LOGGER_CACHE[key]
        cfg = {
            "level": overrides.get("level", _GLOBAL_CONFIG.level),
            "debug": overrides.get("debug", _GLOBAL_CONFIG.debug),
            "timestamp": overrides.get("timestamp", _GLOBAL_CONFIG.timestamp),
            "silent": overrides.get("silent", _GLOBAL_CONFIG.silent),
            "log_to_file": overrides.get("log_to_file", _GLOBAL_CONFIG.log_to_file),
            "file_path": overrides.get("file_path", _GLOBAL_CONFIG.file_path),
            "color": overrides.get("color", _GLOBAL_CONFIG.color),
        }
        logger = CentralLogger(component=component, config=LoggerConfig.from_dict(cfg))
        _LOGGER_CACHE[key] = logger
        return logger


class CentralLogger:
    def __init__(self, component: str, config: LoggerConfig):
        self.component = component
        self._cfg = config

    def is_debug_enabled(self) -> bool:
        return bool(self._cfg.debug) or self._is_enabled(LEVEL_DEBUG)

    def _is_enabled(self, level: str) -> bool:
        if self._cfg.silent:
            return False
        if level == LEVEL_DEBUG and not self._cfg.debug:
            return False
        cur = LEVEL_ORDER.get(self._cfg.level, LEVEL_ORDER[LEVEL_INFO])
        need = LEVEL_ORDER.get(level, LEVEL_ORDER[LEVEL_INFO])
        return need >= cur

    def _safe_value(self, value: Any) -> str:
        try:
            if value is None:
                return "null"
            if isinstance(value, (int, float, bool, str)):
                return str(value)
            if isinstance(value, pathlib.Path):
                return str(value)
            if isinstance(value, dict):
                return json.dumps({k: self._safe_value(v) for k, v in value.items()}, ensure_ascii=False)
            if isinstance(value, (list, tuple, set)):
                seq = list(value)
                short = seq[:6]
                suffix = "..." if len(seq) > 6 else ""
                return f"[{', '.join(self._safe_value(v) for v in short)}{suffix}]"
            if hasattr(value, "shape"):
                return f"shape={tuple(value.shape)}"
            if hasattr(value, "item"):
                return str(value.item())
            return str(value)
        except Exception:
            return "<unserializable>"

    def _kv_text(self, fields: Dict[str, Any]) -> str:
        if not fields:
            return ""
        parts = []
        for k, v in fields.items():
            parts.append(f"{k}={self._safe_value(v)}")
        return " | " + " ".join(parts)

    def _format(self, level: str, message: str, fields: Dict[str, Any]) -> str:
        emoji = EMOJI_MAP.get(level, "ℹ️")
        ts = ""
        if self._cfg.timestamp:
            ts = f"[{_dt.datetime.utcnow().isoformat(timespec='seconds')}Z] "
        line = f"{emoji} {ts}[{level}] [{self.component}] {message}{self._kv_text(fields)}"
        if self._cfg.color and level in _COLOR_MAP:
            return f"{_COLOR_MAP[level]}{line}{_COLOR_END}"
        return line

    def _write(self, level: str, message: str, **fields) -> None:
        if not self._is_enabled(level):
            return
        line = self._format(level, message, fields)
        print(line)
        if self._cfg.log_to_file and self._cfg.file_path:
            try:
                path = pathlib.Path(self._cfg.file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                clean_line = line
                if "[" in clean_line:
                    import re
                    clean_line = re.sub(r"\x1b\[[0-9;]*m", "", clean_line)
                with open(path, "a") as f:
                    f.write(clean_line + "\n")
            except Exception:
                pass

    def info(self, message: str, **fields):
        self._write(LEVEL_INFO, message, **fields)

    def warn(self, message: str, **fields):
        self._write(LEVEL_WARN, message, **fields)

    def warning(self, message: str, **fields):
        self.warn(message, **fields)

    def error(self, message: str, **fields):
        self._write(LEVEL_ERROR, message, **fields)

    def success(self, message: str, **fields):
        self._write(LEVEL_SUCCESS, message, **fields)

    def debug(self, message: str, **fields):
        self._write(LEVEL_DEBUG, message, **fields)

    def metric(self, message: str, **fields):
        self._write(LEVEL_METRIC, message, **fields)

    def config(self, message: str, **fields):
        self._write(LEVEL_CONFIG, message, **fields)

    def run(self, message: str, **fields):
        self._write(LEVEL_RUN, message, **fields)

    def step(self, message: str, **fields):
        self._write(LEVEL_RUN, message, **fields)

    def progress(self, message: str, **fields):
        self._write(LEVEL_RUN, message, **fields)

    def command(self, message: str, **fields):
        self._write(LEVEL_COMMAND, message, **fields)

    def save(self, message: str, **fields):
        self._write(LEVEL_SAVE, message, **fields)
