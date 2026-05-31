# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib

import random

import numpy as np
import torch


def _get_communication_cfg(hypes):
    model_args = hypes.get("model", {}).get("args", {}) if isinstance(hypes, dict) else {}
    comm_cfg = model_args.get("communication", None)
    if isinstance(comm_cfg, dict):
        return comm_cfg
    top_comm = hypes.get("communication", {})
    if isinstance(top_comm, dict):
        return top_comm
    return {}


def log_and_validate_communication_approach(hypes, logger=None):
    comm_cfg = _get_communication_cfg(hypes)
    if not comm_cfg:
        return

    metadata = comm_cfg.get("metadata", {}) if isinstance(comm_cfg.get("metadata", {}), dict) else {}
    rr_cfg = comm_cfg.get("receiver_request", {}) if isinstance(comm_cfg.get("receiver_request", {}), dict) else {}
    rr_loss = rr_cfg.get("loss", {}) if isinstance(rr_cfg.get("loss", {}), dict) else {}

    public_name = metadata.get("public_name", "unknown")
    approach_family = metadata.get("approach_family", "default")
    approach_name = metadata.get("approach_name", "default")
    approach_setting = metadata.get("approach_setting", "default")
    implementation_status = metadata.get("implementation_status", rr_cfg.get("implementation_status", "implemented"))
    strategy = comm_cfg.get("strategy", "none")
    trainable = bool(rr_cfg.get("trainable", False))
    loss_enabled = bool(rr_loss.get("enabled", False))
    comm_enabled = bool(comm_cfg.get("enabled", False))
    rr_enabled = bool(rr_cfg.get("enabled", False))

    if logger is not None:
        logger.info(
            "Communication approach",
            public_name=public_name,
            approach_family=approach_family,
            approach_name=approach_name,
            approach_setting=approach_setting,
            implementation_status=implementation_status,
            strategy=strategy,
            trainable=trainable,
            **{"loss.enabled": loss_enabled},
        )

    implementation_status_l = str(implementation_status).lower()
    requested_policy = (
        str(strategy).lower() != "none"
        or rr_enabled
        or trainable
        or loss_enabled
    )
    if implementation_status_l in {"planned_placeholder", "planned"} and requested_policy:
        raise ValueError(
            "Communication approach is planned_placeholder and cannot run yet. "
            f"public_name={public_name} strategy={strategy} status={implementation_status}"
        )
    if not comm_enabled and str(strategy).lower() != "none":
        raise ValueError(
            "Communication approach is disabled but a strategy was requested. "
            f"public_name={public_name} strategy={strategy} enabled={comm_enabled}"
        )


def set_global_seed(seed, deterministic=False, benchmark=True, logger=None):
    """Apply reproducibility settings for python, numpy, and torch."""
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark and not deterministic)

    if logger is not None:
        logger.config(
            "Reproducibility seed applied",
            seed=seed,
            deterministic=bool(deterministic),
            benchmark=bool(torch.backends.cudnn.benchmark),
        )


def apply_runtime_overrides(hypes, opt, logger=None):
    """Apply CLI overrides to config and return reproducibility tuple."""
    if getattr(opt, "root_dir", None):
        hypes['root_dir'] = opt.root_dir
    if getattr(opt, "validate_dir", None):
        hypes['validate_dir'] = opt.validate_dir

    rep_cfg = hypes.get('reproducibility', {})
    seed = getattr(opt, 'seed', None)
    if seed is None:
        seed = rep_cfg.get('seed', None)

    deterministic = bool(
        getattr(opt, 'deterministic', False) or rep_cfg.get('deterministic', False)
    )
    benchmark = bool(rep_cfg.get('benchmark', True))
    if getattr(opt, 'benchmark', None) is not None:
        benchmark = bool(opt.benchmark)

    if logger is not None:
        logger.config(
            "Runtime overrides applied",
            root_dir=hypes.get('root_dir', ''),
            validate_dir=hypes.get('validate_dir', ''),
            seed=seed,
            deterministic=deterministic,
            benchmark=benchmark,
        )

    return seed, deterministic, benchmark
