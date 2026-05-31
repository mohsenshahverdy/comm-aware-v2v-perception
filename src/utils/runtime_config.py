# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib

import random

import numpy as np
import torch


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
