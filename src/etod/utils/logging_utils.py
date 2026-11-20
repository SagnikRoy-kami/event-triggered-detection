"""Logging to console and to <run>/logs/run.log."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("etod")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("etod")
