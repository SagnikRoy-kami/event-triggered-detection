"""Hardware / software snapshot saved with every run (reproducibility)."""

from __future__ import annotations

import os
import platform
import random
import sys
from datetime import datetime

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def resolve_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def collect_environment(device: str | None = None) -> dict:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "device": device,
    }
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        pass
    for mod in ("numpy", "cv2", "torch", "ultralytics", "matplotlib", "pandas"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except ImportError:
            info[f"{mod}_version"] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info
