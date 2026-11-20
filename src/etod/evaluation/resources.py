"""CPU / RAM / GPU utilisation sampled in a background thread while a method runs
(paper III-E: "CPU/GPU utilization")."""

from __future__ import annotations

import os
import threading
import time

import pandas as pd

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class _GPU:
    def __init__(self) -> None:
        self.handle = None
        self.torch_cuda = False
        try:
            import torch

            self.torch_cuda = torch.cuda.is_available()
        except ImportError:
            pass
        if self.torch_cuda:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nv = pynvml
            except Exception:  # noqa: BLE001 - pynvml optional
                self.handle = None

    def sample(self) -> dict:
        out = {}
        if self.handle is not None:
            try:
                out["gpu_util_pct"] = float(self._nv.nvmlDeviceGetUtilizationRates(self.handle).gpu)
            except Exception:  # noqa: BLE001
                pass
        if self.torch_cuda:
            import torch

            out["gpu_mem_mb"] = torch.cuda.memory_allocated() / 1e6
        return out


class ResourceMonitor:
    def __init__(self, interval: float = 0.25, enabled: bool = True) -> None:
        self.interval = interval
        self.enabled = enabled and psutil is not None
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu = _GPU() if self.enabled else None
        self._ncpu = os.cpu_count() or 1

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if not self.enabled:
            return
        self.samples = []
        self._stop.clear()
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)
        psutil.cpu_percent(None)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            s = {
                "t": time.perf_counter() - self._t0,
                # process CPU normalised to 0-100 % of the whole machine
                "process_cpu_pct": self._proc.cpu_percent(None) / self._ncpu,
                "system_cpu_pct": psutil.cpu_percent(None),
                "rss_mb": self._proc.memory_info().rss / 1e6,
            }
            s.update(self._gpu.sample())
            self.samples.append(s)

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
            self._thread = None

    def summary(self) -> dict:
        if not self.samples:
            return {}
        df = pd.DataFrame(self.samples)
        out = {
            "cpu_process_mean_pct": df["process_cpu_pct"].mean(),
            "cpu_process_max_pct": df["process_cpu_pct"].max(),
            "cpu_system_mean_pct": df["system_cpu_pct"].mean(),
            "ram_peak_mb": df["rss_mb"].max(),
        }
        if "gpu_util_pct" in df:
            out["gpu_util_mean_pct"] = df["gpu_util_pct"].mean()
        try:
            import torch

            if torch.cuda.is_available():
                out["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
        except ImportError:
            pass
        return out

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.samples)
