"""Efficiency metrics (paper Sec. III-E / IV-E).

    invocation rate  = N_d / N_f
    frame-skip rate  S = (1 - N_d / N_f) * 100                     (Eq. 2 / Eq. 5)

N_d = frames processed by the detector, N_f = total input frames.
FPS and latency use the processing time per frame (event module + detector,
or reuse), excluding disk/video decoding which is identical for both methods
and reported separately as ``fps_incl_io``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def skip_rate(n_detected: int, n_frames: int) -> float:
    """Eq. 2: S = (1 - N_d / N_f) x 100."""
    if n_frames <= 0:
        return float("nan")
    return (1.0 - n_detected / n_frames) * 100.0


def efficiency_metrics(log: pd.DataFrame, gflops_per_inference: float | None = None) -> dict:
    n_f = int(len(log))
    n_d = int(log["triggered"].sum())
    total_ms = log["total_ms"].to_numpy(float)
    proc_s = total_ms.sum() / 1000.0
    io_s = log["read_ms"].to_numpy(float).sum() / 1000.0 if "read_ms" in log else 0.0
    det = log.loc[log["triggered"], "detector_ms"].to_numpy(float)
    out = {
        "N_f": n_f,
        "N_d": n_d,
        "invocation_rate": n_d / n_f if n_f else float("nan"),
        "skip_rate": skip_rate(n_d, n_f),
        "fps": n_f / proc_s if proc_s > 0 else float("nan"),
        "fps_incl_io": n_f / (proc_s + io_s) if (proc_s + io_s) > 0 else float("nan"),
        "latency_mean_ms": float(total_ms.mean()) if n_f else float("nan"),
        "latency_p50_ms": float(np.percentile(total_ms, 50)) if n_f else float("nan"),
        "latency_p95_ms": float(np.percentile(total_ms, 95)) if n_f else float("nan"),
        "latency_max_ms": float(total_ms.max()) if n_f else float("nan"),
        "detector_latency_mean_ms": float(det.mean()) if len(det) else float("nan"),
        "event_overhead_mean_ms": float(log["event_ms"].mean()) if n_f else 0.0,
        "processing_time_s": proc_s,
    }
    if gflops_per_inference:
        out["gflops_total"] = n_d * gflops_per_inference
        out["gflops_per_frame"] = out["gflops_total"] / n_f if n_f else float("nan")
    return out


def activity_breakdown(event_log: pd.DataFrame, profile: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    """Invocation rate in low / moderate / high activity periods (paper IV-A).

    Activity level of a frame = its D_t (from the baseline profile) binned by
    the dataset-wide terciles ``bins``.
    """
    df = event_log[["sequence", "frame_index", "triggered"]].merge(
        profile[["sequence", "frame_index", "score"]], on=["sequence", "frame_index"], how="left"
    )
    df = df[np.isfinite(df["score"])]
    labels = ["low", "moderate", "high"]
    df["activity"] = pd.cut(df["score"], [-np.inf, bins[0], bins[1], np.inf], labels=labels)
    g = df.groupby("activity", observed=False)
    return pd.DataFrame({
        "activity": labels,
        "frames": g.size().reindex(labels).fillna(0).astype(int).to_numpy(),
        "mean_D": g["score"].mean().reindex(labels).to_numpy(),
        "invocation_rate": g["triggered"].mean().reindex(labels).to_numpy(),
    })
