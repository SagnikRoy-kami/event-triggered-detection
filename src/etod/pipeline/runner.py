"""Executes one method (baseline or event-triggered at one tau) over all sequences.

Per frame (paper Fig. 1):
    1. the trigger computes D_t and decides E_t            (event module)
    2. E_t = 1 -> run the YOLO detector on the frame        (expensive path)
       E_t = 0 -> skip detection, reuse previous result     (cheap path)
    3. the frame's output (fresh or reused) is recorded and later scored

Timing uses time.perf_counter around steps 1-2. Reading/decoding the frame is
timed separately because it is identical for every method.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from etod.data.base import Boxes, Sequence
from etod.detectors.base import BaseDetector
from etod.evaluation.resources import ResourceMonitor
from etod.event.base import EventTrigger


@dataclass
class MethodRun:
    method: str
    threshold: float
    frame_log: pd.DataFrame
    predictions: dict[tuple[str, int], Boxes]
    resources: dict = field(default_factory=dict)
    resource_samples: pd.DataFrame = field(default_factory=pd.DataFrame)
    wall_time_s: float = 0.0


@dataclass
class DetectionCache:
    """Baseline detections + their measured latency, used by timing_mode=replay."""

    boxes: dict[tuple[str, int], Boxes]
    latency_ms: dict[tuple[str, int], float]


def run_method(
    sequences: list[Sequence],
    detector: BaseDetector,
    trigger: EventTrigger,
    profiler: EventTrigger | None = None,
    monitor: ResourceMonitor | None = None,
    cache: DetectionCache | None = None,
    warmup_iters: int = 3,
    desc: str | None = None,
) -> MethodRun:
    """Run ``trigger`` + ``detector`` over every frame of every sequence.

    profiler : optional trigger used only to *measure* D_t (untimed), so the
               baseline run also yields the D_t profile used for auto-thresholds.
    cache    : if given, detections are taken from the cache instead of running
               the detector (replay mode); the cached latency is charged instead.
    """
    rows = []
    preds: dict[tuple[str, int], Boxes] = {}
    label = desc or trigger.label

    if cache is None and warmup_iters > 0:
        first = next(iter(sequences[0].frames()))
        detector.warmup(first.image, warmup_iters)

    total_frames = sum(s.num_frames for s in sequences) or None
    bar = tqdm(total=total_frames, desc=f"{label:>18s}", unit="f", leave=False, dynamic_ncols=True)
    wall0 = time.perf_counter()
    if monitor:
        monitor.start()
    try:
        for seq in sequences:
            trigger.reset(seq)
            if profiler is not None:
                profiler.reset(seq)
            last = Boxes.empty()
            frames = seq.frames()
            while True:
                r0 = time.perf_counter()
                frame = next(frames, None)
                read_ms = (time.perf_counter() - r0) * 1000.0
                if frame is None:
                    break

                t0 = time.perf_counter()
                decision = trigger.decide(frame.image)            # D_t and E_t (Eq. 1, Eq. 4)
                t1 = time.perf_counter()
                key = (seq.name, frame.frame_id)
                if decision.triggered:
                    if cache is not None:
                        last = cache.boxes[key]
                        det_ms = cache.latency_ms[key]
                    else:
                        last = detector.detect(frame.image)       # expensive path
                        det_ms = (time.perf_counter() - t1) * 1000.0
                else:
                    det_ms = 0.0                                  # skip: reuse previous result
                t2 = time.perf_counter()
                event_ms = (t1 - t0) * 1000.0
                total_ms = event_ms + det_ms if cache is not None else (t2 - t0) * 1000.0
                preds[key] = last

                score = decision.score
                if profiler is not None:
                    score = profiler.decide(frame.image).score   # untimed measurement of D_t
                rows.append((seq.name, frame.index, frame.frame_id, score, bool(decision.triggered), decision.reason,
                             event_ms, det_ms, total_ms, read_ms, len(last)))
                bar.update(1)
    finally:
        if monitor:
            monitor.stop()
        bar.close()

    log = pd.DataFrame(rows, columns=["sequence", "frame_index", "frame_id", "score", "triggered", "reason",
                                      "event_ms", "detector_ms", "total_ms", "read_ms", "num_detections"])
    thr = getattr(trigger, "threshold", math.nan)
    return MethodRun(
        method=trigger.label, threshold=float(thr) if thr is not None else math.nan, frame_log=log, predictions=preds,
        resources=monitor.summary() if monitor else {},
        resource_samples=monitor.dataframe() if monitor else pd.DataFrame(),
        wall_time_s=time.perf_counter() - wall0,
    )


def choose_thresholds(cfg, profile: pd.DataFrame) -> list[float]:
    """Explicit list from the config, or percentiles of the measured D_t distribution."""
    spec = cfg["event_trigger"]["thresholds"]
    if isinstance(spec, (list, tuple)):
        return sorted(float(t) for t in spec)
    if str(spec).lower() != "auto":
        return [float(spec)]
    d = profile["score"].to_numpy(float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        raise RuntimeError("Cannot choose thresholds automatically: no D_t values were measured")
    pcts = cfg["event_trigger"].get("auto_percentiles", [10, 25, 40, 50, 60, 75, 90])
    vals = np.percentile(d, pcts)
    decimals = 3 if np.median(d) < 1 else 2
    return sorted({round(float(v), decimals) for v in vals})
