"""Trigger registry. Add a new triggering mechanism by subclassing
``EventTrigger`` and registering a builder in ``TRIGGERS``."""

from __future__ import annotations

from etod.event.base import AlwaysTrigger, EventTrigger, TriggerDecision
from etod.event.temporal_difference import TemporalDifferenceTrigger


def _temporal_difference(cfg, threshold: float) -> EventTrigger:
    et = cfg["event_trigger"]
    return TemporalDifferenceTrigger(
        threshold=threshold,
        grayscale=et.get("grayscale", True),
        resize_width=et.get("resize_width"),
        blur_kernel=et.get("blur_kernel", 0),
        reference=et.get("reference", "previous"),
        max_skip=et.get("max_skip"),
        first_frame_always_detect=et.get("first_frame_always_detect", True),
        mask_ignore_regions=et.get("mask_ignore_regions", False),
    )


TRIGGERS = {
    "temporal_difference": _temporal_difference,
    "always": lambda cfg, threshold=None: AlwaysTrigger(),
}


def build_trigger(cfg, threshold: float | None) -> EventTrigger:
    method = cfg["event_trigger"]["method"]
    if method not in TRIGGERS:
        raise ValueError(f"Unknown event_trigger.method '{method}'. Available: {sorted(TRIGGERS)}")
    return TRIGGERS[method](cfg, threshold)


def build_profiler(cfg) -> TemporalDifferenceTrigger:
    """A trigger used only to *measure* D_t on every frame (threshold irrelevant)."""
    et = cfg["event_trigger"]
    return TemporalDifferenceTrigger(
        threshold=float("inf"), grayscale=et.get("grayscale", True), resize_width=et.get("resize_width"),
        blur_kernel=et.get("blur_kernel", 0), reference="previous",
        mask_ignore_regions=et.get("mask_ignore_regions", False),
    )


__all__ = ["AlwaysTrigger", "EventTrigger", "TemporalDifferenceTrigger", "TriggerDecision", "TRIGGERS", "build_trigger", "build_profiler"]
