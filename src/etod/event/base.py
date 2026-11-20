"""Event-trigger interface.

A trigger looks at each incoming frame and decides whether the expensive
detector must run (paper Fig. 1: "Change > Threshold tau?"). New triggering
mechanisms (optical flow, adaptive tau, ...) only need to implement
:meth:`decide` and be registered in ``etod.event.TRIGGERS``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TriggerDecision:
    score: float        # change score D_t (NaN when not computed, e.g. first frame)
    triggered: bool     # E_t: True -> run the detector, False -> skip and reuse
    reason: str         # first_frame | event | no_event | max_skip | always


class EventTrigger(ABC):
    name: str = "base"
    threshold: float = math.nan

    def reset(self, sequence=None) -> None:
        """Called at the start of every sequence."""

    @abstractmethod
    def decide(self, image: np.ndarray) -> TriggerDecision:
        """Inspect a BGR frame and decide whether the detector runs."""

    @property
    def label(self) -> str:
        return self.name


class AlwaysTrigger(EventTrigger):
    """Baseline (paper Fig. 2a): the detector runs on every frame."""

    name = "baseline"

    def decide(self, image: np.ndarray) -> TriggerDecision:
        return TriggerDecision(math.nan, True, "always")
