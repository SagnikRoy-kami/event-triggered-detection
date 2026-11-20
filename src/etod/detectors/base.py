"""Detector interface. Both the baseline and the proposed method use the same
detector instance, resolution, confidence threshold and hardware (paper III-D)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from etod.data.base import Boxes


class BaseDetector(ABC):
    name: str = "detector"

    @abstractmethod
    def detect(self, image: np.ndarray) -> Boxes:
        """Run inference on one BGR frame. Labels must be eval-class indices."""

    def warmup(self, image: np.ndarray, iters: int = 3) -> None:
        for _ in range(max(0, iters)):
            self.detect(image)

    def info(self) -> dict:
        """Model facts for the report (params, GFLOPs per inference, ...)."""
        return {"name": self.name}
