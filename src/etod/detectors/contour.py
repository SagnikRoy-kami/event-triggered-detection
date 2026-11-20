"""A tiny classical detector for the synthetic scenes (tests / smoke runs only).

Finds saturated colour blobs and classifies them by hue. It is a real
detector (it only sees pixels), so accuracy changes when stale results are
reused - which is exactly what the pipeline needs to be tested on.
"""

from __future__ import annotations

import cv2
import numpy as np

from etod.data.base import Boxes, label_map
from etod.detectors.base import BaseDetector

HUE_RANGES = {"car": [(95, 130)], "bus": [(18, 40)], "truck": [(0, 10), (165, 180)]}


class ContourDetector(BaseDetector):
    name = "contour (synthetic)"

    def __init__(self, cfg) -> None:
        self.to_eval = label_map(cfg["classes"]["detector_to_eval"], cfg["classes"]["eval_classes"])
        self.min_area = 120

    def _classify(self, hue: float) -> str | None:
        for cls, ranges in HUE_RANGES.items():
            if any(lo <= hue <= hi for lo, hi in ranges):
                return cls
        return None

    def detect(self, image: np.ndarray) -> Boxes:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 110, 60), (180, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        xyxy, labels, scores = [], [], []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            patch = hsv[y : y + h, x : x + w].reshape(-1, 3)
            patch = patch[patch[:, 1] > 110]
            if len(patch) == 0:
                continue
            cls = self._classify(float(np.median(patch[:, 0])))
            if cls is None or cls not in self.to_eval:
                continue
            xyxy.append([x, y, x + w, y + h])
            labels.append(self.to_eval[cls])
            scores.append(min(0.99, 0.5 + 0.5 * area / max(1.0, w * h)))
        return Boxes.from_lists(xyxy, labels, scores) if labels else Boxes.empty()

    def info(self) -> dict:
        return {"name": self.name, "params_m": 0.0}
