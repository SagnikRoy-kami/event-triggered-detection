"""Detector registry. To swap in a different detector family (RT-DETR via
Ultralytics works out of the box by changing detector.weights), implement
``BaseDetector`` and register it in ``DETECTORS``."""

from __future__ import annotations

from etod.detectors.base import BaseDetector


def _ultralytics(cfg):
    from etod.detectors.yolo import UltralyticsDetector

    return UltralyticsDetector(cfg)


def _contour(cfg):
    from etod.detectors.contour import ContourDetector

    return ContourDetector(cfg)


DETECTORS = {"ultralytics": _ultralytics, "contour": _contour}


def build_detector(cfg) -> BaseDetector:
    backend = cfg["detector"]["backend"]
    if backend not in DETECTORS:
        raise ValueError(f"Unknown detector.backend '{backend}'. Available: {sorted(DETECTORS)}")
    return DETECTORS[backend](cfg)


__all__ = ["BaseDetector", "DETECTORS", "build_detector"]
