"""Lightweight temporal-difference event trigger (paper Sec. III-C / IV-D).

    D_t = (1/N) * sum_{i=1..N} | I_t(i) - I_{t-1}(i) |          (Eq. 1 / Eq. 3)

    E_t = 1  if D_t >  tau                                        (Eq. 4)
          0  if D_t <= tau

When E_t = 1 the object detector is executed; otherwise it is skipped and the
previous detection result is reused (Fig. 1, "Skip Detection / Reuse Previous
Result").

Faithful defaults: I_{t-1} is the immediately preceding frame, pixel
intensities are 8-bit grayscale (0-255), the comparison is strict (>), and no
forced re-detection is applied. ``resize_width`` only reduces the cost of the
difference itself; D_t is a mean, so its scale does not depend on resolution.

Optional extensions (all off by default, clearly outside the paper):
  * reference="last_detected": compare with the frame the detector last ran on,
    so slow drifts accumulate until they trigger a detection.
  * max_skip: force a detection after N consecutive skipped frames.
  * blur_kernel: Gaussian smoothing to suppress sensor noise.
  * mask: ignore pixels outside a region of interest.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from etod.event.base import EventTrigger, TriggerDecision


class TemporalDifferenceTrigger(EventTrigger):
    name = "event"

    def __init__(
        self,
        threshold: float,
        grayscale: bool = True,
        resize_width: int | None = 320,
        blur_kernel: int = 0,
        reference: str = "previous",
        max_skip: int | None = None,
        first_frame_always_detect: bool = True,
        mask_ignore_regions: bool = False,
    ) -> None:
        if reference not in ("previous", "last_detected"):
            raise ValueError("reference must be 'previous' or 'last_detected'")
        if blur_kernel and blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd (or 0 to disable)")
        self.threshold = float(threshold)
        self.grayscale = grayscale
        self.resize_width = int(resize_width) if resize_width else None
        self.blur_kernel = int(blur_kernel or 0)
        self.reference = reference
        self.max_skip = int(max_skip) if max_skip else None
        self.first_frame_always_detect = first_frame_always_detect
        self.mask_ignore_regions = mask_ignore_regions
        self.ignore_regions = np.zeros((0, 4), np.float32)
        self.reset()

    @property
    def label(self) -> str:
        return f"event_tau_{self.threshold:g}"

    # ------------------------------------------------------------------ state
    def reset(self, sequence=None) -> None:
        self._ref: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._mask_shape = None
        self._skips = 0
        if sequence is not None:
            regions = getattr(sequence, "ignore_regions", None)
            self.ignore_regions = regions if (self.mask_ignore_regions and regions is not None) else np.zeros((0, 4), np.float32)

    # ------------------------------------------------------------------ helpers
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        x = image
        if self.grayscale and x.ndim == 3:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        if self.resize_width and x.shape[1] > self.resize_width:
            h = max(1, round(x.shape[0] * self.resize_width / x.shape[1]))
            x = cv2.resize(x, (self.resize_width, h), interpolation=cv2.INTER_AREA)
        if self.blur_kernel:
            x = cv2.GaussianBlur(x, (self.blur_kernel, self.blur_kernel), 0)
        return x

    def _build_mask(self, processed: np.ndarray, original_shape) -> np.ndarray | None:
        if not len(self.ignore_regions):
            return None
        h, w = processed.shape[:2]
        sx, sy = w / original_shape[1], h / original_shape[0]
        mask = np.full((h, w), 255, np.uint8)
        for x1, y1, x2, y2 in self.ignore_regions:
            mask[int(y1 * sy) : int(math.ceil(y2 * sy)), int(x1 * sx) : int(math.ceil(x2 * sx))] = 0
        return mask if mask.any() else None

    def difference_score(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Eq. 1: mean absolute pixel difference (over all channels if colour)."""
        diff = cv2.absdiff(a, b)
        if mask is None:
            return float(diff.mean())
        vals = cv2.mean(diff, mask=mask)
        return float(np.mean(vals[: (1 if diff.ndim == 2 else diff.shape[2])]))

    # ------------------------------------------------------------------ decision
    def decide(self, image: np.ndarray) -> TriggerDecision:
        cur = self.preprocess(image)
        if self._mask_shape != cur.shape:
            self._mask = self._build_mask(cur, image.shape)
            self._mask_shape = cur.shape

        if self._ref is None:
            self._ref = cur
            triggered = self.first_frame_always_detect
            self._skips = 0 if triggered else 1
            return TriggerDecision(math.nan, triggered, "first_frame")

        score = self.difference_score(cur, self._ref, self._mask)
        event = score > self.threshold                           # Eq. 4 (strict inequality)
        reason = "event" if event else "no_event"
        if not event and self.max_skip is not None and self._skips >= self.max_skip:
            event, reason = True, "max_skip"

        if self.reference == "previous" or event:
            self._ref = cur
        self._skips = 0 if event else self._skips + 1
        return TriggerDecision(score, event, reason)
