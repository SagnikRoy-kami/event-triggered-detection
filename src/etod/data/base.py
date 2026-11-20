"""Common data structures shared by every dataset loader.

Every loader returns a list of :class:`Sequence` objects. A sequence is an
ordered stream of frames (the temporal order matters: D_t compares frame t
with frame t-1) plus optional ground-truth boxes per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np


@dataclass
class Boxes:
    """A set of boxes for one frame. Labels index into ``classes.eval_classes``."""

    xyxy: np.ndarray            # (N, 4) float32, pixel coordinates x1, y1, x2, y2
    labels: np.ndarray          # (N,) int64
    scores: np.ndarray          # (N,) float32 (1.0 for ground truth)

    @staticmethod
    def empty() -> "Boxes":
        return Boxes(np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.float32))

    @staticmethod
    def from_lists(xyxy, labels, scores=None) -> "Boxes":
        xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        labels = np.asarray(labels, np.int64).reshape(-1)
        scores = np.ones(len(labels), np.float32) if scores is None else np.asarray(scores, np.float32).reshape(-1)
        return Boxes(xyxy, labels, scores)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def filter(self, keep: np.ndarray) -> "Boxes":
        return Boxes(self.xyxy[keep], self.labels[keep], self.scores[keep])

    def areas(self) -> np.ndarray:
        return np.clip(self.xyxy[:, 2] - self.xyxy[:, 0], 0, None) * np.clip(self.xyxy[:, 3] - self.xyxy[:, 1], 0, None)


@dataclass
class Frame:
    index: int                  # position inside the evaluated stream (0, 1, 2, ...)
    frame_id: int               # original frame number in the dataset (used for ground truth lookup)
    image: np.ndarray           # BGR uint8 (OpenCV convention)


@dataclass
class Sequence:
    name: str
    num_frames: int
    fps: float
    frame_factory: Callable[[], Iterator[Frame]]
    ground_truth: dict[int, Boxes] | None = None      # frame_id -> Boxes; None = no annotations
    ignore_regions: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    metadata: dict = field(default_factory=dict)

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth is not None

    def frames(self) -> Iterator[Frame]:
        return self.frame_factory()

    def gt_for(self, frame_id: int) -> Boxes:
        if self.ground_truth is None:
            raise RuntimeError(f"Sequence {self.name} has no ground truth")
        return self.ground_truth.get(frame_id, Boxes.empty())


def label_map(names_to_eval: dict[str, str], eval_classes: list[str]) -> dict[str, int]:
    """dataset/detector label name -> eval class index (labels mapped to unknown classes are dropped)."""
    index = {c: i for i, c in enumerate(eval_classes)}
    out = {}
    for src, dst in (names_to_eval or {}).items():
        if dst in index:
            out[str(src).lower()] = index[dst]
    return out


def select_frames(paths: list, max_frames: int | None, stride: int) -> list:
    stride = max(1, int(stride or 1))
    sel = paths[::stride]
    if max_frames:
        sel = sel[: int(max_frames)]
    return sel
