"""Dataset registry. To add a dataset: write a ``load_x(cfg) -> list[Sequence]``
function in a new module and register it in ``LOADERS``."""

from __future__ import annotations

from etod.data.base import Boxes, Frame, Sequence
from etod.data.image_folder import load_image_folders
from etod.data.synthetic import load_synthetic
from etod.data.ua_detrac import load_ua_detrac
from etod.data.video import load_videos

LOADERS = {
    "ua_detrac": load_ua_detrac,
    "video": load_videos,
    "image_folder": load_image_folders,
    "synthetic": load_synthetic,
}


def load_sequences(cfg) -> list[Sequence]:
    kind = cfg["dataset"]["type"]
    if kind not in LOADERS:
        raise ValueError(f"Unknown dataset.type '{kind}'. Available: {sorted(LOADERS)}")
    return LOADERS[kind](cfg)


__all__ = ["Boxes", "Frame", "Sequence", "LOADERS", "load_sequences"]
