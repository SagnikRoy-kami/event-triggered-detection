"""Saving / loading run artefacts so reports can be rebuilt without re-running."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from etod.data.base import Boxes


def method_slug(method: str) -> str:
    return method.replace(".", "p").replace(" ", "_")


def save_predictions(path: Path, predictions: dict[tuple[str, int], Boxes]) -> None:
    seq_names = sorted({k[0] for k in predictions})
    sidx = {s: i for i, s in enumerate(seq_names)}
    keys, boxes, labels, scores, counts = [], [], [], [], []
    for (seq, fid), b in predictions.items():
        keys.append((sidx[seq], fid))
        counts.append(len(b))
        boxes.append(b.xyxy)
        labels.append(b.labels)
        scores.append(b.scores)
    np.savez_compressed(
        path,
        sequences=np.array(seq_names),
        keys=np.asarray(keys, np.int64).reshape(-1, 2),
        counts=np.asarray(counts, np.int64),
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        labels=np.concatenate(labels) if labels else np.zeros((0,), np.int64),
        scores=np.concatenate(scores) if scores else np.zeros((0,), np.float32),
    )


def load_predictions(path: Path) -> dict[tuple[str, int], Boxes]:
    z = np.load(path, allow_pickle=False)
    names = [str(s) for s in z["sequences"]]
    out, start = {}, 0
    for (si, fid), n in zip(z["keys"], z["counts"]):
        sl = slice(start, start + int(n))
        out[(names[int(si)], int(fid))] = Boxes(z["boxes"][sl], z["labels"][sl], z["scores"][sl])
        start += int(n)
    return out


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(data), fh, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
