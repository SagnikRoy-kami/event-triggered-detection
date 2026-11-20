"""Generic ordered-frame folders with optional YOLO-format labels.

Use this to plug in any other Kaggle traffic dataset that ships as frame
sequences:

    data/processed/my_sequences/
        seq_A/images/000001.jpg ...      (or frames directly in seq_A/)
        seq_A/labels/000001.txt ...      (optional; "<cls> <cx> <cy> <w> <h>" normalised)
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2

from etod.data.base import Boxes, Frame, Sequence, label_map, select_frames
from etod.utils.logging_utils import get_logger
from etod.utils.paths import resolve

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_NUM = re.compile(r"(\d+)")


def _sort_key(p: Path):
    nums = _NUM.findall(p.stem)
    return (int(nums[-1]) if nums else 0, p.name)


def load_image_folders(cfg) -> list[Sequence]:
    ds = cfg["dataset"]
    spec = ds["image_folder"]
    root = resolve(spec["sequences_dir"])
    if not root or not root.exists():
        raise FileNotFoundError(f"image_folder.sequences_dir not found: {root}")
    names = list(ds.get("sequences") or []) or sorted(d.name for d in root.iterdir() if d.is_dir())
    if ds.get("max_sequences"):
        names = names[: int(ds["max_sequences"])]

    class_names = [str(c).lower() for c in spec.get("class_names", [])]
    name_to_idx = label_map(cfg["classes"]["dataset_to_eval"], cfg["classes"]["eval_classes"])

    seqs = []
    for name in names:
        sdir = root / name
        idir = sdir / spec.get("images_subdir", "images")
        idir = idir if idir.exists() else sdir
        files = sorted((p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT), key=_sort_key)
        files = select_frames(files, ds.get("max_frames_per_sequence"), ds.get("frame_stride", 1))
        if not files:
            continue
        ldir = sdir / spec.get("labels_subdir", "labels")
        gt = None
        if ldir.exists():
            h, w = cv2.imread(str(files[0])).shape[:2]
            gt = {}
            for i, f in enumerate(files):
                gt[i] = _read_yolo(ldir / f"{f.stem}.txt", w, h, class_names, name_to_idx)
        seqs.append(
            Sequence(
                name=name,
                num_frames=len(files),
                fps=float(spec.get("fps", 25)) / max(1, int(ds.get("frame_stride") or 1)),
                frame_factory=_iter(files),
                ground_truth=gt,
                metadata={"source": str(sdir)},
            )
        )
    get_logger().info(f"Image folders: {len(seqs)} sequences, {sum(s.num_frames for s in seqs)} frames")
    return seqs


def _read_yolo(path: Path, w: int, h: int, class_names: list[str], name_to_idx: dict[str, int]) -> Boxes:
    if not path.exists():
        return Boxes.empty()
    xyxy, labels = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        if c >= len(class_names) or class_names[c] not in name_to_idx:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        xyxy.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        labels.append(name_to_idx[class_names[c]])
    return Boxes.from_lists(xyxy, labels) if labels else Boxes.empty()


def _iter(files: list[Path]):
    def factory():
        for i, f in enumerate(files):
            yield Frame(index=i, frame_id=i, image=cv2.imread(str(f)))

    return factory
