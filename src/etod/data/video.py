"""Plain video files (mp4/avi/mov/mkv). No ground truth: use evaluation.reference = baseline."""

from __future__ import annotations

from pathlib import Path

import cv2

from etod.data.base import Frame, Sequence
from etod.utils.logging_utils import get_logger
from etod.utils.paths import VIDEOS_DIR, resolve

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _collect(paths: list[str]) -> list[Path]:
    targets = [resolve(p) for p in paths] if paths else [VIDEOS_DIR]
    files: list[Path] = []
    for t in targets:
        if t.is_dir():
            files += sorted(p for p in t.iterdir() if p.suffix.lower() in VIDEO_EXT)
        elif t.exists():
            files.append(t)
        else:
            raise FileNotFoundError(t)
    return files


def load_videos(cfg) -> list[Sequence]:
    ds = cfg["dataset"]
    files = _collect(ds.get("video_paths") or [])
    if not files:
        raise RuntimeError(
            f"No videos found. Put .mp4 files in {VIDEOS_DIR} or set dataset.video_paths, "
            "or run: python scripts/download_data.py --dataset sample-videos"
        )
    if ds.get("max_sequences"):
        files = files[: int(ds["max_sequences"])]
    stride = max(1, int(ds.get("frame_stride") or 1))
    cap_frames = ds.get("max_frames_per_sequence")

    seqs = []
    for f in files:
        cap = cv2.VideoCapture(str(f))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        n = (total + stride - 1) // stride if total > 0 else 0
        if cap_frames:
            n = min(n, int(cap_frames)) if n else int(cap_frames)
        seqs.append(
            Sequence(
                name=f.stem,
                num_frames=n,
                fps=fps / stride,
                frame_factory=_video_iter(f, stride, n),
                ground_truth=None,
                metadata={"source": str(f), "width": w, "height": h},
            )
        )
    get_logger().info(f"Videos: {len(seqs)} files, ~{sum(s.num_frames for s in seqs)} frames")
    return seqs


def _video_iter(path: Path, stride: int, limit: int):
    def factory():
        cap = cv2.VideoCapture(str(path))
        raw_idx, out_idx = 0, 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if raw_idx % stride == 0:
                    yield Frame(index=out_idx, frame_id=raw_idx, image=img)
                    out_idx += 1
                    if limit and out_idx >= limit:
                        break
                raw_idx += 1
        finally:
            cap.release()

    return factory
