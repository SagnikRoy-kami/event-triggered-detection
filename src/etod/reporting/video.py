"""Side-by-side comparison video: baseline (left) vs event-triggered (right).

The right panel shows, for every frame, whether the detector actually ran
(green banner) or the previous result was reused (grey banner), together with
the D_t value and the threshold - a moving version of the paper's Fig. 2.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from etod.data.base import Boxes
from etod.pipeline.storage import load_json, load_predictions, method_slug
from etod.utils.logging_utils import get_logger

# BGR versions of the report palette
CLASS_COLORS = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232), (0, 131, 0), (214, 80, 98), (72, 73, 227)]
GREEN, GREY, DARK = (60, 160, 40), (150, 150, 150), (25, 25, 25)


def _draw(img: np.ndarray, boxes: Boxes, names: list[str]) -> np.ndarray:
    out = img.copy()
    t = max(1, round(img.shape[1] / 480))
    for (x1, y1, x2, y2), lab, sc in zip(boxes.xyxy.astype(int), boxes.labels, boxes.scores):
        c = CLASS_COLORS[int(lab) % len(CLASS_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, t + 1)
        txt = f"{names[int(lab)]} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _banner(img: np.ndarray, text: str, color) -> np.ndarray:
    h = max(26, img.shape[0] // 14)
    bar = np.full((h, img.shape[1], 3), color, np.uint8)
    cv2.putText(bar, text, (8, int(h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, h / 48, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def render_comparison_video(run_dir: Path, cfg, sequences) -> Path | None:
    log = get_logger()
    vcfg = cfg["reporting"]["video"]
    info = load_json(run_dir / "run_info.json") if (run_dir / "run_info.json").exists() else {}
    rec = info.get("recommendation") or {}
    tau = vcfg.get("threshold") if vcfg.get("threshold") is not None else rec.get("threshold")
    if tau is None:
        log.warning("No threshold for the comparison video")
        return None
    method = f"event_tau_{float(tau):g}"
    pred_b, pred_e = run_dir / "predictions" / "baseline.npz", run_dir / "predictions" / f"{method_slug(method)}.npz"
    log_e = run_dir / "per_frame" / f"{method_slug(method)}.csv"
    if not (pred_b.exists() and pred_e.exists() and log_e.exists()):
        log.warning(f"Missing predictions/logs for {method}; enable save_predictions and save_per_frame_log")
        return None
    seq_name = vcfg.get("sequence") or sequences[0].name
    seq = next((s for s in sequences if s.name == seq_name), None)
    if seq is None:
        log.warning(f"Sequence {seq_name} not in this run")
        return None

    names = cfg["classes"]["eval_classes"]
    pb, pe = load_predictions(pred_b), load_predictions(pred_e)
    elog = pd.read_csv(log_e)
    elog = elog[elog["sequence"] == seq_name].set_index("frame_id")
    out_path = run_dir / "videos" / f"comparison_{seq_name}_tau_{float(tau):g}.mp4"
    fps = float(vcfg.get("fps") or seq.fps or 25)
    writer, n_det, n = None, 0, 0
    best_canvas, best_score = None, -1.0   # report preview: most boxes, prefer a skipped (reused) frame
    max_frames = int(vcfg.get("max_frames") or 10**9)
    for frame in seq.frames():
        if n >= max_frames:
            break
        img = frame.image
        scale = min(1.0, 640 / img.shape[1])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        key = (seq_name, frame.frame_id)
        sb, se = pb.get(key, Boxes.empty()), pe.get(key, Boxes.empty())
        sb = Boxes(sb.xyxy * scale, sb.labels, sb.scores)
        se = Boxes(se.xyxy * scale, se.labels, se.scores)
        row = elog.loc[frame.frame_id] if frame.frame_id in elog.index else None
        trig = bool(row["triggered"]) if row is not None else True
        d = row["score"] if row is not None else float("nan")
        n_det += trig
        left = _banner(_draw(img, sb, names), "BASELINE - detector runs on every frame", DARK)
        dtxt = "D_t=n/a" if pd.isna(d) else f"D_t={d:.2f}"
        rtxt = (f"EVENT tau={float(tau):g} | {dtxt} > tau -> DETECTOR RAN" if trig
                else f"EVENT tau={float(tau):g} | {dtxt} <= tau -> SKIPPED (reuse)")
        right = _banner(_draw(img, se, names), rtxt, GREEN if trig else GREY)
        footer = np.full((24, left.shape[1] * 2 + 6, 3), 245, np.uint8)
        cv2.putText(footer, f"frame {frame.index + 1}   detector calls so far: baseline {n + 1}  |  event-triggered {n_det}"
                    f"  ({(1 - n_det / (n + 1)) * 100:.0f}% skipped)", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        canvas = np.vstack([np.hstack([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right]), footer])
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
        score = min(len(sb), 5) + min(len(se), 5) + (1.5 if (not trig and len(se)) else 0.0) + n * 1e-6
        if score > best_score:
            best_score, best_canvas = score, canvas.copy()
        n += 1
    if writer is not None:
        writer.release()
    if best_canvas is not None:
        cv2.imwrite(str(run_dir / "videos" / "comparison_preview.jpg"), best_canvas)
    log.info(f"Comparison video: {out_path} ({n} frames)")
    return out_path
