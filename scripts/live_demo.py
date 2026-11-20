"""LIVE side-by-side view: baseline (detector on every frame) vs event-triggered.

Left  : YOLO runs on every frame.
Right : YOLO runs only when D_t > tau, otherwise the previous result is reused
        (green banner = detector ran, grey banner = skipped).
Bottom: live D_t graph with the threshold line and a strip of detector calls.

Examples
--------
    python scripts/live_demo.py --source data/videos/car-detection.mp4
    python scripts/live_demo.py --source 0                                  # webcam
    python scripts/live_demo.py --source "data/raw/ua-detrac/DETRAC-Images/MVI_39031" ^
                                --config configs/experiments/ua_detrac_quick.yaml
    python scripts/live_demo.py --source data/videos/car-detection.mp4 --tau 0.5 --save outputs/live/demo.mp4

Keys (click the window first)
    q / Esc  quit          space  pause / resume
    + / -    raise / lower tau by 10 %      s  save a screenshot

tau defaults to the recommended threshold of your latest experiment run
(outputs/runs/LATEST.txt), otherwise 1.0. FPS shown on each side is that
method's own processing rate (both run on the same frames, one after the other).
"""
import argparse
import collections
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from etod.config import load_config
from etod.data.base import Boxes
from etod.detectors import build_detector
from etod.event import TemporalDifferenceTrigger
from etod.pipeline.storage import load_json
from etod.reporting.video import DARK, GREEN, GREY, _banner, _draw
from etod.utils.paths import OUTPUTS_DIR, ensure_dir, latest_run, resolve

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
BLUE_BGR, ORANGE_BGR = (214, 120, 42), (52, 104, 235)


# --------------------------------------------------------------------------- sources
def open_source(src: str):
    """Returns (frame iterator, fps, description). Webcam index, video file or folder of frames."""
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(int(src))
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(src))
        return _cap_iter(cap), cap.get(cv2.CAP_PROP_FPS) or 30.0, f"webcam {src}"
    p = resolve(src)
    if p is None or not p.exists():
        raise SystemExit(f"Source not found: {src}")
    if p.is_dir():
        files = [f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT]
        num = lambda f: [int(x) for x in re.findall(r"\d+", f.stem)] or [0]  # noqa: E731
        files.sort(key=lambda f: (str(f.parent), num(f)))
        if not files:
            raise SystemExit(f"No images in {p}")
        return (cv2.imread(str(f)) for f in files), 25.0, f"{p.name} ({len(files)} frames)"
    cap = cv2.VideoCapture(str(p))
    return _cap_iter(cap), cap.get(cv2.CAP_PROP_FPS) or 25.0, p.name


def _cap_iter(cap):
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img
    finally:
        cap.release()


def default_tau() -> tuple[float, str]:
    run = latest_run()
    if run and (run / "run_info.json").exists():
        rec = (load_json(run / "run_info.json").get("recommendation") or {}).get("threshold")
        if rec is not None:
            return float(rec), f"recommended tau of latest run ({run.name})"
    return 1.0, "default (press + / - to tune)"


# --------------------------------------------------------------------------- drawing
def graph(width: int, values, flags, tau: float, height: int = 120) -> np.ndarray:
    g = np.full((height, width, 3), 250, np.uint8)
    strip_h, top, pad = 10, 18, 6
    plot_h = height - strip_h - top - pad
    vals = np.asarray(values, float)
    finite = vals[np.isfinite(vals)]
    ymax = max(tau * 2.0, float(np.percentile(finite, 98)) if len(finite) else 0.0, 1e-3) * 1.1
    n = len(vals)
    xs = np.linspace(8, width - 8, max(n, 2))[-n:] if n else []
    y_of = lambda v: int(top + plot_h - min(v, ymax) / ymax * plot_h)  # noqa: E731
    yt = y_of(tau)
    cv2.line(g, (8, yt), (width - 8, yt), ORANGE_BGR, 1, cv2.LINE_AA)
    cv2.putText(g, f"tau = {tau:.3g}", (width - 110, max(12, yt - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ORANGE_BGR, 1, cv2.LINE_AA)
    pts = [(int(x), y_of(v)) for x, v in zip(xs, vals) if np.isfinite(v)]
    if len(pts) > 1:
        cv2.polylines(g, [np.array(pts, np.int32)], False, BLUE_BGR, 1, cv2.LINE_AA)
    for x, f in zip(xs, flags):  # detector-call strip (paper Fig. 2)
        cv2.line(g, (int(x), height - strip_h - 2), (int(x), height - 3), GREEN if f else (205, 205, 205), 1)
    cv2.putText(g, "D_t (mean |I_t - I_t-1|)   strip: green = detector ran, grey = skipped",
                (8, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    return g


def fps_of(times) -> float:
    return len(times) / sum(times) if times and sum(times) > 0 else 0.0


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, folder of frames, or webcam index (0)")
    ap.add_argument("--config", default="configs/experiments/custom_video.yaml", help="detector + class settings")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. detector.device=cuda:0")
    ap.add_argument("--tau", type=float, default=None, help="event threshold (default: recommended tau of latest run)")
    ap.add_argument("--mode", choices=["both", "event"], default="both", help="'event' shows only the event-triggered side (faster)")
    ap.add_argument("--save", default=None, help="also record the view to this .mp4")
    ap.add_argument("--realtime", action="store_true", help="do not play faster than the source fps")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--display-width", type=int, default=640, help="width of each panel on screen")
    ap.add_argument("--no-window", action="store_true", help="no window (record only, for servers)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tau, tau_src = (args.tau, "--tau") if args.tau is not None else default_tau()
    frames, src_fps, desc = open_source(args.source)
    det = build_detector(cfg)
    et = cfg["event_trigger"]
    trig = TemporalDifferenceTrigger(tau, grayscale=et["grayscale"], resize_width=et["resize_width"],
                                     blur_kernel=et["blur_kernel"], reference=et["reference"], max_skip=et["max_skip"])
    names = cfg["classes"]["eval_classes"]
    print(f"Source: {desc} | tau = {tau:g} ({tau_src}) | keys: q quit, space pause, +/- tau, s screenshot")

    first = next(frames, None)
    if first is None:
        raise SystemExit("Source has no frames")
    det.warmup(first, int(cfg["detector"].get("warmup_iters", 3)))

    def all_frames():
        yield first
        yield from frames

    win = "Event-triggered detection - live (q to quit)"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    writer, last_e, paused = None, Boxes.empty(), False
    tb, te = collections.deque(maxlen=30), collections.deque(maxlen=30)
    dts, flags = collections.deque(maxlen=300), collections.deque(maxlen=300)
    n = n_det = 0
    shot_dir = OUTPUTS_DIR / "live"

    for img in all_frames():
        if args.max_frames and n >= args.max_frames:
            break
        loop_start = time.perf_counter()
        scale = min(1.0, args.display_width / img.shape[1])
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img

        # ---- baseline: detector on every frame
        if args.mode == "both":
            t0 = time.perf_counter()
            b = det.detect(img)
            tb.append(time.perf_counter() - t0)

        # ---- event-triggered: Eq. 1 + Eq. 4, reuse on skip
        t0 = time.perf_counter()
        d = trig.decide(img)
        if d.triggered:
            last_e = det.detect(img)
        te.append(time.perf_counter() - t0)
        n += 1
        n_det += int(d.triggered)
        dts.append(d.score)
        flags.append(d.triggered)

        # ---- compose view
        sc = lambda bx: Boxes(bx.xyxy * scale, bx.labels, bx.scores)  # noqa: E731
        dtxt = "first frame" if np.isnan(d.score) else f"D_t={d.score:.3g} {'>' if d.triggered else '<='} tau"
        right = _banner(_draw(small, sc(last_e), names),
                        f"EVENT  {fps_of(te):5.1f} FPS | {dtxt} -> {'DETECT' if d.triggered else 'SKIP (reuse)'}",
                        GREEN if d.triggered else GREY)
        panels = [right]
        if args.mode == "both":
            left = _banner(_draw(small, sc(b), names), f"BASELINE  {fps_of(tb):5.1f} FPS | detector every frame", DARK)
            panels = [left, np.full((left.shape[0], 6, 3), 255, np.uint8), right]
        top = np.hstack(panels)
        footer = np.full((26, top.shape[1], 3), 245, np.uint8)
        speed = f"   speed-up x{fps_of(te) / fps_of(tb):.2f}" if args.mode == "both" and fps_of(tb) else ""
        cv2.putText(footer, f"frame {n}   detector calls: baseline {n if args.mode == 'both' else '-'} | event {n_det} "
                    f"({(1 - n_det / n) * 100:.0f}% skipped){speed}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
        canvas = np.vstack([top, graph(top.shape[1], dts, flags, trig.threshold), footer])

        if args.save:
            if writer is None:
                out = resolve(args.save)
                out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)

        if not args.no_window:
            cv2.imshow(win, canvas)
            if args.realtime:
                wait = max(1, int((1 / src_fps - (time.perf_counter() - loop_start)) * 1000))
            else:
                wait = 1
            key = cv2.waitKey(wait) & 0xFF
            while True:
                if key in (ord("q"), 27):
                    n = -1
                    break
                if key in (ord("+"), ord("=")):
                    trig.threshold *= 1.1
                elif key in (ord("-"), ord("_")):
                    trig.threshold /= 1.1
                elif key == ord("s"):
                    p = ensure_dir(shot_dir) / f"live_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    cv2.imwrite(str(p), canvas)
                    print(f"screenshot: {p}")
                elif key == ord(" "):
                    paused = not paused
                if not paused:
                    break
                key = cv2.waitKey(50) & 0xFF
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    n = -1
                    break
            if n == -1 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

    if writer is not None:
        writer.release()
        print(f"Recording saved: {resolve(args.save)}")
    if not args.no_window:
        cv2.destroyAllWindows()
    if n > 0:
        print(f"{n} frames | event-triggered ran the detector on {n_det} ({(1 - n_det / n) * 100:.1f}% skipped) | "
              f"final tau = {trig.threshold:.4g}")


if __name__ == "__main__":
    main()
