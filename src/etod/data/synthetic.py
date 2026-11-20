"""Synthetic fixed-camera traffic scenes with exact ground truth.

Used by the unit tests and the smoke-test config so the whole pipeline can be
exercised without downloading anything. Scenes mimic the paper's case study
(Sec. IV-A): a fixed roadside camera with periods of idle, low, moderate and
high activity. Sensor noise is added to every frame so D_t is never exactly 0,
just like a real camera.
"""

from __future__ import annotations

import numpy as np

from etod.data.base import Boxes, Frame, Sequence, label_map

# class name -> (width, height) at 640 px frame width, BGR colour
OBJECT_TYPES = {
    "car": ((60, 34), (200, 70, 30)),      # blue
    "bus": ((150, 56), (20, 200, 235)),    # yellow
    "truck": ((110, 48), (40, 40, 210)),   # red
}
ACTIVITY = {  # n_objects (fraction of max), speed px/frame at 640 width
    "idle": (0.0, 0.0),
    "low": (0.25, 2.0),
    "moderate": (0.5, 4.5),
    "high": (1.0, 8.0),
}


def _background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    bg = np.full((h, w, 3), 95, np.uint8)
    bg = np.clip(bg.astype(np.int16) + rng.integers(-12, 13, (h, w, 1)), 0, 255).astype(np.uint8)
    for k in range(1, 4):  # lane markings
        y = int(h * k / 4)
        for x in range(0, w, 40):
            bg[y - 1 : y + 1, x : x + 22] = 225
    bg[: int(h * 0.12)] = (90, 120, 80)  # verge
    return bg


def _segment_plan(n_frames: int, pattern: list[str]) -> list[str]:
    seg = max(1, n_frames // max(1, len(pattern)))
    return [pattern[min(i // seg, len(pattern) - 1)] for i in range(n_frames)]


def make_sequence(name: str, n_frames: int, w: int, h: int, max_objects: int, pattern: list[str], seed: int, eval_classes: list[str], dataset_to_eval: dict) -> Sequence:
    rng = np.random.default_rng(seed)
    bg = _background(w, h, rng)
    scale = w / 640.0
    lanes = [int(h * (k + 0.5) / 4) for k in range(4)]
    plan = _segment_plan(n_frames, pattern)
    name_to_idx = label_map(dataset_to_eval, eval_classes)

    # Pre-compute object tracks: each activity segment gets its own set of objects.
    tracks: list[list[tuple[str, float, float, int]]] = []  # per frame: (cls, x1, y1, w, h)
    seg_objects: dict[int, list[dict]] = {}
    seg_id, prev = -1, None
    for t, act in enumerate(plan):
        if act != prev:
            seg_id += 1
            frac, speed = ACTIVITY[act]
            n_obj = int(round(frac * max_objects))
            objs = []
            for k in range(n_obj):
                cls = rng.choice(list(OBJECT_TYPES))
                (ow, oh), _ = OBJECT_TYPES[cls]
                objs.append({
                    "cls": cls, "w": ow * scale, "h": oh * scale,
                    "x": rng.uniform(-ow * scale, w * 0.6), "y": lanes[k % 4] - oh * scale / 2,
                    "v": speed * scale * rng.uniform(0.8, 1.2) * (1 if k % 2 == 0 else -1),
                })
            if act == "idle":  # one parked vehicle: present but static (tests result reuse)
                (ow, oh), _ = OBJECT_TYPES["car"]
                objs.append({"cls": "car", "w": ow * scale, "h": oh * scale, "x": w * 0.7, "y": lanes[1] - oh * scale / 2, "v": 0.0})
            seg_objects[seg_id] = objs
            seg_start = t
            prev = act
        frame_objs = []
        for o in seg_objects[seg_id]:
            x = o["x"] + o["v"] * (t - seg_start)
            span = w + o["w"]
            x = ((x + o["w"]) % span) - o["w"] if o["v"] != 0 else x  # wrap around
            frame_objs.append((o["cls"], x, o["y"], o["w"], o["h"]))
        tracks.append(frame_objs)

    gt: dict[int, Boxes] = {}
    for t, objs in enumerate(tracks):
        xyxy, labels = [], []
        for cls, x, y, ow, oh in objs:
            x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + ow), min(h, y + oh)
            if x2 - x1 >= 8 and y2 - y1 >= 8 and cls in name_to_idx:
                xyxy.append([x1, y1, x2, y2])
                labels.append(name_to_idx[cls])
        gt[t] = Boxes.from_lists(xyxy, labels) if labels else Boxes.empty()

    def factory():
        noise_rng = np.random.default_rng(seed + 1)
        for t, objs in enumerate(tracks):
            img = bg.copy()
            for cls, x, y, ow, oh in objs:
                _, color = OBJECT_TYPES[cls]
                x1, y1, x2, y2 = int(max(0, x)), int(max(0, y)), int(min(w, x + ow)), int(min(h, y + oh))
                if x2 > x1 and y2 > y1:
                    img[y1:y2, x1:x2] = color
                    img[y1 + 3 : max(y1 + 4, y2 - 3), x1 + 3 : max(x1 + 4, int(x1 + (x2 - x1) * 0.35))] = (
                        int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))  # windscreen
            noise = noise_rng.normal(0, 1.5, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            yield Frame(index=t, frame_id=t, image=img)

    return Sequence(
        name=name, num_frames=n_frames, fps=25.0, frame_factory=factory, ground_truth=gt,
        metadata={"synthetic": True, "activity_plan": plan},
    )


def load_synthetic(cfg) -> list[Sequence]:
    spec = cfg["dataset"]["synthetic"]
    n_seq = int(cfg["dataset"].get("max_sequences") or spec["num_sequences"])
    n_frames = int(cfg["dataset"].get("max_frames_per_sequence") or spec["frames_per_sequence"])
    pattern = list(spec["activity_pattern"])
    seed = int(cfg["experiment"]["seed"])
    return [
        make_sequence(
            f"SYN_{i + 1:02d}", n_frames, int(spec["width"]), int(spec["height"]), int(spec["max_objects"]),
            pattern[i % len(pattern):] + pattern[: i % len(pattern)], seed + 100 * i,
            cfg["classes"]["eval_classes"], cfg["classes"]["dataset_to_eval"],
        )
        for i in range(n_seq)
    ]
