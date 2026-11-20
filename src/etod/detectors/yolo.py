"""Ultralytics YOLO detector (YOLOv8 by default, any Ultralytics model works)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from etod.data.base import Boxes, label_map
from etod.detectors.base import BaseDetector
from etod.utils.environment import resolve_device
from etod.utils.logging_utils import get_logger
from etod.utils.paths import resolve


def resolve_weights(weights: str, weights_dir: str | Path) -> Path:
    """Use an existing file if given; otherwise download into weights_dir."""
    p = resolve(weights)
    if p.exists():
        return p
    wdir = resolve(weights_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    return wdir / Path(weights).name   # Ultralytics downloads official weights to this path


class UltralyticsDetector(BaseDetector):
    def __init__(self, cfg) -> None:
        from ultralytics import YOLO

        d = cfg["detector"]
        self.weights_path = resolve_weights(d["weights"], d["weights_dir"])
        self.model = YOLO(str(self.weights_path))
        self.name = f"Ultralytics {Path(d['weights']).stem}"
        self.imgsz = int(d["imgsz"])
        self.conf = float(d["conf"])
        self.iou = float(d["iou"])
        self.max_det = int(d.get("max_det", 300))
        self.device = resolve_device(d.get("device", "auto"))
        self.half = bool(d.get("half", False)) and self.device.startswith("cuda")

        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        to_eval = label_map(cfg["classes"]["detector_to_eval"], cfg["classes"]["eval_classes"])
        self._lut = np.full(max(names) + 1, -1, np.int64)
        for cid, cname in names.items():
            if cname in to_eval:
                self._lut[cid] = to_eval[cname]
        self.class_ids = [cid for cid in names if self._lut[cid] >= 0]
        if not self.class_ids:
            raise ValueError(
                f"None of the detector classes map to eval classes. Detector has e.g. {list(names.values())[:10]}; "
                f"check classes.detector_to_eval"
            )
        get_logger().info(
            f"Detector: {self.name} on {self.device} | imgsz={self.imgsz} conf={self.conf} | "
            f"classes kept: {[names[c] for c in self.class_ids]}"
        )

    def detect(self, image: np.ndarray) -> Boxes:
        kwargs = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device,
                      max_det=self.max_det, classes=self.class_ids, verbose=False)
        if self.half:  # only pass when requested (newer Ultralytics versions deprecate the flag)
            kwargs["half"] = True
        res = self.model.predict(image, **kwargs)[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return Boxes.empty()
        xyxy = b.xyxy.cpu().numpy().astype(np.float32)   # .cpu() also synchronises CUDA -> honest timing
        cls = b.cls.cpu().numpy().astype(np.int64)
        conf = b.conf.cpu().numpy().astype(np.float32)
        labels = self._lut[cls]
        keep = labels >= 0
        return Boxes(xyxy[keep], labels[keep], conf[keep])

    def info(self) -> dict:
        info = {"name": self.name, "weights": str(self.weights_path), "imgsz": self.imgsz, "conf": self.conf,
                "iou": self.iou, "device": self.device, "half": self.half}
        try:
            info["params_m"] = round(sum(p.numel() for p in self.model.model.parameters()) / 1e6, 3)
            from ultralytics.utils.torch_utils import get_flops

            info["gflops_per_inference"] = round(float(get_flops(self.model.model, self.imgsz)), 3)
        except Exception:  # noqa: BLE001 - informative only
            pass
        return info
