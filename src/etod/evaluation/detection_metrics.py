"""Accuracy metrics (paper Sec. III-E): precision, recall, mAP, plus F1 and
detection accuracy.

* mAP follows the COCO protocol: greedy score-ordered matching per class, AP
  with 101-point interpolation, mAP@0.5 and mAP@[0.5:0.95].
* Precision / recall / F1 / accuracy are computed at ``pr_iou`` over all
  detections that passed the detector's confidence threshold.
  "Detection accuracy" is defined as TP / (TP + FP + FN).
* Every frame is scored. On skipped frames the evaluated output is the reused
  previous result (Fig. 1), so accuracy loss from skipping is measured exactly.
* UA-DETRAC protocol: detections whose area lies >= 50 % inside an ignored
  region are removed before scoring (identically for every method).

Matching is done once per method; overall and per-sequence numbers are then
aggregated from the same match records.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from etod.data.base import Boxes


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def fraction_inside(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """For each box, the largest fraction of its area covered by a single region."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes), np.float32)
    lt = np.maximum(boxes[:, None, :2], regions[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], regions[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clip(1e-9)
    return (inter / area[:, None]).max(axis=1)


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP for one class at one IoU threshold."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    t = tp[order]
    ctp = np.cumsum(t)
    cfp = np.cumsum(~t)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]    # precision envelope
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rs, side="left")
    q = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(q.mean())


@dataclass
class Matcher:
    iou_thresholds: list[float]
    num_classes: int
    min_gt_area: float = 0.0
    ignore_overlap: float = 0.5
    # accumulated per detection
    _seq: list = field(default_factory=list)
    _cls: list = field(default_factory=list)
    _score: list = field(default_factory=list)
    _tp: list = field(default_factory=list)
    _valid: list = field(default_factory=list)
    # accumulated per ground-truth box
    _gt_seq: list = field(default_factory=list)
    _gt_cls: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.thr = np.asarray(self.iou_thresholds, np.float32)

    def add_frame(self, seq_idx: int, gt: Boxes, pred: Boxes, ignore_regions: np.ndarray | None = None) -> None:
        if ignore_regions is not None and len(ignore_regions) and len(pred):
            pred = pred.filter(fraction_inside(pred.xyxy, ignore_regions) < self.ignore_overlap)

        gt_ign = gt.areas() < self.min_gt_area if self.min_gt_area > 0 else np.zeros(len(gt), bool)
        valid_gt = ~gt_ign
        self._gt_seq.append(np.full(int(valid_gt.sum()), seq_idx, np.int32))
        self._gt_cls.append(gt.labels[valid_gt])

        P, G, T = len(pred), len(gt), len(self.thr)
        if P == 0:
            return
        order = np.argsort(-pred.scores, kind="mergesort")
        pred = pred.filter(order)
        tp = np.zeros((P, T), bool)
        valid = np.ones((P, T), bool)
        if G:
            iou = box_iou(pred.xyxy, gt.xyxy)
            iou[pred.labels[:, None] != gt.labels[None, :]] = -1.0     # class-aware matching
            best_any = iou.max(axis=1)
            for ti, thr in enumerate(self.thr):
                used = gt_ign.copy()
                for p in np.nonzero(best_any >= thr)[0]:               # others are FP for this thr
                    row = np.where(used, -1.0, iou[p])
                    j = int(row.argmax())
                    if row[j] >= thr:
                        tp[p, ti] = True
                        used[j] = True
                    elif gt_ign.any() and (iou[p][gt_ign] >= thr).any():
                        valid[p, ti] = False                           # matched an ignored GT: not an FP
        self._seq.append(np.full(P, seq_idx, np.int32))
        self._cls.append(pred.labels)
        self._score.append(pred.scores)
        self._tp.append(tp)
        self._valid.append(valid)

    # ------------------------------------------------------------------ results
    def _arrays(self):
        T = len(self.thr)
        cat = lambda xs, dt, shape=(0,): np.concatenate(xs) if xs else np.zeros(shape, dt)  # noqa: E731
        return (cat(self._seq, np.int32), cat(self._cls, np.int64), cat(self._score, np.float32),
                cat(self._tp, bool, (0, T)), cat(self._valid, bool, (0, T)),
                cat(self._gt_seq, np.int32), cat(self._gt_cls, np.int64))

    def compute(self, pr_iou: float = 0.5, seq_idx: int | None = None) -> dict:
        seq, cls, score, tp, valid, gseq, gcls = self._arrays()
        if seq_idx is not None:
            m, gm = seq == seq_idx, gseq == seq_idx
            seq, cls, score, tp, valid = seq[m], cls[m], score[m], tp[m], valid[m]
            gcls = gcls[gm]
        ti_pr = int(np.argmin(np.abs(self.thr - pr_iou)))
        i50 = int(np.argmin(np.abs(self.thr - 0.5)))

        per_class = []
        ap = np.full((self.num_classes, len(self.thr)), np.nan)
        for c in range(self.num_classes):
            cm = cls == c
            n_gt = int((gcls == c).sum())
            for ti in range(len(self.thr)):
                v = valid[cm, ti]
                ap[c, ti] = average_precision(score[cm][v], tp[cm, ti][v], n_gt)
            v = valid[cm, ti_pr]
            c_tp = int(tp[cm, ti_pr][v].sum())
            c_fp = int(v.sum()) - c_tp
            per_class.append({
                "class_id": c, "num_gt": n_gt, "num_pred": int(cm.sum()),
                "AP50": ap[c, i50], "AP50_95": float(np.nanmean(ap[c])) if n_gt else float("nan"),
                "precision": c_tp / max(c_tp + c_fp, 1) if (c_tp + c_fp) else float("nan"),
                "recall": c_tp / n_gt if n_gt else float("nan"),
            })

        v = valid[:, ti_pr]
        TP = int(tp[:, ti_pr][v].sum())
        FP = int(v.sum()) - TP
        n_gt_total = int(len(gcls))
        FN = n_gt_total - TP
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / n_gt_total if n_gt_total else 0.0
        has_gt = ~np.isnan(ap[:, i50])
        return {
            "mAP50": float(np.nanmean(ap[has_gt, i50])) if has_gt.any() else float("nan"),
            "mAP50_95": float(np.nanmean(ap[has_gt])) if has_gt.any() else float("nan"),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
            "accuracy": TP / (TP + FP + FN) if (TP + FP + FN) else 0.0,
            "TP": TP, "FP": FP, "FN": FN, "num_gt": n_gt_total, "num_pred": int(len(score)),
            "per_class": per_class,
        }
