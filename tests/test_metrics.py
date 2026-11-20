import numpy as np
import pytest

from etod.data.base import Boxes
from etod.evaluation import Matcher, average_precision, box_iou, skip_rate

THR = [0.5, 0.75]


def B(boxes, labels, scores=None):
    return Boxes.from_lists(boxes, labels, scores)


def test_skip_rate_eq2():
    assert skip_rate(25, 100) == pytest.approx(75.0)
    assert skip_rate(100, 100) == 0.0


def test_iou():
    iou = box_iou(np.array([[0, 0, 10, 10]], np.float32), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32))
    assert iou[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_perfect_predictions():
    m = Matcher(THR, num_classes=2)
    for _ in range(5):
        gt = B([[0, 0, 10, 10], [20, 20, 40, 40]], [0, 1])
        m.add_frame(0, gt, B(gt.xyxy, gt.labels, [0.9, 0.8]))
    r = m.compute()
    assert r["mAP50"] == pytest.approx(1.0) and r["precision"] == 1.0 and r["recall"] == 1.0 and r["accuracy"] == 1.0


def test_no_predictions_zero_ap():
    m = Matcher(THR, num_classes=1)
    m.add_frame(0, B([[0, 0, 10, 10]], [0]), Boxes.empty())
    r = m.compute()
    assert r["mAP50"] == 0.0 and r["recall"] == 0.0 and r["FN"] == 1


def test_wrong_class_is_false_positive():
    m = Matcher(THR, num_classes=2)
    m.add_frame(0, B([[0, 0, 10, 10]], [0]), B([[0, 0, 10, 10]], [1], [0.9]))
    r = m.compute()
    assert r["TP"] == 0 and r["FP"] == 1 and r["FN"] == 1


def test_duplicate_detection_counts_once():
    m = Matcher(THR, num_classes=1)
    m.add_frame(0, B([[0, 0, 10, 10]], [0]), B([[0, 0, 10, 10], [0, 0, 10, 10]], [0, 0], [0.9, 0.8]))
    r = m.compute()
    assert r["TP"] == 1 and r["FP"] == 1


def test_ap_ranking():
    # one TP with high score and one FP with lower score -> AP 1.0; reversed -> lower
    assert average_precision(np.array([0.9, 0.1]), np.array([True, False]), 1) == pytest.approx(1.0)
    assert average_precision(np.array([0.1, 0.9]), np.array([True, False]), 1) < 1.0


def test_ignore_regions_remove_detections():
    m = Matcher(THR, num_classes=1)
    m.add_frame(0, Boxes.empty(), B([[0, 0, 10, 10]], [0], [0.9]), ignore_regions=np.array([[0, 0, 100, 100]], np.float32))
    assert m.compute()["FP"] == 0


def test_stale_box_after_motion_is_penalised():
    """A reused (stale) box loses accuracy once the object has moved - the cost of skipping."""
    m = Matcher(THR, num_classes=1)
    m.add_frame(0, B([[50, 0, 60, 10]], [0]), B([[0, 0, 10, 10]], [0], [0.9]))
    r = m.compute()
    assert r["TP"] == 0 and r["FP"] == 1


def test_per_sequence_split():
    m = Matcher(THR, num_classes=1)
    gt = B([[0, 0, 10, 10]], [0])
    m.add_frame(0, gt, B(gt.xyxy, gt.labels, [0.9]))
    m.add_frame(1, gt, Boxes.empty())
    assert m.compute(seq_idx=0)["recall"] == 1.0
    assert m.compute(seq_idx=1)["recall"] == 0.0
    assert m.compute()["recall"] == 0.5
