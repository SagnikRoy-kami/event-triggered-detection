"""Eq. 1 (D_t) and Eq. 4 (E_t) exactly as written in the paper."""
import math

import numpy as np
import pytest

from etod.event import AlwaysTrigger, TemporalDifferenceTrigger


def gray(v, shape=(40, 60)):
    return np.full(shape + (3,), v, np.uint8)


def trig(tau, **kw):
    kw.setdefault("resize_width", None)
    return TemporalDifferenceTrigger(threshold=tau, **kw)


def test_dt_is_mean_absolute_difference():
    t = trig(1.0)
    a = np.zeros((10, 10), np.uint8)
    b = a.copy()
    b[:5] = 20          # half the pixels differ by 20 -> D = 10
    assert t.difference_score(a, b) == pytest.approx(10.0)
    assert t.difference_score(b, a) == pytest.approx(10.0)   # absolute value, symmetric


def test_first_frame_detects_and_has_no_score():
    d = trig(5).decide(gray(100))
    assert d.triggered and math.isnan(d.score) and d.reason == "first_frame"


def test_strict_inequality_eq4():
    t = trig(10.0)
    t.decide(gray(100))
    d = t.decide(gray(110))            # D = 10, not > 10 -> skip
    assert d.score == pytest.approx(10.0) and not d.triggered
    d = t.decide(gray(121))            # D = 11 > 10 -> event
    assert d.triggered and d.reason == "event"


def test_reference_previous_compares_consecutive_frames():
    t = trig(5.0)
    for v in (100, 103, 106, 109, 112):   # slow drift: each step D=3 -> never triggers
        d = t.decide(gray(v))
    assert not d.triggered and d.score == pytest.approx(3.0)


def test_reference_last_detected_accumulates_drift():
    t = trig(5.0, reference="last_detected")
    t.decide(gray(100))
    assert not t.decide(gray(103)).triggered     # 3
    d = t.decide(gray(106))                      # 6 vs last detected frame -> event
    assert d.triggered and d.score == pytest.approx(6.0)


def test_max_skip_forces_detection():
    t = trig(1000.0, max_skip=2)
    t.decide(gray(1))
    assert [t.decide(gray(1)).triggered for _ in range(6)] == [False, False, True, False, False, True]


def test_resize_keeps_scale_of_dt():
    a, b = gray(100, (200, 400)), gray(130, (200, 400))
    full, small = trig(0, resize_width=None), trig(0, resize_width=100)
    full.decide(a), small.decide(a)
    assert full.decide(b).score == pytest.approx(small.decide(b).score, abs=0.5)


def test_reset_between_sequences():
    t = trig(1.0)
    t.decide(gray(0))
    t.reset()
    assert t.decide(gray(200)).reason == "first_frame"


def test_baseline_always_triggers():
    a = AlwaysTrigger()
    assert all(a.decide(gray(0)).triggered for _ in range(3))
