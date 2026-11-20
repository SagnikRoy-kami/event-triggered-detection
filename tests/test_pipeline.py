"""End-to-end: synthetic data + contour detector through all five phases."""
import numpy as np
import pandas as pd

from etod.config import load_config
from etod.data import load_sequences
from etod.detectors import build_detector
from etod.event import AlwaysTrigger, TemporalDifferenceTrigger
from etod.pipeline import run_experiment, run_method


def small_cfg(tmp_path, *extra):
    return load_config("configs/experiments/synthetic_smoke.yaml", [
        f"experiment.output_root={tmp_path.as_posix()}", "dataset.synthetic.frames_per_sequence=60",
        "reporting.figure_formats=[png]", "evaluation.monitor_resources=false", *extra])


def test_baseline_runs_detector_every_frame(tmp_path):
    cfg = small_cfg(tmp_path)
    seqs = load_sequences(cfg)
    r = run_method(seqs, build_detector(cfg), AlwaysTrigger(), warmup_iters=0)
    assert r.frame_log["triggered"].all() and len(r.frame_log) == 120


def test_skipped_frames_reuse_previous_result(tmp_path):
    cfg = small_cfg(tmp_path)
    seqs = load_sequences(cfg)[:1]
    r = run_method(seqs, build_detector(cfg), TemporalDifferenceTrigger(1e9, resize_width=None), warmup_iters=0)
    log = r.frame_log
    assert log["triggered"].sum() == 1                    # only the first frame
    first = r.predictions[(seqs[0].name, 0)]
    for fid in log["frame_id"]:
        assert r.predictions[(seqs[0].name, int(fid))] is first   # identical reused object


def test_full_experiment_writes_report(tmp_path):
    cfg = small_cfg(tmp_path, "event_trigger.thresholds=[1.2, 2.0, 5.0]", "reporting.video.enabled=true")
    run_dir = run_experiment(cfg)
    s = pd.read_csv(run_dir / "metrics" / "summary.csv")
    assert list(s["method"])[0] == "baseline" and len(s) == 4
    ev = s.iloc[1:]
    assert (np.diff(ev["skip_rate"]) >= 0).all()          # higher tau -> more skipping
    assert s["recommended"].sum() == 1
    for f in ["report/report.html", "report/report.md", "tables/results.tex", "figures/threshold_tradeoff.png",
              "run_info.json", "per_frame/baseline.csv", "predictions/baseline.npz"]:
        assert (run_dir / f).exists(), f
    assert list((run_dir / "videos").glob("*.mp4"))


def test_replay_mode_matches_measured_accuracy(tmp_path):
    a = run_experiment(small_cfg(tmp_path / "a", "event_trigger.thresholds=[2.0]"))
    b = run_experiment(small_cfg(tmp_path / "b", "event_trigger.thresholds=[2.0]", "evaluation.timing_mode=replay"))
    sa, sb = pd.read_csv(a / "metrics" / "summary.csv"), pd.read_csv(b / "metrics" / "summary.csv")
    assert np.allclose(sa["mAP50"], sb["mAP50"]) and np.allclose(sa["skip_rate"], sb["skip_rate"])
