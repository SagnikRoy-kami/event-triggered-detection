"""The full experiment, organised exactly as the paper's procedure (Sec. III-F):

    Phase 1  dataset preparation
    Phase 2  baseline detection            (detector on every frame)
    Phase 3  event-triggered detection     (one run per threshold tau)
    Phase 4  metric collection             (accuracy, efficiency, resources)
    Phase 5  comparative analysis          (deltas vs baseline, trade-off, report)

Both methods share the same detector instance, input resolution, confidence
threshold and hardware (Sec. III-D).
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from etod.config import Config, save_config
from etod.data import load_sequences
from etod.data.base import Boxes, Sequence
from etod.detectors import build_detector
from etod.evaluation import Matcher, ResourceMonitor, activity_breakdown, efficiency_metrics
from etod.evaluation.detection_metrics import fraction_inside
from etod.event import AlwaysTrigger, build_profiler, build_trigger
from etod.pipeline.runner import DetectionCache, MethodRun, choose_thresholds, run_method
from etod.pipeline.storage import method_slug, save_json, save_predictions
from etod.utils.environment import collect_environment, set_seed
from etod.utils.logging_utils import setup_logging
from etod.utils.paths import PROJECT_ROOT, ensure_dir, resolve

RUN_SUBDIRS = ["logs", "metrics", "per_frame", "predictions", "figures", "tables", "videos", "report"]


# =========================================================================== helpers
def create_run_dir(cfg) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolve(cfg["experiment"]["output_root"]) / f"{stamp}_{cfg['experiment']['name']}"
    for sub in RUN_SUBDIRS:
        ensure_dir(run_dir / sub)
    return run_dir


def dataset_summary(sequences: list[Sequence], eval_classes: list[str]) -> dict:
    rows = []
    for s in sequences:
        row = {"sequence": s.name, "frames": s.num_frames, "fps": s.fps, **{k: v for k, v in s.metadata.items() if k != "activity_plan"}}
        if s.has_ground_truth:
            labels = np.concatenate([b.labels for b in s.ground_truth.values()]) if s.ground_truth else np.zeros(0, int)
            row["gt_boxes"] = int(len(labels))
            for i, c in enumerate(eval_classes):
                row[f"gt_{c}"] = int((labels == i).sum())
        rows.append(row)
    return {
        "num_sequences": len(sequences),
        "num_frames": int(sum(s.num_frames for s in sequences)),
        "has_ground_truth": all(s.has_ground_truth for s in sequences),
        "sequences": rows,
    }


def make_reference(cfg, sequences: list[Sequence], baseline: MethodRun):
    """Returns f(seq, frame_id) -> reference Boxes for scoring."""
    if cfg["evaluation"]["reference"] == "ground_truth":
        return lambda seq, fid: seq.gt_for(fid)

    def from_baseline(seq: Sequence, fid: int) -> Boxes:
        b = baseline.predictions[(seq.name, fid)]
        if len(seq.ignore_regions) and len(b):
            b = b.filter(fraction_inside(b.xyxy, seq.ignore_regions) < 0.5)
        return Boxes(b.xyxy, b.labels, np.ones(len(b), np.float32))

    return from_baseline


def evaluate_accuracy(cfg, run: MethodRun, sequences: list[Sequence], reference) -> tuple[dict, list[dict]]:
    ev = cfg["evaluation"]
    thrs = sorted(set([float(t) for t in ev["iou_thresholds"]] + [float(ev["pr_iou"])]))
    matcher = Matcher(thrs, num_classes=len(cfg["classes"]["eval_classes"]), min_gt_area=float(ev.get("min_gt_area", 0)))
    by_name = {s.name: s for s in sequences}
    seq_idx = {s.name: i for i, s in enumerate(sequences)}
    for name, fid in zip(run.frame_log["sequence"].to_numpy(), run.frame_log["frame_id"].to_numpy()):
        seq = by_name[name]
        matcher.add_frame(seq_idx[name], reference(seq, int(fid)), run.predictions[(name, int(fid))], seq.ignore_regions)
    overall = matcher.compute(ev["pr_iou"])
    per_seq = []
    if ev.get("per_sequence", True):
        for s in sequences:
            r = matcher.compute(ev["pr_iou"], seq_idx[s.name])
            r.pop("per_class")
            per_seq.append({"sequence": s.name, **r})
    return overall, per_seq


def recommend_threshold(summary: pd.DataFrame, tolerance_pp: float) -> dict:
    """Largest skip rate whose mAP@0.5 drop stays within ``tolerance_pp``."""
    ev = summary[summary["method"] != "baseline"].copy()
    if ev.empty:
        return {}
    metric = "delta_mAP50_pp" if ev["delta_mAP50_pp"].notna().any() else "delta_f1_pp"
    ok = ev[ev[metric] >= -tolerance_pp]
    if len(ok):
        row, why = ok.loc[ok["skip_rate"].idxmax()], f"highest skip rate with {metric.replace('delta_', '').replace('_pp', '')} drop <= {tolerance_pp} pp"
    else:
        row, why = ev.loc[ev[metric].idxmax()], f"no threshold met the {tolerance_pp} pp tolerance; smallest accuracy drop chosen"
    return {"method": row["method"], "threshold": float(row["threshold"]), "criterion": why, "metric": metric}


# =========================================================================== main
def run_experiment(cfg: Config) -> Path:
    run_dir = create_run_dir(cfg)
    log = setup_logging(run_dir / "logs" / "run.log")
    save_config(cfg, run_dir / "config.yaml")
    set_seed(int(cfg["experiment"]["seed"]))
    phases: dict[str, float] = {}
    ev_cfg = cfg["evaluation"]
    log.info(f"Run folder: {run_dir}")

    # ------------------------------------------------------------- Phase 1
    t = time.perf_counter()
    log.info("=== Phase 1/5: dataset preparation ===")
    sequences = load_sequences(cfg)
    if not sequences:
        raise RuntimeError("No sequences to evaluate")
    reference_mode = ev_cfg["reference"]
    if reference_mode == "ground_truth" and not all(s.has_ground_truth for s in sequences):
        log.warning("Dataset has no ground truth -> scoring against the baseline detector output (reference=baseline)")
        reference_mode = "baseline"
    cfg["evaluation"]["reference"] = reference_mode
    ds_summary = dataset_summary(sequences, cfg["classes"]["eval_classes"])
    log.info(f"{ds_summary['num_sequences']} sequences, {ds_summary['num_frames']} frames, reference = {reference_mode}")

    detector = build_detector(cfg)
    det_info = detector.info()
    gflops = det_info.get("gflops_per_inference")
    monitor = ResourceMonitor(ev_cfg["resource_sample_interval"], ev_cfg["monitor_resources"])
    warmup = int(cfg["detector"].get("warmup_iters", 3))
    phases["1_dataset_preparation"] = time.perf_counter() - t

    # ------------------------------------------------------------- Phase 2
    t = time.perf_counter()
    log.info("=== Phase 2/5: baseline detection (every frame) ===")
    baseline = run_method(sequences, detector, AlwaysTrigger(), profiler=build_profiler(cfg), monitor=monitor,
                          warmup_iters=warmup, desc="baseline")
    log.info(f"baseline: {len(baseline.frame_log)} frames in {baseline.wall_time_s:.1f}s")
    thresholds = choose_thresholds(cfg, baseline.frame_log)
    log.info(f"Event thresholds tau: {thresholds}")
    phases["2_baseline_detection"] = time.perf_counter() - t

    reference = make_reference(cfg, sequences, baseline)
    cache = None
    if ev_cfg.get("timing_mode", "measured") == "replay":
        log.info("timing_mode=replay: detector outputs and latencies are reused from the baseline run")
        cache = DetectionCache(
            boxes=baseline.predictions,
            latency_ms={(s, int(f)): float(d) for s, f, d in baseline.frame_log[["sequence", "frame_id", "detector_ms"]].itertuples(index=False)},
        )

    summary_rows, class_rows, seq_rows, activity_rows, resource_frames = [], [], [], [], []
    names = cfg["classes"]["eval_classes"]

    def collect(run: MethodRun) -> None:
        """Phase 4 (metric collection) for one finished run; artefacts saved immediately."""
        slug = method_slug(run.method)
        if cfg["reporting"].get("save_per_frame_log", True):
            run.frame_log.to_csv(run_dir / "per_frame" / f"{slug}.csv", index=False)
        if cfg["reporting"].get("save_predictions", True):
            save_predictions(run_dir / "predictions" / f"{slug}.npz", run.predictions)
        acc, per_seq = evaluate_accuracy(cfg, run, sequences, reference)
        eff = efficiency_metrics(run.frame_log, gflops)
        per_class = acc.pop("per_class")
        summary_rows.append({"method": run.method, "threshold": run.threshold, **acc, **eff, **run.resources,
                             "wall_time_s": run.wall_time_s})
        for pc in per_class:
            class_rows.append({"method": run.method, "threshold": run.threshold, "class": names[pc["class_id"]], **pc})
        for ps in per_seq:
            sl = run.frame_log[run.frame_log["sequence"] == ps["sequence"]]
            seq_rows.append({"method": run.method, "threshold": run.threshold, **ps,
                             "skip_rate": (1 - sl["triggered"].mean()) * 100 if len(sl) else math.nan,
                             "mean_D": float(np.nanmean(baseline.frame_log.loc[baseline.frame_log["sequence"] == ps["sequence"], "score"]))})
        if not run.resource_samples.empty:
            resource_frames.append(run.resource_samples.assign(method=run.method))
        log.info(f"{run.method:>18s} | mAP50={acc['mAP50']:.4f} mAP50-95={acc['mAP50_95']:.4f} P={acc['precision']:.3f} "
                 f"R={acc['recall']:.3f} | FPS={eff['fps']:.1f} skip={eff['skip_rate']:.1f}%")

    t = time.perf_counter()
    log.info("=== Phase 4/5 (baseline): metric collection ===")
    collect(baseline)
    phases["4_metric_collection"] = time.perf_counter() - t

    # ------------------------------------------------------------- Phase 3
    finite = baseline.frame_log["score"].to_numpy(float)
    finite = finite[np.isfinite(finite)]
    activity_bins = [float(np.percentile(finite, 100 / 3)), float(np.percentile(finite, 200 / 3))] if len(finite) else [0.0, 0.0]
    t3 = 0.0
    log.info("=== Phase 3/5: event-triggered detection ===")
    for tau in thresholds:
        t = time.perf_counter()
        run = run_method(sequences, detector, build_trigger(cfg, tau), monitor=monitor, cache=cache,
                         warmup_iters=0 if cache else warmup)
        t3 += time.perf_counter() - t
        t = time.perf_counter()
        collect(run)
        act = activity_breakdown(run.frame_log, baseline.frame_log, activity_bins)
        activity_rows.append(act.assign(method=run.method, threshold=tau))
        phases["4_metric_collection"] += time.perf_counter() - t
        run.predictions.clear()   # saved to disk; free memory
    phases["3_event_triggered_detection"] = t3

    # ------------------------------------------------------------- Phase 5
    t = time.perf_counter()
    log.info("=== Phase 5/5: comparative analysis + report ===")
    summary = pd.DataFrame(summary_rows)
    base = summary.iloc[0]
    for col in ["mAP50", "mAP50_95", "precision", "recall", "f1", "accuracy"]:
        summary[f"delta_{col}_pp"] = (summary[col] - base[col]) * 100
    summary["fps_speedup"] = summary["fps"] / base["fps"]
    summary["latency_reduction_pct"] = (1 - summary["latency_mean_ms"] / base["latency_mean_ms"]) * 100
    if "gflops_total" in summary:
        summary["compute_saved_pct"] = (1 - summary["gflops_total"] / base["gflops_total"]) * 100
    rec = recommend_threshold(summary, float(ev_cfg["accuracy_drop_tolerance"]))
    summary["recommended"] = summary["method"] == rec.get("method")

    summary.to_csv(run_dir / "metrics" / "summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(run_dir / "metrics" / "per_class.csv", index=False)
    if seq_rows:
        pd.DataFrame(seq_rows).to_csv(run_dir / "metrics" / "per_sequence.csv", index=False)
    if activity_rows:
        pd.concat(activity_rows).to_csv(run_dir / "metrics" / "activity_levels.csv", index=False)
    if resource_frames:
        pd.concat(resource_frames).to_csv(run_dir / "metrics" / "resources_timeseries.csv", index=False)
    phases["5_comparative_analysis"] = time.perf_counter() - t

    run_info = {
        "experiment": cfg["experiment"],
        "reference": reference_mode,
        "timing_mode": ev_cfg.get("timing_mode", "measured"),
        "thresholds": thresholds,
        "activity_bins": activity_bins,
        "recommendation": rec,
        "detector": det_info,
        "dataset": ds_summary,
        "environment": collect_environment(det_info.get("device")),
        "phase_durations_s": phases,
        "eval_classes": names,
        "run_dir": str(run_dir),
    }
    save_json(run_dir / "run_info.json", run_info)

    from etod.reporting import build_report, render_comparison_video

    if cfg["reporting"]["video"].get("enabled"):
        try:
            render_comparison_video(run_dir, cfg, sequences)
        except Exception as exc:  # noqa: BLE001 - a video failure must not lose the run
            log.warning(f"Comparison video failed: {exc}")
    build_report(run_dir)

    (run_dir.parent / "LATEST.txt").write_text(str(run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir), encoding="utf-8")
    log.info(f"Done. Report: {run_dir / 'report' / 'report.html'}")
    if rec:
        r = summary[summary["recommended"]].iloc[0]
        log.info(f"Recommended tau = {rec['threshold']:g}: skip {r['skip_rate']:.1f}% of frames, "
                 f"FPS x{r['fps_speedup']:.2f}, mAP50 {r['delta_mAP50_pp']:+.2f} pp vs baseline")
    return run_dir
