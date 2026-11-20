"""Builds the full report for a finished run from the artefacts on disk.

    report/report.html   self-contained (figures embedded) - open in any browser
    report/report.md     same content as Markdown
    tables/*.tex / *.md  tables ready for the paper (Overleaf)
    figures/*.png|pdf    all figures

Can be re-run at any time:  python scripts/generate_report.py --run <run_dir>
"""

from __future__ import annotations

import base64
import math
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from etod.pipeline.storage import load_json, method_slug
from etod.reporting import plots
from etod.reporting.tables import (_f, main_table_latex, main_table_rows, simple_table, table1_row_latex,
                                   to_markdown)
from etod.utils.logging_utils import get_logger

TEMPLATE_DIR = Path(__file__).parent / "templates"

PAPER_MAPPING = [
    ("Eq. (1)/(3)  D_t = (1/N) Σ |I_t(i) − I_{t−1}(i)|", "src/etod/event/temporal_difference.py → difference_score()"),
    ("Eq. (4)  E_t = 1 if D_t > τ else 0", "src/etod/event/temporal_difference.py → decide()"),
    ("Fig. 1  skip detection / reuse previous result", "src/etod/pipeline/runner.py → run_method()"),
    ("Eq. (2)/(5)  S = (1 − N_d/N_f) × 100", "src/etod/evaluation/efficiency.py → skip_rate()"),
    ("Sec. III-D  same detector, resolution, confidence, hardware", "src/etod/pipeline/experiment.py (one detector instance)"),
    ("Sec. III-E  precision, recall, mAP", "src/etod/evaluation/detection_metrics.py"),
    ("Sec. III-E  FPS, latency, invocation rate, CPU/GPU", "src/etod/evaluation/efficiency.py, resources.py"),
    ("Sec. III-F  five experimental phases", "src/etod/pipeline/experiment.py → run_experiment()"),
    ("Sec. IV-A  low / moderate / high activity", "src/etod/evaluation/efficiency.py → activity_breakdown()"),
    ("Sec. IV-E  multiple thresholds evaluated", "configs/default.yaml → event_trigger.thresholds"),
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _img_b64(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def key_findings(summary: pd.DataFrame, info: dict, activity: pd.DataFrame) -> list[str]:
    base = summary[summary["method"] == "baseline"].iloc[0]
    ev = summary[summary["method"] != "baseline"].sort_values("threshold")
    ds = info["dataset"]
    ref = "ground-truth annotations" if info["reference"] == "ground_truth" else "the baseline detector's own output"
    out = [
        f"{ds['num_frames']:,} frames from {ds['num_sequences']} sequence(s) were processed. Accuracy is scored against "
        f"{ref} on every frame; skipped frames are scored with the reused previous result.",
        f"Baseline (detector on every frame): mAP@0.5 = {base['mAP50'] * 100:.2f}%, mAP@[.5:.95] = {base['mAP50_95'] * 100:.2f}%, "
        f"{base['fps']:.1f} FPS, {base['latency_mean_ms']:.2f} ms mean latency per frame.",
    ]
    if ev.empty:
        return out
    out.append(
        f"Across τ = {ev['threshold'].min():g} … {ev['threshold'].max():g}, the frame-skip rate S ranged from "
        f"{ev['skip_rate'].min():.1f}% to {ev['skip_rate'].max():.1f}% and mAP@0.5 changed by "
        f"{ev['delta_mAP50_pp'].max():+.2f} to {ev['delta_mAP50_pp'].min():+.2f} percentage points."
    )
    rec = summary[summary["recommended"].astype(bool)]
    if len(rec):
        r = rec.iloc[0]
        txt = (f"Recommended τ = {r['threshold']:g} ({info['recommendation'].get('criterion', '')}): the detector ran on "
               f"{r['invocation_rate'] * 100:.1f}% of frames (S = {r['skip_rate']:.1f}%), throughput went from "
               f"{base['fps']:.1f} to {r['fps']:.1f} FPS ({r['fps_speedup']:.2f}×), mean latency fell by "
               f"{r['latency_reduction_pct']:.1f}%")
        if "compute_saved_pct" in r and not pd.isna(r.get("compute_saved_pct")):
            txt += f", detector compute fell by {r['compute_saved_pct']:.1f}%"
        txt += (f", while mAP@0.5 changed by {r['delta_mAP50_pp']:+.2f} pp and mAP@[.5:.95] by "
                f"{r['delta_mAP50_95_pp']:+.2f} pp.")
        out.append(txt)
        if not activity.empty:
            a = activity[activity["method"] == r["method"]].set_index("activity")
            if {"low", "high"} <= set(a.index):
                out.append(
                    f"At this τ the detector ran on {a.loc['low', 'invocation_rate'] * 100:.0f}% of low-activity frames "
                    f"and {a.loc['high', 'invocation_rate'] * 100:.0f}% of high-activity frames, i.e. computation follows "
                    f"scene change as intended."
                )
        out.append(
            f"The event module itself costs {r['event_overhead_mean_ms']:.3f} ms per frame, versus "
            f"{base['detector_latency_mean_ms']:.2f} ms for one detector call."
        )
    if info.get("timing_mode") == "replay":
        out.append("Note: timing_mode = replay. Event-triggered timings were estimated from measured baseline "
                   "per-call latencies; accuracy numbers are exact. Use timing_mode = measured for final results.")
    return out


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    log = get_logger()
    info = load_json(run_dir / "run_info.json")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    rep = cfg.get("reporting", {})
    summary = pd.read_csv(run_dir / "metrics" / "summary.csv")
    per_class = _read_csv(run_dir / "metrics" / "per_class.csv")
    per_seq = _read_csv(run_dir / "metrics" / "per_sequence.csv")
    activity = _read_csv(run_dir / "metrics" / "activity_levels.csv")
    profile = _read_csv(run_dir / "per_frame" / "baseline.csv")
    rec = info.get("recommendation") or {}
    rec_method = rec.get("method")
    rec_log = _read_csv(run_dir / "per_frame" / f"{method_slug(rec_method)}.csv") if rec_method else pd.DataFrame()

    # ---------------------------------------------------------------- figures
    fw = plots.FigureWriter(run_dir / "figures", rep.get("figure_formats", ["png", "pdf"]), rep.get("dpi", 160))
    if (summary["method"] != "baseline").any():
        plots.fig_threshold_tradeoff(fw, summary)
        plots.fig_accuracy_efficiency(fw, summary)
        plots.fig_speed(fw, summary)
        plots.fig_quality_metrics(fw, summary)
        if not profile.empty:
            plots.fig_dt_histogram(fw, profile, summary)
            seqs = list(dict.fromkeys(profile["sequence"]))[: int(rep.get("timeline_sequences", 2))]
            for s in seqs:
                plots.fig_dt_timeline(fw, profile, rec_log if not rec_log.empty else None, rec.get("threshold"), s)
        plots.fig_activity(fw, activity)
        plots.fig_per_class(fw, per_class, rec_method)
        plots.fig_resources(fw, summary)
        plots.fig_per_sequence(fw, per_seq, rec_method)

    # ---------------------------------------------------------------- tables
    headers, rows, hl = main_table_rows(summary)
    md_main = to_markdown(headers, rows, hl)
    (run_dir / "tables" / "results.md").write_text(md_main + "\n", encoding="utf-8")
    ds = info["dataset"]
    caption = (f"Frame-by-frame vs event-triggered detection ({info['detector'].get('name', '')}, "
               f"{ds['num_sequences']} sequences, {ds['num_frames']} frames). Bold: recommended τ.")
    (run_dir / "tables" / "results.tex").write_text(main_table_latex(summary, caption) + "\n", encoding="utf-8")
    if rec_method:
        r = summary[summary["method"] == rec_method].iloc[0]
        (run_dir / "tables" / "table1_proposed_row.tex").write_text(table1_row_latex(r), encoding="utf-8")
    summary.to_csv(run_dir / "tables" / "results_full.csv", index=False)

    pct = lambda v: _f(v, 1, pct=True)  # noqa: E731
    class_tbl = ([], [])
    if not per_class.empty and rec_method:
        pc = per_class[per_class["method"].isin(["baseline", rec_method])].copy()
        pc["method"] = pc["method"].map(lambda m: "Baseline" if m == "baseline" else m.replace("event_tau_", "τ="))
        class_tbl = simple_table(pc, {"method": ("Method", str), "class": ("Class", str), "num_gt": ("GT boxes", lambda v: f"{int(v):,}"),
                                      "num_pred": ("Detections", lambda v: f"{int(v):,}"), "AP50": ("AP@0.5 (%)", pct),
                                      "AP50_95": ("AP@.5:.95 (%)", pct), "precision": ("Precision (%)", pct), "recall": ("Recall (%)", pct)})
    act_tbl = ([], [])
    if not activity.empty:
        act_tbl = simple_table(activity, {"threshold": ("τ", lambda v: f"{v:g}"), "activity": ("Activity", str),
                                          "frames": ("Frames", lambda v: f"{int(v):,}"), "mean_D": ("Mean D_t", lambda v: _f(v, 2)),
                                          "invocation_rate": ("Detector invoked (%)", pct)})
    seq_tbl = ([], [])
    if not per_seq.empty and rec_method:
        b = per_seq[per_seq["method"] == "baseline"].set_index("sequence")
        r = per_seq[per_seq["method"] == rec_method].set_index("sequence").copy()
        r["base_mAP50"] = b["mAP50"]
        r["delta"] = (r["mAP50"] - r["base_mAP50"]) * 100
        meta = {row["sequence"]: row for row in ds["sequences"]}
        r["frames"] = [meta.get(s, {}).get("frames") for s in r.index]
        r["camera"] = [meta.get(s, {}).get("camera_state") or "–" for s in r.index]
        seq_tbl = simple_table(r.reset_index(), {
            "sequence": ("Sequence", str), "frames": ("Frames", lambda v: "–" if v is None or pd.isna(v) else f"{int(v):,}"),
            "camera": ("Camera", str), "mean_D": ("Mean D_t", lambda v: _f(v, 2)), "skip_rate": ("Skipped (%)", lambda v: _f(v, 1)),
            "base_mAP50": ("Baseline mAP@0.5 (%)", pct), "mAP50": ("Event mAP@0.5 (%)", pct), "delta": ("Δ (pp)", lambda v: _f(v, 2, signed=True))})

    findings = key_findings(summary, info, activity)

    # ---------------------------------------------------------------- render
    preview = run_dir / "videos" / "comparison_preview.jpg"
    videos = sorted(p.name for p in (run_dir / "videos").glob("*.mp4"))
    ctx = {
        "title": f"Event-Triggered Object Detection — {info['experiment'].get('name', '')}",
        "info": info,
        "cfg": cfg,
        "findings": findings,
        "main": {"headers": headers, "rows": rows, "highlight": hl},
        "class_tbl": {"headers": class_tbl[0], "rows": class_tbl[1]},
        "act_tbl": {"headers": act_tbl[0], "rows": act_tbl[1]},
        "seq_tbl": {"headers": seq_tbl[0], "rows": seq_tbl[1]},
        "figures": [{**f, "src": _img_b64(run_dir / "figures" / f["file"])} for f in fw.written],
        "preview": _img_b64(preview) if preview.exists() else None,
        "videos": videos,
        "mapping": PAPER_MAPPING,
        "env": info.get("environment", {}),
        "detector": info.get("detector", {}),
        "phases": {k: f"{v:.1f}" for k, v in info.get("phase_durations_s", {}).items()},
        "config_yaml": yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        "notes": info["experiment"].get("notes", ""),
    }
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(["html"]))
    html = env.get_template("report.html.j2").render(**ctx)
    out_html = run_dir / "report" / "report.html"
    out_html.write_text(html, encoding="utf-8")

    md = _markdown(ctx, md_main)
    (run_dir / "report" / "report.md").write_text(md, encoding="utf-8")
    log.info(f"Report written: {out_html}")
    return out_html


def _markdown(ctx: dict, md_main: str) -> str:
    info = ctx["info"]
    lines = [f"# {ctx['title']}", "", f"Run: `{Path(info['run_dir']).name}`  ", f"Generated: {info['environment'].get('timestamp', '')}", "",
             "## Key findings", ""]
    lines += [f"- {f}" for f in ctx["findings"]]
    lines += ["", "## Results", "", md_main, ""]
    for name, tbl in [("Per-class results", ctx["class_tbl"]), ("Detector usage by activity level", ctx["act_tbl"]),
                      ("Per-sequence results (recommended τ)", ctx["seq_tbl"])]:
        if tbl["rows"]:
            lines += [f"## {name}", "", to_markdown(tbl["headers"], tbl["rows"]), ""]
    lines += ["## Figures", ""]
    for f in ctx["figures"]:
        lines += [f"### {f['title']}", "", f"![{f['title']}](../figures/{f['file']})", "", f"_{f['caption']}_", ""]
    lines += ["## Paper → code mapping", "", "| Paper | Implementation |", "|---|---|"]
    lines += [f"| {a} | `{b}` |" for a, b in ctx["mapping"]]
    lines += ["", "## Setup", "", f"- Detector: {ctx['detector'].get('name')} ({ctx['detector'].get('params_m', '?')} M params, "
              f"{ctx['detector'].get('gflops_per_inference', '?')} GFLOPs/inference), device {ctx['detector'].get('device')}",
              f"- Thresholds τ: {info['thresholds']}", f"- Reference: {info['reference']}, timing mode: {info['timing_mode']}",
              f"- Hardware: {ctx['env'].get('processor')}, {ctx['env'].get('cpu_count_logical')} logical CPUs, "
              f"{ctx['env'].get('ram_gb')} GB RAM, GPU: {ctx['env'].get('gpu', 'none')}", ""]
    return "\n".join(lines)
