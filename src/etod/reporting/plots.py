"""All report figures. Every figure is saved as PNG (for the HTML report) and,
optionally, PDF (vector, for LaTeX/Overleaf).

Conventions: one y-axis per panel (two measures -> two stacked panels),
baseline = orange, event-triggered = blue, an ordered blue ramp when several
thresholds are compared, recessive grid, direct labels only where useful.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Validated categorical palette (light mode) + chart chrome
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BASELINE_C, EVENT_C = ORANGE, BLUE
RAMP = LinearSegmentedColormap.from_list("tau", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2.0, "lines.markersize": 6.5, "font.family": "DejaVu Sans",
})


def tau_colors(n: int) -> list:
    return [RAMP(x) for x in (np.linspace(0, 1, n) if n > 1 else [0.6])]


class FigureWriter:
    def __init__(self, out_dir: Path, formats=("png", "pdf"), dpi: int = 160) -> None:
        self.out_dir = Path(out_dir)
        self.formats = list(formats)
        self.dpi = dpi
        self.written: list[dict] = []

    def save(self, fig, name: str, title: str, caption: str) -> None:
        fig.tight_layout()
        for fmt in self.formats:
            fig.savefig(self.out_dir / f"{name}.{fmt}", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.written.append({"name": name, "file": f"{name}.png", "title": title, "caption": caption})


def _split(summary: pd.DataFrame):
    base = summary[summary["method"] == "baseline"].iloc[0]
    ev = summary[summary["method"] != "baseline"].sort_values("threshold").reset_index(drop=True)
    return base, ev


def _rec(ev: pd.DataFrame):
    r = ev[ev["recommended"].astype(bool)] if "recommended" in ev else ev.iloc[0:0]
    return r.iloc[0] if len(r) else None


# --------------------------------------------------------------------------- figures
def fig_threshold_tradeoff(fw: FigureWriter, summary: pd.DataFrame) -> None:
    base, ev = _split(summary)
    rec = _rec(ev)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
    a1.plot(ev["threshold"], ev["mAP50"] * 100, "-o", color=EVENT_C, label="Event-triggered mAP@0.5")
    a1.plot(ev["threshold"], ev["mAP50_95"] * 100, "-s", color=AQUA, label="Event-triggered mAP@[.5:.95]")
    a1.axhline(base["mAP50"] * 100, color=BASELINE_C, ls="--", lw=1.6, label="Baseline mAP@0.5")
    a1.axhline(base["mAP50_95"] * 100, color=BASELINE_C, ls=":", lw=1.6, label="Baseline mAP@[.5:.95]")
    a1.set_ylabel("mAP (%)")
    a1.set_title("Accuracy vs event threshold τ")
    a1.legend(loc="lower left", ncol=2)
    a2.plot(ev["threshold"], ev["invocation_rate"] * 100, "-o", color=EVENT_C, label="Detector invocation rate")
    a2.axhline(100, color=BASELINE_C, ls="--", lw=1.6, label="Baseline (every frame)")
    for _, r in ev.iterrows():
        a2.annotate(f"{r['invocation_rate'] * 100:.0f}%", (r["threshold"], r["invocation_rate"] * 100),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK2)
    a2.set_ylim(0, 112)
    a2.set_ylabel("Frames sent to detector (%)")
    a2.set_xlabel("Event threshold τ (mean absolute grey-level difference, Eq. 1)")
    a2.set_title("Detector invocation rate vs τ")
    a2.legend(loc="upper right")
    if rec is not None:
        for a in (a1, a2):
            a.axvline(rec["threshold"], color=MUTED, lw=1, ls="-.", zorder=0)
        a2.annotate(f"recommended τ={rec['threshold']:g}", (rec["threshold"], 4), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK2)
    fw.save(fig, "threshold_tradeoff", "Accuracy and detector invocation vs threshold τ",
            "Top: mAP of the event-triggered method for each τ, baseline as dashed lines. Bottom: share of frames on "
            "which the detector actually ran (Nd/Nf). Higher τ skips more frames.")


def fig_accuracy_efficiency(fw: FigureWriter, summary: pd.DataFrame) -> None:
    base, ev = _split(summary)
    rec = _rec(ev)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ev["skip_rate"], ev["mAP50"] * 100, "-", color=EVENT_C, lw=1.5, alpha=0.6)
    ax.scatter(ev["skip_rate"], ev["mAP50"] * 100, s=55, color=tau_colors(len(ev)), edgecolor=SURFACE, lw=2, zorder=3,
               label="Event-triggered (darker = higher τ)")
    ax.scatter([0], [base["mAP50"] * 100], s=70, marker="D", color=BASELINE_C, edgecolor=SURFACE, lw=2, zorder=3,
               label="Baseline (frame-by-frame)")
    for _, r in ev.iterrows():
        ax.annotate(f"τ={r['threshold']:g}", (r["skip_rate"], r["mAP50"] * 100), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color=INK2)
    if rec is not None:
        ax.scatter([rec["skip_rate"]], [rec["mAP50"] * 100], s=180, facecolor="none", edgecolor=INK, lw=1.3, zorder=4,
                   label="Recommended τ")
    ax.set_xlabel("Frame-skip rate S (%)  [Eq. 2]")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_xlim(-3, 103)
    ax.set_title("Accuracy–efficiency trade-off")
    ax.legend(loc="lower left")
    fw.save(fig, "accuracy_efficiency_tradeoff", "Accuracy–efficiency trade-off",
            "Each point is one threshold τ: how many frames were skipped (x) and the resulting mAP@0.5 (y). "
            "Points close to the top-right are the best balance.")


def fig_speed(fw: FigureWriter, summary: pd.DataFrame) -> None:
    base, ev = _split(summary)
    labels = ["Baseline"] + [f"τ={t:g}" for t in ev["threshold"]]
    colors = [BASELINE_C] + tau_colors(len(ev))
    fps = [base["fps"]] + list(ev["fps"])
    event_ms = [base["event_overhead_mean_ms"]] + list(ev["event_overhead_mean_ms"])
    total_ms = [base["latency_mean_ms"]] + list(ev["latency_mean_ms"])
    det_ms = [max(t - e, 0) for t, e in zip(total_ms, event_ms)]
    x = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    bars = a1.bar(x, fps, color=colors, width=0.62, edgecolor=SURFACE, lw=2)
    for b, v in zip(bars, fps):
        a1.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    a1.set_ylabel("Frames per second")
    a1.set_title("Processing throughput (FPS)")
    a2.bar(x, det_ms, color=colors, width=0.62, edgecolor=SURFACE, lw=2)
    a2.bar(x, event_ms, bottom=det_ms, color=MUTED, width=0.62, edgecolor=SURFACE, lw=2)
    for xi, v in zip(x, total_ms):
        a2.annotate(f"{v:.1f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    a2.set_ylabel("Mean latency per frame (ms)")
    a2.set_title("Where the per-frame time goes")
    a2.legend(handles=[Patch(facecolor="#d9d8d3", edgecolor=MUTED, label="Detector + result reuse (bar colour = configuration)"),
                       Patch(facecolor=MUTED, label="Event module (computing D_t)")], loc="upper right")
    a2.set_xticks(x, labels, rotation=0)
    fw.save(fig, "speed_latency", "Throughput and latency",
            "Top: FPS of every configuration. Bottom: mean processing time per frame, split into detector time "
            "and the lightweight event module's overhead.")


def fig_quality_metrics(fw: FigureWriter, summary: pd.DataFrame) -> None:
    base, ev = _split(summary)
    labels = ["Baseline"] + [f"τ={t:g}" for t in ev["threshold"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, color, marker, name in [("precision", BLUE, "o", "Precision"), ("recall", ORANGE, "s", "Recall"),
                                      ("f1", AQUA, "^", "F1"), ("accuracy", VIOLET, "D", "Accuracy TP/(TP+FP+FN)")]:
        vals = [base[col] * 100] + list(ev[col] * 100)
        ax.plot(x, vals, marker=marker, color=color, label=name)
        ax.annotate(name.split(" ")[0], (x[-1], vals[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.3, len(x) - 0.4)
    ax.set_ylabel("%")
    ax.set_title("Precision, recall, F1 and detection accuracy (IoU 0.5)")
    ax.legend(loc="lower left", ncol=2)
    fw.save(fig, "precision_recall_f1", "Precision / recall / F1 / accuracy",
            "Detection quality at IoU 0.5 for the baseline and every threshold, scored on all frames "
            "(skipped frames use the reused previous result).")


def fig_dt_histogram(fw: FigureWriter, profile: pd.DataFrame, summary: pd.DataFrame) -> None:
    base, ev = _split(summary)
    rec = _rec(ev)
    d = profile["score"].to_numpy(float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return
    hi = np.percentile(d, 99.5)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(np.clip(d, 0, hi), bins=80, color="#9ec5f4", edgecolor=SURFACE, lw=0.5)
    for t, c in zip(ev["threshold"], tau_colors(len(ev))):
        ax.axvline(t, color=c, lw=1.3 if rec is None or t != rec["threshold"] else 2.4)
    if rec is not None:
        ax.annotate(f"recommended τ={rec['threshold']:g}", (rec["threshold"], ax.get_ylim()[1] * 0.92), xytext=(5, 0),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("D_t  (mean |I_t − I_{t−1}|, grey levels)")
    ax.set_ylabel("Frames")
    ax.set_title("Distribution of the temporal-difference score D_t")
    fw.save(fig, "dt_histogram", "Distribution of D_t",
            "Histogram of D_t over all evaluated frames; vertical lines are the evaluated thresholds τ "
            "(darker = higher). Frames to the left of a line are skipped at that τ.")


def fig_dt_timeline(fw: FigureWriter, profile: pd.DataFrame, event_log: pd.DataFrame | None, tau: float | None,
                    sequence: str, fps: float | None = None) -> None:
    p = profile[profile["sequence"] == sequence]
    if p.empty:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(p["frame_index"], p["score"], color=BLUE, lw=1.2)
    if tau is not None:
        a1.axhline(tau, color=ORANGE, lw=1.6, ls="--")
        a1.annotate(f"τ = {tau:g}", (p["frame_index"].iloc[-1], tau), xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=8.5, color=INK2)
    a1.set_ylabel("D_t")
    a1.set_title(f"{sequence}: temporal change and detector activations")
    if event_log is not None:
        e = event_log[event_log["sequence"] == sequence]
        trig = e["triggered"].to_numpy(bool)
        idx = e["frame_index"].to_numpy()
        a2.vlines(idx[trig], 0.05, 0.95, color=GREEN, lw=0.8, label="detector ran")
        a2.vlines(idx[~trig], 0.05, 0.95, color="#d9d8d3", lw=0.8, label="skipped (result reused)")
        a2.set_yticks([])
        a2.grid(False)
        a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2)
        a2.set_title(f"Detector activations at τ={tau:g}: {trig.sum()} of {len(trig)} frames "
                     f"({(1 - trig.mean()) * 100:.0f}% skipped)", fontsize=9.5, fontweight="normal")
    a2.set_xlabel("Frame index")
    fw.save(fig, f"dt_timeline_{sequence}", f"D_t timeline – {sequence}",
            "Top: D_t for every frame with the threshold τ. Bottom: frames where the detector ran (green) vs "
            "skipped frames that reused the previous result (grey) – the frame-level view of the paper's Fig. 2.")


def fig_per_class(fw: FigureWriter, per_class: pd.DataFrame, rec_method: str | None) -> None:
    if per_class.empty or rec_method is None:
        return
    b = per_class[per_class["method"] == "baseline"].set_index("class")
    r = per_class[per_class["method"] == rec_method].set_index("class")
    classes = [c for c in b.index if b.loc[c, "num_gt"] > 0]
    if not classes:
        return
    x = np.arange(len(classes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    vb = [b.loc[c, "AP50"] * 100 for c in classes]
    vr = [r.loc[c, "AP50"] * 100 for c in classes]
    ax.bar(x - w / 2, vb, w, color=BASELINE_C, edgecolor=SURFACE, lw=2, label="Baseline")
    ax.bar(x + w / 2, vr, w, color=EVENT_C, edgecolor=SURFACE, lw=2, label=f"Event-triggered ({rec_method.replace('event_tau_', 'τ=')})")
    for xi, v in list(zip(x - w / 2, vb)) + list(zip(x + w / 2, vr)):
        ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{c}\n(n={int(b.loc[c, 'num_gt'])})" for c in classes])
    ax.set_ylabel("AP@0.5 (%)")
    ax.set_title("Per-class AP@0.5")
    ax.legend(loc="upper right")
    fw.save(fig, "per_class_ap", "Per-class AP@0.5",
            "AP@0.5 per object class for the baseline and the recommended threshold; n = ground-truth boxes.")


def fig_activity(fw: FigureWriter, activity: pd.DataFrame) -> None:
    if activity.empty:
        return
    taus = sorted(activity["threshold"].unique())
    levels = ["low", "moderate", "high"]
    x = np.arange(len(levels))
    w = 0.8 / len(taus)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (t, c) in enumerate(zip(taus, tau_colors(len(taus)))):
        a = activity[activity["threshold"] == t].set_index("activity").reindex(levels)
        ax.bar(x - 0.4 + w * (i + 0.5), a["invocation_rate"] * 100, w, color=c, edgecolor=SURFACE, lw=1, label=f"τ={t:g}")
    ax.set_xticks(x, ["Low activity", "Moderate activity", "High activity"])
    ax.set_ylabel("Detector invocation rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Detector usage by scene activity level")
    ax.legend(ncol=min(4, len(taus)), loc="upper left")
    fw.save(fig, "activity_levels", "Detector usage by scene activity",
            "Frames are grouped into low / moderate / high activity by D_t terciles (paper Sec. IV-A). "
            "An effective trigger runs the detector rarely in quiet periods and often in busy ones.")


def fig_resources(fw: FigureWriter, summary: pd.DataFrame) -> None:
    if "cpu_process_mean_pct" not in summary:
        return
    base, ev = _split(summary)
    labels = ["Baseline"] + [f"τ={t:g}" for t in ev["threshold"]]
    colors = [BASELINE_C] + tau_colors(len(ev))
    panels = [("cpu_process_mean_pct", "Mean process CPU (% of machine)")]
    if "gpu_util_mean_pct" in summary and summary["gpu_util_mean_pct"].notna().any():
        panels.append(("gpu_util_mean_pct", "Mean GPU utilisation (%)"))
    if "gflops_total" in summary:
        panels.append(("gflops_per_frame", "Detector GFLOPs per input frame"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 2.6 * len(panels)), sharex=True, squeeze=False)
    x = np.arange(len(labels))
    for ax, (col, title) in zip(axes[:, 0], panels):
        vals = [base[col]] + list(ev[col])
        bars = ax.bar(x, vals, color=colors, width=0.62, edgecolor=SURFACE, lw=2)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10)
    axes[-1, 0].set_xticks(x, labels)
    fw.save(fig, "resource_usage", "Computational resource usage",
            "CPU (and GPU when available) utilisation sampled while each configuration ran, and detector compute "
            "per input frame (GFLOPs x invocation rate).")


def fig_per_sequence(fw: FigureWriter, per_seq: pd.DataFrame, rec_method: str | None) -> None:
    if per_seq.empty or rec_method is None:
        return
    b = per_seq[per_seq["method"] == "baseline"].set_index("sequence")
    r = per_seq[per_seq["method"] == rec_method].set_index("sequence")
    seqs = list(r.index)
    if len(seqs) < 2:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    a1.scatter(r["mean_D"], r["skip_rate"], s=40, color=BLUE, edgecolor=SURFACE, lw=1.5)
    a1.set_xlabel("Mean D_t of the sequence")
    a1.set_ylabel("Frames skipped (%)")
    a1.set_title("Busier scenes trigger more often", fontsize=10)
    delta = (r["mAP50"] - b.loc[seqs, "mAP50"]) * 100
    a2.scatter(r["skip_rate"], delta, s=40, color=BLUE, edgecolor=SURFACE, lw=1.5)
    a2.axhline(0, color=BASELINE_C, ls="--", lw=1.4)
    a2.set_xlabel("Frames skipped (%)")
    a2.set_ylabel("Δ mAP@0.5 vs baseline (pp)")
    a2.set_title("Accuracy change per sequence", fontsize=10)
    for s in delta.abs().sort_values().index[-3:]:
        a2.annotate(s, (r.loc[s, "skip_rate"], delta[s]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    fw.save(fig, "per_sequence", "Per-sequence behaviour",
            f"Each point is one sequence at {rec_method.replace('event_tau_', 'τ=')}. Left: scene activity vs share "
            "of skipped frames. Right: skipped frames vs change in mAP@0.5.")
