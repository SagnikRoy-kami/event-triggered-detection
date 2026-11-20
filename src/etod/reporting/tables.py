"""Result tables in Markdown and LaTeX (IEEE-conference friendly, no extra packages)."""

from __future__ import annotations

import math

import pandas as pd


def _f(v, digits=2, pct=False, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    v = float(v) * (100 if pct else 1)
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def method_name(row) -> str:
    return "Baseline (every frame)" if row["method"] == "baseline" else f"Event-triggered, τ={row['threshold']:g}"


MAIN_COLUMNS = [
    # header, function(row) -> str
    ("Method", method_name),
    ("mAP@0.5 (%)", lambda r: _f(r["mAP50"], 2, pct=True)),
    ("mAP@.5:.95 (%)", lambda r: _f(r["mAP50_95"], 2, pct=True)),
    ("Precision (%)", lambda r: _f(r["precision"], 1, pct=True)),
    ("Recall (%)", lambda r: _f(r["recall"], 1, pct=True)),
    ("F1 (%)", lambda r: _f(r["f1"], 1, pct=True)),
    ("FPS", lambda r: _f(r["fps"], 1)),
    ("Latency (ms)", lambda r: _f(r["latency_mean_ms"], 2)),
    ("Invocation (%)", lambda r: _f(r["invocation_rate"], 1, pct=True)),
    ("Skip rate S (%)", lambda r: _f(r["skip_rate"], 1)),
    ("Δ mAP@0.5 (pp)", lambda r: "–" if r["method"] == "baseline" else _f(r["delta_mAP50_pp"], 2, signed=True)),
    ("Speed-up", lambda r: f"{r['fps_speedup']:.2f}×"),
]


def main_table_rows(summary: pd.DataFrame) -> tuple[list[str], list[list[str]], list[bool]]:
    headers = [h for h, _ in MAIN_COLUMNS]
    rows = [[fn(r) for _, fn in MAIN_COLUMNS] for _, r in summary.iterrows()]
    highlight = list(summary.get("recommended", pd.Series([False] * len(summary))).astype(bool))
    return headers, rows, highlight


def to_markdown(headers: list[str], rows: list[list[str]], highlight: list[bool] | None = None) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for i, r in enumerate(rows):
        cells = [f"**{c}**" for c in r] if highlight and highlight[i] else r
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _tex(s: str) -> str:
    return (s.replace("%", r"\%").replace("τ", r"$\tau$").replace("Δ", r"$\Delta$").replace("×", r"$\times$")
            .replace("–", "--").replace("@", "@").replace("_", r"\_"))


def main_table_latex(summary: pd.DataFrame, caption: str, label: str = "tab:event_triggered_results") -> str:
    cols = [("Method", method_name),
            ("mAP@0.5", lambda r: _f(r["mAP50"], 1, pct=True)),
            ("mAP@.5:.95", lambda r: _f(r["mAP50_95"], 1, pct=True)),
            ("P", lambda r: _f(r["precision"], 1, pct=True)),
            ("R", lambda r: _f(r["recall"], 1, pct=True)),
            ("FPS", lambda r: _f(r["fps"], 1)),
            ("Lat. (ms)", lambda r: _f(r["latency_mean_ms"], 1)),
            ("$N_d/N_f$ (\\%)", lambda r: _f(r["invocation_rate"], 1, pct=True)),
            ("$S$ (\\%)", lambda r: _f(r["skip_rate"], 1))]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{" + _tex(caption) + "}",
        r"\label{" + label + "}",
        r"\centering",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}",
        r"\hline",
        " & ".join(c if c.startswith("$") or "\\%" in c else _tex(c) for c, _ in cols) + r" \\",
        r"\hline",
    ]
    for _, r in summary.iterrows():
        cells = [_tex(fn(r)) for _, fn in cols]
        if bool(r.get("recommended", False)):
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def table1_row_latex(rec_row) -> str:
    """Replacement for the 'To be evaluated' row of the paper's Table I."""
    return (
        "% Replace the 'Proposed Framework ... To be evaluated' row of Table I with:\n"
        f"Proposed Framework & 2026 & Temporal event triggering & "
        f"{rec_row['skip_rate']:.1f}\\% frames skipped; {rec_row['fps_speedup']:.2f}$\\times$ FPS; "
        f"mAP@0.5 {rec_row['delta_mAP50_pp']:+.2f} pp \\\\\n"
    )


def simple_table(df: pd.DataFrame, columns: dict[str, tuple[str, callable]]) -> tuple[list[str], list[list[str]]]:
    headers = [h for h, _ in columns.values()]
    rows = [[fn(r[c]) if c in r else "–" for c, (_, fn) in columns.items()] for _, r in df.iterrows()]
    return headers, rows
