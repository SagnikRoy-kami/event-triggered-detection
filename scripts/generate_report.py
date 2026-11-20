"""Rebuild figures, tables and the HTML/Markdown report of an existing run
(e.g. after changing plot styles) without re-running any detection.

    python scripts/generate_report.py                 # latest run
    python scripts/generate_report.py --run outputs/runs/20261120-104500_ua_detrac_quick
    python scripts/generate_report.py --video          # also (re)render the comparison video
"""
import argparse

import _bootstrap  # noqa: F401

from etod.config import load_config
from etod.reporting import build_report, render_comparison_video
from etod.utils.logging_utils import setup_logging
from etod.utils.paths import latest_run, resolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run folder (default: latest)")
    ap.add_argument("--video", action="store_true", help="re-render the side-by-side video (needs the dataset)")
    ap.add_argument("--sequence", default=None, help="sequence for the video")
    ap.add_argument("--threshold", type=float, default=None, help="tau for the video (default: recommended)")
    args = ap.parse_args()
    run_dir = resolve(args.run) if args.run else latest_run()
    if run_dir is None or not run_dir.exists():
        raise SystemExit("No run found. Run scripts/run_experiment.py first.")
    setup_logging(run_dir / "logs" / "report.log")
    if args.video:
        from etod.data import load_sequences

        cfg = load_config(None, None)
        import yaml

        cfg.update(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
        if args.sequence:
            cfg["reporting"]["video"]["sequence"] = args.sequence
            cfg["dataset"]["sequences"] = [args.sequence]
        if args.threshold is not None:
            cfg["reporting"]["video"]["threshold"] = args.threshold
        render_comparison_video(run_dir, cfg, load_sequences(cfg))
    print(build_report(run_dir))


if __name__ == "__main__":
    main()
