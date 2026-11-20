"""Run the full experiment (paper Sec. III-F, five phases) and generate the report.

Examples
--------
    # smoke test, no downloads (seconds)
    python scripts/run_experiment.py --config configs/experiments/synthetic_smoke.yaml

    # UA-DETRAC, a few test sequences
    python scripts/run_experiment.py --config configs/experiments/ua_detrac_quick.yaml

    # full paper experiment
    python scripts/run_experiment.py --config configs/experiments/ua_detrac_paper.yaml

    # override anything from the command line
    python scripts/run_experiment.py --config configs/experiments/ua_detrac_quick.yaml \
        --set detector.weights=yolov8s.pt detector.device=cuda:0 "event_trigger.thresholds=[1,2,4,8]"
"""
import argparse

import _bootstrap  # noqa: F401

from etod.config import load_config
from etod.pipeline import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/ua_detrac_quick.yaml", help="experiment YAML")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    run_dir = run_experiment(cfg)
    print(f"\nReport: {run_dir / 'report' / 'report.html'}")


if __name__ == "__main__":
    main()
