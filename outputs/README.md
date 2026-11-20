# outputs/

* `runs/<timestamp>_<experiment>/`: everything produced by one experiment (report, figures, tables, metrics, per-frame logs, predictions, video, config, log). Open `report/report.html`.
* `runs/LATEST.txt`: path of the newest run (used by `scripts/generate_report.py` with no arguments).
* `training/`: Ultralytics training runs from `scripts/train_detector.py`.
* `dataset_preview/`: annotated sample frames from `scripts/prepare_data.py --preview N`.
