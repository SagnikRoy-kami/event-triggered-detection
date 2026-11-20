# Event-Triggered Object Detection

Implementation of the paper **"Event-Driven Object Detection: the model decides *when* to perform expensive detection rather than detecting every frame."**

A lightweight event module measures how much each video frame differs from the previous one. The expensive YOLO detector runs only when that change exceeds a threshold τ. Otherwise the previous detection result is reused. The project compares this against frame-by-frame detection on a real traffic dataset and generates a full report automatically.

```
Input video ─► frame I_t ─► D_t = (1/N)·Σ|I_t(i) − I_{t−1}(i)| ─► D_t > τ ?
                                                                  ├─ yes ─► YOLO detector ─► detections
                                                                  └─ no  ─► reuse previous detections
```

| Paper | Where it lives |
|---|---|
| Eq. (1)/(3): temporal difference D_t | `src/etod/event/temporal_difference.py` |
| Eq. (4): event E_t = 1 if D_t > τ | `src/etod/event/temporal_difference.py` |
| Fig. 1: skip detection and reuse the previous result | `src/etod/pipeline/runner.py` |
| Eq. (2)/(5): frame-skip rate S = (1 − N_d/N_f)·100 | `src/etod/evaluation/efficiency.py` |
| Sec. III-E: precision, recall, mAP, FPS, latency, invocation rate, CPU/GPU | `src/etod/evaluation/` |
| Sec. III-F: five experimental phases | `src/etod/pipeline/experiment.py` |
| Sec. IV-B: UA-DETRAC traffic dataset | `src/etod/data/ua_detrac.py` |

`docs/METHODOLOGY.md` explains each design decision in detail.

---

## 1. Folder structure

```
event-triggered-object-detection/
├── configs/
│   ├── default.yaml              ← EVERY option, documented. Edit experiments, not this.
│   ├── datasets.yaml             ← where datasets are downloaded from / stored
│   └── experiments/
│       ├── synthetic_smoke.yaml  ← no downloads, runs in ~1 min
│       ├── custom_video.yaml     ← your own .mp4 files (no annotations)
│       ├── ua_detrac_quick.yaml  ← 4 UA-DETRAC test sequences (laptop friendly)
│       └── ua_detrac_paper.yaml  ← the full paper experiment (all 40 test sequences)
│
├── data/                         ← (git-ignored contents)
│   ├── raw/ua-detrac/            ← Kaggle download is extracted here
│   ├── processed/                ← index files, YOLO-format export for fine-tuning
│   ├── interim/                  ← scratch space for your own preprocessing
│   └── videos/                   ← drop any .mp4 here for custom_video.yaml
│
├── models/
│   ├── weights/                  ← pretrained YOLO weights (auto-downloaded)
│   └── finetuned/                ← your fine-tuned detectors
│
├── outputs/
│   ├── runs/<timestamp>_<name>/  ← ONE FOLDER PER EXPERIMENT (see section 5)
│   ├── runs/LATEST.txt           ← pointer to the newest run
│   ├── training/                 ← Ultralytics training logs
│   └── dataset_preview/          ← annotated sample frames from prepare_data.py
│
├── scripts/                      ← command-line entry points
│   ├── download_data.py          ← Kaggle / sample-video download
│   ├── prepare_data.py           ← index, preview, YOLO export of the train split
│   ├── run_experiment.py         ← the full 5-phase experiment + report
│   ├── generate_report.py        ← rebuild the report / video of an existing run
│   └── train_detector.py         ← optional fine-tuning on UA-DETRAC train split
│
├── src/etod/                     ← the library (importable package)
│   ├── config.py                 ← YAML loading, merging, validation, --set overrides
│   ├── data/                     ← UA-DETRAC, video, image-folder, synthetic loaders
│   ├── event/                    ← event triggers (temporal difference, always-on baseline)
│   ├── detectors/                ← YOLO (Ultralytics) + tiny test detector
│   ├── pipeline/                 ← per-frame runner + 5-phase experiment + storage
│   ├── evaluation/               ← mAP/P/R, FPS/latency/skip rate, CPU/GPU monitor
│   ├── reporting/                ← figures, tables (MD/LaTeX), HTML report, video
│   └── utils/                    ← paths, logging, environment info
│
├── tests/                        ← pytest suite (32 tests, ~25 s)
├── docs/METHODOLOGY.md           ← how every part of the paper was implemented
├── notebooks/                    ← your own exploration notebooks
├── .vscode/                      ← run/debug configs, tasks, settings, extensions
├── requirements.txt
└── pyproject.toml
```

Rule of thumb: **raw data goes in `data/`, code in `src/`, settings in `configs/`, results in `outputs/`.** Nothing is ever written outside these folders.

---

## 2. Setup in VS Code (Windows, macOS, Linux)

Python **3.10 to 3.13**.

1. Open the folder in VS Code (`File → Open Folder…`). Accept the recommended extensions.
2. Create the environment: `Terminal → Run Task… → Setup: create virtual environment`.
3. Install everything: `Terminal → Run Task… → Setup: install requirements`.
   *NVIDIA GPU?* First install the CUDA build of PyTorch from <https://pytorch.org>, then run the task.
4. Select the interpreter: `Ctrl+Shift+P → Python: Select Interpreter → .venv`.
5. Check that it works: `Terminal → Run Task… → Run tests`, then in Run & Debug (`Ctrl+Shift+D`) start **"1. Smoke test"**.

If you prefer the terminal:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python scripts/run_experiment.py --config configs/experiments/synthetic_smoke.yaml
```

---

## 3. Getting the data

### UA-DETRAC from Kaggle (the dataset the paper names)

Source: <https://www.kaggle.com/datasets/bratjay/ua-detrac-orig>. This is the original layout: frame sequences plus XML annotations for both train and test. The download is about 11.5 GB, so keep about 25 GB free while it extracts.

One-time Kaggle setup: kaggle.com → *Settings* → *API* → **Create New Token**. Save `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json` on Windows or `~/.kaggle/kaggle.json` on macOS/Linux.

```bash
python scripts/download_data.py --dataset ua-detrac      # download + extract + index
python scripts/prepare_data.py --preview 6               # check annotations visually
```

If you downloaded the zip in your browser instead:

```bash
python scripts/download_data.py --dataset ua-detrac --from-zip "C:/Users/<you>/Downloads/archive.zip"
```

The loader finds the sequences at any nesting depth, so it doesn't matter whether the zip extracts to `DETRAC-Images/DETRAC-Images/MVI_xxxxx`.

### Quick test videos (10 MB, no account needed)

```bash
python scripts/download_data.py --dataset sample-videos   # -> data/videos/
python scripts/run_experiment.py --config configs/experiments/custom_video.yaml
```

### Your own videos

Copy `.mp4` files into `data/videos/` and run `custom_video.yaml`. Without annotations, the event-triggered output is scored against the frame-by-frame output, which measures exactly the accuracy cost of skipping.

---

## 4. Running experiments

```bash
# laptop-sized UA-DETRAC run (4 test sequences x 600 frames)
python scripts/run_experiment.py --config configs/experiments/ua_detrac_quick.yaml

# the full paper experiment (40 test sequences)
python scripts/run_experiment.py --config configs/experiments/ua_detrac_paper.yaml
```

Any option can be changed from the command line without editing files:

```bash
python scripts/run_experiment.py --config configs/experiments/ua_detrac_quick.yaml \
    --set detector.weights=yolov8s.pt detector.device=cuda:0 \
          "event_trigger.thresholds=[0.5, 1, 2, 3, 5, 8]" \
          "dataset.sequences=[MVI_39031, MVI_40701]" experiment.name=yolov8s_test
```

**What happens** (paper Sec. III-F, printed as it runs):

1. **Dataset preparation.** Loads the sequences and ground truth (UA-DETRAC test split).
2. **Baseline detection.** YOLO runs on every frame. D_t is also recorded (untimed) to calibrate τ.
3. **Event-triggered detection.** One full pass per threshold τ, with the same detector, resolution, confidence threshold and hardware.
4. **Metric collection.** Precision, recall, F1, accuracy, mAP@0.5, mAP@[.5:.95], FPS, latency, invocation rate, skip rate S, CPU/GPU/RAM, GFLOPs.
5. **Comparative analysis.** Deltas against the baseline, recommended τ, figures, tables, HTML report and side-by-side video.

**Thresholds.** `event_trigger.thresholds: auto` (the default) picks τ at the 10/25/40/50/60/75/90th percentiles of the measured D_t distribution. That fits any camera without guessing. A fixed list works too.

**Rough run times** with YOLOv8n (measured ~50 ms/frame on a 2-core CPU):

| Config | Frames | CPU | GPU |
|---|---|---|---|
| synthetic_smoke | 300 | ~1 min | ~1 min |
| ua_detrac_quick | 2,400 | 10–20 min | 2–5 min |
| ua_detrac_paper | ~56,000 | several hours | 30–60 min |

Add `--set evaluation.timing_mode=replay` for a fast sweep. Detections and per-call latency are reused from the baseline pass, so accuracy numbers are identical but timings are estimates. Use `measured` (the default) for final paper numbers.

---

## 5. Outputs and the report

Every run creates its own folder, so nothing is ever overwritten:

```
outputs/runs/20261120-104500_ua_detrac_quick/
├── report/report.html      ← OPEN THIS: self-contained report (figures embedded)
├── report/report.md        ← same as Markdown
├── tables/
│   ├── results.tex         ← IEEE-style LaTeX table, paste into Overleaf
│   ├── table1_proposed_row.tex ← replaces "To be evaluated" in the paper's Table I
│   ├── results.md / results_full.csv
├── figures/*.png / *.pdf   ← PDF = vector figures for LaTeX
├── videos/comparison_*.mp4 ← baseline vs event-triggered side by side
├── metrics/                ← summary.csv, per_class.csv, per_sequence.csv,
│                              activity_levels.csv, resources_timeseries.csv
├── per_frame/<method>.csv  ← D_t, trigger decision, latencies for every frame
├── predictions/<method>.npz← every box of every method (re-scoring without re-running)
├── config.yaml             ← exact configuration used (reproducibility)
├── run_info.json           ← thresholds, recommendation, detector, hardware, timings
└── logs/run.log
```

**Figures produced:** accuracy & invocation rate vs τ, accuracy–efficiency trade-off (skip rate vs mAP), FPS & latency breakdown, precision/recall/F1/accuracy, D_t histogram with thresholds, per-sequence D_t timelines with detector on/off strips (a frame-level version of the paper's Fig. 2), detector usage by activity level (low/moderate/high, Sec. IV-A), per-class AP, CPU/GPU/GFLOPs, and per-sequence behaviour.

**Recommended τ** is the threshold with the highest skip rate whose mAP@0.5 drop stays within `evaluation.accuracy_drop_tolerance` (default 1.0 percentage point).

Rebuild a report without re-running detection (for example, after changing plot styles):

```bash
python scripts/generate_report.py                         # latest run
python scripts/generate_report.py --run outputs/runs/<folder> --video --sequence MVI_39031 --threshold 2.5
```

---

## 6. Making changes later

| You want to… | Change |
|---|---|
| try another detector (YOLOv8s/m, YOLO11, RT-DETR) | `detector.weights` (any Ultralytics model name or `.pt` path) |
| use a fine-tuned detector | `scripts/prepare_data.py --export-yolo` → `scripts/train_detector.py` → `detector.weights=models/finetuned/.../best.pt` |
| change thresholds | `event_trigger.thresholds` (list or `auto`) |
| evaluate other classes (person, motorcycle…) | `classes.eval_classes`, `classes.detector_to_eval`, `classes.dataset_to_eval` |
| add a new dataset | write `load_x(cfg)` in `src/etod/data/`, register it in `src/etod/data/__init__.py`, add a source to `configs/datasets.yaml` |
| any frame-folder dataset with YOLO labels | `dataset.type: image_folder` (no code needed) |
| add a new triggering mechanism (optical flow, adaptive τ…) | subclass `EventTrigger` in `src/etod/event/`, register it in `TRIGGERS` |
| add a new detector family | subclass `BaseDetector` in `src/etod/detectors/`, register it in `DETECTORS` |
| add a metric | `src/etod/evaluation/`; it appears in `summary.csv` automatically |
| add a figure | a function in `src/etod/reporting/plots.py`, called from `report.py` |
| add a config option | add it with a default to `configs/default.yaml` first (unknown keys are rejected to catch typos) |

**Paper-faithful defaults vs extensions.** The defaults follow the paper exactly: D_t against the previous frame, strict `>`, no forced re-detection, raw grey-level differences. Optional extensions are present but **off**: `event_trigger.reference: last_detected`, `max_skip`, `blur_kernel`, `mask_ignore_regions`. They are useful for a follow-up study (the paper's "future work" mentions adaptive thresholds and tracking).

---

## 7. Troubleshooting

* **`UA-DETRAC not found`**: run the download script or set `dataset.root` to your extracted folder.
* **Kaggle 401/403**: `kaggle.json` is missing or in the wrong place, or you haven't accepted the dataset's terms on its web page.
* **Slow on CPU**: use `ua_detrac_quick.yaml`, `--set dataset.max_frames_per_sequence=300`, or `evaluation.timing_mode=replay`.
* **CUDA not used**: `python -c "import torch; print(torch.cuda.is_available())"` must print `True`. Otherwise reinstall PyTorch from pytorch.org with CUDA.
* **Video won't play in your browser**: the `.mp4` uses the `mp4v` codec. Open it in VLC or the Windows Media Player app.
