# Methodology: how the paper is implemented

This document records every decision made when turning the paper into code, so the
experiment can be described accurately in the paper and defended in a viva.

## 1. Event detection (Sec. III-C, IV-D)

**Temporal difference, Eq. (1)/(3).** For the current frame I_t and the previous frame I_{t−1}:

    D_t = (1/N) · Σ_{i=1..N} |I_t(i) − I_{t−1}(i)|

* Implemented in `TemporalDifferenceTrigger.difference_score()` as `cv2.absdiff(...).mean()`.
* Pixels are 8-bit grey levels (0–255), so D_t is "mean absolute grey-level change per pixel".
* `resize_width` (default 320 px) shrinks the frame **before** differencing. Because D_t is a *mean*, its scale barely changes; only the cost of computing it drops (about 0.3 ms/frame). Set it to `null` for full resolution.
* No smoothing by default (`blur_kernel: 0`), as in the paper.

**Event, Eq. (4).** `E_t = 1 if D_t > τ else 0`. The comparison is strict (`>`), exactly as written.

**First frame.** There is no I_{t−1}, so the detector always runs on the first frame of each sequence (`first_frame_always_detect: true`). The trigger state is reset between sequences.

**Skipped frames (Fig. 1, "Skip Detection / Reuse Previous Result").** When E_t = 0, the most recent detection result is output again for frame t.

## 2. Experimental comparison (Sec. III-D)

* **One** detector instance serves both methods: same weights, input resolution (`imgsz` 640), confidence threshold (0.25), NMS IoU (0.45) and device.
* Baseline = the same pipeline with a trigger that always fires (`AlwaysTrigger`). The only difference between the two methods is therefore the trigger.
* Several thresholds are evaluated (Sec. IV-E). With `thresholds: auto` they are the 10/25/40/50/60/75/90th percentiles of the D_t distribution measured during the baseline pass, so they cover "skip almost nothing" to "skip almost everything" on any camera.
* Every run is preceded by untimed warm-up inferences so model loading and CUDA initialisation don't count against either method.

## 3. Dataset (Sec. III-B, IV-B)

* **UA-DETRAC**, from Kaggle `bratjay/ua-detrac-orig` (original layout: frame folders + XML). 100 sequences from fixed roadside cameras at 25 fps and 960×540. The official split is 60 train / 40 test.
* The **test split** is used for the comparison, as the paper states. The train split is only used if you choose to fine-tune (`prepare_data.py --export-yolo` splits it into train/validation **by sequence**, so no near-identical frames leak between them).
* **Frame order is preserved**, which D_t requires. Frames are read in their original numbering.
* **Classes.** UA-DETRAC labels {car, van, bus, others} are mapped to the evaluation classes {car, bus, truck} as car→car, van→car, bus→bus, others→truck. The COCO-pretrained detector's {car, bus, truck} map onto the same classes. All mappings are in `configs/default.yaml → classes` and can be changed.
  *Note:* UA-DETRAC annotates vehicles only, so pedestrians and motorcycles (mentioned in the paper as "depending on the chosen dataset") are evaluated only on datasets that annotate them (e.g. via `custom_video.yaml` in baseline-reference mode).
* **Ignored regions.** UA-DETRAC marks image areas that were not annotated. Following the official protocol, any detection lying ≥ 50 % inside such a region is discarded before scoring, identically for every method.

## 4. Metrics (Sec. III-E, IV-E)

All metrics are computed over **every input frame**. A skipped frame is scored with the result it actually outputs, the reused previous detections. Accuracy loss caused by skipping is therefore measured directly, not estimated.

| Metric | Definition |
|---|---|
| mAP@0.5, mAP@[.5:.95] | COCO protocol: per class, detections sorted by confidence, greedy matching to unmatched GT of the same class, 101-point interpolated AP, mean over classes with GT (and IoU 0.50:0.05:0.95) |
| Precision, Recall | at IoU 0.5 over all detections above the confidence threshold: TP/(TP+FP), TP/(TP+FN) |
| F1 | 2PR/(P+R) |
| Detection accuracy | TP/(TP+FP+FN) |
| Invocation rate | N_d / N_f |
| Frame-skip rate S | (1 − N_d/N_f) × 100, Eq. (2)/(5) |
| FPS | N_f / Σ processing time, where processing = event module + detector (or reuse). Frame decoding is excluded because it is identical for all methods; `fps_incl_io` includes it |
| Latency | per-frame processing time: mean, median, p95, max |
| Event overhead | mean cost of computing D_t |
| CPU / RAM / GPU | sampled every 0.25 s in a background thread (`psutil`; GPU via `torch.cuda` and `pynvml` if installed) |
| Detector compute | GFLOPs per inference × N_d (total) and ÷ N_f (per frame) |

**Activity levels (Sec. IV-A).** Frames are grouped into low / moderate / high activity by the terciles of D_t over the dataset. The report shows how often the detector ran in each group. An effective trigger runs rarely in quiet periods and often in busy ones.

**Recommended τ.** The highest skip rate whose mAP@0.5 is within `accuracy_drop_tolerance` percentage points (default 1.0) of the baseline. If no threshold qualifies, the one with the smallest drop is chosen and the report says so.

## 5. Timing modes

* `measured` (default, use for the paper): every threshold is executed for real and timed.
* `replay`: the event module runs for real, but on triggered frames the detections and measured per-call latency are taken from the baseline pass. Accuracy is bit-identical to `measured` (the detector is deterministic); timings are estimates. Use it for fast exploration of many thresholds.

## 6. Without ground truth

For videos without annotations (`evaluation.reference: baseline`), the frame-by-frame detector's output is treated as the reference. The baseline then scores 100 % by definition. Event-triggered scores measure how closely skipping reproduces the frame-by-frame output, which is exactly the accuracy cost of the method.

## 7. Extensions (off by default)

Not part of the evaluated method in the paper; available for follow-up work:

* `reference: last_detected`: compare with the last frame the detector ran on, so slow drifts accumulate.
* `max_skip: N`: force a detection after N consecutive skipped frames.
* `blur_kernel`: Gaussian smoothing to suppress sensor noise.
* `mask_ignore_regions`: compute D_t only outside UA-DETRAC ignored regions.

## 8. Verification

`pytest` runs 32 tests, including:
* D_t equals the mean absolute difference, the comparison is strict, and the first-frame and reset behaviour is correct,
* skipped frames output exactly the previous result object,
* AP / precision / recall on hand-computed cases (perfect, empty, wrong class, duplicates, ignore regions, stale boxes),
* the UA-DETRAC XML parser on a fake dataset with the Kaggle double-nested layout,
* a complete five-phase run that produces every report file, with skip rate increasing monotonically with τ,
* `replay` and `measured` give identical accuracy.
