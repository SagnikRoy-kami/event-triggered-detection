"""Event-Triggered Object Detection (ETOD).

Implementation of the framework described in
"Event-Driven Object Detection: the model decides *when* to perform expensive
detection rather than detecting every frame".

Package layout
--------------
etod.config       configuration loading / merging / validation
etod.data         dataset loaders (UA-DETRAC, video files, image folders, synthetic)
etod.event        event triggers (temporal difference, Eq. 1 and Eq. 4)
etod.detectors    object detectors (Ultralytics YOLO, tiny contour detector)
etod.pipeline     baseline vs event-triggered execution + the 5-phase experiment
etod.evaluation   accuracy (P/R/mAP), efficiency (FPS, latency, skip rate), resources
etod.reporting    figures, tables, HTML/Markdown/LaTeX reports, comparison videos
"""

__version__ = "1.0.0"
