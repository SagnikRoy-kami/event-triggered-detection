# data/

| Folder | Contents | Created by |
|---|---|---|
| `raw/ua-detrac/` | extracted Kaggle UA-DETRAC (frames + XML annotations), never modified | `scripts/download_data.py --dataset ua-detrac` |
| `processed/ua_detrac/index.json` | list of sequences, split, frame count, camera state | `download_data.py` / `prepare_data.py` |
| `processed/ua_detrac_yolo/` | train-split export in YOLO format for fine-tuning | `prepare_data.py --export-yolo` |
| `interim/` | free scratch space for your own preprocessing | you |
| `videos/` | `.mp4` files used by `configs/experiments/custom_video.yaml` | `download_data.py --dataset sample-videos` or you |

Expected UA-DETRAC layout after extraction (any extra nesting level is fine):

```
raw/ua-detrac/
├── DETRAC-Images/MVI_20011/img00001.jpg ...
├── DETRAC-Train-Annotations-XML/MVI_20011.xml ...
└── DETRAC-Test-Annotations-XML/MVI_39031.xml ...
```

The contents of these folders are git-ignored (large files).
