"""Phase 1 helpers: index, inspect, and (optionally) split UA-DETRAC for training.

    # index + summary (fast; also done automatically by download_data.py)
    python scripts/prepare_data.py

    # save a few annotated frames to check the ground truth visually
    python scripts/prepare_data.py --preview 6

    # export the TRAIN split as a YOLO dataset (train/val by sequence) for fine-tuning
    python scripts/prepare_data.py --export-yolo --stride 10 --val-fraction 0.15

The TEST split is never exported: it is reserved for the baseline vs
event-triggered comparison (paper Sec. III-B: train / validation / test).
"""
import argparse
import random
from collections import Counter

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from etod.config import load_config
from etod.data.ua_detrac import IMG_RE, build_index, parse_annotations
from etod.data.base import label_map
from etod.pipeline.storage import save_json
from etod.utils.paths import OUTPUTS_DIR, ensure_dir, resolve


def summarise(idx: dict) -> None:
    seqs = idx["sequences"]
    by_split = Counter(v["split"] for v in seqs.values())
    frames = Counter()
    for v in seqs.values():
        frames[v["split"]] += v["num_frames"]
    print(f"\nUA-DETRAC at {idx['root']}")
    for s in sorted(by_split):
        print(f"  {s:6s}: {by_split[s]:3d} sequences, {frames[s]:7,d} frames")
    if idx.get("missing_images"):
        print(f"  ! {len(idx['missing_images'])} annotated sequences without frames: {idx['missing_images'][:5]}...")
    cams = Counter(v.get("camera_state") for v in seqs.values())
    print(f"  camera_state: {dict(cams)}")


def preview(cfg, idx: dict, n: int) -> None:
    root = resolve(cfg["dataset"]["root"])
    out = ensure_dir(OUTPUTS_DIR / "dataset_preview")
    names = cfg["classes"]["eval_classes"]
    name_to_idx = label_map(cfg["classes"]["dataset_to_eval"], names)
    test = [s for s, v in idx["sequences"].items() if v["split"] == cfg["dataset"]["split"]] or list(idx["sequences"])
    random.seed(0)
    for seq in random.sample(test, min(n, len(test))):
        info = idx["sequences"][seq]
        gt, ignore = parse_annotations(root / info["xml"], name_to_idx)
        fid = sorted(gt)[len(gt) // 2]
        img = cv2.imread(str(root / info["image_dir"] / f"img{fid:05d}.jpg"))
        for x1, y1, x2, y2 in ignore.astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), -1)
        for (x1, y1, x2, y2), lab in zip(gt[fid].xyxy.astype(int), gt[fid].labels):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, names[lab], (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        cv2.imwrite(str(out / f"{seq}_frame{fid}.jpg"), img)
    print(f"Previews (grey = ignored regions) -> {out}")


def export_yolo(cfg, idx: dict, stride: int, val_fraction: float, out_name: str) -> None:
    root = resolve(cfg["dataset"]["root"])
    out = ensure_dir(resolve(f"data/processed/{out_name}"))
    names = cfg["classes"]["eval_classes"]
    name_to_idx = label_map(cfg["classes"]["dataset_to_eval"], names)
    train = sorted(s for s, v in idx["sequences"].items() if v["split"] == "train")
    if not train:
        raise SystemExit("No train-split sequences found")
    random.seed(int(cfg["experiment"]["seed"]))
    val = set(random.sample(train, max(1, round(len(train) * val_fraction))))
    counts = Counter()
    for k, seq in enumerate(train, 1):
        split = "val" if seq in val else "train"
        info = idx["sequences"][seq]
        gt, ignore = parse_annotations(root / info["xml"], name_to_idx)
        idir, ldir = ensure_dir(out / "images" / split), ensure_dir(out / "labels" / split)
        files = sorted((f for f in (root / info["image_dir"]).iterdir() if IMG_RE.search(f.name)),
                       key=lambda p: int(IMG_RE.search(p.name).group(1)))[::stride]
        for f in files:
            fid = int(IMG_RE.search(f.name).group(1))
            img = cv2.imread(str(f))
            h, w = img.shape[:2]
            for x1, y1, x2, y2 in ignore.astype(int):   # ignored regions are not annotated -> mask them
                img[max(0, y1):y2, max(0, x1):x2] = 0
            stem = f"{seq}_img{fid:05d}"
            cv2.imwrite(str(idir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            b = gt.get(fid)
            lines = []
            if b is not None:
                for (x1, y1, x2, y2), lab in zip(np.clip(b.xyxy, 0, [w, h, w, h]), b.labels):
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    lines.append(f"{lab} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (ldir / f"{stem}.txt").write_text("\n".join(lines))
            counts[split] += 1
        print(f"  [{k}/{len(train)}] {seq} -> {split}", end="\r")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8")
    print(f"\nYOLO dataset: {counts['train']:,} train / {counts['val']:,} val images -> {data_yaml}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/ua_detrac_quick.yaml")
    ap.add_argument("--preview", type=int, default=0, help="save N annotated sample frames")
    ap.add_argument("--export-yolo", action="store_true", help="export train split in YOLO format for fine-tuning")
    ap.add_argument("--stride", type=int, default=10, help="keep every k-th frame when exporting (consecutive frames are near-duplicates)")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--out-name", default="ua_detrac_yolo")
    ap.add_argument("--root", default=None, help="UA-DETRAC folder if not data/raw/ua-detrac")
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.root={args.root}"] if args.root else None)
    root = resolve(cfg["dataset"]["root"])
    if not root.exists():
        raise SystemExit(f"{root} not found - run: python scripts/download_data.py --dataset ua-detrac")
    idx = build_index(root)
    out = resolve(cfg["dataset"]["index_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, idx)
    summarise(idx)
    print(f"Index -> {out}")
    if args.preview:
        preview(cfg, idx, args.preview)
    if args.export_yolo:
        export_yolo(cfg, idx, args.stride, args.val_fraction, args.out_name)


if __name__ == "__main__":
    main()
