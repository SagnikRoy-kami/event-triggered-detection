"""Optional: fine-tune YOLO on the UA-DETRAC train split (paper III-B mentions
train / validation / test subsets). The COCO-pretrained detector already works;
fine-tuning raises absolute mAP for BOTH methods equally, so it does not change
the baseline vs event-triggered comparison itself.

    python scripts/prepare_data.py --export-yolo
    python scripts/train_detector.py --model yolov8n.pt --epochs 30 --device 0

Then evaluate with the fine-tuned weights (its classes are already the eval classes):
    python scripts/run_experiment.py --config configs/experiments/ua_detrac_quick.yaml \
        --set detector.weights=models/finetuned/<name>/best.pt \
              "classes.detector_to_eval={car: car, bus: bus, truck: truck}"
"""
import argparse
import shutil

import _bootstrap  # noqa: F401

from etod.utils.environment import resolve_device
from etod.utils.paths import ensure_dir, resolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/ua_detrac_yolo/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    from etod.detectors.yolo import resolve_weights

    data = resolve(args.data)
    if not data.exists():
        raise SystemExit(f"{data} not found - run: python scripts/prepare_data.py --export-yolo")
    name = args.name or f"{args.model.replace('.pt', '')}_uadetrac_e{args.epochs}"
    device = resolve_device(args.device)
    model = YOLO(str(resolve_weights(args.model, "models/weights")))
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device.replace("cuda:", ""), project=str(resolve("outputs/training")), name=name, exist_ok=True)
    best = resolve(f"outputs/training/{name}/weights/best.pt")
    dst = ensure_dir(resolve(f"models/finetuned/{name}")) / "best.pt"
    shutil.copy2(best, dst)
    print(f"\nFine-tuned weights: {dst}")


if __name__ == "__main__":
    main()
