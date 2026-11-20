# models/

* `weights/`: pretrained detector weights. Downloaded automatically the first time a name such as `yolov8n.pt` is used in `detector.weights`.
* `finetuned/<name>/best.pt`: written by `scripts/train_detector.py`. Use with `--set detector.weights=models/finetuned/<name>/best.pt`.
