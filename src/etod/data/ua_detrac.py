"""UA-DETRAC loader (paper Sec. IV-B: "A publicly accessible traffic dataset,
such as BDD100K or UA-DETRAC, is chosen for evaluation").

Expected raw layout (any nesting depth works, folders are discovered):

    data/raw/ua-detrac/
        DETRAC-Images/[DETRAC-Images/]MVI_20011/img00001.jpg ...
        DETRAC-Train-Annotations-XML/MVI_20011.xml ...
        DETRAC-Test-Annotations-XML/MVI_39031.xml ...

The original "Insight-MVT_Annotation_Train/Test" image folders are also
recognised. The split (train/test) is decided by which annotation folder a
sequence's XML lives in. Each XML contains per-frame vehicle boxes with
``vehicle_type`` in {car, van, bus, others} and a list of *ignored regions*
(areas that were not annotated). Following the official protocol, detections
inside ignored regions are discarded before scoring.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from etod.data.base import Boxes, Frame, Sequence, label_map, select_frames
from etod.utils.logging_utils import get_logger
from etod.utils.paths import resolve

SEQ_RE = re.compile(r"(MVI_\d+)")
IMG_RE = re.compile(r"img(\d+)\.jpg$", re.IGNORECASE)
UA_DETRAC_FPS = 25.0


# --------------------------------------------------------------------------- discovery
def discover(root: Path) -> dict:
    """Walk ``root`` once and return {'images': {seq: dir}, 'xml': {seq: (path, split)}}."""
    images: dict[str, str] = {}
    xmls: dict[str, tuple[str, str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        m = SEQ_RE.fullmatch(base)
        if m and any(IMG_RE.search(f) for f in filenames[:50]):
            # keep the first copy found (Kaggle zips sometimes contain duplicates)
            images.setdefault(m.group(1), dirpath)
            dirnames[:] = []           # do not descend into a frame folder
            continue
        for f in filenames:
            if f.lower().endswith(".xml"):
                sm = SEQ_RE.search(f)
                if not sm:
                    continue
                parent = dirpath.lower()
                split = "train" if "train" in parent else "test" if "test" in parent else "unknown"
                xmls.setdefault(sm.group(1), (os.path.join(dirpath, f), split))
    return {"images": images, "xml": xmls}


def build_index(root: Path) -> dict:
    """Index of every sequence that has both frames and annotations."""
    found = discover(root)
    sequences = {}
    for seq, (xml_path, split) in sorted(found["xml"].items()):
        img_dir = found["images"].get(seq)
        if img_dir is None:
            continue
        n = sum(1 for f in os.listdir(img_dir) if IMG_RE.search(f))
        attrs = _sequence_attributes(Path(xml_path))
        sequences[seq] = {
            "split": split,
            "image_dir": os.path.relpath(img_dir, root),
            "xml": os.path.relpath(xml_path, root),
            "num_frames": n,
            **attrs,
        }
    return {
        "root": str(root),
        "num_sequences": len(sequences),
        "missing_images": sorted(set(found["xml"]) - set(found["images"])),
        "sequences": sequences,
    }


def _sequence_attributes(xml_path: Path) -> dict:
    try:
        for _, elem in ET.iterparse(xml_path, events=("start",)):
            if elem.tag == "sequence_attribute":
                return {"camera_state": elem.get("camera_state"), "weather": elem.get("sence_weather")}
    except ET.ParseError:
        pass
    return {}


# --------------------------------------------------------------------------- parsing
def parse_annotations(xml_path: Path, name_to_idx: dict[str, int]) -> tuple[dict[int, Boxes], np.ndarray]:
    """Return (frame_id -> Boxes, ignored regions as (M,4) xyxy)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ignore = []
    ign = root.find("ignored_region")
    if ign is not None:
        for b in ign.findall("box"):
            x, y = float(b.get("left")), float(b.get("top"))
            w, h = float(b.get("width")), float(b.get("height"))
            ignore.append([x, y, x + w, y + h])

    gt: dict[int, Boxes] = {}
    for frame in root.findall("frame"):
        fid = int(frame.get("num"))
        xyxy, labels = [], []
        tl = frame.find("target_list")
        if tl is not None:
            for t in tl.findall("target"):
                b = t.find("box")
                a = t.find("attribute")
                vtype = (a.get("vehicle_type") if a is not None else "car") or "car"
                cls = name_to_idx.get(vtype.lower())
                if cls is None:
                    continue
                x, y = float(b.get("left")), float(b.get("top"))
                w, h = float(b.get("width")), float(b.get("height"))
                xyxy.append([x, y, x + w, y + h])
                labels.append(cls)
        gt[fid] = Boxes.from_lists(xyxy, labels) if labels else Boxes.empty()
    return gt, np.asarray(ignore, np.float32).reshape(-1, 4)


# --------------------------------------------------------------------------- loader
def load_ua_detrac(cfg) -> list[Sequence]:
    log = get_logger()
    ds = cfg["dataset"]
    root = resolve(ds["root"])
    if root is None or not root.exists():
        raise FileNotFoundError(
            f"UA-DETRAC not found at '{root}'. Download it first:\n"
            "    python scripts/download_data.py --dataset ua-detrac\n"
            "or place the extracted Kaggle folder there (see data/README.md)."
        )

    index_path = resolve(ds.get("index_file"))
    index = None
    if index_path and index_path.exists():
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
        if Path(index.get("root", "")).resolve() != root.resolve():
            log.info("Index file was built for a different root; re-scanning.")
            index = None
    if index is None:
        log.info(f"Scanning {root} for UA-DETRAC sequences (run scripts/prepare_data.py to cache this)...")
        index = build_index(root)

    seqs = index["sequences"]
    wanted = list(ds.get("sequences") or [])
    if wanted:
        missing = [s for s in wanted if s not in seqs]
        if missing:
            raise ValueError(f"Requested sequences not found: {missing}")
        names = wanted
    else:
        split = ds.get("split", "test")
        names = [s for s, info in seqs.items() if split in ("all", info["split"])]
    if ds.get("max_sequences"):
        names = names[: int(ds["max_sequences"])]
    if not names:
        raise RuntimeError(f"No UA-DETRAC sequences matched split='{ds.get('split')}' in {root}")

    name_to_idx = label_map(cfg["classes"]["dataset_to_eval"], cfg["classes"]["eval_classes"])
    out = []
    for name in names:
        info = seqs[name]
        img_dir = root / info["image_dir"]
        files = sorted((f for f in os.listdir(img_dir) if IMG_RE.search(f)), key=lambda f: int(IMG_RE.search(f).group(1)))
        files = select_frames(files, ds.get("max_frames_per_sequence"), ds.get("frame_stride", 1))
        gt, ignore = parse_annotations(root / info["xml"], name_to_idx)
        frame_ids = [int(IMG_RE.search(f).group(1)) for f in files]
        gt = {fid: gt.get(fid, Boxes.empty()) for fid in frame_ids}
        out.append(
            Sequence(
                name=name,
                num_frames=len(files),
                fps=UA_DETRAC_FPS / max(1, int(ds.get("frame_stride", 1) or 1)),
                frame_factory=_image_iter(img_dir, files, frame_ids),
                ground_truth=gt,
                ignore_regions=ignore if ds.get("apply_ignore_regions", True) else np.zeros((0, 4), np.float32),
                metadata={"split": info["split"], "camera_state": info.get("camera_state"), "weather": info.get("weather")},
            )
        )
    log.info(f"UA-DETRAC: {len(out)} sequences, {sum(s.num_frames for s in out)} frames")
    return out


def _image_iter(img_dir: Path, files: list[str], frame_ids: list[int]):
    def factory():
        for i, (f, fid) in enumerate(zip(files, frame_ids)):
            img = cv2.imread(str(img_dir / f), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Could not read {img_dir / f}")
            yield Frame(index=i, frame_id=fid, image=img)

    return factory
