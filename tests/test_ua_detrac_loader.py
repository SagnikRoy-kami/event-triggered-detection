"""Builds a tiny fake UA-DETRAC tree (with the Kaggle double-nesting) and loads it."""
import cv2
import numpy as np

from etod.config import load_config
from etod.data import load_sequences

XML = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="{name}">
  <sequence_attribute camera_state="stable" sence_weather="sunny"/>
  <ignored_region><box left="0" top="0" width="20" height="20"/></ignored_region>
  <frame density="2" num="1"><target_list>
    <target id="1"><box left="30" top="30" width="40" height="20"/><attribute orientation="0" speed="1" trajectory_length="1" truncation_ratio="0" vehicle_type="car"/></target>
    <target id="2"><box left="80" top="40" width="30" height="30"/><attribute orientation="0" speed="1" trajectory_length="1" truncation_ratio="0" vehicle_type="van"/></target>
  </target_list></frame>
  <frame density="1" num="2"><target_list>
    <target id="3"><box left="10" top="50" width="60" height="25"/><attribute orientation="0" speed="1" trajectory_length="1" truncation_ratio="0" vehicle_type="bus"/></target>
  </target_list></frame>
</sequence>"""


def make_tree(root):
    for name, split in [("MVI_20011", "Train"), ("MVI_39031", "Test")]:
        img_dir = root / "DETRAC-Images" / "DETRAC-Images" / name
        img_dir.mkdir(parents=True)
        for i in range(1, 4):
            cv2.imwrite(str(img_dir / f"img{i:05d}.jpg"), np.full((90, 160, 3), 50 * i, np.uint8))
        ann = root / f"DETRAC-{split}-Annotations-XML"
        ann.mkdir(exist_ok=True)
        (ann / f"{name}.xml").write_text(XML.format(name=name))


def test_load(tmp_path):
    make_tree(tmp_path)
    cfg = load_config(None, [f"dataset.root={tmp_path.as_posix()}", "dataset.index_file=", "dataset.split=test"])
    seqs = load_sequences(cfg)
    assert [s.name for s in seqs] == ["MVI_39031"]
    s = seqs[0]
    assert s.num_frames == 3 and s.metadata["camera_state"] == "stable"
    g1 = s.gt_for(1)
    assert len(g1) == 2 and list(g1.labels) == [0, 0]          # car + van -> car
    assert g1.xyxy[0].tolist() == [30, 30, 70, 50]
    assert list(s.gt_for(2).labels) == [1]                       # bus
    assert len(s.gt_for(3)) == 0                                 # frame without annotation
    assert s.ignore_regions.tolist() == [[0, 0, 20, 20]]
    frames = list(s.frames())
    assert [f.frame_id for f in frames] == [1, 2, 3]
