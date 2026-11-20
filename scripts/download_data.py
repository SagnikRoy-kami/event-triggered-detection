"""Download datasets into data/ (sources are listed in configs/datasets.yaml).

    python scripts/download_data.py --list
    python scripts/download_data.py --dataset sample-videos        # ~10 MB, quick test
    python scripts/download_data.py --dataset ua-detrac            # ~11.5 GB from Kaggle

Kaggle authentication (one-time):
    1. kaggle.com -> Settings -> API -> "Create New Token" (downloads kaggle.json)
    2. put it at  ~/.kaggle/kaggle.json   (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
       or set the environment variables KAGGLE_USERNAME and KAGGLE_KEY
    3. open the dataset page once in the browser and accept its terms if asked

Manual alternative: download the zip from the Kaggle page yourself and run
    python scripts/download_data.py --dataset ua-detrac --from-zip path/to/archive.zip
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from etod.utils.paths import CONFIG_DIR, resolve


def load_registry() -> dict:
    return yaml.safe_load((CONFIG_DIR / "datasets.yaml").read_text(encoding="utf-8"))


def extract(zip_path: Path, dest: Path) -> None:
    print(f"Extracting {zip_path.name} -> {dest} (this can take a while)...")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for i, m in enumerate(members, 1):
            zf.extract(m, dest)
            if i % 5000 == 0 or i == len(members):
                print(f"  {i:,}/{len(members):,} files", end="\r")
    print()


def download_kaggle(handle: str, dest: Path, keep_zip: bool) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The 'kaggle' package is missing:  pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Kaggle authentication failed ({exc}).\nSee the instructions at the top of this script.")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading kaggle dataset '{handle}' to {dest} ...")
    api.dataset_download_files(handle, path=str(dest), unzip=False, quiet=False)
    zips = sorted(dest.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    if not zips:
        sys.exit("Download finished but no zip was found")
    extract(zips[0], dest)
    if not keep_zip:
        zips[0].unlink()


def download_http(urls: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for url in urls:
        target = dest / url.rsplit("/", 1)[-1]
        if target.exists() and target.stat().st_size > 0:
            print(f"  exists: {target.name}")
            continue
        print(f"  {url}")
        with urllib.request.urlopen(url) as r, open(target, "wb") as fh:
            shutil.copyfileobj(r, fh)


def main() -> None:
    reg = load_registry()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(reg), help="dataset key from configs/datasets.yaml")
    ap.add_argument("--list", action="store_true", help="list available datasets")
    ap.add_argument("--dest", default=None, help="override destination folder")
    ap.add_argument("--from-zip", default=None, help="extract an already-downloaded zip instead of downloading")
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip after extracting")
    ap.add_argument("--no-index", action="store_true", help="skip building the dataset index afterwards")
    args = ap.parse_args()

    if args.list or not args.dataset:
        for k, v in reg.items():
            print(f"{k:15s} {v.get('approx_size', ''):>30s}   ->  {v['dest']}\n{'':15s} {v['description'].strip()}\n")
        return

    spec = reg[args.dataset]
    dest = resolve(args.dest or spec["dest"])
    if args.from_zip:
        extract(resolve(args.from_zip), dest)
    elif spec["source"] == "kaggle":
        download_kaggle(spec["kaggle_handle"], dest, args.keep_zip)
    elif spec["source"] == "http":
        download_http(spec["files"], dest)

    if spec.get("config_type") == "ua_detrac" and not args.no_index:
        from etod.data.ua_detrac import build_index
        from etod.pipeline.storage import save_json

        idx = build_index(dest)
        out = resolve("data/processed/ua_detrac/index.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_json(out, idx)
        print(f"Indexed {idx['num_sequences']} sequences -> {out}")
    print(f"Done: {dest}")


if __name__ == "__main__":
    main()
