"""Project-wide path helpers. Every relative path in the configs is resolved
against the project root, so scripts work no matter where they are launched from."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VIDEOS_DIR = DATA_DIR / "videos"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"


def resolve(path: str | Path | None) -> Path | None:
    """Resolve a path relative to the project root (absolute paths pass through)."""
    if path is None or path == "":
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_run(runs_dir: Path = RUNS_DIR) -> Path | None:
    """Return the most recent run folder (via LATEST.txt, else newest folder)."""
    pointer = runs_dir / "LATEST.txt"
    if pointer.exists():
        candidate = resolve(pointer.read_text(encoding="utf-8").strip())
        if candidate and candidate.exists():
            return candidate
    runs = sorted([d for d in runs_dir.glob("*") if d.is_dir()], key=lambda d: d.stat().st_mtime)
    return runs[-1] if runs else None
