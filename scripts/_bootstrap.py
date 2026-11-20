"""Lets every script import the `etod` package without installing it.
(`pip install -e .` also works; then this is a no-op.)"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
