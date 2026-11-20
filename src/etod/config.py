"""Configuration: default.yaml < experiment yaml < command-line overrides.

The merged configuration is a plain nested dict wrapped in :class:`Config`,
which allows both ``cfg["detector"]["conf"]`` and ``cfg.detector.conf``.
Keys that do not exist in ``configs/default.yaml`` are rejected so a typo
never silently falls back to a default.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

from etod.utils.paths import CONFIG_DIR, resolve

DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"

# Sub-trees whose keys are user-defined (class names) and therefore not validated.
_FREEFORM_KEYS = {("classes", "dataset_to_eval"), ("classes", "detector_to_eval")}


class Config(dict):
    """dict with attribute access (read-only convenience)."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _validate(override: dict, base: dict, trail: tuple[str, ...] = ()) -> None:
    for key, value in override.items():
        here = trail + (key,)
        if trail in _FREEFORM_KEYS:
            continue
        if key not in base:
            close = ", ".join(sorted(base.keys()))
            raise KeyError(f"Unknown config key '{'.'.join(here)}'. Valid keys here: {close}")
        if isinstance(value, dict) and isinstance(base[key], dict) and here not in _FREEFORM_KEYS:
            _validate(value, base[key], here)


def _deep_merge(base: dict, override: dict, trail: tuple[str, ...] = ()) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        here = trail + (key,)
        if here in _FREEFORM_KEYS:
            out[key] = copy.deepcopy(value)  # class maps are replaced, not merged
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value, here)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_overrides(pairs: Iterable[str]) -> dict:
    """Turn ``["detector.conf=0.3", "dataset.sequences=[MVI_39031]"]`` into a nested dict."""
    result: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must look like key.subkey=value")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw) if raw.strip() != "" else ""
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def load_config(path: str | Path | None = None, overrides: Iterable[str] | dict | None = None) -> Config:
    """Load default.yaml, then the experiment file, then overrides."""
    base = _read_yaml(DEFAULT_CONFIG)
    cfg = base
    if path:
        exp_path = resolve(path)
        if not exp_path.exists():
            raise FileNotFoundError(f"Config not found: {exp_path}")
        exp = _read_yaml(exp_path)
        _validate(exp, base)
        cfg = _deep_merge(cfg, exp)
    if overrides:
        ov = overrides if isinstance(overrides, dict) else parse_overrides(overrides)
        _validate(ov, base)
        cfg = _deep_merge(cfg, ov)
    return Config(cfg)


def save_config(cfg: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False, allow_unicode=True)
