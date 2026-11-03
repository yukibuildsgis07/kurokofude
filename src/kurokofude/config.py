"""YAML 設定の読み込み。指定した設定は既定値（configs/default.yaml）に上書きする。"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_config(path: Path | None = None) -> dict:
    config = _read_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        config = _merge(config, _read_yaml(Path(path)))
    return config
