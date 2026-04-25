from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start.parent


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    raw = dict(raw)
    inherit = raw.pop("inherit", None)
    if not inherit:
        return raw
    root = find_repo_root(path)
    base_path = (root / inherit).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Inherit YAML not found: {base_path} (from {path})")
    base_cfg = load_config(base_path)
    return deep_merge(base_cfg, raw)
