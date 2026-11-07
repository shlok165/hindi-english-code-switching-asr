"""YAML configuration with includes, overrides and reproducibility metadata.

A config may list `includes`, which are merged first, in order. The file's own
keys then win, and `--set a.b=value` overrides win over everything.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src import ROOT


def deep_merge(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve(path: str | Path) -> Path:
    """Resolve a config path against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def data_path(path: str | Path) -> Path:
    """Resolve a data path, honouring MUCS_DATA_ROOT (e.g. Drive on Colab)."""
    path = Path(path)
    root = os.environ.get("MUCS_DATA_ROOT")
    if path.is_absolute():
        return path
    if root and path.parts[0] == "data":
        return Path(root).joinpath(*path.parts[1:])
    return ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    merged: dict[str, Any] = {}
    for include in config.pop("includes", []):
        merged = deep_merge(merged, load_yaml(include))
    return deep_merge(merged, config)


def parse_value(raw: str) -> Any:
    """YAML value, except that 1e-4 is a number, which YAML 1.1 reads as text."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"override must look like a.b=value: {item}")
        node = config
        *parents, leaf = key.split(".")
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = parse_value(raw)
    return config


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    return apply_overrides(load_yaml(path), overrides or [])


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def config_hash(config: dict) -> str:
    text = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_metadata(config: dict) -> dict[str, Any]:
    """Everything needed to tie a result back to code, config and packages."""
    import torch
    import transformers

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "config_hash": config_hash(config),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
