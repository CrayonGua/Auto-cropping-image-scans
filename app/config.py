"""集中式配置。

所有可调参数都放在项目根目录的 config.json，改参数不需要改代码。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
DATA_DIR = ROOT / "data"

DEFAULTS: dict[str, Any] = {
    "detector": {
        "detect_max_side": 1600,
        "border_ratio": 0.05,
        "min_area_ratio": 0.015,
        "max_area_ratio": 0.92,
        "min_solidity": 0.85,
        "close_kernel": 9,
        "max_regions": 4,
        "min_confidence": 0.45,
        "inset_px": 2,
    },
    "upload": {
        "max_files": 30,
        "max_file_mb": 60,
        "preview_max_side": 1600,
        "allowed_ext": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
    },
    "export": {
        "default_format": "jpg",
        "default_quality": 92,
        "write_sidecar": True,
    },
}


def load_config() -> dict[str, Any]:
    """读取 config.json，缺失的键回落到默认值。"""
    cfg = copy.deepcopy(DEFAULTS)
    if not CONFIG_PATH.exists():
        return cfg
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"config.json 格式错误：{exc}") from exc
    for section, values in raw.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg
