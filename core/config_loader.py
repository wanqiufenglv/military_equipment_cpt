from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_py_config(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("_military_equipment_cpt_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Invalid config module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = json.loads(getattr(module, "CONFIG_JSON"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config payload: {path}")
    return data


def _load_text_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Please install PyYAML or use JSON/Python config files.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config payload: {path}")
    return data


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    default_cfg = _load_py_config(package_root() / "config.py")
    if not config_path:
        return default_cfg
    override_path = Path(config_path)
    override = _load_py_config(override_path) if override_path.suffix == ".py" else _load_text_config(override_path)
    return deep_merge(default_cfg, override)


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return copy.deepcopy(base)
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

