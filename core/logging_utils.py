from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl, write_json


def configure_logger(name: str, log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class RunState:
    def __init__(self, logger: logging.Logger, output_dir: Path | None = None, cfg: dict[str, Any] | None = None) -> None:
        self.logger = logger
        self.output_dir = output_dir
        self.cfg = cfg or {}
        self.metrics: dict[str, Any] = {}
        self.checkpoints: dict[str, Any] = {}

    def _path(self, key: str) -> Path | None:
        if self.output_dir is None:
            return None
        rel = self.cfg.get("paths", {}).get(key)
        return self.output_dir / rel if rel else None

    def error(self, payload: dict[str, Any]) -> None:
        self.logger.error("pipeline_error %s", payload)
        path = self._path("errors")
        if path is not None:
            append_jsonl(path, payload)

    def increment(self, key: str, amount: int = 1) -> None:
        self.metrics[key] = int(self.metrics.get(key, 0)) + amount
        self.logger.info("metric %s=%s", key, self.metrics[key])
        path = self._path("metrics")
        if path is not None:
            write_json(path, self.metrics)

    def checkpoint(self, key: str, payload: Any) -> None:
        self.checkpoints[key] = payload
        self.logger.info("checkpoint %s=%s", key, payload)
        path = self._path("checkpoints")
        if path is not None:
            write_json(path, self.checkpoints)

