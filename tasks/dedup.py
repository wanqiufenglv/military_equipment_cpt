from __future__ import annotations

import hashlib
import json
from typing import Any


def deduplicate(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_payloads: set[str] = set()
    per_page_task: dict[tuple[str, str, int, str], int] = {}
    default_max_per_page = int(cfg["validation"]["max_samples_per_page_task"])
    max_per_page_by_task = cfg["validation"].get("max_samples_per_page_task_by_task", {})
    result: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        if sample_id in seen_ids:
            continue
        task_type = str(sample.get("task_type"))
        max_per_page = int(max_per_page_by_task.get(task_type, default_max_per_page))
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        key = (
            str(metadata.get("dataset_id")),
            str(metadata.get("pdf_id")),
            int(metadata.get("page_no") or 0),
            task_type,
        )
        if per_page_task.get(key, 0) >= max_per_page:
            continue
        text_key = hashlib.sha256(json.dumps(sample.get("payload", {}), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if text_key in seen_payloads:
            continue
        seen_ids.add(sample_id)
        seen_payloads.add(text_key)
        per_page_task[key] = per_page_task.get(key, 0) + 1
        result.append(sample)
    return result
