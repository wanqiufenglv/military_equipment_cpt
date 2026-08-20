from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.io_utils import append_jsonl, write_json, write_jsonl


def _to_sharegpt(sample: dict[str, Any]) -> dict[str, Any]:
    payload = dict(sample.get("payload") or {})
    payload["id"] = sample.get("sample_id")
    return payload


def _to_alpaca(sample: dict[str, Any]) -> dict[str, Any]:
    payload = dict(sample.get("payload") or {})
    record = {
        "instruction": payload.get("instruction", ""),
        "input": payload.get("input", ""),
        "output": payload.get("output", ""),
    }
    if payload.get("images"):
        record["images"] = payload["images"]
    return record


def _to_pt(sample: dict[str, Any]) -> dict[str, str]:
    payload = sample.get("payload") if isinstance(sample.get("payload"), dict) else {}
    return {"text": str(payload.get("text") or "").strip()}


def _dataset_info(cfg: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for task_type in cfg["export_formats"]["sharegpt"]:
        info[f"equipment_cpt_{task_type}"] = {
            "file_name": f"sharegpt/{task_type}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
        }
    for task_type in cfg["export_formats"]["alpaca"]:
        info[f"equipment_cpt_{task_type}"] = {
            "file_name": f"alpaca/{task_type}.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    for task_type in cfg["export_formats"].get("pt", []):
        info[f"equipment_cpt_{task_type}"] = {
            "file_name": f"pt/{task_type}.jsonl",
            "stage": "pt",
            "columns": {"prompt": "text"},
        }
    return info


def export_task_files(output_dir: Path, samples: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    sharegpt_task_types = set(cfg["export_formats"]["sharegpt"])
    alpaca_task_types = set(cfg["export_formats"]["alpaca"])
    pt_task_types = set(cfg["export_formats"].get("pt", []))
    provenance_path = output_dir / cfg["paths"]["provenance"]
    if provenance_path.exists():
        provenance_path.unlink()
    for sample in samples:
        append_jsonl(provenance_path, {"sample_id": sample.get("sample_id"), "task_type": sample.get("task_type"), "metadata": sample.get("metadata")})
    for task_type in cfg["task_types"]:
        task_samples = [sample for sample in samples if sample.get("task_type") == task_type]
        if task_type in sharegpt_task_types:
            write_jsonl(output_dir / cfg["paths"]["export_sharegpt"].format(task_type=task_type), (_to_sharegpt(sample) for sample in task_samples))
        if task_type in alpaca_task_types:
            write_jsonl(output_dir / cfg["paths"]["export_alpaca"].format(task_type=task_type), (_to_alpaca(sample) for sample in task_samples))
        if task_type in pt_task_types:
            write_jsonl(
                output_dir / cfg["paths"]["export_pt"].format(task_type=task_type),
                (_to_pt(sample) for sample in task_samples if _to_pt(sample)["text"]),
            )
    write_json(output_dir / cfg["paths"]["dataset_info"], _dataset_info(cfg))
