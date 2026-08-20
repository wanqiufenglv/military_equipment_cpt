from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIXED_INSTRUCTION_TEXTS = {
    "请根据页面图片整理其中的资料内容。",
    "请将以下资料证据清洗并重建为连续、准确的资料文本。",
    "请将以下属性或参数片段整理为自然语言资料说明。",
    "请结合图片和相关资料整理一段资料说明。",
    "请综合这些图片和相关资料整理完整说明。",
    "请根据以下资料生成实体-属性知识句。",
}


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _text_payload(sample: dict[str, Any]) -> str:
    return json.dumps(sample.get("payload", {}), ensure_ascii=False)


def _answer_text(sample: dict[str, Any]) -> str:
    payload = sample.get("payload") if isinstance(sample.get("payload"), dict) else {}
    if sample.get("format") == "pt":
        return str(payload.get("text") or "").strip()
    if sample.get("format") == "sharegpt":
        conversations = payload.get("conversations")
        if not isinstance(conversations, list):
            return ""
        return " ".join(
            str(item.get("value", "")).strip()
            for item in conversations
            if isinstance(item, dict) and item.get("from") == "gpt"
        ).strip()
    return str(payload.get("output") or "").strip()


def _valid_images(sample: dict[str, Any], output_dir: Path) -> bool:
    payload = sample.get("payload") if isinstance(sample.get("payload"), dict) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    images = payload.get("images") or metadata.get("image_paths") or []
    if not isinstance(images, list):
        return False
    return all((output_dir / str(image)).exists() for image in images)


def _sharegpt_image_count_ok(sample: dict[str, Any]) -> bool:
    payload = sample.get("payload") if isinstance(sample.get("payload"), dict) else {}
    if sample.get("format") != "sharegpt":
        return True
    conversations = payload.get("conversations")
    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    if not isinstance(conversations, list) or len(conversations) < 2:
        return False
    human_text = "\n".join(str(item.get("value", "")) for item in conversations if isinstance(item, dict) and item.get("from") == "human")
    return human_text.count("<image>") == len(images)


def _instruction_text(sample: dict[str, Any]) -> str:
    payload = sample.get("payload") if isinstance(sample.get("payload"), dict) else {}
    if sample.get("format") == "sharegpt":
        conversations = payload.get("conversations")
        if not isinstance(conversations, list):
            return ""
        return " ".join(
            str(item.get("value", "")).replace("<image>", "").replace("<|image|>", "").strip()
            for item in conversations
            if isinstance(item, dict) and item.get("from") == "human"
        ).strip()
    return str(payload.get("instruction") or "").strip()


def _pt_task_types(cfg: dict[str, Any]) -> set[str]:
    return {str(task_type) for task_type in cfg.get("export_formats", {}).get("pt", [])}


def _pt_payload_is_pure(payload: dict[str, Any]) -> bool:
    return set(payload.keys()) == {"text"} and bool(str(payload.get("text") or "").strip())


def validate_sample(sample: dict[str, Any], output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    task_type = str(sample.get("task_type") or "")
    if task_type not in set(cfg["task_types"]):
        return None
    payload = sample.get("payload") if isinstance(sample.get("payload"), dict) else {}
    if not _has_value(sample.get("sample_id")) or not payload:
        return None
    for field in cfg["validation"]["required_payload_fields"].get(task_type, []):
        if not _has_value(payload.get(field)):
            return None
    is_pt = sample.get("format") == "pt" or task_type in _pt_task_types(cfg)
    if is_pt:
        if not _pt_payload_is_pure(payload):
            return None
        min_pt_chars = int(cfg["validation"].get("min_pt_text_chars", cfg["validation"].get("min_output_chars", 12)))
        if len(_answer_text(sample)) < min_pt_chars:
            return None
        forbidden = {"instruction", "input", "output", "messages", "conversations", "images", "metadata"}
        if any(key in payload for key in forbidden):
            return None
    else:
        min_output_chars = int(
            cfg["validation"].get("min_output_chars_by_task", {}).get(
                task_type,
                cfg["validation"]["min_output_chars"],
            )
        )
        if len(_answer_text(sample)) < min_output_chars:
            return None
        instruction_text = _instruction_text(sample)
        if len(instruction_text) < int(cfg["validation"].get("min_instruction_chars", 4)):
            return None
    if any(pattern in _text_payload(sample) for pattern in cfg["validation"].get("reject_patterns", [])):
        return None
    question_reject_patterns = cfg["validation"].get("question_reject_patterns_by_task", {}).get(task_type, [])
    if any(str(pattern) in _instruction_text(sample) for pattern in question_reject_patterns):
        return None
    answer_reject_patterns = cfg["validation"].get("answer_reject_patterns_by_task", {}).get(task_type, [])
    if any(str(pattern) in _answer_text(sample) for pattern in answer_reject_patterns):
        return None
    if task_type in set(cfg["validation"].get("image_required_tasks", [])) and not _valid_images(sample, output_dir):
        return None
    if not _sharegpt_image_count_ok(sample):
        return None
    if task_type in set(cfg["validation"].get("single_image_tasks", [])):
        payload = sample.get("payload") if isinstance(sample.get("payload"), dict) else {}
        images = payload.get("images") if isinstance(payload.get("images"), list) else []
        if len(images) != 1:
            return None
    if task_type in set(cfg["validation"].get("multi_image_tasks", [])):
        images = payload.get("images") if isinstance(payload.get("images"), list) else []
        if len(images) < 2:
            return None
    metadata = sample.setdefault("metadata", {})
    metadata["validator"] = "rule"
    metadata["quality_score"] = float(cfg["validation"]["default_quality_score"])
    return sample
