from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any

from .clients import MinerUClient
from ..core.io_utils import maybe_json, read_json, write_json
from ..core.models import PdfJob


def strip_embedded_images(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: strip_embedded_images(value) for key, value in payload.items() if key != "images"}
    if isinstance(payload, list):
        return [strip_embedded_images(item) for item in payload]
    return payload


def find_content_list(payload: Any) -> list[dict[str, Any]]:
    candidates: list[list[dict[str, Any]]] = []

    def visit(value: Any) -> None:
        value = maybe_json(value)
        if isinstance(value, dict):
            content = maybe_json(value.get("content_list"))
            if isinstance(content, list) and all(isinstance(item, dict) for item in content):
                candidates.append(content)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value) and any("type" in item for item in value):
                candidates.append(value)
            for child in value:
                visit(child)

    visit(payload)
    return max(candidates, key=len) if candidates else []


def _iter_image_maps(payload: Any):
    payload = maybe_json(payload)
    if isinstance(payload, dict):
        for key in ("images", "image_dict", "img_dict"):
            image_map = maybe_json(payload.get(key))
            if isinstance(image_map, dict):
                yield image_map
        for value in payload.values():
            yield from _iter_image_maps(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_image_maps(value)


def _decode_image(value: Any) -> tuple[bytes, str] | None:
    value = maybe_json(value)
    if isinstance(value, dict):
        for key in ("data", "base64", "content", "image"):
            decoded = _decode_image(value.get(key))
            if decoded:
                return decoded
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    mime = "image/png"
    if text.startswith("data:image/"):
        header, _, text = text.partition(",")
        match = re.search(r"data:([^;]+)", header)
        if match:
            mime = match.group(1)
    try:
        return base64.b64decode(text, validate=False), mime
    except Exception:
        return None


def _image_suffix(name: str, mime: str, cfg: dict[str, Any]) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in set(cfg["extracted_image_naming"]["allowed_suffixes"]):
        suffix = mimetypes.guess_extension(mime) or ".png"
    return suffix


def _safe_image_name(name: str, image_no: int, mime: str, cfg: dict[str, Any]) -> str:
    suffix = _image_suffix(name, mime, cfg)
    page_label = cfg["extracted_image_naming"]["unknown_page_label"]
    return cfg["extracted_image_naming"]["template"].format(page_label=page_label, page_no=0, image_no=image_no, source_stem=Path(name).stem, suffix=suffix)


def save_extracted_images(payload: Any, output_dir: Path, cfg: dict[str, Any]) -> dict[str, str]:
    images_dir = output_dir / cfg["paths"]["extracted_images"]
    images_dir.mkdir(parents=True, exist_ok=True)
    image_map: dict[str, str] = {}
    image_no = 0
    for mineru_images in _iter_image_maps(payload):
        for original_name, encoded in mineru_images.items():
            decoded = _decode_image(encoded)
            if not decoded:
                continue
            image_no += 1
            image_bytes, mime = decoded
            filename = _safe_image_name(str(original_name), image_no, mime, cfg)
            target = images_dir / filename
            if not target.exists():
                target.write_bytes(image_bytes)
            image_map[str(original_name)] = str(target.relative_to(output_dir)).replace("\\", "/")
    return image_map


def parse_pdf_with_mineru(pdf_job: PdfJob, cfg: dict[str, Any]) -> tuple[Any, list[dict[str, Any]], str, dict[str, str]]:
    output_dir = Path(pdf_job.output_dir)
    raw_path = output_dir / cfg["paths"]["mineru_raw"].format(pdf_id=pdf_job.pdf_id)
    parsed_path = output_dir / cfg["paths"]["mineru_parsed"].format(pdf_id=pdf_job.pdf_id)
    image_map_path = output_dir / cfg["paths"]["image_map"]
    if bool(cfg["runtime"].get("skip_mineru")):
        required_paths = [raw_path, parsed_path, image_map_path]
        missing = [str(path.relative_to(output_dir)) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "skip_mineru=true requires existing MinerU outputs: "
                + ", ".join(missing)
            )
        return read_json(raw_path), read_json(parsed_path), str(parsed_path.relative_to(output_dir)), read_json(image_map_path)
    if bool(cfg["runtime"]["reuse_mineru"]) and raw_path.exists() and parsed_path.exists() and image_map_path.exists():
        return read_json(raw_path), read_json(parsed_path), str(parsed_path.relative_to(output_dir)), read_json(image_map_path)
    payload = MinerUClient(cfg["mineru"]).parse_pdf(Path(pdf_job.source_pdf))
    content = find_content_list(payload)
    image_map = save_extracted_images(payload, output_dir, cfg)
    write_json(raw_path, strip_embedded_images(payload))
    write_json(parsed_path, content)
    write_json(image_map_path, image_map)
    return payload, content, str(parsed_path.relative_to(output_dir)), image_map
