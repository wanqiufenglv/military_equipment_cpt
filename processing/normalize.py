from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.io_utils import clean_text, read_json, write_json
from ..core.models import DocumentUnit, PageAsset, PdfJob


def _item_page(item: dict[str, Any], fallback: int) -> int:
    for key in ("page_idx", "page_index", "page", "page_no"):
        value = item.get(key)
        if isinstance(value, int):
            return value + 1 if key == "page_idx" else value
        if isinstance(value, str) and value.isdigit():
            number = int(value)
            return number + 1 if key == "page_idx" else number
    return fallback


def _bbox(item: dict[str, Any]) -> list[float]:
    value = item.get("bbox") or item.get("box") or item.get("poly")
    if isinstance(value, list):
        flat: list[float] = []
        for part in value:
            if isinstance(part, list):
                flat.extend(float(x) for x in part if isinstance(x, (int, float)))
            elif isinstance(part, (int, float)):
                flat.append(float(part))
        return flat[:4]
    return []


def _unit_type(raw: str, cfg: dict[str, Any]) -> str:
    return str(cfg["block_types"].get(raw.lower(), cfg["block_types"]["unknown"]))


def _item_text(item: dict[str, Any]) -> str:
    keys = ("text", "content", "md_content", "markdown", "caption", "img_caption", "image_caption", "table_caption", "latex", "html")
    return clean_text(" ".join(clean_text(item.get(key)) for key in keys))


def _caption(item: dict[str, Any]) -> str:
    return clean_text(item.get("caption") or item.get("img_caption") or item.get("image_caption") or item.get("table_caption"))


def _iter_image_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"img_path", "image_path", "table_img_path", "figure_path", "path", "file_path"}:
                text = clean_text(child)
                if text:
                    refs.append(text)
            else:
                refs.extend(_iter_image_refs(child))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_iter_image_refs(item))
    return refs


def _resolve_image_refs(item: dict[str, Any], image_map: dict[str, str]) -> list[str]:
    images: list[str] = []
    for ref in _iter_image_refs(item):
        normalized = ref.replace("\\", "/")
        value = image_map.get(ref) or image_map.get(normalized) or image_map.get(Path(normalized).name)
        if value and value not in images:
            images.append(value)
    return images


def _nearby_text(items: list[dict[str, Any]], index: int, radius: int = 2) -> str:
    start = max(0, index - radius)
    end = min(len(items), index + radius + 1)
    return clean_text(" ".join(_item_text(item) for item in items[start:end] if item is not items[index]))


def normalize_units(
    pdf_job: PdfJob,
    page_assets: list[PageAsset],
    content: list[dict[str, Any]],
    mineru_parse_path: str,
    image_map: dict[str, str],
    cfg: dict[str, Any],
) -> list[DocumentUnit]:
    output_dir = Path(pdf_job.output_dir)
    page_asset_map = {asset.page_no: asset for asset in page_assets}
    grouped: dict[int, list[dict[str, Any]]] = {}
    fallback_page = 1
    for item in content:
        page_no = _item_page(item, fallback_page)
        grouped.setdefault(page_no, []).append(item)
        fallback_page = page_no
    total_pages = max(pdf_job.page_count, max(grouped.keys(), default=0))
    units: list[DocumentUnit] = []
    for page_no in range(1, total_pages + 1):
        normalized_path = output_dir / cfg["paths"]["normalized_page"].format(page_no=page_no)
        if bool(cfg["runtime"].get("reuse_normalized")) and normalized_path.exists():
            cached = read_json(normalized_path)
            if isinstance(cached, list):
                page_units = [DocumentUnit(**item) for item in cached if isinstance(item, dict)]
                units.extend(page_units)
                continue
        page_items = grouped.get(page_no, [])
        page_asset = page_asset_map.get(page_no)
        page_image_path = page_asset.page_image_path if page_asset else cfg["paths"]["page_images"].format(page_no=page_no)
        page_units: list[DocumentUnit] = []
        for index, item in enumerate(page_items, start=1):
            raw_type = clean_text(item.get("type")) or cfg["block_types"]["unknown"]
            unit_type = _unit_type(raw_type, cfg)
            text = _item_text(item)
            caption = _caption(item)
            images = _resolve_image_refs(item, image_map)
            evidence_text = clean_text(" ".join(part for part in [text, caption] if part))
            region_id = f"r{index:04d}"
            unit = DocumentUnit(
                dataset_id=pdf_job.dataset_id,
                pdf_id=pdf_job.pdf_id,
                source_pdf=pdf_job.source_pdf,
                page_no=page_no,
                unit_id=f"p{page_no:04d}_{region_id}",
                unit_type=unit_type,
                region_id=region_id,
                bbox=_bbox(item),
                reading_order=index,
                image_paths=images,
                page_image_path=page_image_path,
                caption=caption,
                ocr_text=text if unit_type in {"text", "ocr", "caption"} else "",
                text=text,
                nearby_text=_nearby_text(page_items, index - 1),
                evidence_text=evidence_text,
                mineru_block_ids=[clean_text(item.get("block_id") or item.get("id") or region_id)],
                normalized_page_path=str(normalized_path.relative_to(output_dir)).replace("\\", "/"),
                mineru_parse_path=mineru_parse_path,
            )
            page_units.append(unit)
        if not page_units and page_asset:
            page_units.append(
                DocumentUnit(
                    dataset_id=pdf_job.dataset_id,
                    pdf_id=pdf_job.pdf_id,
                    source_pdf=pdf_job.source_pdf,
                    page_no=page_no,
                    unit_id=f"p{page_no:04d}_page",
                    unit_type="page",
                    region_id="page",
                    page_image_path=page_image_path,
                    normalized_page_path=str(normalized_path.relative_to(output_dir)).replace("\\", "/"),
                    mineru_parse_path=mineru_parse_path,
                )
            )
        write_json(normalized_path, [asdict(unit) for unit in page_units])
        units.extend(page_units)
    return units

