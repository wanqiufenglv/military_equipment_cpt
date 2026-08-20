from __future__ import annotations

from dataclasses import asdict

from ..core.io_utils import clean_text, truncate_text
from ..core.models import DocumentUnit, PdfJob, TaskContext


def _enabled(enabled_tasks: dict[str, bool], task_type: str) -> bool:
    return bool(enabled_tasks.get(task_type, False))


def _page_units(units: list[DocumentUnit]) -> dict[int, list[DocumentUnit]]:
    grouped: dict[int, list[DocumentUnit]] = {}
    for unit in units:
        grouped.setdefault(unit.page_no, []).append(unit)
    for page_units in grouped.values():
        page_units.sort(key=lambda item: item.reading_order)
    return grouped


def _unit_text(units: list[DocumentUnit]) -> str:
    return clean_text("\n".join(unit.evidence_text or unit.text or unit.ocr_text for unit in units))


def _domain_unit_text(unit: DocumentUnit) -> str:
    return clean_text(unit.evidence_text or unit.text or unit.ocr_text or unit.caption)


def _domain_page_text(page_units: list[DocumentUnit], cfg: dict) -> str:
    allowed_types = set(cfg["routing"].get("domain_corpus_unit_types", []))
    skipped_types = set(cfg["routing"].get("domain_corpus_skip_unit_types", []))
    lines: list[str] = []
    for unit in page_units:
        if unit.unit_type in skipped_types:
            continue
        if allowed_types and unit.unit_type not in allowed_types:
            continue
        text = _domain_unit_text(unit)
        if not text or text.upper() == "OSID:":
            continue
        lines.append(text)
    return "\n".join(lines).strip()


def _page_image(units: list[DocumentUnit]) -> str:
    for unit in units:
        if unit.page_image_path:
            return unit.page_image_path
    return ""


def _visual_units(units: list[DocumentUnit], cfg: dict) -> list[DocumentUnit]:
    visual_types = set(cfg["routing"].get("visual_unit_types", []))
    return [unit for unit in units if unit.unit_type in visual_types and unit.image_paths]


def _unique_images(units: list[DocumentUnit]) -> list[str]:
    images: list[str] = []
    for unit in units:
        for image in unit.image_paths:
            if image and image not in images:
                images.append(image)
    return images


def _multi_image_groups(images: list[str], max_group_size: int = 3) -> list[list[str]]:
    limited_images = images[: max(2, max_group_size)]
    groups: list[list[str]] = []
    if len(limited_images) >= 2:
        for left in range(0, len(limited_images)):
            for right in range(left + 1, len(limited_images)):
                groups.append([limited_images[left], limited_images[right]])
    if len(limited_images) >= 3:
        groups.append(limited_images[:max_group_size])
    return groups


def _image_description(unit: DocumentUnit, image: str) -> dict[str, str]:
    return {
        "image_path": image,
        "caption": clean_text(unit.caption),
        "visible_or_ocr_text": clean_text(unit.evidence_text or unit.text or unit.ocr_text),
        "nearby_text": clean_text(unit.nearby_text),
        "page_no": str(unit.page_no),
        "region_id": unit.region_id,
    }


def _image_descriptions(units: list[DocumentUnit], images: list[str]) -> list[dict[str, str]]:
    descriptions: list[dict[str, str]] = []
    for image in images:
        owner = next((unit for unit in units if image in unit.image_paths), None)
        if owner is None:
            descriptions.append({"image_path": image, "caption": "", "visible_or_ocr_text": "", "nearby_text": "", "page_no": "", "region_id": ""})
        else:
            descriptions.append(_image_description(owner, image))
    return descriptions


def _attribute_like(unit: DocumentUnit, cfg: dict) -> bool:
    text = clean_text(unit.evidence_text or unit.text)
    if len(text) < int(cfg["routing"]["min_unit_text_chars"]):
        return False
    keywords = cfg["routing"].get("attribute_keywords", [])
    if any(str(keyword) in text for keyword in keywords):
        return True
    return any(char.isdigit() for char in text)


def _window_pages(sorted_pages: list[int], start: int, size: int) -> list[int]:
    idx = sorted_pages.index(start)
    return sorted_pages[idx : idx + max(1, size)]


def _context(
    task_type: str,
    pdf_job: PdfJob,
    units: list[DocumentUnit],
    page_no: int,
    region_id: str,
    images: list[str],
    evidence_text: str,
    source: dict,
) -> TaskContext:
    return TaskContext(task_type, pdf_job, units, page_no, region_id, images, evidence_text, source)


def _domain_region_id(page_numbers: list[int]) -> str:
    if not page_numbers:
        return "domain_corpus"
    if len(page_numbers) == 1:
        return f"p{page_numbers[0]:04d}_domain_corpus"
    return f"p{page_numbers[0]:04d}_p{page_numbers[-1]:04d}_domain_corpus"


def _build_domain_corpus_contexts(
    pdf_job: PdfJob,
    grouped: dict[int, list[DocumentUnit]],
    sorted_pages: list[int],
    cfg: dict,
) -> list[TaskContext]:
    routing = cfg["routing"]
    min_chars = int(routing.get("min_domain_corpus_text_chars", routing.get("min_page_text_chars", 120)))
    target_chars = int(routing.get("domain_corpus_target_input_chars", cfg["generation"].get("max_context_chars", 3600)))
    max_pages = max(1, int(routing.get("domain_corpus_window", routing.get("document_window", 4))))
    contexts: list[TaskContext] = []
    window_pages: list[int] = []
    window_units: list[DocumentUnit] = []
    window_texts: list[str] = []
    window_chars = 0

    def emit() -> None:
        nonlocal window_pages, window_units, window_texts, window_chars
        if not window_pages:
            return
        raw_text = "\n\n".join(f"[page {page_no}]\n{text}" for page_no, text in zip(window_pages, window_texts) if text).strip()
        if len(clean_text(raw_text)) >= min_chars:
            contexts.append(
                _context(
                    "domain_knowledge_corpus",
                    pdf_job,
                    list(window_units),
                    window_pages[0],
                    _domain_region_id(window_pages),
                    [],
                    clean_text(raw_text),
                    {
                        "raw_text": raw_text,
                        "page_numbers": list(window_pages),
                        "window_units": [asdict(unit) for unit in window_units],
                    },
                )
            )
        window_pages = []
        window_units = []
        window_texts = []
        window_chars = 0

    for page_no in sorted_pages:
        page_units = grouped[page_no]
        text = _domain_page_text(page_units, cfg)
        if not text:
            continue
        if window_pages and (len(window_pages) >= max_pages or (window_chars + len(text) > target_chars and window_chars >= min_chars)):
            emit()
        window_pages.append(page_no)
        window_units.extend(page_units)
        window_texts.append(text)
        window_chars += len(text)
        if window_chars >= target_chars or len(window_pages) >= max_pages:
            emit()
    emit()
    return contexts


def build_task_contexts(pdf_job: PdfJob, units: list[DocumentUnit], cfg: dict) -> list[TaskContext]:
    contexts: list[TaskContext] = []
    routing = cfg["routing"]
    enabled = routing["enabled_tasks"]
    grouped = _page_units(units)
    sorted_pages = sorted(grouped)
    min_page_text = int(routing["min_page_text_chars"])
    min_document_text = int(routing["min_document_text_chars"])
    document_window = int(routing["document_window"])
    multi_image_window = int(routing["multi_image_window"])
    multi_image_group_size = int(routing.get("multi_image_group_size", 3))

    for page_no in sorted_pages:
        page_units = grouped[page_no]
        page_text = truncate_text(_unit_text(page_units), int(cfg["generation"]["max_context_chars"]))
        page_image = _page_image(page_units)

        if _enabled(enabled, "cpt_page_reconstruction") and page_image:
            contexts.append(
                _context(
                    "cpt_page_reconstruction",
                    pdf_job,
                    page_units,
                    page_no,
                    "page",
                    [page_image],
                    page_text,
                    {"page_units": [asdict(unit) for unit in page_units]},
                )
            )

        if _enabled(enabled, "cpt_document_rewrite") and len(page_text) >= min_page_text:
            contexts.append(
                _context(
                    "cpt_document_rewrite",
                    pdf_job,
                    page_units,
                    page_no,
                    "page",
                    [],
                    page_text,
                    {"page_units": [asdict(unit) for unit in page_units]},
                )
            )

        if _enabled(enabled, "cpt_attribute_to_text"):
            attribute_units = [unit for unit in page_units if _attribute_like(unit, cfg)]
            if attribute_units:
                attribute_text = "\n".join(
                    clean_text(unit.evidence_text or unit.text)
                    for unit in attribute_units
                    if clean_text(unit.evidence_text or unit.text)
                )
                contexts.append(
                    _context(
                        "cpt_attribute_to_text",
                        pdf_job,
                        attribute_units,
                        page_no,
                        f"p{page_no:04d}_attributes",
                        _unique_images(attribute_units),
                        truncate_text(attribute_text, int(cfg["generation"]["max_context_chars"])),
                        {"attribute_units": [asdict(unit) for unit in attribute_units]},
                    )
                )

        if _enabled(enabled, "cpt_single_image_brief"):
            for unit in _visual_units(page_units, cfg):
                for image_index, image in enumerate(_unique_images([unit]), start=1):
                    description = _image_description(unit, image)
                    text = clean_text(" ".join([description["caption"], description["visible_or_ocr_text"], description["nearby_text"]]))
                    contexts.append(
                        _context(
                            "cpt_single_image_brief",
                            pdf_job,
                            [unit],
                            page_no,
                            f"{unit.region_id}_img{image_index:02d}",
                            [image],
                            truncate_text(text, int(cfg["generation"]["max_unit_chars"])),
                            {"image_description": description, "unit": asdict(unit)},
                        )
                    )

        if _enabled(enabled, "sft_single_image_vqa"):
            for unit in _visual_units(page_units, cfg):
                for image_index, image in enumerate(_unique_images([unit]), start=1):
                    description = _image_description(unit, image)
                    text = clean_text(" ".join([description["caption"], description["visible_or_ocr_text"], description["nearby_text"]]))
                    contexts.append(
                        _context(
                            "sft_single_image_vqa",
                            pdf_job,
                            [unit],
                            page_no,
                            f"{unit.region_id}_vqa_img{image_index:02d}",
                            [image],
                            truncate_text(text, int(cfg["generation"]["max_unit_chars"])),
                            {"image_description": description, "unit": asdict(unit)},
                        )
                    )

        if _enabled(enabled, "cpt_multi_image_brief"):
            window_page_numbers = _window_pages(sorted_pages, page_no, multi_image_window)
            window_units = [unit for current_page in window_page_numbers for unit in grouped[current_page]]
            visual_units = _visual_units(window_units, cfg)
            visual_images = _unique_images(visual_units)
            for chunk_index, image_chunk in enumerate(_multi_image_groups(visual_images, multi_image_group_size), start=1):
                chunk_units = [
                    unit
                    for unit in visual_units
                    if any(image in image_chunk for image in unit.image_paths)
                ]
                image_descriptions = _image_descriptions(chunk_units or window_units, image_chunk)
                contexts.append(
                    _context(
                        "cpt_multi_image_brief",
                        pdf_job,
                        chunk_units or window_units,
                        page_no,
                        f"p{page_no:04d}_multi_image_{chunk_index:02d}",
                        image_chunk,
                        truncate_text(_unit_text(chunk_units or window_units), int(cfg["generation"]["max_context_chars"])),
                        {
                            "image_count": len(image_chunk),
                            "image_descriptions": image_descriptions,
                            "window_units": [asdict(unit) for unit in chunk_units or window_units],
                        },
                    )
                )

        if _enabled(enabled, "cpt_entity_attribute_sentences"):
            window_page_numbers = _window_pages(sorted_pages, page_no, document_window)
            window_units = [unit for current_page in window_page_numbers for unit in grouped[current_page]]
            window_text = truncate_text(_unit_text(window_units), int(cfg["generation"]["max_context_chars"]))
            if len(window_text) >= min_document_text:
                contexts.append(
                    _context(
                        "cpt_entity_attribute_sentences",
                        pdf_job,
                        window_units,
                        page_no,
                        f"p{page_no:04d}_facts",
                        [],
                        window_text,
                        {"window_units": [asdict(unit) for unit in window_units]},
                    )
                )
    if _enabled(enabled, "domain_knowledge_corpus"):
        contexts.extend(_build_domain_corpus_contexts(pdf_job, grouped, sorted_pages, cfg))
    return contexts
