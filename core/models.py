from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PdfJob:
    dataset_id: str
    pdf_id: str
    source_pdf: str
    output_dir: str
    file_hash: str
    file_size: int
    page_count: int
    task_types: list[str]
    status: str
    created_at: str
    pipeline_version: str


@dataclass
class PageAsset:
    dataset_id: str
    pdf_id: str
    source_pdf: str
    page_no: int
    page_image_path: str
    width: int = 0
    height: int = 0
    image_hash: str = ""


@dataclass
class DocumentUnit:
    dataset_id: str
    pdf_id: str
    source_pdf: str
    page_no: int
    unit_id: str
    unit_type: str
    region_id: str
    bbox: list[float] = field(default_factory=list)
    reading_order: int = 0
    image_paths: list[str] = field(default_factory=list)
    page_image_path: str = ""
    caption: str = ""
    ocr_text: str = ""
    text: str = ""
    nearby_text: str = ""
    evidence_text: str = ""
    mineru_block_ids: list[str] = field(default_factory=list)
    normalized_page_path: str = ""
    mineru_parse_path: str = ""


@dataclass(frozen=True)
class TaskContext:
    task_type: str
    pdf_job: PdfJob
    units: list[DocumentUnit]
    page_no: int
    region_id: str
    image_paths: list[str] = field(default_factory=list)
    evidence_text: str = ""
    source: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    task_type: str
    format: str
    payload: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PipelineOptions:
    config_path: Path | None = None
    input_dir: Path | None = None
    input_pdfs: list[dict[str, str]] | None = None
    output_root: Path | None = None
    reuse_mineru: bool | None = None
    skip_mineru: bool | None = None
    skip_model: bool | None = None
    pdf_workers: int | None = None
    model_workers: int | None = None
    page_workers: int | None = None
    min_page_text_chars: int | None = None
    min_unit_text_chars: int | None = None
    task_types: list[str] | None = None
    limit_pdfs: int | None = None
