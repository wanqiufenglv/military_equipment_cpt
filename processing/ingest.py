from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.io_utils import file_sha256, safe_name, stable_slug, utc_now
from ..core.models import PdfJob


def page_count(pdf_path: Path, cfg: dict[str, Any]) -> int:
    try:
        from pypdf import PdfReader  # type: ignore

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        try:
            import fitz  # type: ignore

            with fitz.open(pdf_path) as document:
                return int(document.page_count)
        except Exception:
            pattern = str(cfg["pdf_page_regex"]).encode("ascii")
            return len(re.findall(pattern, pdf_path.read_bytes()))


def _dataset_id_for(pdf_path: Path, item: dict[str, str] | None, cfg: dict[str, Any]) -> str:
    if item and item.get("dataset_id"):
        return safe_name(item["dataset_id"], cfg["unknown_dataset_id"])
    return safe_name(pdf_path.parent.name, cfg["unknown_dataset_id"])


def _pdf_id_for(pdf_path: Path, file_hash: str, item: dict[str, str] | None, cfg: dict[str, Any]) -> str:
    if item and item.get("pdf_id"):
        return safe_name(item["pdf_id"], cfg["unknown_pdf_id"])
    stem = stable_slug(pdf_path.stem, cfg["unknown_pdf_id"])
    return f"{stem}_{file_hash[:8]}"


def build_pdf_job(
    *,
    pdf_path: Path,
    output_root: Path,
    cfg: dict[str, Any],
    item: dict[str, str] | None = None,
) -> PdfJob:
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    file_hash = file_sha256(pdf_path, int(cfg["hash_chunk_size"]))
    dataset_id = _dataset_id_for(pdf_path, item, cfg)
    pdf_id = _pdf_id_for(pdf_path, file_hash, item, cfg)
    output_dir = output_root / safe_name(f"{dataset_id}_{pdf_id}", pdf_id)
    return PdfJob(
        dataset_id=dataset_id,
        pdf_id=pdf_id,
        source_pdf=str(pdf_path),
        output_dir=str(output_dir),
        file_hash=file_hash,
        file_size=pdf_path.stat().st_size,
        page_count=page_count(pdf_path, cfg),
        task_types=list(cfg["task_types"]),
        status=cfg["statuses"]["ready"],
        created_at=utc_now(cfg["timestamp_format"]),
        pipeline_version=cfg["pipeline_version"],
    )


def scan_pdfs(input_dir: Path, output_root: Path, cfg: dict[str, Any]) -> list[PdfJob]:
    pattern = f"**/*{cfg['pdf_suffix']}" if bool(cfg["runtime"]["recursive"]) else f"*{cfg['pdf_suffix']}"
    jobs: list[PdfJob] = []
    if not input_dir.exists():
        return jobs
    for pdf_path in sorted(input_dir.glob(pattern)):
        if pdf_path.is_file():
            jobs.append(build_pdf_job(pdf_path=pdf_path, output_root=output_root, cfg=cfg))
    return jobs


def scan_input_pdfs(input_pdfs: list[dict[str, str]], output_root: Path, cfg: dict[str, Any]) -> list[PdfJob]:
    jobs: list[PdfJob] = []
    for item in input_pdfs:
        raw_path = item.get("path") or item.get("source_pdf") or item.get("pdf")
        if not raw_path:
            raise ValueError(f"Missing PDF path in input item: {item}")
        jobs.append(build_pdf_job(pdf_path=Path(raw_path), output_root=output_root, cfg=cfg, item=item))
    return jobs

