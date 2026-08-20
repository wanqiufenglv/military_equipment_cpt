from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.io_utils import append_jsonl, file_sha256
from ..core.models import PageAsset, PdfJob


def _render_page_worker(args: tuple[dict[str, Any], dict[str, Any], int]) -> dict[str, Any] | None:
    job_payload, cfg, page_index = args
    try:
        import fitz  # type: ignore
    except ImportError:
        return None

    job = PdfJob(**job_payload)
    output_dir = Path(job.output_dir)
    page_no = page_index + 1
    relative = cfg["paths"]["page_images"].format(page_no=page_no)
    target = output_dir / relative
    if not target.exists():
        scale = float(cfg["render"]["dpi"]) / 72.0
        target.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open(job.source_pdf) as document:
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pixmap.save(str(target))
    width = height = 0
    try:
        from PIL import Image  # type: ignore

        with Image.open(target) as image:
            width, height = image.size
    except Exception:
        pass
    return asdict(
        PageAsset(
            dataset_id=job.dataset_id,
            pdf_id=job.pdf_id,
            source_pdf=job.source_pdf,
            page_no=page_no,
            page_image_path=relative,
            width=width,
            height=height,
            image_hash=file_sha256(target, int(cfg["hash_chunk_size"])) if target.exists() else "",
        )
    )


def render_pages(pdf_job: PdfJob, cfg: dict[str, Any]) -> list[PageAsset]:
    if not bool(cfg["runtime"].get("render_pages", True)):
        return []
    try:
        import fitz  # type: ignore
    except ImportError:
        return []
    with fitz.open(pdf_job.source_pdf) as document:
        total_pages = int(document.page_count)
    workers = max(1, int(cfg["runtime"].get("page_workers", 1)))
    args = [(asdict(pdf_job), cfg, page_index) for page_index in range(total_pages)]
    if workers == 1 or len(args) <= 1:
        rows = [_render_page_worker(arg) for arg in args]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_render_page_worker, arg) for arg in args]
            for future in as_completed(futures):
                rows.append(future.result())
    assets = [PageAsset(**row) for row in rows if isinstance(row, dict)]
    assets.sort(key=lambda item: item.page_no)
    manifest = Path(pdf_job.output_dir) / cfg["paths"]["pages_manifest"]
    for asset in assets:
        append_jsonl(manifest, asdict(asset))
    return assets

