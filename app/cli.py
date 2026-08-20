from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from military_equipment_cpt.app.pipeline import run_pipeline
    from military_equipment_cpt.core.models import PipelineOptions
else:
    from .pipeline import run_pipeline
    from ..core.models import PipelineOptions


def parse_pdf_spec(value: str) -> dict[str, str]:
    if "=" not in value:
        return {"path": value}
    left, path = value.split("=", 1)
    if ":" in left:
        dataset_id, pdf_id = left.split(":", 1)
        return {"dataset_id": dataset_id.strip(), "pdf_id": pdf_id.strip(), "path": path.strip()}
    return {"dataset_id": left.strip(), "path": path.strip()}


def parse_task_specs(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    task_types: list[str] = []
    for value in values:
        for item in value.split(","):
            task_type = item.strip()
            if task_type:
                task_types.append(task_type)
    return task_types or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Military equipment CPT SFT data pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--pdf", action="append", default=None, help="指定 PDF，格式可为 path、dataset=path 或 dataset:pdf_id=path。")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reuse-mineru", action="store_true")
    parser.add_argument(
        "--skip-mineru",
        "--skip_mineru",
        dest="skip_mineru",
        action="store_true",
        help="跳过 MinerU 服务调用，只读取已有 MinerU raw/parsed/image_map 结果。",
    )
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--pdf-workers", "--pdf_workers", dest="pdf_workers", type=int, default=None)
    parser.add_argument("--model-workers", "--model_workers", dest="model_workers", type=int, default=None)
    parser.add_argument("--page-workers", "--page_workers", dest="page_workers", type=int, default=None)
    parser.add_argument("--min-page-text-chars", "--min_page_text_chars", dest="min_page_text_chars", type=int, default=None)
    parser.add_argument("--min-unit-text-chars", "--min_unit_text_chars", dest="min_unit_text_chars", type=int, default=None)
    parser.add_argument("--task", dest="tasks", action="append", default=None)
    parser.add_argument("--tasks", dest="tasks", action="append")
    parser.add_argument("--limit-pdfs", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_pipeline(
        PipelineOptions(
            config_path=Path(args.config) if args.config else None,
            input_dir=Path(args.input_dir) if args.input_dir else None,
            input_pdfs=[parse_pdf_spec(item) for item in args.pdf] if args.pdf else None,
            output_root=Path(args.output_root) if args.output_root else None,
            reuse_mineru=True if args.reuse_mineru else None,
            skip_mineru=True if args.skip_mineru else None,
            skip_model=True if args.skip_model else None,
            pdf_workers=args.pdf_workers,
            model_workers=args.model_workers,
            page_workers=args.page_workers,
            min_page_text_chars=args.min_page_text_chars,
            min_unit_text_chars=args.min_unit_text_chars,
            task_types=parse_task_specs(args.tasks),
            limit_pdfs=args.limit_pdfs,
        )
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
