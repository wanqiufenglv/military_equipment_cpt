from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.config_loader import deep_merge, load_config, package_root
from ..core.io_utils import append_jsonl, iter_jsonl, write_jsonl
from ..core.logging_utils import RunState, configure_logger
from ..core.models import PipelineOptions
from ..processing.ingest import scan_input_pdfs, scan_pdfs
from ..processing.normalize import normalize_units
from ..processing.render import render_pages
from ..services.clients import ModelPool
from ..services.mineru import parse_pdf_with_mineru
from ..tasks.dedup import deduplicate
from ..tasks.exporters import export_task_files
from ..tasks.generation import generate_for_context
from ..tasks.routing import build_task_contexts
from ..tasks.validation import validate_sample


def _select_task_types(cfg: dict[str, Any], task_types: list[str]) -> dict[str, Any]:
    requested = [str(task_type).strip() for task_type in task_types if str(task_type).strip()]
    known_tasks = list(cfg.get("task_types", []))
    unknown = [task_type for task_type in requested if task_type not in set(known_tasks)]
    if unknown:
        raise ValueError(f"Unknown task type(s): {', '.join(unknown)}")
    selected = list(dict.fromkeys(requested))
    if not selected:
        raise ValueError("At least one task type must be selected.")
    selected_set = set(selected)
    cfg["task_types"] = [task_type for task_type in known_tasks if task_type in selected_set]
    for key, task_list in list(cfg.get("export_formats", {}).items()):
        if isinstance(task_list, list):
            cfg["export_formats"][key] = [task_type for task_type in task_list if task_type in selected_set]
    enabled_tasks = cfg.setdefault("routing", {}).setdefault("enabled_tasks", {})
    for task_type in known_tasks:
        enabled_tasks[task_type] = task_type in selected_set
    for provider in cfg.get("vlm_pool", {}).get("providers", []):
        if isinstance(provider, dict) and isinstance(provider.get("task_types"), list):
            provider["task_types"] = [task_type for task_type in provider["task_types"] if task_type in selected_set]
    return cfg


def _apply_options(cfg: dict[str, Any], options: PipelineOptions) -> dict[str, Any]:
    override: dict[str, Any] = {"runtime": {}}
    if options.input_dir is not None:
        override["runtime"]["input_dir"] = str(options.input_dir)
    if options.input_pdfs is not None:
        override["runtime"]["input_pdfs"] = options.input_pdfs
    if options.output_root is not None:
        override["runtime"]["output_root"] = str(options.output_root)
    if options.reuse_mineru is not None:
        override["runtime"]["reuse_mineru"] = bool(options.reuse_mineru)
    if options.skip_mineru is not None:
        override["runtime"]["skip_mineru"] = bool(options.skip_mineru)
    if options.skip_model is not None:
        override["runtime"]["skip_model"] = bool(options.skip_model)
    if options.pdf_workers is not None:
        override["runtime"]["pdf_workers"] = int(options.pdf_workers)
    if options.model_workers is not None:
        override["runtime"]["model_workers"] = int(options.model_workers)
    if options.page_workers is not None:
        override["runtime"]["page_workers"] = int(options.page_workers)
    if options.min_page_text_chars is not None:
        override.setdefault("routing", {})["min_page_text_chars"] = int(options.min_page_text_chars)
    if options.min_unit_text_chars is not None:
        override.setdefault("routing", {})["min_unit_text_chars"] = int(options.min_unit_text_chars)
    cfg = deep_merge(cfg, override)
    if options.task_types is not None:
        cfg = _select_task_types(cfg, options.task_types)
    return cfg


def _resolve_config_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else base_dir / path)


def _resolve_runtime_paths(cfg: dict[str, Any], options: PipelineOptions) -> dict[str, Any]:
    base_dir = Path(options.config_path).resolve().parent if options.config_path else package_root().parent
    if options.input_dir is None:
        cfg["runtime"]["input_dir"] = _resolve_config_path(str(cfg["runtime"]["input_dir"]), base_dir)
    if options.output_root is None:
        cfg["runtime"]["output_root"] = _resolve_config_path(str(cfg["runtime"]["output_root"]), base_dir)
    if options.input_pdfs is None:
        input_pdfs = cfg["runtime"].get("input_pdfs")
        if isinstance(input_pdfs, list):
            for item in input_pdfs:
                if not isinstance(item, dict):
                    continue
                raw_path = item.get("path") or item.get("source_pdf") or item.get("pdf")
                if raw_path and not Path(str(raw_path)).is_absolute():
                    item["path"] = _resolve_config_path(str(raw_path), base_dir)
    return cfg


def _expected_export_paths(output_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for task_type in cfg["export_formats"]["sharegpt"]:
        paths.append(output_dir / cfg["paths"]["export_sharegpt"].format(task_type=task_type))
    for task_type in cfg["export_formats"]["alpaca"]:
        paths.append(output_dir / cfg["paths"]["export_alpaca"].format(task_type=task_type))
    for task_type in cfg["export_formats"].get("pt", []):
        paths.append(output_dir / cfg["paths"]["export_pt"].format(task_type=task_type))
    return paths


def _exports_exist(output_dir: Path, cfg: dict[str, Any]) -> bool:
    expected = _expected_export_paths(output_dir, cfg)
    if not (bool(expected) and all(path.exists() for path in expected) and any(path.stat().st_size > 0 for path in expected)):
        return False
    provenance_path = output_dir / cfg["paths"]["provenance"]
    return _provenance_matches_question_version(provenance_path, cfg)


def _deduped_paths_exist(output_dir: Path, cfg: dict[str, Any]) -> bool:
    paths = [output_dir / cfg["paths"]["sample_deduped"].format(task_type=task_type) for task_type in cfg["task_types"]]
    if not (all(path.exists() for path in paths) and any(path.stat().st_size > 0 for path in paths)):
        return False
    expected_version = cfg.get("generation", {}).get("question_generation_version")
    if not expected_version:
        return True
    for path in paths:
        for sample in iter_jsonl(path):
            metadata = sample.get("metadata") if isinstance(sample, dict) else {}
            if isinstance(metadata, dict) and metadata.get("question_generation_version") != expected_version:
                return False
    return True


def _provenance_matches_question_version(path: Path, cfg: dict[str, Any]) -> bool:
    expected_version = cfg.get("generation", {}).get("question_generation_version")
    if not expected_version:
        return True
    if not path.exists():
        return False
    for item in iter_jsonl(path):
        metadata = item.get("metadata") if isinstance(item, dict) else {}
        if isinstance(metadata, dict) and metadata.get("question_generation_version") != expected_version:
            return False
    return True


def _load_deduped_samples(output_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for task_type in cfg["task_types"]:
        path = output_dir / cfg["paths"]["sample_deduped"].format(task_type=task_type)
        samples.extend(item for item in iter_jsonl(path) if isinstance(item, dict))
    return samples


def _build_model_pool(cfg: dict[str, Any]) -> ModelPool:
    cfg = deep_merge(
        cfg,
        {
            "vlm_pool": {
                "provider_defaults": {
                    "min_interval_seconds": float(cfg["runtime"].get("model_min_interval_seconds", 0.0)),
                    "max_image_bytes": int(cfg["generation"].get("max_image_bytes", 0)),
                    "max_image_side": int(cfg["generation"].get("max_image_side", 1600)),
                    "image_jpeg_quality": int(cfg["generation"].get("image_jpeg_quality", 85)),
                }
            }
        },
    )
    return ModelPool.from_config(cfg, cfg["prompts"])


def _generate_samples_for_contexts(
    *,
    contexts: list[Any],
    output_dir: Path,
    cfg: dict[str, Any],
    model_pool: ModelPool | None,
    state: RunState,
) -> list[dict[str, Any]]:
    raw_samples: list[dict[str, Any]] = []
    workers = max(1, int(cfg["runtime"].get("model_workers", 1)))
    max_pending = max(workers, int(cfg["runtime"].get("model_max_pending", workers)))
    if workers == 1 or len(contexts) <= 1:
        for context in contexts:
            try:
                samples = generate_for_context(context, output_dir, cfg, model_pool)
                for sample in samples:
                    raw_samples.append(sample)
                    append_jsonl(output_dir / cfg["paths"]["sample_raw"].format(task_type=context.task_type), sample)
            except Exception as exc:
                state.error({"stage": "generate", "pdf_id": context.pdf_job.pdf_id, "task_type": context.task_type, "error": str(exc)})
        return raw_samples
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Any, Any] = {}
        context_iter = iter(contexts)
        exhausted = False
        while futures or not exhausted:
            while not exhausted and len(futures) < max_pending:
                try:
                    context = next(context_iter)
                except StopIteration:
                    exhausted = True
                    break
                futures[executor.submit(generate_for_context, context, output_dir, cfg, model_pool)] = context
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                context = futures.pop(future)
                try:
                    samples = future.result()
                except Exception as exc:
                    state.error({"stage": "generate", "pdf_id": context.pdf_job.pdf_id, "task_type": context.task_type, "error": str(exc)})
                    continue
                for sample in samples:
                    raw_samples.append(sample)
                    append_jsonl(output_dir / cfg["paths"]["sample_raw"].format(task_type=context.task_type), sample)
    return raw_samples


def _process_pdf(pdf_job: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(pdf_job.output_dir)
    logger = configure_logger(f"{cfg['logger_name']}.{pdf_job.pdf_id}", output_dir / cfg["paths"]["pipeline_log"])
    state = RunState(logger, output_dir, cfg)
    model_pool = None if bool(cfg["runtime"]["skip_model"]) else _build_model_pool(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(output_dir / cfg["paths"]["manifest"], asdict(pdf_job))
    try:
        if bool(cfg["runtime"].get("reuse_exports")) and _exports_exist(output_dir, cfg):
            return {"pdf_id": pdf_job.pdf_id, "output_dir": str(output_dir), "sample_count": 0, "reused": "exports"}
        page_assets = render_pages(pdf_job, cfg)
        state.checkpoint("render_pages", len(page_assets))
        _, content, parsed_path, image_map = parse_pdf_with_mineru(pdf_job, cfg)
        state.checkpoint("mineru", parsed_path)
        units = normalize_units(pdf_job, page_assets, content, parsed_path, image_map, cfg)
        state.checkpoint("normalize", len(units))
        if bool(cfg["runtime"].get("reuse_samples")) and _deduped_paths_exist(output_dir, cfg):
            deduped = _load_deduped_samples(output_dir, cfg)
        else:
            contexts = build_task_contexts(pdf_job, units, cfg)
            raw_samples = _generate_samples_for_contexts(contexts=contexts, output_dir=output_dir, cfg=cfg, model_pool=model_pool, state=state)
            validated: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            for sample in raw_samples:
                valid = validate_sample(sample, output_dir, cfg)
                if valid is None:
                    rejected.append(sample)
                else:
                    validated.append(valid)
            for task_type in cfg["task_types"]:
                write_jsonl(output_dir / cfg["paths"]["sample_validated"].format(task_type=task_type), (sample for sample in validated if sample.get("task_type") == task_type))
                write_jsonl(output_dir / cfg["paths"]["sample_rejected"].format(task_type=task_type), (sample for sample in rejected if sample.get("task_type") == task_type))
            deduped = deduplicate(validated, cfg)
            for task_type in cfg["task_types"]:
                write_jsonl(output_dir / cfg["paths"]["sample_deduped"].format(task_type=task_type), (sample for sample in deduped if sample.get("task_type") == task_type))
        export_task_files(output_dir, deduped, cfg)
        state.increment("pdfs_completed")
        state.increment("samples_exported", len(deduped))
        state.checkpoint("export", cfg["statuses"]["done"])
        return {"pdf_id": pdf_job.pdf_id, "output_dir": str(output_dir), "sample_count": len(deduped)}
    except Exception as exc:
        state.error({"stage": "pdf", "pdf_id": pdf_job.pdf_id, "error": str(exc)})
        logger.exception("failed pdf_id=%s", pdf_job.pdf_id)
        return {"pdf_id": pdf_job.pdf_id, "output_dir": str(output_dir), "sample_count": 0, "error": str(exc)}


def run_pipeline(options: PipelineOptions | None = None) -> list[dict[str, Any]]:
    options = options or PipelineOptions()
    cfg = _apply_options(load_config(options.config_path), options)
    cfg = _resolve_runtime_paths(cfg, options)
    input_dir = Path(cfg["runtime"]["input_dir"])
    output_root = Path(cfg["runtime"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    input_pdfs = cfg["runtime"].get("input_pdfs")
    if isinstance(input_pdfs, list) and input_pdfs:
        pdf_jobs = scan_input_pdfs(input_pdfs, output_root, cfg)
    else:
        pdf_jobs = scan_pdfs(input_dir, output_root, cfg)
    if options.limit_pdfs is not None:
        pdf_jobs = pdf_jobs[: options.limit_pdfs]
    workers = max(1, int(cfg["runtime"].get("pdf_workers", 1)))
    if workers == 1 or len(pdf_jobs) <= 1:
        return [_process_pdf(pdf_job, cfg) for pdf_job in pdf_jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_pdf, pdf_job, cfg): index for index, pdf_job in enumerate(pdf_jobs)}
        ordered_results: list[dict[str, Any] | None] = [None] * len(pdf_jobs)
        for future in as_completed(futures):
            ordered_results[futures[future]] = future.result()
    return [result for result in ordered_results if result is not None]
