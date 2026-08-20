from __future__ import annotations

from pathlib import Path

from military_equipment_cpt.core.config_loader import load_config
from military_equipment_cpt.core.io_utils import iter_jsonl, read_json
from military_equipment_cpt.core.models import DocumentUnit, PdfJob
from military_equipment_cpt.app.cli import build_parser
from military_equipment_cpt.services.mineru import parse_pdf_with_mineru
from military_equipment_cpt.tasks.dedup import deduplicate
from military_equipment_cpt.tasks.exporters import export_task_files
from military_equipment_cpt.tasks.generation import build_prompt, generate_for_context
from military_equipment_cpt.tasks.routing import build_task_contexts
from military_equipment_cpt.tasks.validation import validate_sample


def _pdf_job(tmp_path: Path) -> PdfJob:
    return PdfJob(
        dataset_id="dataset",
        pdf_id="pdf_12345678",
        source_pdf=str(tmp_path / "a.pdf"),
        output_dir=str(tmp_path),
        file_hash="12345678",
        file_size=1,
        page_count=1,
        task_types=load_config()["task_types"],
        status="ready",
        created_at="2026-06-27T00:00:00Z",
        pipeline_version="test",
    )


def test_load_config_has_eight_tasks() -> None:
    cfg = load_config()
    assert cfg["task_types"] == [
        "cpt_page_reconstruction",
        "cpt_document_rewrite",
        "cpt_attribute_to_text",
        "cpt_single_image_brief",
        "sft_single_image_vqa",
        "cpt_multi_image_brief",
        "cpt_entity_attribute_sentences",
        "domain_knowledge_corpus",
    ]
    assert cfg["export_formats"]["pt"] == ["domain_knowledge_corpus"]
    assert "sft_single_image_vqa" in cfg["export_formats"]["sharegpt"]
    assert "domain_knowledge_corpus" in cfg["prompts"]
    assert cfg["runtime"]["skip_mineru"] is False


def test_cli_accepts_skip_mineru_flag() -> None:
    args = build_parser().parse_args(["--skip-mineru"])

    assert args.skip_mineru is True


def test_skip_mineru_requires_existing_outputs(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["runtime"]["skip_mineru"] = True
    job = _pdf_job(tmp_path)

    try:
        parse_pdf_with_mineru(job, cfg)
    except FileNotFoundError as exc:
        assert "skip_mineru=true requires existing MinerU outputs" in str(exc)
    else:
        raise AssertionError("skip_mineru=true should fail when MinerU outputs are missing")


def test_routing_builds_text_contexts(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["routing"]["min_page_text_chars"] = 10
    cfg["routing"]["min_document_text_chars"] = 10
    job = _pdf_job(tmp_path)
    unit = DocumentUnit(
        dataset_id=job.dataset_id,
        pdf_id=job.pdf_id,
        source_pdf=job.source_pdf,
        page_no=1,
        unit_id="p0001_r0001",
        unit_type="text",
        region_id="r0001",
        text="型号 A 的参数为 10 kg，尺寸为 2 m。",
        evidence_text="型号 A 的参数为 10 kg，尺寸为 2 m。",
        page_image_path="images/pages/p0001.png",
    )
    contexts = build_task_contexts(job, [unit], cfg)
    task_types = {context.task_type for context in contexts}
    assert "cpt_document_rewrite" in task_types
    assert "cpt_attribute_to_text" in task_types
    assert "cpt_entity_attribute_sentences" in task_types


def test_routing_builds_domain_knowledge_corpus_context(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["routing"]["min_domain_corpus_text_chars"] = 20
    cfg["routing"]["domain_corpus_target_input_chars"] = 80
    cfg["routing"]["domain_corpus_window"] = 2
    job = _pdf_job(tmp_path)
    units = [
        DocumentUnit(
            dataset_id=job.dataset_id,
            pdf_id=job.pdf_id,
            source_pdf=job.source_pdf,
            page_no=1,
            unit_id="p0001_r0001",
            unit_type="text",
            region_id="r0001",
            text="Fujian carrier displacement and propulsion overview. " * 2,
            evidence_text="Fujian carrier displacement and propulsion overview. " * 2,
            page_image_path="images/pages/p0001.png",
        ),
        DocumentUnit(
            dataset_id=job.dataset_id,
            pdf_id=job.pdf_id,
            source_pdf=job.source_pdf,
            page_no=2,
            unit_id="p0002_r0001",
            unit_type="text",
            region_id="r0001",
            text="Electromagnetic launch system operating notes and sortie data.",
            evidence_text="Electromagnetic launch system operating notes and sortie data.",
            page_image_path="images/pages/p0002.png",
        ),
    ]

    contexts = build_task_contexts(job, units, cfg)
    domain_contexts = [context for context in contexts if context.task_type == "domain_knowledge_corpus"]

    assert domain_contexts
    assert domain_contexts[0].image_paths == []
    assert "raw_text" in domain_contexts[0].source
    assert "Fujian carrier" in domain_contexts[0].evidence_text


def test_multi_image_routing_builds_pair_and_three_image_groups(tmp_path: Path) -> None:
    cfg = load_config()
    for task_type in cfg["routing"]["enabled_tasks"]:
        cfg["routing"]["enabled_tasks"][task_type] = task_type == "cpt_multi_image_brief"
    cfg["routing"]["multi_image_window"] = 1
    cfg["routing"]["multi_image_group_size"] = 3
    job = _pdf_job(tmp_path)
    unit = DocumentUnit(
        dataset_id=job.dataset_id,
        pdf_id=job.pdf_id,
        source_pdf=job.source_pdf,
        page_no=1,
        unit_id="p0001_r0001",
        unit_type="image",
        region_id="r0001",
        image_paths=["images/extracted/a.png", "images/extracted/b.png", "images/extracted/c.png"],
        caption="三张装备图片组合说明。",
        evidence_text="三张装备图片组合说明。",
    )

    contexts = build_task_contexts(job, [unit], cfg)

    assert [context.image_paths for context in contexts] == [
        ["images/extracted/a.png", "images/extracted/b.png"],
        ["images/extracted/a.png", "images/extracted/c.png"],
        ["images/extracted/b.png", "images/extracted/c.png"],
        ["images/extracted/a.png", "images/extracted/b.png", "images/extracted/c.png"],
    ]


def test_multi_image_dedup_keeps_twelve_pair_and_three_image_samples(tmp_path: Path) -> None:
    cfg = load_config()
    samples = []
    for index in range(12):
        pair_index = index // 3
        groups = [
            ["images/extracted/a.png", "images/extracted/b.png"],
            ["images/extracted/a.png", "images/extracted/c.png"],
            ["images/extracted/b.png", "images/extracted/c.png"],
            ["images/extracted/a.png", "images/extracted/b.png", "images/extracted/c.png"],
        ]
        image_tokens = "\n".join("<image>" for _ in groups[pair_index])
        samples.append(
            {
                "sample_id": f"id-{index}",
                "task_type": "cpt_multi_image_brief",
                "format": "sharegpt",
                "payload": {
                    "conversations": [
                        {"from": "human", "value": f"{image_tokens}\n问题 {index}"},
                        {"from": "gpt", "value": f"答案 {index}，用于形成唯一 payload。"},
                    ],
                    "images": groups[pair_index],
                },
                "metadata": {"dataset_id": "dataset", "pdf_id": "pdf", "page_no": 1},
            }
        )

    assert len(deduplicate(samples, cfg)) == 12


def test_image_brief_prompt_requires_three_model_generated_samples(tmp_path: Path) -> None:
    cfg = load_config()
    job = _pdf_job(tmp_path)
    context = next(
        item
        for item in build_task_contexts(
            job,
            [
                DocumentUnit(
                    dataset_id=job.dataset_id,
                    pdf_id=job.pdf_id,
                    source_pdf=job.source_pdf,
                    page_no=1,
                    unit_id="p0001_r0001",
                    unit_type="image",
                    region_id="r0001",
                    image_paths=["images/extracted/a.png"],
                    caption="单张装备图片说明。",
                    evidence_text="单张装备图片说明。",
                )
            ],
            cfg,
        )
        if item.task_type == "cpt_single_image_brief"
    )

    prompt = build_prompt(context, cfg)

    assert cfg["generation"]["samples_per_job"]["cpt_single_image_brief"] == 3
    assert '"expected_sample_count": 3' in prompt
    assert "必须返回 JSON 数组" in prompt
    assert "先生成 3 个问题" in prompt
    assert "逐题生成答案" in prompt
    assert "问题中不得出现图片装备的具体型号名称" in prompt
    assert "答案必须说明该图片展示或描述的是哪类或哪型装备" in prompt
    assert "答案必须明确写出该具体型号名称" in prompt
    assert "概括该装备在外观结构、可见部件、场景用途、标注信息或资料中可确认参数方面的特点" in prompt
    assert "答案必须说明图中是什么装备" in prompt
    assert "问题和答案都必须根据单张图片" in prompt
    assert "主要围绕图片内容展开" in prompt
    assert "问题必须主要针对图片内容本身提问" in prompt
    assert "根据旁边的文字信息" in prompt
    assert "其附带的文字说明" in prompt
    assert "旁边的文字" in prompt
    assert "参数类型不固定" in prompt
    assert "不要预设或枚举固定参数字段" in prompt


def test_sft_single_image_vqa_routes_and_prompts_three_samples(tmp_path: Path) -> None:
    cfg = load_config()
    for task_type in cfg["routing"]["enabled_tasks"]:
        cfg["routing"]["enabled_tasks"][task_type] = task_type == "sft_single_image_vqa"
    job = _pdf_job(tmp_path)
    unit = DocumentUnit(
        dataset_id=job.dataset_id,
        pdf_id=job.pdf_id,
        source_pdf=job.source_pdf,
        page_no=1,
        unit_id="p0001_r0001",
        unit_type="image",
        region_id="r0001",
        image_paths=["images/extracted/a.png"],
        caption="单张装备图片参数问答。",
        evidence_text="型号 A 最大航速 30 节。",
    )

    context = next(item for item in build_task_contexts(job, [unit], cfg) if item.task_type == "sft_single_image_vqa")
    prompt = build_prompt(context, cfg)

    assert context.image_paths == ["images/extracted/a.png"]
    assert cfg["generation"]["samples_per_job"]["sft_single_image_vqa"] == 3
    assert cfg["validation"]["min_output_chars_by_task"]["sft_single_image_vqa"] == 30
    assert '"expected_sample_count": 3' in prompt
    assert "基于图中可见装备内容和相关装备参数生成问答" in prompt
    assert "重点是围绕当前输入中实际出现且与图片相关的参数进行问答" in prompt
    assert "问题必须基于图片内容本身以及相关装备参数提出" in prompt
    assert "以单张图片中的可见主体、外观部件、标注位置、场景或装备身份作为提问入口" in prompt
    assert "围绕可确认的装备参数提问" in prompt
    assert "不要生成脱离参数的泛泛图片描述题" in prompt
    assert "human.value 问题必须基于图片内容本身和相关装备参数提问" in prompt
    assert "答案必须包含对图片内容的简单描述" in prompt
    assert "答案开头必须明确给出图片装备名称" in prompt
    assert "不要像 cpt_single_image_brief 那样展开详细图片说明" in prompt
    assert "明确给出图片装备名称" in prompt
    assert "不得省略装备名称" in prompt
    assert "不要把图片内容描述写成主体" in prompt


def test_image_brief_validation_rejects_source_cue_in_answer(tmp_path: Path) -> None:
    cfg = load_config()
    image_path = tmp_path / "images" / "extracted" / "a.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-a-real-image")
    sample = {
        "sample_id": "id",
        "task_type": "cpt_single_image_brief",
        "format": "sharegpt",
        "payload": {
            "conversations": [
                {"from": "human", "value": "<image>\n这张图片展示了什么？"},
                {"from": "gpt", "value": "相邻正文补充说明了图中装备的资料。"},
            ],
            "images": ["images/extracted/a.png"],
        },
        "metadata": {"page_no": 1},
    }

    assert validate_sample(sample, tmp_path, cfg) is None


def test_image_brief_validation_rejects_source_cue_in_question(tmp_path: Path) -> None:
    cfg = load_config()
    image_path = tmp_path / "images" / "extracted" / "a.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-a-real-image")
    sample = {
        "sample_id": "id",
        "task_type": "cpt_single_image_brief",
        "format": "sharegpt",
        "payload": {
            "conversations": [
                {"from": "human", "value": "<image>\n根据旁边的文字信息，这张图片展示了什么？"},
                {"from": "gpt", "value": "图中装备外形轮廓清晰，主体结构和标注位置可辨。"},
            ],
            "images": ["images/extracted/a.png"],
        },
        "metadata": {"page_no": 1},
    }

    assert validate_sample(sample, tmp_path, cfg) is None


def test_validate_alpaca_sample(tmp_path: Path) -> None:
    cfg = load_config()
    sample = {
        "sample_id": "id",
        "task_type": "cpt_document_rewrite",
        "format": "alpaca",
        "payload": {
            "instruction": "请清洗资料。",
            "input": "原始证据文本",
            "output": "cleaned equipment reference text with enough detail",
        },
        "metadata": {"page_no": 1},
    }
    assert validate_sample(sample, tmp_path, cfg) is not None


def test_validate_pt_sample_accepts_pure_text_payload(tmp_path: Path) -> None:
    cfg = load_config()
    sample = {
        "sample_id": "pt-id",
        "task_type": "domain_knowledge_corpus",
        "format": "pt",
        "payload": {"text": "Fujian carrier technical corpus text. " * 6},
        "metadata": {"page_no": 1, "source_pages": [1]},
    }

    valid = validate_sample(sample, tmp_path, cfg)

    assert valid is not None
    assert valid["payload"] == {"text": sample["payload"]["text"]}
    assert valid["metadata"]["validator"] == "rule"


def test_domain_knowledge_corpus_generation_omits_dataset_header(tmp_path: Path) -> None:
    cfg = load_config()
    cfg["routing"]["min_domain_corpus_text_chars"] = 20
    job = _pdf_job(tmp_path)
    unit = DocumentUnit(
        dataset_id=job.dataset_id,
        pdf_id=job.pdf_id,
        source_pdf=job.source_pdf,
        page_no=1,
        unit_id="p0001_r0001",
        unit_type="text",
        region_id="r0001",
        text="Fujian carrier technical corpus text with propulsion details. " * 3,
        evidence_text="Fujian carrier technical corpus text with propulsion details. " * 3,
    )
    context = next(
        item for item in build_task_contexts(job, [unit], cfg)
        if item.task_type == "domain_knowledge_corpus"
    )

    samples = generate_for_context(context, tmp_path, cfg, None)
    text = samples[0]["payload"]["text"]

    assert "数据集：" not in text
    assert "文档：" in text
    assert "页码：" in text


def test_export_writes_pt_file_and_dataset_info(tmp_path: Path) -> None:
    cfg = load_config()
    sample = {
        "sample_id": "pt-id",
        "task_type": "domain_knowledge_corpus",
        "format": "pt",
        "payload": {"text": "Fujian carrier technical corpus text. " * 6},
        "metadata": {"page_no": 1, "source_pages": [1]},
    }

    export_task_files(tmp_path, [sample], cfg)

    pt_path = tmp_path / "exports" / "pt" / "domain_knowledge_corpus.jsonl"
    assert list(iter_jsonl(pt_path)) == [{"text": sample["payload"]["text"].strip()}]
    dataset_info = read_json(tmp_path / "exports" / "dataset_info.json")
    assert dataset_info["equipment_cpt_domain_knowledge_corpus"]["file_name"] == "pt/domain_knowledge_corpus.jsonl"
    assert dataset_info["equipment_cpt_domain_knowledge_corpus"]["stage"] == "pt"
