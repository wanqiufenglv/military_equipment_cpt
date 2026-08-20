from __future__ import annotations

import json


TASK_TYPES = [
    "cpt_page_reconstruction",
    "cpt_document_rewrite",
    "cpt_attribute_to_text",
    "cpt_single_image_brief",
    "sft_single_image_vqa",
    "cpt_multi_image_brief",
    "cpt_entity_attribute_sentences",
    "domain_knowledge_corpus",
]

PROMPTS = {
    "system": (
        "你是装备资料多模态增量预训练式 SFT 数据构建专家。所有输出必须严格依据当前 PDF、"
        "页面图片、OCR、图注、正文和解析证据，不得引入外部知识，不得预设固定装备参数字段。"
    ),
    "cpt_page_reconstruction": (
        "请根据页面截图、OCR 和版面解析证据，生成页面资料文本重建样本。"
        "每个页面都要尽量完整重建；输出必须是资料化文本，不要写成问答，不要补充页面中没有的信息。"
    ),
    "cpt_document_rewrite": (
        "请根据 PDF 解析文本、OCR、图注和相邻上下文，生成原文清洗与资料段落重建样本。"
        "要求清理断行、噪声和重复，但不得改变专业含义。"
    ),
    "cpt_attribute_to_text": (
        "请根据资料中实际出现的属性、参数、编号、数值、单位、条件、备注或说明片段，"
        "生成自然、准确且相对完整的资料说明。不要一个参数生成一条孤立说明；应尽量把输入证据中出现的多个参数、"
        "单位、条件、备注和相互关系合并介绍。不得预设参数字段。"
    ),
    "cpt_single_image_brief": (
        "请根据单张图片、图片描述、图注、OCR 和相邻正文，先由大模型生成 3 个不同问题，"
        "再由大模型分别生成对应答案，形成 3 条单图文 ShareGPT 样本。问题和答案都应主要围绕图片内容展开，"
        "问题必须围绕图片主体、外观、标注、场景、图中可见细节或资料说明关系自然提出，不要套用固定模板。"
        "问题中不得出现图片装备的具体型号名称；如果图片描述、图注、OCR 或相邻正文能够确认具体型号名称，"
        "具体型号名称必须写在答案中。答案必须说明该图片展示或描述的是哪类或哪型装备；能够确认具体名称时写出具体名称，"
        "只能确认类别时写出类别名称，不得省略装备名称，也不得编造名称。"
        "问题主要针对图片内容本身进行提问，不要出现“根据旁边的文字信息”“旁边的文字”“其附带的文字说明”“附带的文字说明”“表格”“上下文说明”“上下文”“相邻正文”“文字内容”“OCR”等来源或文本位置表达，"
        "也不要把上下文信息或文字说明来源写进问题；上下文只作为作答参考。答案必须以图片可见内容为主，"
        "结合图注、OCR 和相邻正文中的可确认信息补充说明；不得编造图片或资料中没有的信息。"
        "答案重点是描述图片内容，并概括该装备在外观结构、可见部件、场景用途、标注信息或资料中可确认参数方面的特点；"
        "可在描述基础上顺带简单介绍当前输入中实际出现且与图片相关的参数，参数类型不固定，不要预设或枚举固定参数字段，没有可确认参数时不要编造。"
        "答案正文只写自然的资料说明，不要出现“证据”“上下文”“图片描述”“文字证据”“上述内容”“输入信息”"
        "“提供的信息”“相邻正文”“OCR”“根据上述”“依据以上”等来源提示字眼。"
    ),
    "sft_single_image_vqa": (
        "请根据单张图片、图片描述、图注、OCR 和相邻正文，先由大模型生成 3 个围绕图片装备参数的自然问题，"
        "再由大模型分别生成对应答案，形成 3 条单图 VQA ShareGPT 样本。问题必须基于图片内容本身以及相关装备参数提出，"
        "以图中可见装备、部件、标注、场景或装备身份作为提问入口，自然引出相关参数提问；不要生成脱离参数的泛泛图片描述题。"
        "参数可以来自图片描述、图注、OCR 或相邻正文中可确认且与图片装备相关的信息。"
        "答案开头必须明确给出图片装备名称，并包含对图片内容的简单描述，但描述应简短，不要像 cpt_single_image_brief 那样展开详细图片说明。"
        "能够确认具体名称时写出具体名称，只能确认类别时写出类别名称，不得省略装备名称，也不得编造名称。"
        "再进入参数问答主体；该任务重点是问答，不是长篇图片描述。"
        "问题应结合图中可见对象与当前输入中实际出现且与图片相关的参数提问，参数类型不固定，不要预设或枚举固定参数字段。"
        "问题不要出现“表格”“上下文说明”“上下文”“旁边的文字”“相邻正文”“文字内容”“OCR”等来源或文本位置表达。"
        "答案应在点明装备名称或类别并简要描述图片内容后，直接回答问题中的参数点，不要把图片内容描述写成主体。"
        "所有参数必须来自图片描述、图注、OCR 或相邻正文中的可确认信息；没有可确认参数时应说明未见明确参数，不得编造。"
        "答案正文不要出现“证据”“上下文”“图片描述”“文字证据”“上述内容”“输入信息”“提供的信息”“相邻正文”“OCR”“根据上述”“依据以上”等来源提示字眼。"
    ),
    "cpt_multi_image_brief": (
        "请根据当前输入中实际包含的多张图片、图片描述、图注、OCR 和相邻正文，先由大模型生成 3 个不同问题，"
        "再由大模型分别生成对应答案，形成 3 条多图综合 ShareGPT 样本。问题必须围绕多图共同点、差异、"
        "互补关系、连续关系、细节对照或资料说明关系自然提出，不要套用固定模板。问题中不得出现图片装备的具体型号名称；"
        "如果图片描述、图注、OCR 或相邻正文能够确认具体型号名称，具体型号名称必须写在答案中。"
        "问题中不要出现“表格”“上下文说明”“上下文”“旁边的文字”“相邻正文”“文字内容”“OCR”等来源或文本位置表达，"
        "也不要把上下文信息写进问题；上下文只作为作答参考。答案必须以多张图片可见内容为主，"
        "结合图注、OCR 和相邻正文中的可确认信息补充说明。"
        "答案正文只写自然的综合资料说明，"
        "本任务每条样本必须包含路由传入的全部图片，图片数量可以是 2 张或 3 张；当输入为 3 张图片时，必须综合 3 张图片整体作答，不要退化为只比较其中 2 张。"
        "并可在综合图片内容描述基础上顺带简单介绍当前输入中实际出现且与图片相关的参数；参数类型不固定，不要预设或枚举固定参数字段，没有可确认参数时不要编造。"
        "不要机械逐图罗列，不要忽略任何输入图片，也不要出现“证据”“上下文”“图片描述”“文字证据”“上述内容”"
        "“输入信息”“提供的信息”“相邻正文”“OCR”“根据上述”“依据以上”等来源提示字眼。"
    ),
    "cpt_entity_attribute_sentences": (
        "请根据资料证据生成实体-属性知识句。应尽量覆盖输入证据中包含的所有实体、属性、数值、单位、条件、"
        "备注和限定范围；内容要丰富，不要只输出一两条过短事实。每条知识句只表达一个清晰事实，所有信息必须来自当前证据。"
    ),
    "domain_knowledge_corpus": (
        "你是装备资料领域知识注入语料清洗助手。请把 PDF 解析后的装备资料正文、参数、图注、表格文字和说明性段落"
        "处理成适合继续预训练的纯文本语料。不要生成问答、instruction/input/output、messages 或 conversations；"
        "不要补充资料中没有的外部知识、推测、作战建议或弱点分析。可以修正明显 OCR 噪声、断行、重复页眉页脚和页码；"
        "保留装备名称、型号、参数、单位、条件、说明、标题层级和语义完整段落。最终导出 JSONL 每行只能包含 text 字段，"
        "用于 LLaMA-Factory stage: pt。"
    ),
    "validator": "请检查训练样本是否忠实、格式合法、证据充分、没有外部知识注入或安全边界问题。",
}

CFG = {
    "pipeline_version": "military_equipment_cpt_v1",
    "encoding": "utf-8",
    "sharegpt_image_token": "<image>",
    "pdf_suffix": ".pdf",
    "pdf_page_regex": r"/Type\s*/Page\b",
    "timestamp_format": "%Y-%m-%dT%H:%M:%SZ",
    "hash_chunk_size": 1048576,
    "logger_name": "military_equipment_cpt",
    "statuses": {"ready": "ready", "done": "done"},
    "unknown_dataset_id": "equipment_dataset",
    "unknown_pdf_id": "untitled_pdf",
    "default_region_id": "page",
    "default_unit_type": "page",
    "paths": {
        "manifest": "manifest.jsonl",
        "pages_manifest": "pages/page_index.jsonl",
        "page_images": "images/pages/p{page_no:04d}.png",
        "extracted_images": "images/extracted",
        "image_map": "mineru/image_map.json",
        "mineru_raw": "mineru/raw/{pdf_id}.json",
        "mineru_parsed": "mineru/parsed/{pdf_id}.json",
        "normalized_page": "normalized/p{page_no:04d}.json",
        "prompts": "prompts/{task_type}.jsonl",
        "raw_responses": "raw_responses/{task_type}.jsonl",
        "sample_raw": "samples/raw/{task_type}.jsonl",
        "sample_validated": "samples/validated/{task_type}.jsonl",
        "sample_deduped": "samples/deduped/{task_type}.jsonl",
        "sample_rejected": "samples/rejected/{task_type}.jsonl",
        "export_sharegpt": "exports/sharegpt/{task_type}.jsonl",
        "export_alpaca": "exports/alpaca/{task_type}.jsonl",
        "export_pt": "exports/pt/{task_type}.jsonl",
        "dataset_info": "exports/dataset_info.json",
        "provenance": "provenance/provenance_index.jsonl",
        "pipeline_log": "logs/pipeline.log",
        "errors": "logs/errors.jsonl",
        "metrics": "logs/metrics.json",
        "checkpoints": "logs/checkpoints.json",
    },
    "extracted_image_naming": {
        "template": "p{page_label}_img{image_no:03d}{suffix}",
        "unknown_page_label": "unknown",
        "allowed_suffixes": [".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    },
    "task_types": TASK_TYPES,
    "export_formats": {
        "sharegpt": ["cpt_page_reconstruction", "cpt_single_image_brief", "sft_single_image_vqa", "cpt_multi_image_brief"],
        "alpaca": ["cpt_document_rewrite", "cpt_attribute_to_text", "cpt_entity_attribute_sentences"],
        "pt": ["domain_knowledge_corpus"],
    },
    "block_types": {
        "title": "title",
        "section": "section_title",
        "section_title": "section_title",
        "text": "text",
        "paragraph": "text",
        "image": "image",
        "figure": "image",
        "figure_caption": "caption",
        "caption": "caption",
        "table": "table",
        "table_caption": "caption",
        "formula": "formula",
        "equation": "formula",
        "list": "list",
        "header": "header",
        "footer": "footer",
        "page_number": "page_number",
        "reference": "reference",
        "unknown": "unknown",
    },
    "runtime": {
        "input_dir": "military_equipment_cpt/data",
        "input_pdfs": [],
        "output_root": "military_equipment_cpt/outputs",
        "recursive": False,
        "pdf_workers": 1,
        "model_workers": 4,
        "page_workers": 2,
        "model_max_pending": 8,
        "model_min_interval_seconds": 0.0,
        "reuse_mineru": True,
        "skip_mineru": False,
        "reuse_normalized": True,
        "reuse_samples": True,
        "reuse_exports": True,
        "skip_model": False,
        "render_pages": True,
    },
    "render": {"dpi": 180, "image_format": "png", "max_side": 2200, "retry_count": 2},
    "mineru": {
        "url": "http://192.168.78.36:7086",
        "backend": "hybrid-auto-engine",
        "parse_method": "ocr",
        "lang_list": ["ch"],
        "timeout": 2400,
        "formula_enable": True,
        "table_enable": True,
        "return_md": True,
        "return_middle_json": True,
        "return_content_list": True,
        "return_images": True,
        "response_format_zip": False,
        "return_original_file": False,
    },
    "vlm_pool": {
        "strategy": "least_busy_weighted_fallback",
        "fallback": {"enabled": True, "max_attempts": 2, "cooldown_seconds": 300},
        "providers": [
            {
                "name": "fx_q3_235",
                "url": "https://mcc-pre.3xmt.com/gateway/ai-service/v1/chat/completions",
                "model": "fx-q3-235",
                "api_key": "sk-2h1RwMjqYcdh6F5Fs5",
                "stream": False,
                "temperature": 0.4,
                "max_tokens": 8192,
                "timeout": 2400,
                "chat_template_kwargs": {"enable_thinking": False},
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 1,
                "max_concurrency": 4,
            },
            {
                "name": "Qwen3.6-27B",
                "url": "http://192.168.78.36:3012/v1/chat/completions",
                "model": "Qwen3.6-27B",
                "api_key": "sk-bveYeVn6NAdRRElTWCqhtyJbkTL5XwweedczV9FJ05kDqhqX",
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 8192,
                "timeout": 2400,
                "chat_template_kwargs": {"enable_thinking": False},
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 2,
                "max_concurrency": 4,
            },
            {
                "name": "InternVL3_5-38B",
                "url": "http://192.168.78.35:8879/intern-vl/v1/chat/completions",
                "model": "InternVL3_5-38B",
                "api_key": "",
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 8192,
                "timeout": 2400,
                "chat_template_kwargs": {},
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 3,
                "max_concurrency": 6,
            },
        ],
    },
    "routing": {
        "enabled_tasks": {task_type: True for task_type in TASK_TYPES},
        "min_page_text_chars": 120,
        "min_unit_text_chars": 12,
        "min_document_text_chars": 300,
        "min_domain_corpus_text_chars": 120,
        "multi_image_window": 3,
        "multi_image_group_size": 3,
        "document_window": 4,
        "domain_corpus_window": 4,
        "domain_corpus_target_input_chars": 3600,
        "visual_unit_types": ["image", "table", "mixed"],
        "text_unit_types": ["text", "ocr", "caption", "mixed"],
        "domain_corpus_unit_types": ["title", "section_title", "text", "ocr", "caption", "table", "formula", "list", "mixed"],
        "domain_corpus_skip_unit_types": ["header", "footer", "page_number", "reference", "unknown"],
        "attribute_keywords": [
            "参数", "指标", "性能", "尺寸", "重量", "质量", "速度", "高度", "航程", "射程", "功率",
            "载荷", "编号", "型号", "单位", "范围", "直径", "长度", "宽度", "数量", "时间",
        ],
    },
    "generation": {
        "question_generation_version": "question_answer_model_only_image_grounded_triples_v17",
        "samples_per_job": {
            **{task_type: 1 for task_type in TASK_TYPES},
            "cpt_single_image_brief": 3,
            "sft_single_image_vqa": 3,
            "cpt_multi_image_brief": 3,
        },
        "max_context_chars": 16000,
        "max_pt_context_chars": 10000,
        "max_unit_chars": 3000,
        "max_image_bytes": 2097152,
        "max_image_side": 1600,
        "image_jpeg_quality": 85,
    },
    "validation": {
        "min_instruction_chars": 4,
        "min_output_chars": 12,
        "min_output_chars_by_task": {
            "cpt_page_reconstruction": 120,
            "cpt_attribute_to_text": 180,
            "sft_single_image_vqa": 30,
            "cpt_multi_image_brief": 80,
            "cpt_entity_attribute_sentences": 240,
            "domain_knowledge_corpus": 80
        },
        "min_pt_text_chars": 80,
        "default_quality_score": 0.7,
        "max_samples_per_page_task": 8,
        "max_samples_per_page_task_by_task": {
            "cpt_multi_image_brief": 12,
        },
        "required_payload_fields": {
            "cpt_page_reconstruction": ["conversations"],
            "cpt_document_rewrite": ["instruction", "input", "output"],
            "cpt_attribute_to_text": ["instruction", "input", "output"],
            "cpt_single_image_brief": ["conversations"],
            "sft_single_image_vqa": ["conversations"],
            "cpt_multi_image_brief": ["conversations"],
            "cpt_entity_attribute_sentences": ["instruction", "input", "output"],
            "domain_knowledge_corpus": ["text"],
        },
        "image_required_tasks": [
            "cpt_page_reconstruction",
            "cpt_single_image_brief",
            "sft_single_image_vqa",
            "cpt_multi_image_brief",
        ],
        "single_image_tasks": ["cpt_single_image_brief", "sft_single_image_vqa"],
        "multi_image_tasks": ["cpt_multi_image_brief"],
        "reject_patterns": ["攻击方案", "打击建议", "弱点分析", "作战建议", "规避侦察"],
        "answer_reject_patterns_by_task": {
            "cpt_single_image_brief": [
                "证据", "上下文", "图片描述", "文字证据", "上述内容", "输入信息", "提供的信息",
                "相邻正文", "根据上述", "依据上述", "根据以上", "依据以上", "从以上", "由以上",
                "如上", "资料显示", "材料显示", "文本显示", "文字显示", "图注显示", "OCR",
            ],
            "sft_single_image_vqa": [
                "证据", "上下文", "图片描述", "文字证据", "上述内容", "输入信息", "提供的信息",
                "相邻正文", "根据上述", "依据上述", "根据以上", "依据以上", "从以上", "由以上",
                "如上", "资料显示", "材料显示", "文本显示", "文字显示", "图注显示", "OCR",
            ],
            "cpt_multi_image_brief": [
                "证据", "上下文", "图片描述", "文字证据", "上述内容", "输入信息", "提供的信息",
                "相邻正文", "根据上述", "依据上述", "根据以上", "依据以上", "从以上", "由以上",
                "如上", "资料显示", "材料显示", "文本显示", "文字显示", "图注显示", "OCR",
            ],
        },
        "question_reject_patterns_by_task": {
            "cpt_single_image_brief": [
                "表格", "上下文说明", "上下文", "旁边的文字", "旁边文字", "相邻正文",
                "旁边的文字信息", "根据旁边", "附带的文字说明", "其附带的文字说明",
                "文字内容", "文字说明", "OCR", "图注", "根据上述", "依据以上",
            ],
            "sft_single_image_vqa": [
                "表格", "上下文说明", "上下文", "旁边的文字", "旁边文字", "相邻正文",
                "文字内容", "文字说明", "OCR", "图注", "根据上述", "依据以上",
            ],
            "cpt_multi_image_brief": [
                "表格", "上下文说明", "上下文", "旁边的文字", "旁边文字", "相邻正文",
                "文字内容", "文字说明", "OCR", "图注", "根据上述", "依据以上",
            ],
        },
    },
    "prompts": PROMPTS,
}

CONFIG_JSON = json.dumps(CFG, ensure_ascii=False, indent=2)
