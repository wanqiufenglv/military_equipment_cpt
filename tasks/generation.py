from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.io_utils import append_jsonl, clean_text, safe_name, text_sha256, truncate_text, utc_now
from ..core.models import TaskContext, TrainingSample
from ..services.clients import ModelPool, parse_json_objects


TASK_FORMATS = {
    "cpt_page_reconstruction": "sharegpt",
    "cpt_document_rewrite": "alpaca",
    "cpt_attribute_to_text": "alpaca",
    "cpt_single_image_brief": "sharegpt",
    "sft_single_image_vqa": "sharegpt",
    "cpt_multi_image_brief": "sharegpt",
    "cpt_entity_attribute_sentences": "alpaca",
    "domain_knowledge_corpus": "pt",
}


def _source_pages(context: TaskContext) -> list[int]:
    return sorted({unit.page_no for unit in context.units}) or [context.page_no]


def _sample_id(context: TaskContext, sample_no: int, payload: dict[str, Any]) -> str:
    digest = text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))[:8]
    dataset_id = safe_name(context.pdf_job.dataset_id, "dataset")
    pdf_id = safe_name(context.pdf_job.pdf_id, "pdf")
    region_id = safe_name(context.region_id, "region")
    return f"{context.task_type}__{dataset_id}__{pdf_id}__p{context.page_no:04d}__{region_id}__{sample_no:04d}__{digest}"


def _metadata(context: TaskContext, response_model: str, response_provider: str) -> dict[str, Any]:
    first = context.units[0] if context.units else None
    return {
        "dataset_id": context.pdf_job.dataset_id,
        "pdf_id": context.pdf_job.pdf_id,
        "source_pdf": context.pdf_job.source_pdf,
        "page_no": context.page_no,
        "source_pages": _source_pages(context),
        "region_id": context.region_id,
        "bbox": first.bbox if first else [],
        "image_paths": list(context.image_paths),
        "page_image_path": first.page_image_path if first else "",
        "mineru_block_ids": [block_id for unit in context.units for block_id in unit.mineru_block_ids],
        "evidence_text": context.evidence_text,
        "generator_model": response_model,
        "generator_provider": response_provider,
        "created_at": utc_now("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_version": context.pdf_job.pipeline_version,
    }


def _instruction_for(task_type: str) -> str:
    instructions = {
        "cpt_page_reconstruction": "请根据页面图片整理其中的资料内容。",
        "cpt_document_rewrite": "请将以下资料证据清洗并重建为连续、准确的资料文本。",
        "cpt_attribute_to_text": "请将以下属性或参数片段整理为自然语言资料说明。",
        "cpt_single_image_brief": "请围绕单张图片内容生成自然问题，并回答图中装备及其特点。",
        "sft_single_image_vqa": "请基于图中可见装备内容和相关装备参数生成问答，答案简述图片内容后回答参数。",
        "cpt_multi_image_brief": "请围绕多张图片组合生成自然问题，并基于多图可见内容作答。",
        "cpt_entity_attribute_sentences": "请根据以下资料生成实体-属性知识句。",
        "domain_knowledge_corpus": "请将以下装备资料证据清洗为继续预训练使用的纯文本语料。",
    }
    return instructions[task_type]


def _task_output_requirements(task_type: str) -> list[str]:
    requirements = {
        "cpt_page_reconstruction": [
            "这是页面级任务，每个输入页面都要重建，不能因为页面信息少就跳过。",
            "输出应覆盖页面中可确认的标题、正文、图注、表格、编号、参数和备注。",
            "保持资料表达，不要改成问答答案。",
        ],
        "cpt_document_rewrite": [
            "清理 OCR 噪声、断行和重复内容。",
            "保留原资料中出现的名称、编号、数值、单位、条件和备注。",
        ],
        "cpt_attribute_to_text": [
            "尽量覆盖输入证据中出现的多个参数、属性、数值、单位、条件和备注。",
            "不要一个参数一句极短说明；应组织成较完整的资料说明。",
            "如果多个参数存在同一实体或同一主题下，应合并介绍它们的关系和限定条件。",
        ],
        "cpt_single_image_brief": [
            "先由模型生成 3 个不同问题，再由模型分别生成答案，最终返回 3 条 ShareGPT 样本。",
            "问题和答案都必须根据单张图片、图片描述、图注、OCR 和相邻正文生成，并主要围绕图片内容展开，不要使用固定模板问题。",
            "问题必须主要针对图片内容本身提问，不要出现具体型号名称，也不要出现“根据旁边的文字信息”“旁边的文字”“其附带的文字说明”“附带的文字说明”“表格”“上下文说明”“上下文”“相邻正文”“OCR”等来源或文本位置表达。",
            "上下文、图注、OCR 和文字说明只作为生成答案的参考，不要把这些来源写进问题。",
            "答案以图片可见信息为主，结合文字信息补充说明，不要只描述图片外观。",
            "答案必须说明该图片展示或描述的是哪类或哪型装备；如果输入信息能够确认图片装备的具体型号名称，答案必须明确写出该具体型号名称，只能确认类别时写出类别名称，不得省略装备名称，也不得编造名称。",
            "答案重点是图片内容描述，并概括该装备在外观结构、可见部件、场景用途、标注信息或资料中可确认参数方面的特点；可在描述基础上顺带简单介绍当前输入中实际出现且与图片相关的参数。",
            "参数类型不固定，不要预设或枚举固定参数字段，没有可确认参数时不要编造。",
            "输出中必须包含对该图片本身的描述内容，例如可见主体、外观、位置、场景、标注或图中可确认元素。",
            "答案正文不要出现“证据”“上下文”“图片描述”“文字证据”“上述内容”“输入信息”“提供的信息”“相邻正文”“OCR”“根据上述”“依据以上”等来源提示字眼。",
            "每条样本只能包含 1 张图片，不能把多张图片合并进单图任务。",
        ],
        "sft_single_image_vqa": [
            "先由模型生成 3 个不同问题，再由模型分别生成答案，最终返回 3 条 ShareGPT 样本。",
            "本任务重点是围绕当前输入中实际出现且与图片相关的参数进行问答，不是生成长篇图片描述。",
            "问题必须基于图片内容本身以及相关装备参数提出，以单张图片中的可见主体、外观部件、标注位置、场景或装备身份作为提问入口，并围绕可确认的装备参数提问。",
            "不要使用固定模板问题，也不要生成脱离参数的泛泛图片描述题。",
            "参数类型不固定，不要预设或枚举固定参数字段；问题由当前输入中实际出现的图中可见对象和参数共同决定。",
            "问题不要出现“表格”“上下文说明”“上下文”“旁边的文字”“相邻正文”“OCR”等来源或文本位置表达。",
            "答案开头必须明确给出图片装备名称，并包含对图片内容的简单描述；描述应简短，不要像 cpt_single_image_brief 那样展开详细图片说明。能够确认具体名称时写出具体名称，只能确认类别时写出类别名称，不得省略装备名称，也不得编造名称。",
            "随后直接回答问题中的参数点，不要把图片内容描述写成主体。",
            "参数必须来自图片描述、图注、OCR 或相邻正文中的可确认信息；没有可确认参数时应说明未见明确参数，不得编造。",
            "每条样本只能包含 1 张图片。",
            "答案正文不要出现“证据”“上下文”“图片描述”“文字证据”“上述内容”“输入信息”“提供的信息”“相邻正文”“OCR”“根据上述”“依据以上”等来源提示字眼。",
        ],
        "cpt_multi_image_brief": [
            "先由模型生成 3 个不同问题，再由模型分别生成答案，最终返回 3 条 ShareGPT 样本。",
            "问题和答案都必须根据多张图片、图片描述、图注、OCR 和相邻正文生成，不要使用固定模板问题。",
            "问题必须主要围绕多张图片本身提问，不要出现具体型号名称，也不要出现“表格”“上下文说明”“上下文”“旁边的文字”“相邻正文”“OCR”等来源或文本位置表达。",
            "上下文、图注和 OCR 只作为生成答案的参考，不要把这些来源写进问题。",
            "本任务处理当前输入中实际包含的多张图片，图片数量以路由传入的 image_paths 为准。",
            "多图任务每条样本必须包含路由传入的全部图片；图片数量可以是 2 张或 3 张，3 张图片输入必须综合 3 张图片整体生成问答，不要退化为只比较其中 2 张。",
            "输出必须综合图片之间的共同点、差异、互补关系或连续关系。",
            "答案以多张图片可见信息为主，结合文字信息补充说明。",
            "如果输入信息能够确认图片装备的具体型号名称，答案必须明确写出相关具体型号名称。",
            "答案重点是综合图片内容描述，可在描述基础上顺带简单介绍当前输入中实际出现且与图片相关的参数；参数类型不固定，不要预设或枚举固定参数字段，没有可确认参数时不要编造。",
            "不要机械逐图罗列，也不要忽略任何一张输入图片。",
            "输出中必须包含对每张输入图片的描述内容，并说明这些图片之间的关系。",
            "答案正文不要出现“证据”“上下文”“图片描述”“文字证据”“上述内容”“输入信息”“提供的信息”“相邻正文”“OCR”“根据上述”“依据以上”等来源提示字眼。",
        ],
        "cpt_entity_attribute_sentences": [
            "尽量覆盖输入证据中包含的所有实体-属性事实。",
            "输出多条较完整的知识句，不要只给一两条过短句子。",
            "每条知识句应保留实体、属性、数值、单位、条件、备注或限定范围中可确认的内容。",
        ],
        "domain_knowledge_corpus": [
            "输出用于 LLaMA-Factory stage: pt 的纯文本，不要生成问答、instruction、input、output、messages 或 conversations。",
            "只保留资料证据中已有的装备名称、型号、参数、单位、条件、说明、标题层级、表格文字和图注信息。",
            "清理 OCR 断行、重复页眉页脚、页码和明显噪声，不得添加外部知识、作战建议、弱点分析或推测。",
        ],
    }
    return requirements[task_type]


def _expected_sample_count(task_type: str, cfg: dict[str, Any]) -> int:
    return int(cfg.get("generation", {}).get("samples_per_job", {}).get(task_type, 1))


def _expected_schema(task_type: str, image_count: int = 1) -> str:
    if TASK_FORMATS[task_type] == "pt":
        return '{"text":"用于继续预训练的纯文本"}'
    if TASK_FORMATS[task_type] == "sharegpt":
        image_tokens = "\\n".join("<image>" for _ in range(max(1, image_count)))
        image_list = ",".join('"..."' for _ in range(max(1, image_count)))
        if task_type in {"cpt_single_image_brief", "sft_single_image_vqa", "cpt_multi_image_brief"}:
            return (
                f'[{{"conversations":[{{"from":"human","value":"{image_tokens}\\n问题1"}},'
                f'{{"from":"gpt","value":"答案1"}}],"images":[{image_list}]}},'
                f'{{"conversations":[{{"from":"human","value":"{image_tokens}\\n问题2"}},'
                f'{{"from":"gpt","value":"答案2"}}],"images":[{image_list}]}},'
                f'{{"conversations":[{{"from":"human","value":"{image_tokens}\\n问题3"}},'
                f'{{"from":"gpt","value":"答案3"}}],"images":[{image_list}]}}]'
            )
        return (
            '{"conversations":[{"from":"human","value":"<image>\\n..."},'
            '{"from":"gpt","value":"资料说明文本"}],"images":["..."]}'
        )
    return '{"instruction":"...","input":"原始证据文本","output":"清洗或整理后的资料文本"}'


def build_prompt(context: TaskContext, cfg: dict[str, Any]) -> str:
    payload = {
        "task_type": context.task_type,
        "target_format": TASK_FORMATS[context.task_type],
        "format_schema": _expected_schema(context.task_type, len(context.image_paths)),
        "expected_sample_count": _expected_sample_count(context.task_type, cfg),
        "task_intent_hint": _instruction_for(context.task_type),
        "task_output_requirements": _task_output_requirements(context.task_type),
        "evidence_text": truncate_text(context.evidence_text, int(cfg["generation"]["max_context_chars"])),
        "image_paths": context.image_paths,
        "source_pages": _source_pages(context),
        "source": context.source,
    }
    return (
        f"{cfg['prompts'][context.task_type]}\n\n"
        "生成要求：\n"
        "1. 只使用 Input JSON 中的证据，不得补充资料中没有的信息。\n"
        "2. 不得预设固定装备参数字段；输出结构由当前证据决定。\n"
        "3. 不要生成作战建议、攻击方案、弱点分析或超出资料范围的推测。\n"
        "4. 问题/指令必须由你根据当前证据自行生成，保持表达多样性，不要复制 task_intent_hint。\n"
        "5. 同一任务的不同样本应使用不同问法，可从整理、重建、说明、归纳、提取等角度自然变化。\n"
        "6. ShareGPT 格式必须返回 conversations；human.value 中的问题必须是自生成问题，并包含正确数量的 <image>。\n"
        "7. Alpaca 格式必须返回 instruction；instruction 必须是自生成问题或任务指令。\n"
        "8. 对图像任务，必须利用 source.image_description 或 source.image_descriptions 中的图片描述证据。\n"
        "9. cpt_single_image_brief、sft_single_image_vqa 和 cpt_multi_image_brief 必须在本次大模型输出中先生成 3 个问题，再逐题生成答案；不要复用固定问题或外部预设答案。\n"
        "10. cpt_single_image_brief、sft_single_image_vqa 和 cpt_multi_image_brief 必须返回 JSON 数组，数组长度必须等于 expected_sample_count；每个对象对应 1 个自生成问题和 1 个模型生成答案。\n"
        "11. cpt_single_image_brief 的 human.value 问题必须主要针对图片内容本身提问，不要出现“根据旁边的文字信息”“其附带的文字说明”等来源表达；cpt_multi_image_brief 的 human.value 问题必须主要围绕图片内容提问；sft_single_image_vqa 的 human.value 问题必须基于图片内容本身和相关装备参数提问。问题中不要出现“表格”“上下文说明”“上下文”“旁边的文字”“相邻正文”“文字内容”“OCR”“图注”等来源或文本位置表达。\n"
        "12. cpt_single_image_brief 和 cpt_multi_image_brief 的 human.value 问题不要出现具体型号名称；如果能确认具体型号名称，答案必须写出具体型号名称。\n"
        "13. cpt_single_image_brief 的重点是单张图片内容描述，答案必须说明图中是什么装备，并概括该装备的可确认特点；cpt_multi_image_brief 的重点是综合全部输入图片内容描述，输入 3 张图片时必须覆盖 3 张图片整体；sft_single_image_vqa 的重点是图片装备参数问答，答案必须包含对图片内容的简单描述，但描述不要像 cpt_single_image_brief 那样详细，再回答参数问题。三类任务都不得预设或枚举固定参数字段；答案中不要写“证据”“上下文”“图片描述”“文字证据”“上述内容”“输入信息”“提供的信息”“相邻正文”“OCR”“根据上述”“依据以上”等来源提示字眼。\n"
        "14. 只返回合法 JSON 对象或 JSON 数组，不要使用 Markdown 代码块。\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _generated_instruction(row: dict[str, Any], context: TaskContext) -> str:
    value = clean_text(row.get("instruction") or row.get("question") or row.get("prompt") or row.get("user_prompt"))
    if not value:
        return ""
    return value


def _strip_image_tokens(value: Any) -> str:
    text = str(value or "")
    for token in ("<image>", "<|image|>"):
        text = text.replace(token, "")
    return text.strip()


def _extract_answer_text(row: dict[str, Any]) -> str:
    for key in ("output", "answer", "text", "brief", "content", "summary", "description", "result", "reconstruction", "page_reconstruction"):
        value = row.get(key)
        if isinstance(value, str) and clean_text(value):
            return clean_text(value)
    return ""


def _normalize_payload(row: dict[str, Any], context: TaskContext) -> dict[str, Any] | None:
    target_format = TASK_FORMATS[context.task_type]
    if target_format == "sharegpt":
        expected_images = list(context.image_paths)
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            text = _extract_answer_text(row)
            if not text:
                return None
            instruction = _generated_instruction(row, context)
            if not instruction:
                return None
            image_tokens = "\n".join("<image>" for _ in context.image_paths)
            conversations = [
                {"from": "human", "value": f"{image_tokens}\n{_strip_image_tokens(instruction)}".strip()},
                {"from": "gpt", "value": text},
            ]
        else:
            human_values = [
                clean_text(item.get("value"))
                for item in conversations
                if isinstance(item, dict) and item.get("from") == "human"
            ]
            if not human_values:
                return None
            human_without_image = " ".join(_strip_image_tokens(value) for value in human_values)
            if not clean_text(human_without_image):
                return None
            image_tokens = "\n".join("<image>" for _ in expected_images)
            normalized = []
            human_rewritten = False
            for item in conversations:
                if not isinstance(item, dict):
                    continue
                copied = dict(item)
                if copied.get("from") == "human" and not human_rewritten:
                    text = _strip_image_tokens(copied.get("value", ""))
                    copied["value"] = f"{image_tokens}\n{text}".strip()
                    human_rewritten = True
                elif copied.get("from") == "human":
                    copied["value"] = _strip_image_tokens(copied.get("value", ""))
                normalized.append(copied)
            conversations = normalized
        return {"conversations": conversations, "images": expected_images}
    instruction = _generated_instruction(row, context)
    if not instruction:
        return None
    input_text = clean_text(row.get("input") or context.evidence_text)
    output = row.get("output")
    if not isinstance(output, str):
        output = json.dumps(row.get("sentences") or row.get("result") or row.get("content") or row, ensure_ascii=False)
    if not clean_text(output):
        return None
    payload = {"instruction": instruction, "input": input_text, "output": clean_text(output)}
    if context.image_paths:
        payload["images"] = list(context.image_paths)
    return payload


def _clean_pt_corpus_text(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line.upper() == "OSID:":
            continue
        if re.fullmatch(r"(?:page\s*)?\d{1,4}", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def _truncate_pt_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    breakpoint = max(cut.rfind("\n\n"), cut.rfind("\n"), cut.rfind("。"), cut.rfind("."))
    if breakpoint > limit // 2:
        return cut[: breakpoint + 1].strip()
    return cut


def _page_range(context: TaskContext) -> str:
    pages = _source_pages(context)
    if not pages:
        return str(context.page_no)
    return str(pages[0]) if pages[0] == pages[-1] else f"{pages[0]}-{pages[-1]}"


def _generate_domain_corpus_for_context(context: TaskContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw_text = str(context.source.get("raw_text") or context.evidence_text or "")
    max_chars = int(cfg.get("generation", {}).get("max_pt_context_chars", cfg.get("generation", {}).get("max_context_chars", 10000)))
    body = _truncate_pt_text(_clean_pt_corpus_text(raw_text), max_chars)
    if not body:
        return []
    payload = {
        "text": (
            f"文档：{context.pdf_job.pdf_id}\n"
            f"页码：{_page_range(context)}\n\n"
            f"{body}"
        )
    }
    metadata = _metadata(context, "rule", "heuristic")
    metadata["question_generation_version"] = cfg.get("generation", {}).get("question_generation_version", "")
    metadata["target_format"] = "pt"
    metadata["llamafactory_stage"] = "pt"
    sample = TrainingSample(
        sample_id=_sample_id(context, 1, payload),
        task_type=context.task_type,
        format="pt",
        payload=payload,
        metadata=metadata,
    )
    return [asdict(sample)]


def generate_for_context(context: TaskContext, output_dir: Path, cfg: dict[str, Any], model_pool: ModelPool | None) -> list[dict[str, Any]]:
    if context.task_type == "domain_knowledge_corpus":
        return _generate_domain_corpus_for_context(context, cfg)
    if model_pool is None or bool(cfg["runtime"]["skip_model"]):
        return []
    prompt = build_prompt(context, cfg)
    append_jsonl(output_dir / cfg["paths"]["prompts"].format(task_type=context.task_type), {"task_type": context.task_type, "prompt": prompt, "metadata": asdict(context)})
    image_paths = [output_dir / image for image in context.image_paths if (output_dir / image).exists()]
    response = model_pool.chat(context.task_type, prompt, image_paths)
    append_jsonl(
        output_dir / cfg["paths"]["raw_responses"].format(task_type=context.task_type),
        {"task_type": context.task_type, "text": response.text, "provider": response.provider_name, "model": response.model, "metadata": asdict(context)},
    )
    rows = parse_json_objects(response.text)
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        payload = _normalize_payload(row, context)
        if payload is None:
            continue
        metadata = _metadata(context, response.model, response.provider_name)
        metadata["question_generation_version"] = cfg.get("generation", {}).get("question_generation_version", "")
        sample = TrainingSample(
            sample_id=_sample_id(context, index, payload),
            task_type=context.task_type,
            format=TASK_FORMATS[context.task_type],
            payload=payload,
            metadata=metadata,
        )
        samples.append(asdict(sample))
    return samples
