# 八类装备多模态增量预训练式 SFT/领域知识注入数据处理流水线

本目录实现一套面向装备资料 PDF 的多模态 SFT 与领域知识注入数据生成流水线。系统将 PDF 转为页面图片，调用 MinerU 做 OCR/版面解析，标准化为 `DocumentUnit`，再按八类任务生成 ShareGPT、Alpaca 或 `stage: pt` 纯文本样本，并保留完整溯源、日志、checkpoint 和中间文件。

## 1. 处理流程

```text
PDF
  -> 页面图片渲染
  -> MinerU/OCR/版面解析
  -> DocumentUnit 标准化
  -> 八类任务路由
  -> 大模型生成 SFT 样本 / 本地清洗 PT 语料
  -> 格式校验、证据校验、安全边界校验
  -> 去重
  -> ShareGPT/Alpaca/PT 导出
```

代码入口：

- `app/cli.py`：命令行入口。
- `app/pipeline.py`：端到端流水线。
- `config.py`：默认配置、任务类型、MinerU 和大模型接口。
- `core/`：数据结构、配置加载、IO、日志。
- `processing/`：PDF 扫描、页面渲染、标准化。
- `services/`：MinerU 和 OpenAI-compatible 大模型接口。
- `tasks/`：任务路由、生成、校验、去重、导出。

## 2. 快速执行

默认读取 `military_equipment_cpt/data` 下的 PDF，输出到 `military_equipment_cpt/outputs`：

```powershell
python -m military_equipment_cpt.app.cli
```

指定输入目录和输出目录：

```powershell
python -m military_equipment_cpt.app.cli `
  --input-dir military_equipment_cpt/data `
  --output-root military_equipment_cpt/outputs
```
``
python app\cli.py --task cpt_single_image_brief,cpt_multi_image_brief,sft_single_image_vqa  --input-dir data --output-root outputs --pdf-workers 1  --page-workers 2 --model-workers 2 --skip-mineru

```
指定单个 PDF：

```powershell
python -m military_equipment_cpt.app.cli `
  --pdf military_equipment_cpt/data/example.pdf
```

指定数据集 ID：

```powershell
python -m military_equipment_cpt.app.cli `
  --pdf equipment_set_a=military_equipment_cpt/data/example.pdf
```

指定数据集 ID 和 PDF ID：

```powershell
python -m military_equipment_cpt.app.cli `
  --pdf equipment_set_a:carrier_doc=military_equipment_cpt/data/example.pdf
```

只跑前处理，不调用大模型：

```powershell
python -m military_equipment_cpt.app.cli --skip-model
```

MinerU 已经提取完成，只复用已有 MinerU 结果并跳过 MinerU 服务调用：

```powershell
python -m military_equipment_cpt.app.cli --skip-mineru
```

## 3. 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 可选配置文件路径。支持 `.py`、`.json`、`.yaml`、`.yml`，会覆盖 `config.py` 默认配置。 |
| `--input-dir` | PDF 输入目录。未指定时使用 `runtime.input_dir`。 |
| `--pdf` | 指定 PDF，可重复传入。格式：`path`、`dataset_id=path`、`dataset_id:pdf_id=path`。 |
| `--output-root` | 输出根目录。未指定时使用 `runtime.output_root`。 |
| `--reuse-mineru` | 复用已有 MinerU 解析结果。 |
| `--skip-mineru` | 跳过 MinerU 服务调用，只读取已有 `mineru/raw/{pdf_id}.json`、`mineru/parsed/{pdf_id}.json` 和 `mineru/image_map.json`；缺失任一文件会报错。 |
| `--skip-model` | 跳过大模型样本生成，只执行 PDF 扫描、渲染、MinerU、标准化等前处理。 |
| `--pdf-workers` | PDF 级并发数，对应 `runtime.pdf_workers`。 |
| `--model-workers` | 大模型调用并发数，对应 `runtime.model_workers`。 |
| `--page-workers` | 页面渲染并发数，对应 `runtime.page_workers`。 |
| `--min-page-text-chars` | 页面级任务最小证据文本长度，对应 `routing.min_page_text_chars`。 |
| `--min-unit-text-chars` | 局部证据单元最小文本长度，对应 `routing.min_unit_text_chars`。 |
| `--task` | 只运行指定任务。可重复传入，也可逗号分隔。 |
| `--tasks` | 同 `--task`。 |
| `--limit-pdfs` | 限制处理 PDF 数量，调试时常用。 |

示例：

```powershell
python -m military_equipment_cpt.app.cli `
  --task cpt_document_rewrite,cpt_attribute_to_text `
  --model-workers 2 `
  --page-workers 4
```

## 4. 输入规范

默认输入目录：

```text
military_equipment_cpt/
  data/
    example.pdf
```

系统默认只扫描当前目录下的 `.pdf` 文件。若需要递归扫描子目录，在配置文件中设置：

```json
{
  "runtime": {
    "recursive": true
  }
}
```

输入 PDF 会生成：

- `dataset_id`：默认取 PDF 父目录名，也可通过 `--pdf dataset_id=path` 指定。
- `pdf_id`：默认由 PDF 文件名 slug 和文件 hash 前 8 位组成，也可通过 `--pdf dataset_id:pdf_id=path` 指定。
- `file_hash`：PDF 内容 SHA256，用于溯源和稳定命名。

## 5. 输出目录

每个 PDF 单独一个输出目录：

```text
outputs/
  {dataset_id}_{pdf_id}/
    manifest.jsonl
    pages/
      page_index.jsonl
    images/
      pages/
        p0001.png
      extracted/
    mineru/
      raw/{pdf_id}.json
      parsed/{pdf_id}.json
      image_map.json
    normalized/
      p0001.json
    prompts/
      {task_type}.jsonl
    raw_responses/
      {task_type}.jsonl
    samples/
      raw/{task_type}.jsonl
      validated/{task_type}.jsonl
      deduped/{task_type}.jsonl
      rejected/{task_type}.jsonl
    exports/
      sharegpt/{task_type}.jsonl
      alpaca/{task_type}.jsonl
      pt/domain_knowledge_corpus.jsonl
      dataset_info.json
    provenance/
      provenance_index.jsonl
    logs/
      pipeline.log
      errors.jsonl
      metrics.json
      checkpoints.json
```

关键文件说明：

- `manifest.jsonl`：PDF 级元数据。
- `pages/page_index.jsonl`：页面图片索引。
- `mineru/raw/`：MinerU 原始响应，已去除嵌入图片字段。
- `mineru/parsed/`：MinerU `content_list` 解析结果。
- `normalized/`：标准化后的 `DocumentUnit`。
- `prompts/`：发送给大模型的 prompt。
- `raw_responses/`：大模型原始响应。
- `samples/raw/`：模型输出解析后的样本。
- `samples/validated/`：规则校验通过样本。
- `samples/rejected/`：规则校验拒绝样本。
- `samples/deduped/`：去重后的最终中间样本。
- `exports/sharegpt/`：ShareGPT 训练数据。
- `exports/alpaca/`：Alpaca 训练数据。
- `exports/pt/`：继续预训练/领域知识注入纯文本数据，每行只包含 `text` 字段。
- `provenance/provenance_index.jsonl`：样本到 PDF、页面、图片、证据文本的溯源索引。

## 6. 八类任务类型

| 任务类型 | 默认导出格式 | 输入证据 | 目标 |
| --- | --- | --- | --- |
| `cpt_page_reconstruction` | ShareGPT | 页面图片、页面 OCR、版面块 | 按实际解析到的页面生成页面截图到资料文本重建样本，不预设固定页数。 |
| `cpt_document_rewrite` | Alpaca | 页面或文档片段文本、OCR、图注 | 清洗原文并重建连续资料段落。 |
| `cpt_attribute_to_text` | Alpaca | 同页内包含参数、编号、数值、单位、条件等信息的证据集合 | 尽量合并介绍多个属性/参数，不再一个参数生成一条极短说明。 |
| `cpt_single_image_brief` | ShareGPT | 单张图片、图注、相邻正文、OCR | 问题主要针对图片内容本身提问，不出现“根据旁边的文字信息”“其附带的文字说明”等来源表达；答案必须说明该图片展示或描述的是哪类或哪型装备，并概括该装备的可确认特点。 |
| `sft_single_image_vqa` | ShareGPT | 单张图片、图注、相邻正文、OCR | 问题基于图片内容本身以及相关装备参数提问；答案包含对图片内容的简单描述并给出图片装备名称，再回答对应参数，但描述不如 `cpt_single_image_brief` 详细。 |
| `cpt_multi_image_brief` | ShareGPT | 相邻页面或同一窗口内实际解析到的多张图片及文字证据 | 每条样本包含至少 2 张图片；3 张图片可生成两两组合和 3 图整体组合，每个组合生成 3 条数据。 |
| `cpt_entity_attribute_sentences` | Alpaca | 页面或多页窗口内可确认事实 | 尽量覆盖证据中包含的所有实体属性，生成内容更丰富的知识句。 |
| `domain_knowledge_corpus` | PT | 连续页面窗口内的正文、参数、表格文字、图注和说明性文本 | 清洗为 LLaMA-Factory `stage: pt` 使用的纯文本领域知识注入语料，每行只保留 `text` 字段。 |

默认启用全部任务。指定任务后，路由、模型 provider 任务白名单、中间文件和导出文件都会收窄到指定任务：

```powershell
python -m military_equipment_cpt.app.cli --task cpt_single_image_brief
python -m military_equipment_cpt.app.cli --tasks cpt_document_rewrite,cpt_entity_attribute_sentences
python -m military_equipment_cpt.app.cli --task domain_knowledge_corpus
```

SFT 样本的用户问题或任务指令由大模型根据当前证据生成。代码只向模型提供抽象任务意图和输出要求，不提供具体问题示例；校验阶段会保证图片数量、输出长度和格式正确，并优先保留页面重建与多图组合的完整性。`domain_knowledge_corpus` 不生成问答，提示词要求只保留资料中已有的装备名称、型号、参数、单位、条件、标题层级、表格文字和图注信息，并拒绝外部知识、作战建议、弱点分析或推测。

## 7. 配置文件

默认配置在 `config.py` 的 `CFG` 中。可以用外部 JSON/YAML 覆盖配置：

```powershell
python -m military_equipment_cpt.app.cli --config my_config.json
```

覆盖示例：

```json
{
  "runtime": {
    "input_dir": "military_equipment_cpt/data",
    "output_root": "military_equipment_cpt/outputs",
    "recursive": true,
    "pdf_workers": 1,
    "model_workers": 2,
    "page_workers": 4,
    "reuse_mineru": true,
    "skip_mineru": false,
    "reuse_normalized": true,
    "reuse_samples": true,
    "reuse_exports": true
  },
  "routing": {
    "min_page_text_chars": 120,
    "min_unit_text_chars": 12
  },
  "generation": {
    "max_context_chars": 6000,
    "max_image_bytes": 2097152
  }
}
```

配置文件会和默认配置做深度合并，只需要写需要覆盖的字段。

## 8. Runtime 参数

`runtime` 控制输入输出、并发和复用策略。

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `input_dir` | `military_equipment_cpt/data` | 默认 PDF 输入目录。 |
| `input_pdfs` | `[]` | 显式 PDF 列表，通常由 `--pdf` 覆盖。 |
| `output_root` | `military_equipment_cpt/outputs` | 输出根目录。 |
| `recursive` | `false` | 是否递归扫描输入目录。 |
| `pdf_workers` | `1` | PDF 级并发数。多个 PDF 可并行处理。 |
| `model_workers` | `4` | 大模型请求并发数。 |
| `page_workers` | `2` | 页面渲染并发数。 |
| `model_max_pending` | `8` | 大模型请求队列最大积压数量，用于背压控制。 |
| `model_min_interval_seconds` | `0.0` | 单 provider 请求最小间隔。 |
| `reuse_mineru` | `true` | 如果 MinerU raw/parsed/image_map 已存在，则复用。 |
| `skip_mineru` | `false` | 跳过 MinerU 服务调用，只使用已有 MinerU raw/parsed/image_map；用于 MinerU 提取已完成后继续生成或重跑下游流程。 |
| `reuse_normalized` | `true` | 如果 normalized 页面文件已存在，则复用。 |
| `reuse_samples` | `true` | 如果 deduped 样本已存在，则复用并直接导出。 |
| `reuse_exports` | `true` | 如果最终导出文件已存在，则跳过该 PDF。 |
| `skip_model` | `false` | 跳过大模型生成。 |
| `render_pages` | `true` | 是否渲染页面图片。 |

## 9. MinerU 配置

`mineru` 控制 PDF 解析服务：

| 配置项 | 说明 |
| --- | --- |
| `url` | MinerU 服务地址。代码会自动拼接 `/file_parse`。 |
| `backend` | MinerU 后端，例如 `hybrid-auto-engine`。 |
| `parse_method` | 解析方式，例如 `ocr`。 |
| `lang_list` | OCR 语言列表，默认 `["ch"]`。 |
| `timeout` | 请求超时时间，单位秒。 |
| `formula_enable` | 是否解析公式。 |
| `table_enable` | 是否解析表格。 |
| `return_md` | 是否返回 Markdown。 |
| `return_middle_json` | 是否返回中间 JSON。 |
| `return_content_list` | 是否返回 `content_list`。 |
| `return_images` | 是否返回提取图片。 |
| `response_format_zip` | 是否返回 zip 格式。 |
| `return_original_file` | 是否返回原文件。 |

如果网络环境不能访问 MinerU，流水线会在解析阶段失败。若对应 PDF 的 MinerU 输出已经完整存在，可使用 `--skip-mineru` 直接进入后续标准化、路由和样本生成流程：

```powershell
python -m military_equipment_cpt.app.cli --skip-mineru
```

调试命令行和配置时可用：

```powershell
python -m military_equipment_cpt.app.cli --skip-model --limit-pdfs 0
```

## 10. 大模型配置

大模型接口位于 `vlm_pool.providers`，兼容 OpenAI Chat Completions 格式。

| 配置项 | 说明 |
| --- | --- |
| `name` | provider 名称，用于日志和 metadata。 |
| `url` | Chat Completions URL。 |
| `model` | 模型名。 |
| `api_key` | API key。也可以改造为 `api_key_env` 使用环境变量。 |
| `stream` | 是否流式返回。 |
| `temperature` | 采样温度。 |
| `max_tokens` | 最大生成 token。 |
| `timeout` | 请求超时。 |
| `chat_template_kwargs` | 传给服务端的额外模板参数。 |
| `capabilities` | 能力列表，包含 `text`、`image`。需要图片的任务会筛选 `image` provider。 |
| `task_types` | provider 可处理的任务类型。命令行指定任务后会同步收窄。 |
| `weight` | 负载均衡权重。 |
| `max_concurrency` | 单 provider 最大并发。 |

`vlm_pool.fallback` 控制失败切换：

| 配置项 | 说明 |
| --- | --- |
| `enabled` | 是否启用 fallback。 |
| `max_attempts` | 单次请求最多尝试多少个 provider。 |
| `cooldown_seconds` | provider 失败后的冷却时间。 |

## 11. 路由参数

`routing` 控制哪些页面、证据块或图片会进入某类任务：

| 配置项 | 说明 |
| --- | --- |
| `enabled_tasks` | 每个任务的启停开关。 |
| `min_page_text_chars` | 页面级任务最小文本长度。 |
| `min_unit_text_chars` | 局部单元最小文本长度。 |
| `min_document_text_chars` | 文档窗口任务最小文本长度。 |
| `multi_image_window` | 多图任务聚合相邻页窗口大小。 |
| `multi_image_group_size` | 多图任务组合单元大小，默认 `3`；`cpt_multi_image_brief` 会从该单元生成两两组合，并在有 3 张图片时生成 3 图整体组合。 |
| `document_window` | 文档片段/知识句任务聚合相邻页窗口大小。 |
| `visual_unit_types` | 哪些 `DocumentUnit.unit_type` 视为视觉单元。 |
| `text_unit_types` | 哪些 `DocumentUnit.unit_type` 视为文本单元。 |
| `attribute_keywords` | 属性/参数片段识别关键词。仅用于判断证据是否适合任务，不预设输出字段。 |

## 12. 渲染参数

`render` 控制页面图片：

| 配置项 | 说明 |
| --- | --- |
| `dpi` | PDF 渲染 DPI。 |
| `image_format` | 页面图片格式，目前路径模板使用 PNG。 |
| `max_side` | 预留参数，用于限制最大边长。 |
| `retry_count` | 预留参数，用于渲染重试。 |

## 13. 生成与校验参数

`generation`：

| 配置项 | 说明 |
| --- | --- |
| `question_generation_version` | 问题生成策略版本。版本不匹配的旧 deduped 样本和导出结果不会被自动复用。 |
| `samples_per_job` | 每个任务上下文期望生成样本数。 |
| `max_context_chars` | 发送给模型的最大上下文字符数。 |
| `max_pt_context_chars` | `domain_knowledge_corpus` 每个 PT 清洗窗口允许保留的最大原文字符数。 |
| `max_unit_chars` | 单个证据单元最大字符数。 |
| `max_image_bytes` | 图片发送模型前的最大字节数，超过会尝试压缩。 |
| `max_image_side` | 图片压缩后的最大边长。 |
| `image_jpeg_quality` | 图片压缩 JPEG 质量。 |

`validation`：

| 配置项 | 说明 |
| --- | --- |
| `min_instruction_chars` | instruction 最小字符数。 |
| `min_output_chars` | 输出 payload 最小字符数。 |
| `min_output_chars_by_task` | 按任务覆盖最小输出长度，用于约束属性说明、实体属性句、多图说明和 PT 纯文本不要过短。 |
| `min_pt_text_chars` | `domain_knowledge_corpus` 纯文本语料最小字符数。 |
| `default_quality_score` | 规则校验通过后的默认质量分。 |
| `max_samples_per_page_task` | 同一 PDF、同一页、同一任务最大保留样本数。 |
| `max_samples_per_page_task_by_task` | 按任务覆盖页级最大保留样本数；`cpt_multi_image_brief` 默认为 `12`，避免 3 组两两组合加 1 组 3 图整体组合各 3 条时被默认上限截断。 |
| `required_payload_fields` | 各任务必须包含的 payload 字段。 |
| `image_required_tasks` | 必须包含有效图片路径的任务。 |
| `single_image_tasks` | 必须恰好包含一张图片的任务。 |
| `multi_image_tasks` | 必须包含至少 2 张图片的多图任务，图片上限由实际解析和路由窗口决定。 |
| `reject_patterns` | 安全边界拒绝关键词。 |

## 14. 断点续跑与复用

系统通过文件存在性和配置开关实现断点续跑：

- `reuse_exports=true`：如果所有目标导出文件存在且非空，直接跳过该 PDF。
- `reuse_samples=true`：如果 `samples/deduped/{task_type}.jsonl` 已存在，则跳过生成、校验、去重，直接重新导出。
- `reuse_mineru=true`：复用 `mineru/raw/`、`mineru/parsed/` 和 `mineru/image_map.json`。
- `skip_mineru=true`：不调用 MinerU；必须已经存在 `mineru/raw/{pdf_id}.json`、`mineru/parsed/{pdf_id}.json` 和 `mineru/image_map.json`。
- `reuse_normalized=true`：复用 `normalized/pXXXX.json`。
- 页面图片如果已存在，也会尽量复用。

checkpoint 和日志位置：

```text
logs/
  pipeline.log
  errors.jsonl
  metrics.json
  checkpoints.json
```

常见续跑方式：

```powershell
python -m military_equipment_cpt.app.cli --reuse-mineru
```

MinerU 已完成且不希望再次触发解析服务时：

```powershell
python -m military_equipment_cpt.app.cli --skip-mineru
```

如果只想重新导出已有去重样本，保持 `reuse_samples=true`，删除 `exports/` 下对应文件后重新运行即可。

如果要强制重新生成模型样本，需要删除对应 PDF 输出目录下的：

```text
samples/raw/
samples/validated/
samples/deduped/
samples/rejected/
exports/
```

## 15. 并发建议

推荐从保守并发开始：

```powershell
python -m military_equipment_cpt.app.cli `
  --pdf-workers 1 `
  --page-workers 2 `
  --model-workers 2
```

参数建议：

- PDF 数量较多且机器资源充足时，提高 `pdf_workers`。
- 页面渲染慢时，提高 `page_workers`，但注意内存占用。
- 大模型服务吞吐足够时，提高 `model_workers` 和 provider 的 `max_concurrency`。
- 大模型服务不稳定时，降低 `model_workers`，设置 `model_min_interval_seconds`，并保持 fallback 开启。

## 16. LLaMA-Factory 使用

导出文件已经按任务类型分开保存：

```text
exports/sharegpt/cpt_page_reconstruction.jsonl
exports/sharegpt/cpt_single_image_brief.jsonl
exports/sharegpt/sft_single_image_vqa.jsonl
exports/sharegpt/cpt_multi_image_brief.jsonl
exports/alpaca/cpt_document_rewrite.jsonl
exports/alpaca/cpt_attribute_to_text.jsonl
exports/alpaca/cpt_entity_attribute_sentences.jsonl
exports/pt/domain_knowledge_corpus.jsonl
exports/dataset_info.json
```

`dataset_info.json` 会按任务生成 LLaMA-Factory 数据集注册项。ShareGPT/Alpaca 任务用于 `stage: sft`，`domain_knowledge_corpus` 注册为 `stage: pt`，列映射为 `{"prompt": "text"}`。复制或引用对应输出目录中的 `exports/` 即可注册训练数据。

## 17. 调试命令

检查 CLI 参数：

```powershell
python -m military_equipment_cpt.app.cli --help
```

只检查配置和入口，不处理 PDF：

```powershell
python -m military_equipment_cpt.app.cli --skip-model --limit-pdfs 0
```

只处理一个 PDF：

```powershell
python -m military_equipment_cpt.app.cli --limit-pdfs 1 --skip-model
```

只生成文本任务：

```powershell
python -m military_equipment_cpt.app.cli `
  --tasks cpt_document_rewrite,cpt_attribute_to_text,cpt_entity_attribute_sentences
```

只生成领域知识注入 PT 语料：

```powershell
python -m military_equipment_cpt.app.cli  --task domain_knowledge_corpus
```

只生成多模态任务：

```powershell
python -m military_equipment_cpt.app.cli `
  --tasks cpt_page_reconstruction,cpt_single_image_brief,sft_single_image_vqa,cpt_multi_image_brief
```

## 18. 注意事项

- 真实执行需要能访问 `mineru.url` 和 `vlm_pool.providers[*].url`。
- `--skip-model` 只跳过大模型，仍会调用 MinerU。
- `--skip-mineru` 只跳过 MinerU 服务调用，仍会执行页面渲染、标准化、任务路由和后续样本流程；前提是已有 MinerU 输出完整。
- 若只想测试 CLI，不调用 MinerU，请加 `--limit-pdfs 0`。
- 任务 prompt 和 schema 不预设固定装备字段，输出应由当前证据动态决定。
- `domain_knowledge_corpus` 的 prompt 明确禁止问答格式和训练对话字段，导出 JSONL 每行只允许 `text` 字段。
- 安全边界校验会拒绝包含攻击方案、打击建议、弱点分析等内容的样本。
- 生成样本的图片路径是相对 PDF 输出目录的路径，移动数据集时需保持 `exports/` 与 `images/` 的相对关系。
