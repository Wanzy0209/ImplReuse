# RQ3 Ablation Study

本目录用于补充 RQ3 的消融实验代码。RQ3 关注 TeRL 方法中两个关键组件对最终测试用例复用效果的影响：

1. 去掉核心 API 识别模块。
2. 将代码相似度替换为文档相似度。

两个脚本都会生成与 Step2 原始输出兼容的 `similar_api_list` JSON 文件，因此后续可以直接复用 Step3 的 prompt 生成、LLM 生成和结果解析流程。

## 文件说明

### `1_no_coreapi_identification.py`

该脚本对应第一个消融实验：去掉核心 API 识别。

原始 TeRL 流程中，Step1 会先从 issue 中识别一个 `core_api`，Step2 再围绕这个核心 API 查找相似 API。

该消融实验不再使用 Step1 识别出的 `core_api`，而是从 issue 自身的信息中提取所有显式出现的 API，并用这些 API 替代核心 API 作为相似 API 检索入口。

脚本会从以下字段和文本中提取候选 API：

- `apis`
- `affected_api`
- `best_code`
- `repro_code`
- `bug_description`
- `error_description`
- `repro_output`
- `title`

提取到的 API 会和原始 `core_api` 做去重，然后使用 Step2 已生成的代码相似度文件查找候选 API：

- `api_pairwise_similarity.jsonl`
- `issue_api_similarity.jsonl`
- `pytorch_pairwise_similarity.jsonl`

默认输出目录：

```text
Step5-Evaluate/RQ3/data/no_coreapi_identification/similar_api_list
```

输出 JSON 仍保持 Step3 可读取的格式：

```json
{
  "issue_id": "159974",
  "core_api": "NO_CORE_API",
  "ablation": "no_coreapi_identification",
  "original_core_api": "torch.compile",
  "replacement_apis": ["torch.randn", "torch.compile"],
  "similar_apis": {
    "pytorch_to_tensorflow": [],
    "issue_to_api": [],
    "pytorch_to_pytorch": []
  }
}
```

其中 `replacement_apis` 记录本次消融实际使用的替代 API；每个相似 API 条目中还会包含 `replacement_api` 字段，用来追踪该候选 API 是由哪个替代 API 检索得到的。

### `2_no_similarity_calculation.py`

该脚本对应第二个消融实验：将代码相似度替换为文档相似度。

原始 TeRL 流程中，Step2 使用 API 源码或实现代码计算代码相似度。该消融实验保留 Step1 的核心 API 识别结果，但不再使用代码相似度，而是直接使用 CSV 文件中预计算的 API 文档相似度。

脚本会从以下 CSV 文件读取相似度数据：

- `Step5-Evaluate/RQ3/data/api_documentation_db.csv`
- `Step5-Evaluate/RQ3/data/equivalent_api_pairs.csv`

其中 `api_documentation_db.csv` 包含 `pytorch_api`、`tf_api`、`pytorch_doc`、`tf_doc` 和 `similarity` 列，用于构建 `pytorch_to_tensorflow` 候选集。

`equivalent_api_pairs.csv` 包含显式等价 API 对，并会额外补充 `pytorch_to_tensorflow` 和 `pytorch_to_pytorch` 候选项。

默认输出目录：

```text
Step5-Evaluate/RQ3/data/doc_similarity/similar_api_list
```

输出 JSON 同样保持 Step3 可读取的格式，并在每个相似 API 条目中加入：

```json
"similarity_method": "doc_similarity_csv"
```

用于标记该分数来自 CSV 文档相似度数据。

### `3_generate_prompt_all_replacements.py`

该脚本基于 RQ3 `no_coreapi_identification` 输出，生成 prompt 文件时同时包含：

- `similar_apis` 中的 top-N 统一候选
- `similar_apis_by_replacement` 中每个替代 API 对应的候选

因此它不仅复用核心 API 的类似候选，还会为 issue 中提取出的其他替代 API 单独生成 prompt。

默认输出目录：

```text
Step5-Evaluate/RQ3/data/no_coreapi_identification/prompts_all_replacements
```

### `4_execute_prompt_all_replacements.py`

该脚本读取 prompt 文件目录中的所有 `.txt` 文件，调用 LLM 并将生成的回复与原始事件数据一起保存到：

```text
Step5-Evaluate/RQ3/data/no_coreapi_identification/llm_responses_all_replacements
```

## 运行方式

在仓库根目录运行：

```powershell
python Step5-Evaluate\RQ3\1_no_coreapi_identification.py
python Step5-Evaluate\RQ3\2_no_similarity_calculation.py
```

如果只想先测试少量 issue：

```powershell
python Step5-Evaluate\RQ3\1_no_coreapi_identification.py --limit 10
python Step5-Evaluate\RQ3\2_no_similarity_calculation.py --limit 10
```

如果想调整每个 issue 保留的相似 API 数量：

```powershell
python Step5-Evaluate\RQ3\1_no_coreapi_identification.py --top-n 10
python Step5-Evaluate\RQ3\2_no_similarity_calculation.py --top-n 10
```

## 接入 Step3

两个脚本生成的 `similar_api_list` 可以直接传给 Step3 的 prompt 生成脚本。

### 消融 1：无核心 API 识别

```powershell
python Step3-ReuseTestCase\1-generate-prompt.py `
  --similarity-dir Step5-Evaluate\RQ3\data\no_coreapi_identification\similar_api_list `
  --output-dir Step5-Evaluate\RQ3\data\no_coreapi_identification\prompts
```

### 消融 2：文档相似度

```powershell
python Step3-ReuseTestCase\1-generate-prompt.py `
  --similarity-dir Step5-Evaluate\RQ3\data\doc_similarity\similar_api_list `
  --output-dir Step5-Evaluate\RQ3\data\doc_similarity\prompts
```

之后可以继续复用 Step3 的 LLM 生成和解析流程，再将生成的测试用例交给 Step4 执行。

### `3_generate_prompt_all_replacements.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--step1-dir` | `Step1-CoreAPIIdentification/data/pytorch_core_api_identification` | Step1 issue JSON 输入目录 |
| `--test-case-dir` | `Step0-DataCollection/data/pytorch_test_case` | Step0 测试用例目录 |
| `--similarity-dir` | `Step5-Evaluate/RQ3/data/no_coreapi_identification/similar_api_list` | RQ3 `no_coreapi_identification` 输出目录 |
| `--output-dir` | `Step5-Evaluate/RQ3/data/no_coreapi_identification/prompts_all_replacements` | prompt 输出目录 |
| `--limit` | `0` | 只处理前 N 个 issue；0 表示全部处理 |

### `4_execute_prompt_all_replacements.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--prompt-dir` | `Step5-Evaluate/RQ3/data/no_coreapi_identification/prompts_all_replacements` | prompt 文本文件目录 |
| `--output-dir` | `Step5-Evaluate/RQ3/data/no_coreapi_identification/llm_responses_all_replacements` | LLM 输出目录 |
| `--api-key` | `$env:ZHIPUAI_API_KEY` | Zhipu API key |
| `--model` | `glm-4.7` | LLM 模型 |
| `--temperature` | `0.0` | 采样温度 |
| `--max-tokens` | `8192` | 最大 token 数 |
| `--limit` | `0` | 只处理前 N 个 prompt |
| `--overwrite` | 关闭 | 覆盖已存在回复文件 |

## 常用参数

### `1_no_coreapi_identification.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--step1-dir` | `Step1-CoreAPIIdentification/data/pytorch_core_api_identification` | Step1 issue JSON 输入目录 |
| `--similarity-dir` | `Step2-CalculateSimilarity/data` | Step2 相似度 JSONL 文件目录 |
| `--output-dir` | `Step5-Evaluate/RQ3/data/no_coreapi_identification/similar_api_list` | 消融结果输出目录 |
| `--top-n` | `11` | 每类候选 API 保留数量 |
| `--limit` | `0` | 只处理前 N 个 issue；0 表示全部处理 |
| `--split-by-replacement` | 关闭 | 为每个替代 API 单独输出一个 JSON 文件 |

### `2_no_similarity_calculation.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--step1-dir` | `Step1-CoreAPIIdentification/data/pytorch_core_api_identification` | Step1 issue JSON 输入目录 |
| `--doc-similarity-csv` | `Step5-Evaluate/RQ3/data/api_documentation_db.csv` | 文档相似度 CSV 文件 |
| `--equivalent-api-csv` | `Step5-Evaluate/RQ3/data/equivalent_api_pairs.csv` | 等价 API 对 CSV 文件 |
| `--pytorch-source-dir` | `Step2-CalculateSimilarity/data/pytorch-api-extracted` | 仅用于缺失文档时回退的 PyTorch API 文档目录 |
| `--output-dir` | `Step5-Evaluate/RQ3/data/doc_similarity/similar_api_list` | 消融结果输出目录 |
| `--top-n` | `11` | 每类候选 API 保留数量 |
| `--limit` | `0` | 只处理前 N 个 issue；0 表示全部处理 |
| `--no-source-fallback` | 关闭 | 禁止使用提取源代码/文档作为回退文本 |

## 输出目录建议

建议保留每个消融实验自己的输出目录，避免覆盖原始 Step2 结果：

```text
Step5-Evaluate/RQ3/data/
  no_coreapi_identification/
    similar_api_list/
    prompts/
    test_cases/
  doc_similarity/
    similar_api_list/
    prompts/
    test_cases/
```

## 注意事项

- `1_no_coreapi_identification.py` 依赖 Step2 已经生成的三个相似度 JSONL 文件。如果这些文件不存在，脚本会给出 warning，并且对应类别的候选结果可能为空。
- `2_no_similarity_calculation.py` 会为 PyTorch 和 TensorFlow API 文档建立 TF-IDF 索引。完整运行时会比普通 JSON 处理慢，但只需要在脚本启动时构建一次索引。
- 两个脚本默认不会覆盖 Step2 的正式结果，只会写入 `Step5-Evaluate/RQ3/data/...`。
- 如果输出目录中已有旧结果，脚本会覆盖同名 JSON 文件。需要保留旧实验时，请通过 `--output-dir` 指定新的目录。
