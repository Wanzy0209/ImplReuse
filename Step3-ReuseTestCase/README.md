# Step 3: 测试用例复用

## 概述
Step 3 的目标是基于 Step 0-2 的数据，通过 LLM 生成测试用例复用建议，识别可复用的现有测试用例，以及生成新的测试用例。

## 工作流程

### 1️⃣ 第一步：生成复用提示词 (`1-generate-prompt.py`)
**功能**：基于 issue 数据和相似 API 列表生成 LLM 提示词

**输入**：
- Step 0 生成的 issue 数据
- Step 2 生成的相似 API 列表

**输出**：
- `prompts/` 文件夹中存储 `.txt` 文件
- 每个 issue 生成多个提示词（带不同相似 API）

**运行示例**：
```bash
python 1-generate-prompt.py --issue-dir ../Step0-DataCollection/data/pytorch_llm_processed_data --similar-api-dir ../Step2-CalculateSimilarity/data/similar_api_list --output-dir prompts
```

---

### 2️⃣ 第二步：LLM 复用生成 (`2-llm-reuse.py`)
**功能**：调用 LLM 根据提示词生成测试用例复用建议

**输入**：
- 第一步生成的提示词文件
- API Key（需从环境变量或配置中获取）

**输出**：
- LLM 返回的测试用例复用结果

**运行示例**：
```bash
python 2-llm-reuse.py --prompt-dir prompts --api-key YOUR_API_KEY
```

---

### 3️⃣ 第三步：解析 LLM 结果 (`3-parse-llm-results.py`)
**功能**：解析 LLM 返回的复用结果，提取测试用例代码

**输入**：
- 第二步生成的 LLM 输出结果

**输出**：
- 结构化的测试用例文件
- 包含可复用测试和新生成测试

**运行示例**：
```bash
python 3-parse-llm-results.py --input-dir prompts --output-dir data/reuse_results
```

---

## 完整运行流程

```bash
# 1. 生成复用提示词
python 1-generate-prompt.py --issue-dir ../Step0-DataCollection/data/pytorch_llm_processed_data --similar-api-dir ../Step2-CalculateSimilarity/data/similar_api_list --output-dir prompts

# 2. LLM 复用生成
python 2-llm-reuse.py --prompt-dir prompts --api-key YOUR_API_KEY

# 3. 解析 LLM 结果
python 3-parse-llm-results.py --input-dir prompts --output-dir data/reuse_results
```

---

## 文件夹结构

```
Step3-ReuseTestCase/
├── 1-generate-prompt.py   # 生成复用提示词
├── 2-llm-reuse.py         # LLM 复用生成
├── 3-parse-llm-results.py # 解析 LLM 结果
├── prompts/               # 提示词文件目录
│   ├── {issue_id}_issue_to_api_0.txt
│   ├── {issue_id}_issue_to_api_1.txt
│   └── ...
├── data/
│   └── reuse_results/     # 复用结果
└── README.md
```

---

## 支持的框架

- PyTorch
- TensorFlow
