# Step 0: 数据收集与处理

## 概述
Step 0 的目标是从 GitHub issue 中提取代码片段和结构化信息，通过 LLM 处理生成可执行的测试用例。

## 工作流程

### 1️⃣ 第一步：爬取 Issue 列表 (`1-issue-crawler.py`)
**功能**：从 GitHub 仓库爬取已关闭的 issue HTML 页面并缓存

**输入**：
- GitHub 仓库 URL（如：`https://github.com/pytorch/pytorch`）
- 要爬取的页数

**输出**：
- `data/{framework}_issue/` 文件夹中存储 HTML 文件
- 文件命名格式：`{issue_id}_issue_ori_data.html`

**运行示例**：
```bash
python 1-issue-crawler.py --url "https://github.com/pytorch/pytorch" --pages 10
```

---

### 2️⃣ 第二步：提取代码片段 (`2-file-processor.py`)
**功能**：从 issue HTML 中提取 title、description 和代码片段

**输入**：
- 第一步生成的 HTML 文件

**输出**：
- `data/{framework}_processed_data/` 文件夹中存储 JSON 文件
- 字段包括：title, issue_id, bug_description, best_code 等

**处理逻辑**：
- 优先提取包含 PyTorch/TensorFlow 关键字的代码片段
- 如果没有优先框架，选择最长的代码片段
- 提取 issue title 和问题描述

**运行示例**：
```bash
python 2-file-processor.py --input-dir data/pytorch_issue --output-dir data/pytorch_processed_data
```

---

### 3️⃣ 第三步：生成 LLM 提示词 (`3-prompt-generator.py`)
**功能**：基于提取的信息生成结构化的 LLM 提示词

**输入**：
- 第二步生成的 JSON 文件

**输出**：
- `data/{framework}_prompts/` 文件夹中存储 `.txt` 文件
- 包含详细的 LLM prompt，指导 LLM 提取结构化信息

**Prompt 内容**：
- 要求 LLM 以 JSON 格式返回：
  - error_description（错误描述）
  - repro_code（可复现代码）
  - repro_output（输出）
  - trigger_conditions（触发条件：backend、dtype、input_shape）
  - affected_api（受影响的 API）
  - environment（环境信息）
  - labels（标签）

**运行示例**：
```bash
python 3-prompt-generator.py --input-dir data/pytorch_processed_data --output-dir data/pytorch_prompts
```

---

### 4️⃣ 第四步：调用 LLM 处理 (`4-llm-processor.py`)
**功能**：调用智谱 AI (Zhipu GLM-4.7) LLM 处理提示词，生成结构化信息

**输入**：
- 第三步生成的 prompt 文件
- API Key（需从环境变量或配置中获取）

**输出**：
- `data/{framework}_llm_processed_data/` 文件夹中存储 JSON 文件
- 包含 LLM 返回的结构化信息
- 同时保存 LLM 的输入输出日志在 `llm_io/` 子文件夹

**处理流程**：
- 批量处理多个 prompt 文件
- 添加错误处理和重试机制
- 记录处理时间和成功/失败状态

**运行示例**：
```bash
python 4-llm-processor.py --prompt-dir data/pytorch_prompts --output-dir data/pytorch_llm_processed_data --api-key YOUR_API_KEY
```

---

### 🔧 第四步（备选）：修复解析错误 (`4-fix-parse-errors.py`)
**功能**：修复 LLM 返回的 JSON 格式错误或解析失败的记录

**输入**：
- 第四步生成的 JSON 文件中标记为 parse_error=true 的文件

**输出**：
- 修复后的 JSON 文件

**修复策略**：
- 提取 JSON 有效部分
- 重新调用 LLM 修复格式
- 确保结构完整性

**运行示例**：
```bash
python 4-fix-parse-errors.py --input-dir data/pytorch_llm_processed_data --output-dir data/pytorch_llm_processed_data_fixed
```

---

### 5️⃣ 第五步：生成测试用例 (`5-test-case-generator.py`)
**功能**：从处理后的数据中提取代码片段，分类生成可执行的测试用例

**输入**：
- 第四步生成的 JSON 文件（已处理的数据）

**输出**：
- `data/{framework}_test_case/` 文件夹，包含 4 个子文件夹：
  - `executable/` - 可执行的 Python 代码
  - `non_executable/` - 语法有效但有错误的代码
  - `non_code/` - 错误信息、文档等非代码内容
  - `empty/` - 空文件

**分类逻辑**：
1. 空文件 → `empty`
2. 包含错误关键字（error, failed, 💥 等）→ `non_code`
3. 包含代码关键字（import, def, class 等）且通过编译 → `executable`
4. 包含代码关键字但编译失败 → `non_executable`

**运行示例**：
```bash
# 处理单个文件
python 5-test-case-generator.py --input-dir data/pytorch_llm_processed_data/159972_issue_ori_data.json --output-dir data/pytorch_test_case

# 批量处理整个目录
python 5-test-case-generator.py --input-dir data/pytorch_llm_processed_data --output-dir data/pytorch_test_case --overwrite
```

---

## 完整运行流程

```bash
# 1. 爬取 issue
python 1-issue-crawler.py --url "https://github.com/pytorch/pytorch" --pages 50

# 2. 提取代码片段
python 2-file-processor.py --input-dir data/pytorch_issue --output-dir data/pytorch_processed_data

# 3. 生成 prompt
python 3-prompt-generator.py --input-dir data/pytorch_processed_data --output-dir data/pytorch_prompts

# 4. 调用 LLM
python 4-llm-processor.py --prompt-dir data/pytorch_prompts --output-dir data/pytorch_llm_processed_data --api-key YOUR_API_KEY

# 5. 生成测试用例
python 5-test-case-generator.py --input-dir data/pytorch_llm_processed_data --output-dir data/pytorch_test_case
```

---

## 数据格式说明

### JSON 最终结构示例
```json
{
  "title": "问题标题",
  "issue_id": "159972",
  "bug_description": "详细的bug描述",
  "apis": ["torch.api1", "torch.api2"],
  "best_code": "代码片段",
  "error_description": "错误描述",
  "repro_code": "可复现的最小代码",
  "repro_output": "预期输出",
  "affected_api": "直接相关的API",
  "trigger_conditions": {
    "backend": "MPS/CUDA/CPU",
    "dtype": "float32/float64",
    "input_shape": "128",
    "other_conditions": "其他条件"
  },
  "environment": {
    "pytorch_version": "2.8.0",
    "python_version": "3.10",
    "os": "Linux",
    "hardware": "GPU/CPU"
  }
}
```

---

## 文件夹结构

```
Step1-DataCollection/
├── 1-issue-crawler.py           # 爬虫脚本
├── 2-file-processor.py          # 提取器
├── 3-prompt-generator.py        # Prompt 生成
├── 4-llm-processor.py           # LLM 处理
├── 4-fix-parse-errors.py        # 错误修复
├── 5-test-case-generator.py     # 测试用例生成
├── data/
│   ├── pytorch_issue/           # HTML 原始数据
│   ├── pytorch_processed_data/  # 处理后的 JSON
│   ├── pytorch_prompts/         # LLM Prompt
│   ├── pytorch_llm_processed_data/  # LLM 输出
│   └── pytorch_test_case/       # 最终测试用例
│       ├── executable/
│       ├── non_executable/
│       ├── non_code/
│       └── empty/
└── README.md
```

---

## 支持的框架

目前支持的框架：
- PyTorch (`pytorch_*`)
- TensorFlow (`tensorflow_*`)

可通过修改文件名前缀支持其他框架。
