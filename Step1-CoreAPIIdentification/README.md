# Step 1: 核心 API 识别

## 概述
Step 1 的目标是从 Step 0 处理后的数据中识别出与 bug 相关的核心 API，通过 LLM 分析代码片段和错误描述，提取关键的 API 调用及其上下文信息。

## 工作流程

### 1️⃣ 第一步：核心 API 识别 (`1-core-api-identification.py`)
**功能**：从处理后的 issue 数据中识别与 bug 相关的核心 API，生成结构化的 API 信息

**输入**：
- Step 0 生成的 JSON 文件（`data/{framework}_llm_processed_data/`）

**输出**：
- `data/{framework}_core_api_identification/` 文件夹中存储 JSON 文件
- 包含识别出的核心 API、调用参数、上下文信息等

**识别逻辑**：
- 从代码片段中提取 API 调用
- 分析 API 的输入输出类型
- 识别与错误直接相关的 API
- 构建 API 调用链和依赖关系

**运行示例**：
```bash
python 1-core-api-identification.py --input-dir ../Step0-DataCollection/data/pytorch_llm_processed_data --output-dir data/pytorch_core_api_identification
```

---

### 2️⃣ 第二步：解析 LLM 结果 (`2-parse-llm-results.py`)
**功能**：解析 LLM 返回的 API 识别结果，进行数据清洗和标准化

**输入**：
- 第一步生成的 JSON 文件

**输出**：
- 标准化后的 JSON 文件，包含：
  - `core_apis` - 核心 API 列表
  - `api_context` - API 调用上下文
  - `api_relationships` - API 之间的关系

**处理逻辑**：
- 去重和合并重复的 API 识别结果
- 标准化 API 名称格式
- 验证 API 存在性
- 构建 API 依赖图

**运行示例**：
```bash
python 2-parse-llm-results.py --input-dir data/pytorch_core_api_identification --output-dir data/pytorch_core_api_identification
```

---

## 完整运行流程

```bash
# 1. 核心 API 识别
python 1-core-api-identification.py --input-dir ../Step0-DataCollection/data/pytorch_llm_processed_data --output-dir data/pytorch_core_api_identification

# 2. 解析 LLM 结果
python 2-parse-llm-results.py --input-dir data/pytorch_core_api_identification --output-dir data/pytorch_core_api_identification
```

---

## 数据格式说明

### JSON 输出结构示例
```json
{
  "issue_id": "159972",
  "title": "问题标题",
  "core_apis": [
    {
      "api_name": "torch.nn.functional.conv2d",
      "module": "torch.nn.functional",
      "function": "conv2d",
      "parameters": {
        "input": "Tensor",
        "weight": "Tensor",
        "bias": "Optional[Tensor]",
        "stride": "Union[int, Tuple[int, int]]",
        "padding": "Union[str, int, Tuple[int, int]]"
      },
      "return_type": "Tensor",
      "context": "在第5行被调用，输入shape为[1, 3, 224, 224]",
      "confidence": 0.95
    }
  ],
  "api_relationships": [
    {
      "source": "torch.nn.functional.conv2d",
      "target": "torch.Tensor",
      "relationship_type": "input"
    }
  ],
  "affected_api": "torch.nn.functional.conv2d",
  "trigger_conditions": {
    "backend": "MPS",
    "dtype": "float32",
    "input_shape": "[1, 3, 224, 224]"
  }
}
```

---

## 文件夹结构

```
Step1-CoreAPIIdentification/
├── 1-core-api-identification.py  # 核心 API 识别脚本
├── 2-parse-llm-results.py        # LLM 结果解析脚本
├── data/
│   └── pytorch_core_api_identification/  # API 识别结果
├── __pycache__/                  # Python 缓存文件
└── README.md
```

---

## 支持的框架

目前支持的框架：
- PyTorch (`pytorch_*`)
- TensorFlow (`tensorflow_*`)

可通过修改输入目录参数支持其他框架。
