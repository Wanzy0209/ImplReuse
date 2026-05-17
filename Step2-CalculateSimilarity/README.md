# Step 2: API 相似度计算

## 概述
Step 2 的目标是计算 API 之间的相似度，通过 CodeBERT 模型对代码进行嵌入表示，然后利用余弦相似度等方法衡量不同 API 之间的相似程度，从而为测试用例生成提供参考。

## 工作流程

### 1️⃣ 第一步：提取框架代码 (`1-extract-*.py`)
**功能**：从框架源代码中提取 API 定义和实现

**输入**：
- 框架源代码仓库（PyTorch 或 TensorFlow）

**输出**：
- 提取的 API 定义文件
- API 文档和实现代码

**脚本说明**：
- `1-extract-pytorch-code.py` - 提取 PyTorch API 代码
- `1-extract-tensorflow-code.py` - 提取 TensorFlow API 代码

**运行示例**：
```bash
# 提取 PyTorch 代码
python 1-extract-pytorch-code.py --source-dir data/pytorch-code --output-dir data/pytorch-api-extracted

# 提取 TensorFlow 代码
python 1-extract-tensorflow-code.py --source-dir data/tensorflow-code --output-dir data/tensorflow-api-extracted
```

---

### 2️⃣ 第二步：计算相似度 (`2-calculate-similarity-*.py`)
**功能**：使用 CodeBERT 模型计算 API 之间的相似度

**输入**：
- 第一步提取的 API 代码
- CodeBERT 预训练模型

**输出**：
- API 相似度矩阵
- 相似度得分文件

**脚本说明**：
- `2-calculate-similarity-cross.py` - 计算跨框架 API 相似度
- `2-calculate-similarity-pr.py` - 计算 PR 相关 API 相似度

**运行示例**：
```bash
# 计算跨框架相似度
python 2-calculate-similarity-cross.py --model-dir codebert-base --input-dir data/pytorch-api-extracted --output-dir data/api_pairwise_similarity

# 计算 PR 相似度
python 2-calculate-similarity-pr.py --model-dir codebert-base --input-dir data/pytorch-api-extracted --output-dir data/issue_api_similarity
```

---

### 3️⃣ 第三步：获取相似 API 列表 (`3-get-similar-api-list.py`)
**功能**：根据相似度得分获取与目标 API 相似的 API 列表

**输入**：
- 第二步生成的相似度矩阵
- Step 1 识别的核心 API

**输出**：
- `data/{framework}_similar_api_list/` 文件夹中存储 JSON 文件
- 每个文件包含与该 issue 相关 API 的相似 API 列表

**运行示例**：
```bash
python 3-get-similar-api-list.py --similarity-dir data/api_pairwise_similarity --input-dir ../Step1-CoreAPIIdentification/data/pytorch_core_api_identification --output-dir data/similar_api_list
```

---

## 完整运行流程

```bash
# 1. 提取 PyTorch 代码（可选，如已提取可跳过）
python 1-extract-pytorch-code.py --source-dir data/pytorch-code --output-dir data/pytorch-api-extracted

# 2. 计算 API 相似度
python 2-calculate-similarity-cross.py --model-dir codebert-base --input-dir data/pytorch-api-extracted --output-dir data/api_pairwise_similarity

# 3. 获取相似 API 列表
python 3-get-similar-api-list.py --similarity-dir data/api_pairwise_similarity --input-dir ../Step1-CoreAPIIdentification/data/pytorch_core_api_identification --output-dir data/similar_api_list
```

---

## 数据格式说明

### 相似 API 列表 JSON 结构
```json
{
  "issue_id": "159972",
  "core_api": "torch.nn.functional.conv2d",
  "similar_apis": [
    {
      "api_name": "torch.nn.Conv2d",
      "similarity_score": 0.92,
      "module": "torch.nn",
      "function": "Conv2d"
    },
    {
      "api_name": "torch.nn.functional.conv1d",
      "similarity_score": 0.88,
      "module": "torch.nn.functional",
      "function": "conv1d"
    },
    {
      "api_name": "torch.nn.functional.conv3d",
      "similarity_score": 0.85,
      "module": "torch.nn.functional",
      "function": "conv3d"
    }
  ]
}
```

---

## 文件夹结构

```
Step2-CalculateSimilarity/
├── 1-extract-pytorch-code.py       # PyTorch 代码提取
├── 1-extract-tensorflow-code.py    # TensorFlow 代码提取
├── 2-calculate-similarity-cross.py # 跨框架相似度计算
├── 2-calculate-similarity-pr.py    # PR 相似度计算
├── 3-get-similar-api-list.py       # 获取相似 API 列表
├── codebert-base/                  # CodeBERT 预训练模型
├── data/
│   ├── pytorch-code/               # PyTorch 源代码
│   ├── similar_api_list/           # 相似 API 列表
│   ├── api_pairwise_similarity.zip # API 两两相似度（网盘存储）
│   ├── issue_api_similarity.jsonl  # Issue-API 相似度
│   ├── issue_api_similarity.zip    # Issue-API 相似度（网盘存储）
│   ├── pytorch-api-extracted.zip   # 提取的 PyTorch API（网盘存储）
│   └── pytorch_API_def.txt         # PyTorch API 定义
└── README.md
```

---

## 数据文件说明

### 网盘存储文件
由于部分数据文件较大，以下文件存储在 Google 网盘中：

| 文件 | 说明 |
|------|------|
| `api_pairwise_similarity.zip` | API 两两相似度矩阵 |
| `issue_api_similarity.zip` | Issue 与 API 的相似度数据 |
| `pytorch-api-extracted.zip` | 提取的 PyTorch API 定义 |

### 本地文件
- `pytorch_API_def.txt` - PyTorch API 定义列表
- `issue_api_similarity.jsonl` - Issue-API 相似度（JSON Lines 格式）
- `similar_api_list/` - 每个 issue 的相似 API 列表

---

## 依赖模型

### CodeBERT 模型
- 位置：`codebert-base/`
- 来源：[microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- 用途：代码嵌入和相似度计算

---

## 支持的框架

- PyTorch
- TensorFlow
