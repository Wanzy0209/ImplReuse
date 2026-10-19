# TeRL 项目说明

## 项目简介

本项目是一套面向 GitHub Issue 的自动化分析与测试用例生成流水线，包含：
- 从 issue 页面抓取与解析数据
- 通过 LLM 提取结构化 bug 信息与复现代码
- 识别核心 API 及其上下文
- 计算 API 相似度以发现复用机会
- 基于相似 API 生成测试用例复用建议
- 执行测试用例并对结果分类分析

项目目标是构建一个可扩展的研究型流程，用于分析深度学习框架（如 PyTorch / TensorFlow）中的 issue、bug 和测试用例复用问题。

## 目录结构

```
Step0-DataCollection/
Step1-CoreAPIIdentification/
Step2-CalculateSimilarity/
Step3-ReuseTestCase/
Step4-RunTestCase/
Step5-Evaluate/
```

### 主要模块说明

- `Step0-DataCollection/`
  - 从 GitHub issue 抓取 HTML 数据
  - 提取 issue 元信息和代码片段
  - 生成 LLM prompt 并调用 LLM 解析
  - 生成初步测试用例

- `Step1-CoreAPIIdentification/`
  - 从 Step0 数据中识别核心 API
  - 提取 API 调用、参数及上下文
  - 对 LLM 结果做清洗与标准化

- `Step2-CalculateSimilarity/`
  - 提取框架源代码中的 API
  - 使用 CodeBERT 等模型计算 API 相似度
  - 生成与目标 API 的相似 API 列表

- `Step3-ReuseTestCase/`
  - 基于 issue 数据与相似 API 生成复用提示词
  - 调用 LLM 生成测试用例复用建议
  - 解析 LLM 输出，得到可复用测试用例

- `Step4-RunTestCase/`
  - 执行生成的测试用例
  - 记录执行结果日志
  - 对结果进行分类统计

- `Step5-Evaluate/`
  - 评估阶段目录，存放 RQ 分析结果（RQ1/RQ2/RQ3/RQ4）

## 快速开始

### 1. 环境准备

建议使用 Python 3.8 及以上版本。

所需依赖请根据步骤脚本补充安装，典型依赖包括：
- `requests`
- `beautifulsoup4`
- `openai` / `openai` 兼容库
- `transformers` / `torch`（如果使用 CodeBERT 模型）
- `numpy`

> 注意：本仓库当前没有顶层 `requirements.txt`，请根据实际脚本补齐依赖。


### 2. 运行流程

#### Step0: 数据收集与处理

```bash
cd Step0-DataCollection
python 1-issue-crawler.py --url "https://github.com/pytorch/pytorch" --pages 10
python 2-file-processor.py --input-dir data/pytorch_issue --output-dir data/pytorch_processed_data
python 3-prompt-generator.py --input-dir data/pytorch_processed_data --output-dir data/pytorch_prompts
python 4-llm-processor.py --prompt-dir data/pytorch_prompts --output-dir data/pytorch_llm_processed_data --api-key YOUR_API_KEY
python 5-test-case-generator.py --input-dir data/pytorch_llm_processed_data --output-dir data/pytorch_test_case
python 6-pr-code-linker.py --input-dir data/pytorch_llm_processed_data --output-dir data/pytorch_pr_data --repo "pytorch/pytorch"
```

#### Step1: 核心 API 识别

```bash
cd ../Step1-CoreAPIIdentification
python 1-core-api-identification.py --input-dir ../Step0-DataCollection/data/pytorch_llm_processed_data --output-dir data/pytorch_core_api_identification
python 2-parse-llm-results.py --input-dir data/pytorch_core_api_identification --output-dir data/pytorch_core_api_identification
```

#### Step2: API 相似度计算

```bash
cd ../Step2-CalculateSimilarity
python 1-extract-pytorch-code.py --source-dir data/pytorch-code --output-dir data/pytorch-api-extracted
python 2-calculate-similarity-cross.py --model-dir codebert-base --input-dir data/pytorch-api-extracted --output-dir data/api_pairwise_similarity
python 3-get-similar-api-list.py --similarity-dir data/api_pairwise_similarity --input-dir ../Step1-CoreAPIIdentification/data/pytorch_core_api_identification --output-dir data/similar_api_list
```

#### Step3: 测试用例复用

```bash
cd ../Step3-ReuseTestCase
python 1-generate-prompt.py --issue-dir ../Step0-DataCollection/data/pytorch_llm_processed_data --similar-api-dir ../Step2-CalculateSimilarity/data/similar_api_list --output-dir prompts
python 2-llm-reuse.py --prompt-dir prompts --api-key YOUR_API_KEY
python 3-parse-llm-results.py --input-dir prompts --output-dir data/reuse_results
```

#### Step4: 测试用例执行

```bash
cd ../Step4-RunTestCase
python 1-run-test-case.py --input-dir ../Step3-ReuseTestCase/data/reuse_results --output-dir test_results --framework pytorch
python 2-category.py --input-dir test_results --output-dir data/category_results
```

## 关键数据目录说明

- `Step0-DataCollection/data/pytorch_issue/` - 抓取的 issue HTML 数据
- `Step0-DataCollection/data/pytorch_processed_data/` - 提取后的 issue 结构化数据
- `Step0-DataCollection/data/pytorch_prompts/` - 生成的 LLM prompt
- `Step0-DataCollection/data/pytorch_llm_processed_data/` - LLM 解析输出数据
- `Step0-DataCollection/data/pytorch_test_case/` - 生成的测试用例
- `Step1-CoreAPIIdentification/data/pytorch_core_api_identification/` - 核心 API 识别结果
- `Step2-CalculateSimilarity/data/similar_api_list/` - 相似 API 列表
- `Step3-ReuseTestCase/prompts/` - 复用提示词
- `Step3-ReuseTestCase/data/reuse_results/` - 复用结果
- `Step4-RunTestCase/test_results/` - 测试执行日志
- `Step5-Evaluate/` - 评估与研究结果存放区

## 运行建议

- 请先补齐 `Step0-DataCollection` 中的 LLM API Key 配置。
- 如涉及模型嵌入计算，请确保 `Step2-CalculateSimilarity/codebert-base/` 已准备好模型权重。
- 若只关注特定阶段，可按顺序单独执行对应 Step 子目录脚本。
