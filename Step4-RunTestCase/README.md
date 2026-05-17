# Step 4: 测试用例执行

## 概述
Step 4 的目标是执行生成的测试用例，验证测试用例的有效性，并收集执行结果用于后续分析。

## 工作流程

### 1️⃣ 第一步：执行测试用例 (`1-run-test-case.py`)
**功能**：执行生成的测试用例，记录执行结果

**输入**：
- Step 3 生成的测试用例文件
- 目标框架环境（PyTorch/TensorFlow）

**输出**：
- `test_results/` 文件夹中存储 `.output.log` 文件
- 包含测试执行日志、输出和错误信息

**运行示例**：
```bash
# 执行 PyTorch 测试用例
python 1-run-test-case.py --input-dir ../Step3-ReuseTestCase/data/reuse_results --output-dir test_results --framework pytorch

# 执行 TensorFlow 测试用例
python 1-run-test-case.py --input-dir ../Step3-ReuseTestCase/data/reuse_results --output-dir test_results --framework tensorflow
```

---

### 2️⃣ 第二步：结果分类 (`2-category.py`)
**功能**：对测试执行结果进行分类和统计

**输入**：
- 第一步生成的测试结果日志

**输出**：
- 分类统计报告
- 测试用例质量评估

**分类标准**：
- `pass` - 测试通过
- `fail` - 测试失败（检测到 bug）
- `error` - 执行错误（环境问题等）
- `timeout` - 超时

**运行示例**：
```bash
python 2-category.py --input-dir test_results --output-dir data/category_results
```

---

## 完整运行流程

```bash
# 1. 执行测试用例
python 1-run-test-case.py --input-dir ../Step3-ReuseTestCase/data/reuse_results --output-dir test_results --framework pytorch

# 2. 结果分类
python 2-category.py --input-dir test_results --output-dir data/category_results
```

---

## 测试结果文件命名规范

测试结果文件命名格式：
```
{issue_id}_{source_framework}_to_{target_framework}_{index}.output.log
```

示例：
- `159974_pytorch_to_pytorch_0.output.log` - PyTorch 到 PyTorch 的测试
- `159974_pytorch_to_tensorflow_0.output.log` - PyTorch 到 TensorFlow 的测试
- `159995_issue_to_api_0.output.log` - Issue API 测试

---

## 文件夹结构

```
Step4-RunTestCase/
├── 1-run-test-case.py   # 执行测试用例
├── 2-category.py        # 结果分类
├── test_results/        # 测试执行日志
│   ├── {issue_id}_pytorch_to_pytorch_{index}.output.log
│   ├── {issue_id}_pytorch_to_tensorflow_{index}.output.log
│   └── {issue_id}_issue_to_api_{index}.output.log
├── data/
│   └── category_results/ # 分类结果
└── README.md
```

---

## 支持的框架

- PyTorch
- TensorFlow

---

## 测试环境要求

- Python 3.8+
- PyTorch 2.0+（如执行 PyTorch 测试）
- TensorFlow 2.0+（如执行 TensorFlow 测试）
- 足够的内存和 GPU 资源（如测试涉及 GPU 操作）
