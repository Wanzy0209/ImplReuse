"""
CrossProbe Results: Run Test Cases and Statistics
批量执行 CrossProbe-master/results 下的测试用例，统计结果。
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import re

FAILURE_RULES = {
    "Execution Failure": [
        "size of tensor", "must match the size", "expected input", "expected .* channels",
        "non-singleton dimension", "output.shape", "number of steps must be non-negative",
        "steps must be non-negative", "Only Tensors of floating point", "does not require grad",
        "does not have a grad_fn", "Boolean value of Tensor with more than one value is ambiguous",
        "select_cols should match output size", "InvalidArgumentError", "DecodeCSV",
        "Length of .* do not match", "record_defaults", "select_cols",
        "expected to be a .* tensor but is a .* tensor", "DecodeImage", "InvalidArgumentError",
    ],
    "Assertion Mismatch": [
        "AssertionError", "assert ", "Expected shape", "Output values do not match expected values",
        "should produce different", "should change after", "should be reproducible", "FAILED (errors=",
        "raise_error", "raise fut_result",
    ],
    "API Misuse": [
        "got an unexpected keyword argument", "missing 1 required positional argument",
        "remove\(\) got an unexpected keyword", "Expected op in `op_list` to be an OpOverload",
        "Expected one of", "device type at start of device string", "invalid_device", "infer_schema",
        "must have a type annotation", "Got func with signature", "has no attribute", "AttributeError",
        "missing .* required positional arguments", "TypeError", "Got an unexpected keyword argument",
    ],
    "Async Misuse": [
        "torch._C.Future", "object is not iterable", "has no len",
    ],
    "Environment Failure": [
        "post_hook", "step_post_hook", "torch._dynamo", "does not know how to trace", "FutureWarning",
        "UserWarning", "deprecated", "Can't pickle local object", "multiprocessing", "mp.spawn",
    ],
    "Timeout": ["timed out", "Timeout:"],
    "Code Generation Error": ["SyntaxError", "nonlocal .* not found", "NameError: name '*' is not defined"],
    "Missing Dependency": ["ModuleNotFoundError", "No module named", "ImportError"],
    "Unknown Failure": []
}

def classify_failure(log: str) -> str:
    for category, patterns in FAILURE_RULES.items():
        for p in patterns:
            try:
                if p in log:
                    return category
                if re.search(re.escape(p), log):
                    return category
            except re.error:
                continue
    return "Unknown Failure"

def extract_python_code_block(file_path: Path) -> str:
    """
    提取文件中第一个 fenced 代码块内容，若无代码块则返回整个文件内容。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配任意 fenced 代码块 ```...```，包括 ```python 和 ``` 或 ```txt 等。
    pattern = r'```(?:[^\n]*)\n(.*?)```'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted + '\n'

    # 如果文件本身就是一个纯 Python 文件，直接返回全文。
    if re.search(r'^\s*(import|from|def|class|async|@)', content, re.MULTILINE):
        return content

    # 最后回退到全文，避免错过没有 fences 的代码。
    return content

def run_test_file(py_file, timeout):
    start = time.time()
    try:
        proc = subprocess.run([sys.executable, str(py_file)], capture_output=True, text=True, cwd=str(py_file.parent), timeout=timeout)
        end = time.time()
        return {
            'returncode': proc.returncode,
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'duration': end - start,
            'timed_out': False
        }
    except subprocess.TimeoutExpired as e:
        end = time.time()
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or '')
        return {
            'returncode': None,
            'stdout': stdout,
            'stderr': stderr + f"\n[Timeout after {timeout}s]",
            'duration': end - start,
            'timed_out': True
        }
    except Exception as e:
        end = time.time()
        return {
            'returncode': None,
            'stdout': '',
            'stderr': str(e),
            'duration': end - start,
            'timed_out': False
        }

def main():
    results_dir = Path(__file__).parent / 'crossprobe_results_output'
    results_dir.mkdir(exist_ok=True)
    detail_dir = results_dir / 'details'
    detail_dir.mkdir(exist_ok=True)

    # 超时设置（秒）
    default_timeout = int(os.environ.get('CROSSPROBE_TIMEOUT', '120'))

    # 扫描 results 目录
    test_dir = Path(__file__).parent.parent / 'CrossProbe-master' / 'results'
    py_files = list(test_dir.rglob('*.py'))
    print(f"Found {len(py_files)} test files in CrossProbe-master/results")

    # 先提取并覆盖所有py文件中的```python```代码块
    for f in py_files:
        code = extract_python_code_block(f)
        if code is not None:
            with open(f, 'w', encoding='utf-8') as wf:
                wf.write(code)

    all_results = []
    processed = 0
    success = 0
    failed = 0

    for f in py_files:
        processed += 1
        ident = f.stem
        print(f"[{processed:4d}/{len(py_files)}] Running {f.name}...", end=' ', flush=True)
        result = run_test_file(f, timeout=default_timeout)
        fail_type = ''
        if result.get('returncode') != 0:
            fail_type = classify_failure(result.get('stdout', '') + result.get('stderr', ''))
        record = {
            'id': ident,
            'file': str(f),
            'returncode': result.get('returncode'),
            'stdout': result.get('stdout'),
            'stderr': result.get('stderr'),
            'duration': result.get('duration'),
            'timed_out': result.get('timed_out'),
            'failure_type': fail_type,
            'ran_at': datetime.now().isoformat()
        }
        # 保存单个结果
        result_file = detail_dir / f'run_{ident}.json'
        with open(result_file, 'w', encoding='utf-8') as rf:
            json.dump(record, rf, ensure_ascii=False, indent=2)
        all_results.append(record)
        if record['returncode'] == 0:
            print('OK')
            success += 1
        else:
            print('FAIL')
            failed += 1
    # 保存索引和统计
    index_file = results_dir / 'run_index.json'
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    stats = {
        'total': processed,
        'success': success,
        'failed': failed,
        'start_time': None,
        'end_time': datetime.now().isoformat()
    }
    stats_file = results_dir / 'run_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    # SUMMARY
    summary_file = results_dir / 'SUMMARY.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('='*60 + '\n')
        f.write('CrossProbe Results - Summary\n')
        f.write('='*60 + '\n\n')
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输出目录: {results_dir}\n\n")
        f.write('运行统计:\n')
        f.write('-'*40 + '\n')
        f.write(f"总用例数: {processed}\n")
        f.write(f"运行成功: {success}\n")
        f.write(f"运行失败: {failed}\n")
        f.write(f"详细结果目录: {detail_dir}\n")
    print('\n' + '='*60)
    print('CrossProbe test run completed!')
    print('='*60)
    print(f"Output directory: {results_dir}")

if __name__ == '__main__':
    main()
