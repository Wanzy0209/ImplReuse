#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Categorize test results for each subdirectory under test_results.
Process cross_baseline_results, cross_results, single_baseline_results, single_results separately.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter

MISSING_DEPENDENCY = [
    "ModuleNotFoundError",
    "No module named",
    "ImportError",
]

EXECUTION_FAILURE = [
    "size of tensor",
    "must match the size",
    "expected input",
    "expected .* channels",
    "non-singleton dimension",
    "output.shape",
    "number of steps must be non-negative",
    "steps must be non-negative",
    "Only Tensors of floating point",
    "does not require grad",
    "does not have a grad_fn",
    "Boolean value of Tensor with more than one value is ambiguous",
    "select_cols should match output size",
    "InvalidArgumentError",
    "DecodeCSV",
    "Length of .* do not match",
    "record_defaults",
    "select_cols",
    "expected to be a .* tensor but is a .* tensor",
    "DecodeImage",
]

ASSERTION_MISMATCH = [
    "AssertionError",
    "assert ",
    "Expected shape",
    "Output values do not match expected values",
    "should produce different",
    "should change after",
    "should be reproducible",
    "FAILED (errors=",
    "raise_error",
    "raise fut_result",
]

API_MISUSE = [
    "got an unexpected keyword argument",
    "missing 1 required positional argument",
    "remove\\(\\) got an unexpected keyword",
    "Expected op in `op_list` to be an OpOverload",
    "Expected one of",
    "device type at start of device string",
    "invalid_device",
    "infer_schema",
    "must have a type annotation",
    "Got func with signature",
    "has no attribute",
    "AttributeError",
    "missing .* required positional arguments",
    "TypeError",
    "Got an unexpected keyword argument",
]

ENV_FAILURE = [
    "post_hook",
    "step_post_hook",
    "torch._dynamo",
    "does not know how to trace",
    "FutureWarning",
    "UserWarning",
    "deprecated",
    "Can't pickle local object",
    "multiprocessing",
    "mp.spawn",
]

ASYNC_MISUSE = [
    "torch._C.Future",
    "object is not iterable",
    "has no len",
]

TIMEOUT = [
    "timed out",
    "Timeout:",
]

CODE_GEN_ERROR = [
    "SyntaxError",
    "nonlocal .* not found",
    "NameError: name '*' is not defined"
]

FAILURE_RULES = {
    "Execution Failure": EXECUTION_FAILURE,
    "Assertion Mismatch": ASSERTION_MISMATCH,
    "API Misuse": API_MISUSE,
    "Async Misuse": ASYNC_MISUSE,
    "Environment Failure": ENV_FAILURE,
    "Timeout": TIMEOUT,
    "Code Generation Error": CODE_GEN_ERROR,
    "Missing Dependency": MISSING_DEPENDENCY,
    "Unknown Failure": []
}

GENERATION_STATUS_MAP = {
    "Assertion Mismatch": "生成成功，并且找到错误",
    "Missing Dependency": "执行失败",
    "API Misuse": "生成失败",
    "Unknown Failure": "生成失败",
    "Execution Failure": "生成失败",
    "Async Misuse": "生成失败",
    "Environment Failure": "生成失败",
    "Code Generation Error": "生成失败",
    "Timeout": "生成失败"
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


def load_log(log_path: Path):
    try:
        return json.loads(log_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Failed to load {log_path}: {e}")
        return {}


def categorize_results(log_dir: Path, output_file: Path, category_label: str):
    log_files = list(log_dir.glob("*.output.log"))
    json_files = [f for f in log_dir.glob("run_*.json") if "run_index" not in f.name and "run_stats" not in f.name]

    total_files = len(log_files) + len(json_files)
    print(f"Processing {total_files} files ({len(log_files)} .log + {len(json_files)} .json) in {log_dir}...")

    results = []

    for log_file in log_files:
        log_data = load_log(log_file)
        if not log_data:
            continue

        returncode = log_data.get('returncode', -999)
        stdout = log_data.get('stdout') or ''
        stderr = log_data.get('stderr') or ''
        if not isinstance(stdout, str):
            stdout = str(stdout)
        if not isinstance(stderr, str):
            stderr = str(stderr)
        full_log = stdout + stderr

        if returncode == 0:
            category = "Success"
            generation_status = "生成成功"
        else:
            category = classify_failure(full_log)
            generation_status = GENERATION_STATUS_MAP.get(category, "生成失败")

        results.append({
            'file': log_file.name,
            'returncode': returncode,
            'category': category,
            'generation_status': generation_status,
            'stdout': stdout[:500],
            'stderr': stderr[:500]
        })

    for json_file in json_files:
        log_data = load_log(json_file)
        if not log_data:
            continue

        returncode = log_data.get('returncode', -999)
        stdout = log_data.get('stdout') or ''
        stderr = log_data.get('stderr') or ''
        if not isinstance(stdout, str):
            stdout = str(stdout)
        if not isinstance(stderr, str):
            stderr = str(stderr)
        full_log = stdout + stderr

        if returncode == 0:
            category = "Success"
            generation_status = "生成成功"
        else:
            category = classify_failure(full_log)
            generation_status = GENERATION_STATUS_MAP.get(category, "生成失败")

        results.append({
            'file': json_file.name,
            'returncode': returncode,
            'category': category,
            'generation_status': generation_status,
            'stdout': stdout[:500],
            'stderr': stderr[:500]
        })

    if results:
        fieldnames = ['file', 'returncode', 'category', 'generation_status', 'stdout', 'stderr']
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved categorized results to {output_file}")

        categories = Counter(r['category'] for r in results)
        print(f"\n{category_label} Category Summary:")
        for cat, count in categories.items():
            print(f"  {cat}: {count}")

        return categories
    return {}


def main():
    parser = argparse.ArgumentParser(description="Categorize test results for each subdirectory.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).parent / "test_results",
        help="Directory containing test results subdirectories."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for categorized results."
    )
    args = parser.parse_args()

    subdirs = ['cross_baseline_results', 'cross_results', 'single_baseline_results', 'single_results']
    all_categories = {}

    print("=" * 70)
    print("Categorizing Test Results")
    print("=" * 70)

    for subdir in subdirs:
        log_dir = args.log_dir / subdir
        if not log_dir.exists():
            print(f"\n{log_dir} 不存在，跳过")
            continue

        output_file = args.output_dir / f"{subdir}_categorized_results.csv"
        print(f"\n=== Processing {subdir} ===")
        categories = categorize_results(log_dir, output_file, subdir)
        all_categories[subdir] = categories

    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)

    for subdir, categories in all_categories.items():
        print(f"\n{subdir}:")
        total = sum(categories.values())
        success = categories.get("Success", 0)
        for cat, count in categories.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {cat}: {count} ({percentage:.1f}%)")
        if total > 0:
            print(f"  Success Rate: {success}/{total} ({success/total*100:.1f}%)")

    combined = Counter()
    for categories in all_categories.values():
        combined.update(categories)

    print("\n" + "=" * 70)
    print("Combined Summary")
    print("=" * 70)
    total = sum(combined.values())
    success = combined.get("Success", 0)
    for cat, count in combined.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    if total > 0:
        print(f"  Overall Success Rate: {success}/{total} ({success/total*100:.1f}%)")


if __name__ == "__main__":
    main()
