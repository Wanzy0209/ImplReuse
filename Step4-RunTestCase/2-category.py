#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Categorize test cases based on execution results and logs.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter


# Failure type classification rules (from generate_stats.py)
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
    """Classify failure type from log content."""
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


def load_log(log_path: Path) -> Dict[str, Any]:
    """Load log file content."""
    try:
        return json.loads(log_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Failed to load {log_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Categorize test cases based on logs.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).parent / "test_results",
        help="Directory containing .log files."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(__file__).parent / "categorized_results.csv",
        help="Output CSV file for categorized results."
    )
    args = parser.parse_args()

    log_files = list(args.log_dir.glob("*.log"))
    print(f"Processing {len(log_files)} log files...")

    results = []
    for log_file in log_files:
        log_data = load_log(log_file)
        if not log_data:
            continue

        returncode = log_data.get('returncode', -999)
        stdout = log_data.get('stdout', '')
        stderr = log_data.get('stderr', '')
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
            'stdout': stdout[:500],  # Truncate for CSV
            'stderr': stderr[:500]
        })

    # Write to CSV
    if results:
        fieldnames = ['file', 'returncode', 'category', 'generation_status', 'stdout', 'stderr']
        with open(args.output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved categorized results to {args.output_file}")

        # Print summary
        categories = Counter(r['category'] for r in results)
        print("\nCategory Summary:")
        for cat, count in categories.items():
            print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()