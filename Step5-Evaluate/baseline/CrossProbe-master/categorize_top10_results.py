#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Categorize top-10 cross-probe results based on execution results and logs.
Process results from crossprobe_results_output_top10 directory.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

FAILURE_RULES = {
    "Execution Failure": [
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
    ],
    "Assertion Mismatch": [
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
    ],
    "API Misuse": [
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
    ],
    "Async Misuse": [
        "torch._C.Future",
        "object is not iterable",
        "has no len",
    ],
    "Environment Failure": [
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
    ],
    "Timeout": [
        "timed out",
        "Timeout:",
    ],
    "Code Generation Error": [
        "SyntaxError",
        "nonlocal .* not found",
        "NameError: name '*' is not defined"
    ],
    "Missing Dependency": [
        "ModuleNotFoundError",
        "No module named",
        "ImportError",
    ],
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


def parse_top10_id(file_name: str) -> Tuple[str, int]:
    """Parse issue_id and top rank from filename like 'run_159974_issue_ori_data_top1.json'."""
    match = re.match(r'run_(\d+)_issue_ori_data_top(\d+)\.json', file_name)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def load_log(log_path: Path):
    try:
        return json.loads(log_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Failed to load {log_path}: {e}")
        return {}


def categorize_top10_results(details_dir: Path, output_dir: Path):
    log_files = list(details_dir.glob("*.json"))
    print(f"Processing {len(log_files)} log files in {details_dir}...")

    results = []
    issue_results = defaultdict(list)
    rank_stats = defaultdict(Counter)

    for log_file in log_files:
        log_data = load_log(log_file)
        if not log_data:
            continue

        issue_id, top_rank = parse_top10_id(log_file.name)
        if not issue_id:
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

        result = {
            'issue_id': issue_id,
            'top_rank': top_rank,
            'file': log_file.name,
            'returncode': returncode,
            'category': category,
            'generation_status': generation_status,
            'stdout': stdout[:500],
            'stderr': stderr[:500]
        }
        results.append(result)
        issue_results[issue_id].append(result)
        rank_stats[top_rank][category] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_output = output_dir / "top10_categorized_results.csv"
    if results:
        fieldnames = ['issue_id', 'top_rank', 'file', 'returncode', 'category', 'generation_status', 'stdout', 'stderr']
        with open(csv_output, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved categorized results to {csv_output}")

    print("\n=== Top-10 Category Statistics by Rank ===")
    for rank in sorted(rank_stats.keys()):
        stats = rank_stats[rank]
        total = sum(stats.values())
        print(f"\n--- Top-{rank} ---")
        print(f"Total: {total}")
        for cat, count in stats.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {cat}: {count} ({percentage:.1f}%)")

    print("\n=== Overall Category Statistics ===")
    overall_stats = Counter()
    for rank in rank_stats:
        overall_stats.update(rank_stats[rank])
    total_all = sum(overall_stats.values())
    for cat, count in overall_stats.items():
        percentage = (count / total_all) * 100 if total_all > 0 else 0
        print(f"  {cat}: {count} ({percentage:.1f}%)")

    print("\n=== Issue Success Analysis ===")
    issues_with_success = 0
    issues_all_top_fail = 0
    issues_top1_success = 0

    for issue_id, results in issue_results.items():
        has_success = any(r['returncode'] == 0 for r in results)
        all_fail = all(r['returncode'] != 0 for r in results)
        top1_result = next((r for r in results if r['top_rank'] == 1), None)
        top1_success = top1_result and top1_result['returncode'] == 0

        if has_success:
            issues_with_success += 1
        if all_fail:
            issues_all_top_fail += 1
        if top1_success:
            issues_top1_success += 1

    total_issues = len(issue_results)
    print(f"Total issues: {total_issues}")
    print(f"Issues with at least one success (any top-k): {issues_with_success} ({issues_with_success/total_issues*100:.1f}%)")
    print(f"Issues with top-1 success: {issues_top1_success} ({issues_top1_success/total_issues*100:.1f}%)")
    print(f"Issues with all top-10 failed: {issues_all_top_fail} ({issues_all_top_fail/total_issues*100:.1f}%)")

    print("\n=== Success Rate by Rank ===")
    for rank in sorted(rank_stats.keys()):
        stats = rank_stats[rank]
        total = sum(stats.values())
        success_count = stats.get("Success", 0)
        rate = (success_count / total) * 100 if total > 0 else 0
        print(f"  Top-{rank}: {success_count}/{total} ({rate:.1f}%)")

    print("\n=== Issue-level Detailed Analysis (Sample) ===")
    sample_count = min(10, total_issues)
    print(f"First {sample_count} issues:")
    for i, (issue_id, results) in enumerate(sorted(issue_results.items())[:sample_count]):
        success_ranks = [r['top_rank'] for r in results if r['returncode'] == 0]
        fail_ranks = [r['top_rank'] for r in results if r['returncode'] != 0]
        print(f"  Issue {issue_id}:")
        print(f"    Success ranks: {success_ranks if success_ranks else 'None'}")
        print(f"    Failed ranks: {fail_ranks if fail_ranks else 'None'}")


def main():
    parser = argparse.ArgumentParser(description="Categorize top-10 cross-probe results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "crossprobe_results_output_top10",
        help="Directory containing crossprobe top-10 results (with details subdirectory)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "crossprobe_results_output_top10",
        help="Output directory for categorized results."
    )
    args = parser.parse_args()

    details_dir = args.results_dir / "details"
    if not details_dir.exists():
        print(f"Error: details directory not found at {details_dir}")
        return

    categorize_top10_results(details_dir, args.output_dir)


if __name__ == "__main__":
    main()