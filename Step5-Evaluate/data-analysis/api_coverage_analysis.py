#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Coverage Analysis for test cases.
- cross: only count TensorFlow APIs
- single: only count PyTorch APIs
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Set


SCRIPT_DIR = Path(__file__).parent


def normalize_api(api: str) -> str:
    """标准化API名称"""
    if api.endswith('.'):
        api = api[:-1]
    if api.startswith('tensorflow.'):
        api = 'tf.' + api[len('tensorflow.'):]
    return api


def is_valid_api(api: str) -> bool:
    """检查API是否有效"""
    if not api:
        return False
    if api.endswith('.'):
        return False
    parts = api.split('.')
    if len(parts) < 2:
        return False
    for part in parts:
        if not part:
            return False
        if not part[0].isalpha():
            return False
    return True


def extract_apis_from_code(code: str, framework: str) -> Set[str]:
    """从代码中提取指定框架的 API 调用"""
    apis = set()

    if framework == 'tf':
        patterns = [
            r'\btf\.[a-zA-Z0-9_.]+',
            r'\btensorflow\.[a-zA-Z0-9_.]+',
        ]
    elif framework == 'torch':
        patterns = [
            r'\btorch\.[a-zA-Z0-9_.]+',
        ]
    else:
        patterns = [
            r'\btorch\.[a-zA-Z0-9_.]+',
            r'\btf\.[a-zA-Z0-9_.]+',
            r'\btensorflow\.[a-zA-Z0-9_.]+',
        ]

    for pattern in patterns:
        matches = re.findall(pattern, code)
        for api in matches:
            normalized_api = normalize_api(api)
            if is_valid_api(normalized_api):
                apis.add(normalized_api)

    return apis


def process_baseline_directory(dir_path: Path, framework: str) -> Dict:
    """处理 baseline 目录"""
    all_apis = set()
    issue_apis = {}

    py_files = sorted(dir_path.glob("*_issue_ori_data.py"))
    print(f"  Found {len(py_files)} files")

    for py_file in py_files:
        try:
            code = py_file.read_text(encoding='utf-8', errors='ignore')
            apis = extract_apis_from_code(code, framework)
            issue_id = py_file.stem.replace("_issue_ori_data", "")
            issue_apis[issue_id] = apis
            all_apis.update(apis)
        except Exception as e:
            print(f"    Failed to read {py_file.name}: {e}")

    return {
        'total_files': len(py_files),
        'total_unique_apis': len(all_apis),
        'framework_apis': all_apis,
        'issue_apis': issue_apis,
        'framework': framework
    }


def process_reuse_directory(dir_path: Path, framework: str) -> Dict:
    """处理 reuse 目录 - 统计所有脚本"""
    all_apis = set()
    issue_apis = {}

    py_files = sorted(dir_path.glob("*.output.py"))
    print(f"  Found {len(py_files)} files")

    for py_file in py_files:
        try:
            code = py_file.read_text(encoding='utf-8', errors='ignore')
            apis = extract_apis_from_code(code, framework)

            filename = py_file.stem
            parts = filename.split('_')
            issue_id = parts[0]

            if issue_id not in issue_apis:
                issue_apis[issue_id] = set()
            issue_apis[issue_id].update(apis)
            all_apis.update(apis)
        except Exception as e:
            print(f"    Failed to read {py_file.name}: {e}")

    return {
        'total_files': len(py_files),
        'total_unique_apis': len(all_apis),
        'framework_apis': all_apis,
        'issue_apis': issue_apis,
        'framework': framework
    }


def save_stats(stats: Dict, output_file: Path):
    """保存统计结果到 CSV"""
    fieldnames = ['issue_id', 'api_count', 'apis']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for issue_id, apis in sorted(stats['issue_apis'].items()):
            writer.writerow({
                'issue_id': issue_id,
                'api_count': len(apis),
                'apis': ','.join(sorted(apis))
            })


def save_summary(summary: Dict, output_file: Path):
    """保存汇总统计"""
    fieldnames = ['method', 'framework', 'total_files', 'total_unique_apis']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, stats in summary.items():
            writer.writerow({
                'method': method,
                'framework': stats['framework'],
                'total_files': stats['total_files'],
                'total_unique_apis': stats['total_unique_apis'],
            })


def save_api_details(baseline_apis: Set[str], reuse_apis: Set[str], output_prefix: Path, framework: str):
    """保存交集和独有API列表"""
    baseline_only = baseline_apis - reuse_apis
    reuse_only = reuse_apis - baseline_apis
    intersection = baseline_apis & reuse_apis

    intersection_file = output_prefix.parent / f"{output_prefix.stem}_intersection_apis.txt"
    baseline_only_file = output_prefix.parent / f"{output_prefix.stem}_baseline_only_apis.txt"
    reuse_only_file = output_prefix.parent / f"{output_prefix.stem}_reuse_only_apis.txt"

    with open(intersection_file, 'w', encoding='utf-8') as f:
        f.write(f"# {framework} APIs - Intersection (Both Baseline and Reuse)\n")
        f.write(f"# Total: {len(intersection)}\n")
        f.write("#" * 70 + "\n")
        for api in sorted(intersection):
            f.write(f"{api}\n")
    print(f"  Saved intersection APIs to {intersection_file}")

    with open(baseline_only_file, 'w', encoding='utf-8') as f:
        f.write(f"# {framework} APIs - Only in Baseline\n")
        f.write(f"# Total: {len(baseline_only)}\n")
        f.write("#" * 70 + "\n")
        for api in sorted(baseline_only):
            f.write(f"{api}\n")
    print(f"  Saved baseline-only APIs to {baseline_only_file}")

    with open(reuse_only_file, 'w', encoding='utf-8') as f:
        f.write(f"# {framework} APIs - Only in Reuse\n")
        f.write(f"# Total: {len(reuse_only)}\n")
        f.write("#" * 70 + "\n")
        for api in sorted(reuse_only):
            f.write(f"{api}\n")
    print(f"  Saved reuse-only APIs to {reuse_only_file}")


def main():
    parser = argparse.ArgumentParser(description="API Coverage Analysis for test cases")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Base directory containing test case directories."
    )
    args = parser.parse_args()

    directories = {
        'cross_baseline': {
            'path': args.base_dir / "cross_baseline_test_case",
            'framework': 'tf'
        },
        'cross_reuse': {
            'path': args.base_dir / "cross_reuse_test_case",
            'framework': 'tf'
        },
        'single_baseline': {
            'path': args.base_dir / "single_baseline_test_case",
            'framework': 'torch'
        },
        'single_reuse': {
            'path': args.base_dir / "single_reuse_test_case",
            'framework': 'torch'
        },
    }

    summary = {}

    print("=" * 70)
    print("API Coverage Analysis")
    print("=" * 70)

    for method_name, config in directories.items():
        dir_path = config['path']
        framework = config['framework']
        print(f"\nProcessing {method_name} (framework: {framework})...")

        if not dir_path.exists():
            print(f"  Directory not found: {dir_path}")
            continue

        if 'baseline' in method_name:
            stats = process_baseline_directory(dir_path, framework)
        else:
            stats = process_reuse_directory(dir_path, framework)

        summary[method_name] = stats

        output_file = args.base_dir / f"{method_name}_api_coverage.csv"
        save_stats(stats, output_file)
        print(f"  Saved detailed coverage to {output_file}")

        print(f"  Summary:")
        print(f"    Total files: {stats['total_files']}")
        print(f"    Total unique {framework} APIs: {stats['total_unique_apis']}")

    summary_file = args.base_dir / "api_coverage_summary.csv"
    save_summary(summary, summary_file)
    print(f"\nSaved summary to {summary_file}")

    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)

    cross_baseline = summary.get('cross_baseline')
    cross_reuse = summary.get('cross_reuse')
    if cross_baseline and cross_reuse:
        cross_baseline_apis = cross_baseline['framework_apis']
        cross_reuse_apis = cross_reuse['framework_apis']
        cross_baseline_only = cross_baseline_apis - cross_reuse_apis
        cross_reuse_only = cross_reuse_apis - cross_baseline_apis
        cross_intersection = cross_baseline_apis & cross_reuse_apis
        cross_union = cross_baseline_apis | cross_reuse_apis
        print(f"\n[Cross - TensorFlow]")
        print(f"  cross_baseline unique APIs: {len(cross_baseline_apis)}")
        print(f"  cross_reuse unique APIs: {len(cross_reuse_apis)}")
        print(f"  Only in baseline: {len(cross_baseline_only)}")
        print(f"  Only in reuse: {len(cross_reuse_only)}")
        print(f"  Intersection (both): {len(cross_intersection)}")
        print(f"  Union (total): {len(cross_union)}")
        print(f"  Formula check: {len(cross_baseline_only)} + {len(cross_reuse_only)} + {len(cross_intersection)} = {len(cross_union)}")

        cross_output_prefix = args.base_dir / "cross"
        save_api_details(cross_baseline_apis, cross_reuse_apis, cross_output_prefix, 'TensorFlow')

    single_baseline = summary.get('single_baseline')
    single_reuse = summary.get('single_reuse')
    if single_baseline and single_reuse:
        single_baseline_apis = single_baseline['framework_apis']
        single_reuse_apis = single_reuse['framework_apis']
        single_baseline_only = single_baseline_apis - single_reuse_apis
        single_reuse_only = single_reuse_apis - single_baseline_apis
        single_intersection = single_baseline_apis & single_reuse_apis
        single_union = single_baseline_apis | single_reuse_apis
        print(f"\n[Single - PyTorch]")
        print(f"  single_baseline unique APIs: {len(single_baseline_apis)}")
        print(f"  single_reuse unique APIs: {len(single_reuse_apis)}")
        print(f"  Only in baseline: {len(single_baseline_only)}")
        print(f"  Only in reuse: {len(single_reuse_only)}")
        print(f"  Intersection (both): {len(single_intersection)}")
        print(f"  Union (total): {len(single_union)}")
        print(f"  Formula check: {len(single_baseline_only)} + {len(single_reuse_only)} + {len(single_intersection)} = {len(single_union)}")

        single_output_prefix = args.base_dir / "single"
        save_api_details(single_baseline_apis, single_reuse_apis, single_output_prefix, 'PyTorch')


if __name__ == "__main__":
    main()
