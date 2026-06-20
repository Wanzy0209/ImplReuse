#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execute test cases from cross_reuse_test_case and single_reuse_test_case directories.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CROSS_DIR = SCRIPT_DIR.parent / "Step3-ReuseTestCase/data-final/cross_reuse_test_case"
DEFAULT_SINGLE_DIR = SCRIPT_DIR.parent / "Step3-ReuseTestCase/data-final/single_reuse_test_case"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test_results"


def execute_test_case(test_case_path: Path, result_dir: Path) -> Dict[str, Any]:
    """Run the test case as a python subprocess."""
    result = {
        'path': str(test_case_path),
        'returncode': None,
        'stdout': '',
        'stderr': ''
    }
    try:
        proc = subprocess.run(
            ['python', str(test_case_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        result['returncode'] = proc.returncode
        result['stdout'] = proc.stdout
        result['stderr'] = proc.stderr
    except subprocess.TimeoutExpired as e:
        result['returncode'] = -1
        result['stderr'] = f'Timeout: {e}'
    except Exception as e:
        result['returncode'] = -2
        result['stderr'] = str(e)

    # Write log
    log_path = result_dir / (test_case_path.stem + '.log')
    log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def execute_test_cases(test_case_dir: Path, output_dir: Path, limit: int = 0):
    """Execute all test cases in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    test_case_files = list(test_case_dir.glob("*.py"))
    if limit > 0:
        test_case_files = test_case_files[:limit]

    print(f"Executing {len(test_case_files)} test cases in {test_case_dir}...")

    success_count = 0
    fail_count = 0

    for test_case_file in test_case_files:
        print(f"Running {test_case_file.name}...")
        try:
            result = execute_test_case(test_case_file, output_dir)
            if result['returncode'] == 0:
                status = "Success"
                success_count += 1
            else:
                status = "Failed"
                fail_count += 1
            print(f"  {status} (returncode: {result['returncode']})")
        except Exception as exc:
            print(f"  Error: {exc}")
            fail_count += 1

    print(f"\nSummary for {test_case_dir.name}:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {success_count + fail_count}")

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description="Execute test cases from cross and single directories.")
    parser.add_argument(
        "--cross-dir",
        type=Path,
        default=DEFAULT_CROSS_DIR,
        help="Directory containing cross-framework reuse test cases."
    )
    parser.add_argument(
        "--single-dir",
        type=Path,
        default=DEFAULT_SINGLE_DIR,
        help="Directory containing single-framework reuse test cases."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for result logs."
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N test cases per directory.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Execute cross-framework test cases
    cross_output_dir = args.output_dir / "cross_results"
    print("\n=== Executing Cross-Framework Reuse Test Cases ===")
    cross_success, cross_fail = execute_test_cases(args.cross_dir, cross_output_dir, args.limit)

    # Execute single-framework test cases
    single_output_dir = args.output_dir / "single_results"
    print("\n=== Executing Single-Framework Reuse Test Cases ===")
    single_success, single_fail = execute_test_cases(args.single_dir, single_output_dir, args.limit)

    # Overall summary
    print("\n=== Overall Summary ===")
    print(f"Cross-Framework: {cross_success} success, {cross_fail} fail")
    print(f"Single-Framework: {single_success} success, {single_fail} fail")
    print(f"Total: {cross_success + single_success} success, {cross_fail + single_fail} fail")


if __name__ == "__main__":
    main()