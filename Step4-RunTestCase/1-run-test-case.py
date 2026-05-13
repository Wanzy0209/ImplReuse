#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execute test cases and save logs and results.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any


SCRIPT_DIR = Path(__file__).parent
DEFAULT_TEST_CASE_DIR = SCRIPT_DIR.parent / "Step3-ReuseTestCase/test_cases"
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


def main():
    parser = argparse.ArgumentParser(description="Execute test cases and save results.")
    parser.add_argument(
        "--test-case-dir",
        type=Path,
        default=DEFAULT_TEST_CASE_DIR,
        help="Directory containing test case .py files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for result logs."
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N test cases.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_case_files = list(args.test_case_dir.glob("*.py"))
    if args.limit > 0:
        test_case_files = test_case_files[:args.limit]

    print(f"Executing {len(test_case_files)} test cases...")

    for test_case_file in test_case_files:
        print(f"Running {test_case_file.name}...")
        try:
            result = execute_test_case(test_case_file, args.output_dir)
            status = "Success" if result['returncode'] == 0 else "Failed"
            print(f"  {status} (returncode: {result['returncode']})")
        except Exception as exc:
            print(f"  Error: {exc}")


if __name__ == "__main__":
    main()