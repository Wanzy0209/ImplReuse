#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation: Parse LLM responses, extract test cases, execute them, and categorize results.

This script combines functionality from:
- Step3-ReuseTestCase/3-parse-llm-results.py (code extraction)
- Step4-RunTestCase/1-run-test-case.py (test execution)
- Step4-RunTestCase/2-category.py (result classification)

Test cases are separated into cross (TensorFlow) and single (PyTorch) categories.
"""

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESPONSE_DIR = SCRIPT_DIR / "data" / "no_coreapi_identification" / "llm_responses_all_replacements" / "llm_io"
DEFAULT_TEST_CASE_DIR = SCRIPT_DIR / "data" / "no_coreapi_identification" / "test_cases"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "no_coreapi_identification" / "test_results"


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


def detect_framework(code: str) -> str:
    """Detect whether code uses PyTorch (single) or TensorFlow (cross)."""
    tf_patterns = [
        r"import tensorflow",
        r"from tensorflow",
        r"\btf\.",
        r"\bTensorFlow\b",
    ]
    torch_patterns = [
        r"import torch",
        r"from torch",
        r"\btorch\.",
        r"\bPyTorch\b",
    ]

    has_tf = any(re.search(pattern, code, re.IGNORECASE) for pattern in tf_patterns)
    has_torch = any(re.search(pattern, code, re.IGNORECASE) for pattern in torch_patterns)

    if has_tf and not has_torch:
        return "cross"
    elif has_torch and not has_tf:
        return "single"
    elif has_tf and has_torch:
        tf_count = sum(len(re.findall(pattern, code, re.IGNORECASE)) for pattern in tf_patterns)
        torch_count = sum(len(re.findall(pattern, code, re.IGNORECASE)) for pattern in torch_patterns)
        return "cross" if tf_count > torch_count else "single"
    else:
        return "single"


def parse_reply_to_testcase(reply_text: str) -> Optional[str]:
    m_resp = re.search(r"###Response(.*)$", reply_text, re.DOTALL)
    txt = m_resp.group(1) if m_resp else reply_text

    m_code = re.search(r"```python(.*?)(?:```|$)", txt, re.DOTALL)
    if m_code:
        code = m_code.group(1)
        code = re.sub(r"[^\x00-\x7F]+", '', code)
        return code.strip()

    m_def = re.search(r"(def\s+test_[\w_\d]+\s*\([^\)]*\)\s*:\s*(?:\n[ \t]+.*)*)", txt, re.DOTALL)
    if m_def:
        code = m_def.group(1)
        code = re.sub(r"[^\x00-\x7F]+", '', code)
        return code.strip()

    return None


def _find_assigned_variables(code: str) -> List[str]:
    return re.findall(r"\b([a-zA-Z_][\w_]*)\s*=", code)


def _has_import(code: str, name: str) -> bool:
    return re.search(rf"^\s*import\s+{re.escape(name)}\b", code, re.MULTILINE) is not None or \
           re.search(rf"^\s*from\s+{re.escape(name)}\b", code, re.MULTILINE) is not None


def is_incomplete_assert_or_truncated(code: str) -> bool:
    if not code:
        return True
    if code.count('(') > code.count(')'):
        return True
    last_line = code.rstrip().splitlines()[-1].strip()
    if re.match(r"self\.assert\w*\s*$", last_line) or last_line.endswith('assert') or last_line.endswith('assert True'):
        return True
    if len(code.strip().splitlines()) <= 1:
        return True
    return False


def repair_using_reply(parsed_code: Optional[str], reply_text: str) -> Optional[str]:
    code = parsed_code
    if not code:
        m_python = re.search(r"```python(.*?)(?:```|$)", reply_text, re.DOTALL)
        if m_python:
            code = m_python.group(1).strip()
        else:
            return None

    if not code:
        return None

    if is_incomplete_assert_or_truncated(code):
        vars = _find_assigned_variables(code)
        candidate = None
        for name in ('within_range', 'out1', 'out2', 'out', 'result'):
            if name in vars:
                candidate = name
                break
        if candidate is None and vars:
            candidate = vars[0]

        code_lines = code.rstrip().splitlines()
        last = code_lines[-1].strip()
        if re.match(r"self\.assert\w*\s*$", last) or last == 'assert' or last.endswith('assert'):
            if candidate:
                code_lines[-1] = f"    assert {candidate}"
            else:
                code_lines[-1] = "    assert True"
        else:
            if candidate:
                code_lines.append(f"    assert {candidate}")
            else:
                code_lines.append("    assert True")
        code = '\n'.join(code_lines) + '\n'

    if 'torch' in code and not _has_import(code, 'torch'):
        code = "import torch\n" + code

    if 'tensorflow' in code.lower() or 'tf' in code:
        if not _has_import(code, 'tensorflow') and not _has_import(code, 'tf'):
            code = "import tensorflow as tf\n" + code

    if not code.strip():
        return None

    return code


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def load_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding='utf-8')
    except Exception as exc:
        print(f"Failed to load {path}: {exc}")
        return None


def extract_test_cases(response_dir: Path, output_dir: Path, limit: int = 0) -> Dict[str, List[Path]]:
    cross_dir = output_dir / "cross"
    single_dir = output_dir / "single"
    cross_dir.mkdir(parents=True, exist_ok=True)
    single_dir.mkdir(parents=True, exist_ok=True)

    response_files = sorted(response_dir.glob("*.output.txt"))
    if limit > 0:
        response_files = response_files[:limit]

    print(f"Extracting test cases from {len(response_files)} responses...")

    extracted_files = {
        'cross': [],
        'single': []
    }

    for response_file in response_files:
        reply_text = load_text(response_file)
        if not reply_text:
            continue

        base_name = response_file.stem.replace('.output', '')
        parsed_code = parse_reply_to_testcase(reply_text)

        if parsed_code:
            repaired_code = repair_using_reply(parsed_code, reply_text)
            if repaired_code:
                framework = detect_framework(repaired_code)
                target_dir = cross_dir if framework == 'cross' else single_dir
                test_case_file = target_dir / f"{base_name}.py"
                save_text(test_case_file, repaired_code)
                extracted_files[framework].append(test_case_file)
                print(f"  Extracted ({framework}): {test_case_file.name}")
            else:
                print(f"  Failed to repair code for {base_name}")
        else:
            print(f"  No code found in {base_name}")

    print(f"\nExtracted {len(extracted_files['cross'])} cross (TensorFlow) test cases")
    print(f"Extracted {len(extracted_files['single'])} single (PyTorch) test cases")
    return extracted_files


def execute_test_case(test_case_path: Path, result_dir: Path) -> Dict[str, Any]:
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

    log_path = result_dir / (test_case_path.stem + '.log')
    log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def execute_test_cases(test_case_dir: Path, output_dir: Path, label: str, limit: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)

    test_case_files = sorted(test_case_dir.glob("*.py"))
    if limit > 0:
        test_case_files = test_case_files[:limit]

    print(f"Executing {len(test_case_files)} {label} test cases...")

    success_count = 0
    fail_count = 0
    skipped_count = 0

    for test_case_file in test_case_files:
        log_path = output_dir / (test_case_file.stem + '.log')
        try:
            if log_path.exists():
                try:
                    existing_result = json.loads(log_path.read_text(encoding='utf-8'))
                    print(f"Skipping {test_case_file.name}: result already exists")
                    if existing_result.get('returncode', -999) == 0:
                        success_count += 1
                    else:
                        fail_count += 1
                    skipped_count += 1
                    continue
                except Exception:
                    pass

            print(f"Running {test_case_file.name}...")
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

    print(f"\n{label} Execution Summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Skipped (already executed): {skipped_count}")
    print(f"  Total: {success_count + fail_count + skipped_count}")

    return success_count, fail_count, skipped_count


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


def categorize_results(log_dir: Path, output_file: Path, label: str):
    log_files = sorted(log_dir.glob("*.log"))
    print(f"Categorizing {len(log_files)} {label} result logs...")

    results = []
    for log_file in log_files:
        try:
            log_data = json.loads(log_file.read_text(encoding='utf-8'))
        except Exception:
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

    if results:
        fieldnames = ['file', 'returncode', 'category', 'generation_status', 'stdout', 'stderr']
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved categorized {label} results to {output_file}")

        categories = Counter(r['category'] for r in results)
        print(f"\n{label} Category Summary:")
        for cat, count in categories.items():
            print(f"  {cat}: {count}")

        return categories
    return {}


def main():
    parser = argparse.ArgumentParser(description="Extract, execute, and categorize test cases (cross/single separated).")
    parser.add_argument(
        "--response-dir",
        type=Path,
        default=DEFAULT_RESPONSE_DIR,
        help="Directory containing LLM response files (llm_io)."
    )
    parser.add_argument(
        "--test-case-dir",
        type=Path,
        default=DEFAULT_TEST_CASE_DIR,
        help="Output directory for extracted test case .py files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for test results and categorization."
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N items.")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip test case extraction (assume test cases already exist)."
    )
    parser.add_argument(
        "--skip-execute",
        action="store_true",
        help="Skip test execution (assume results already exist)."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        print("\n=== Step 1: Extracting Test Cases from LLM Responses ===")
        extract_test_cases(args.response_dir, args.test_case_dir, args.limit)

    if not args.skip_execute:
        print("\n=== Step 2: Executing Cross-Framework (TensorFlow) Test Cases ===")
        cross_test_case_dir = args.test_case_dir / "cross"
        cross_output_dir = args.output_dir / "cross_results"
        cross_success, cross_fail, cross_skipped = execute_test_cases(cross_test_case_dir, cross_output_dir, "cross", args.limit)

        print("\n=== Step 2: Executing Single-Framework (PyTorch) Test Cases ===")
        single_test_case_dir = args.test_case_dir / "single"
        single_output_dir = args.output_dir / "single_results"
        single_success, single_fail, single_skipped = execute_test_cases(single_test_case_dir, single_output_dir, "single", args.limit)

    print("\n=== Step 3: Categorizing Results ===")
    cross_output_dir = args.output_dir / "cross_results"
    single_output_dir = args.output_dir / "single_results"

    cross_category_file = args.output_dir.parent / "cross_categorized_results.csv"
    single_category_file = args.output_dir.parent / "single_categorized_results.csv"

    cross_categories = categorize_results(cross_output_dir, cross_category_file, "cross")
    single_categories = categorize_results(single_output_dir, single_category_file, "single")

    print("\n=== Overall Summary ===")
    cross_total = sum(cross_categories.values())
    cross_success = cross_categories.get("Success", 0)
    cross_fail = cross_total - cross_success

    single_total = sum(single_categories.values())
    single_success = single_categories.get("Success", 0)
    single_fail = single_total - single_success

    print("\nCross-Framework (TensorFlow):")
    print(f"  Total: {cross_total}")
    print(f"  Success: {cross_success} ({cross_success/cross_total*100:.1f}%)")
    print(f"  Failed: {cross_fail} ({cross_fail/cross_total*100:.1f}%)")

    print("\nSingle-Framework (PyTorch):")
    print(f"  Total: {single_total}")
    print(f"  Success: {single_success} ({single_success/single_total*100:.1f}%)")
    print(f"  Failed: {single_fail} ({single_fail/single_total*100:.1f}%)")

    print("\nCombined:")
    total = cross_total + single_total
    success = cross_success + single_success
    fail = cross_fail + single_fail
    print(f"  Total: {total}")
    print(f"  Success: {success} ({success/total*100:.1f}%)")
    print(f"  Failed: {fail} ({fail/total*100:.1f}%)")


if __name__ == "__main__":
    main()
