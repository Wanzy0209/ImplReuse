#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse LLM responses to extract test case Python files.
Handles streaming responses with thinking process.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional


SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESPONSE_DIR = SCRIPT_DIR / "llm_responses_old" / "llm_io"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test_cases"


def parse_reply_to_testcase(reply_text: str) -> Optional[str]:
    """Extract python test code from a reply string.

    Tries multiple strategies:
    - extract fenced ```python ... ``` block (robust to missing closing fence)
    - extract def test_ blocks including indented body
    """
    # Remove thinking process if present (assuming it's before the actual response)
    # Look for patterns like "###Response" or direct code
    m_resp = re.search(r"###Response(.*)$", reply_text, re.DOTALL)
    txt = m_resp.group(1) if m_resp else reply_text

    # Remove any thinking content that might be before the code
    # Assuming thinking is in <think> tags or similar, but since it's streaming, it might be mixed
    # For simplicity, assume the code is at the end or in fenced blocks

    # Strategy 1: fenced python code block (```python ... ```). Be tolerant if closing fence missing.
    m_code = re.search(r"```python(.*?)(?:```|$)", txt, re.DOTALL)
    if m_code:
        code = m_code.group(1)
        code = re.sub(r"[^\x00-\x7F]+", '', code)  # Remove non-ASCII
        return code.strip()

    # Strategy 2: look for a def test_ and capture following indented lines as body
    m_def = re.search(r"(def\s+test_[\w_\d]+\s*\([^\)]*\)\s*:\s*(?:\n[ \t]+.*)*)", txt, re.DOTALL)
    if m_def:
        code = m_def.group(1)
        code = re.sub(r"[^\x00-\x7F]+", '', code)
        return code.strip()

    return None


def _find_assigned_variables(code: str) -> List[str]:
    # find left-hand-side variable names from simple assignments
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
    """Attempt to repair parsed code using only the content available in the reply."""
    code = parsed_code
    if not code:
        # Try to extract any python-like block
        m_python = re.search(r"```python(.*?)(?:```|$)", reply_text, re.DOTALL)
        if m_python:
            code = m_python.group(1).strip()
        else:
            return None

    if not code:
        return None

    # Fix incomplete assertions
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

    # Add torch import if needed
    if 'torch' in code and not _has_import(code, 'torch'):
        code = "import torch\n" + code

    # Ensure sufficient
    if not code.strip():
        return None

    return code


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Parse LLM responses to extract test cases.")
    parser.add_argument(
        "--response-dir",
        type=Path,
        default=DEFAULT_RESPONSE_DIR,
        help="Directory containing LLM response files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for test case .py files."
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N responses.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    response_files = list(args.response_dir.glob("*.output.txt"))
    if args.limit > 0:
        response_files = response_files[:args.limit]

    print(f"Processing {len(response_files)} responses...")

    for response_file in response_files:
        reply_text = response_file.read_text(encoding='utf-8')
        base_name = response_file.stem

        parsed_code = parse_reply_to_testcase(reply_text)
        if parsed_code:
            repaired_code = repair_using_reply(parsed_code, reply_text)
            if repaired_code:
                test_case_file = args.output_dir / f"{base_name}.py"
                save_text(test_case_file, repaired_code)
                print(f"Saved test case: {test_case_file}")
            else:
                print(f"Failed to repair code for {base_name}")
        else:
            print(f"No code found in {base_name}")


if __name__ == "__main__":
    main()