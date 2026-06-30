#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix failed test cases using LLM and save all test cases to final-data directory.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent
DEFAULT_TEST_RESULTS_DIR = SCRIPT_DIR / "test_results"
DEFAULT_SOURCE_DIR = SCRIPT_DIR.parent / "Step3-ReuseTestCase" / "data-final"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "final-data"
DEFAULT_IO_SUBDIR = "llm_io"


def extract_zhipu_chunk_content(chunk: Any) -> Optional[str]:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta")
            if isinstance(delta, dict) and delta.get("content") is not None:
                return delta["content"]
            message = choice.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return message["content"]
            if isinstance(choice, dict) and choice.get("text") is not None:
                return choice["text"]
        return None

    try:
        choice = chunk.choices[0]
    except Exception:
        return None

    delta = getattr(choice, "delta", None)
    if delta is not None:
        content = getattr(delta, "content", None)
        if content is not None:
            return content
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return content
    return getattr(choice, "text", None)


def serialize_zhipu_event(event: Any) -> Any:
    if isinstance(event, dict):
        return event
    try:
        return json.loads(json.dumps(event, ensure_ascii=False))
    except Exception:
        return repr(event)


def call_zhipu_llm(
    api_key: str,
    prompt: str,
    model: str = "glm-4.7",
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> Tuple[str, List[Any]]:
    try:
        from zai import ZhipuAiClient
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'zai'. Install it before running LLM inference."
        ) from exc

    client = ZhipuAiClient(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in fixing Python test cases. You should analyze the error and provide a corrected version of the test case.",
        },
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        thinking={"type": "enabled"},
    )

    chunks: List[str] = []
    raw_events: List[Any] = []
    for chunk in response:
        raw_events.append(chunk)
        content = extract_zhipu_chunk_content(chunk)
        if content is not None:
            chunks.append(content)

    return "".join(chunks).strip(), raw_events


def parse_reply_to_testcase(reply_text: str) -> Optional[str]:
    """Extract python test code from a reply string."""
    # Strategy 1: fenced python code block (```python ... ```). Be tolerant if closing fence missing.
    m_code = re.search(r"```python(.*?)(?:```|$)", reply_text, re.DOTALL)
    if m_code:
        code = m_code.group(1)
        code = re.sub(r"[^\x00-\x7F]+", '', code)  # Remove non-ASCII
        return code.strip()

    # Strategy 2: look for a def test_ and capture following indented lines as body
    m_def = re.search(r"(def\s+test_[\w_\d]+\s*\([^\)]*\)\s*:\s*(?:\n[ \t]+.*)*)", reply_text, re.DOTALL)
    if m_def:
        code = m_def.group(1)
        code = re.sub(r"[^\x00-\x7F]+", '', code)
        return code.strip()

    return None


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def build_fix_prompt(test_case_code: str, error_log: str) -> str:
    """Build prompt for LLM to fix the test case."""
    prompt = f"""Please fix the following Python test case based on the error message.

## Original Test Case:
```python
{test_case_code}
```

## Error Message:
```
{error_log}
```

## Requirements:
1. Analyze the error message carefully
2. Fix the test case to resolve the error
3. Keep the test logic intact - do not change the core test functionality
4. If the error is due to missing dependencies or environment issues, try to add mock or skip conditions
5. Output only the fixed Python code, wrapped in ```python ... ``` blocks

## Fixed Test Case:
"""
    return prompt


def process_test_case(
    test_case_file: Path,
    log_file: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    io_dir: Path,
    overwrite: bool = False,
) -> Tuple[str, bool]:
    """Process a test case: fix if failed, copy if success."""
    test_case_name = test_case_file.name
    output_file = output_dir / test_case_name

    # Read log file
    try:
        log_data = json.loads(log_file.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Failed to read log {log_file}: {e}")
        # Copy original file
        shutil.copy(test_case_file, output_file)
        return "copy_failed", False

    returncode = log_data.get('returncode', -999)
    stderr = log_data.get('stderr') or ''
    stdout = log_data.get('stdout') or ''

    # Success case: just copy
    if returncode == 0:
        shutil.copy(test_case_file, output_file)
        return "success", False

    # Failed case: try to fix with LLM
    test_case_code = test_case_file.read_text(encoding='utf-8')
    error_log = stderr + stdout

    # Check if already fixed
    fixed_file = io_dir / f"{test_case_file.stem}_fixed.output.py"
    if fixed_file.exists() and not overwrite:
        shutil.copy(fixed_file, output_file)
        return "already_fixed", True

    # Build prompt and call LLM
    prompt = build_fix_prompt(test_case_code, error_log)

    input_file = io_dir / f"{test_case_file.stem}_fix.input.txt"
    response_file = io_dir / f"{test_case_file.stem}_fix.output.txt"
    events_file = io_dir / f"{test_case_file.stem}_fix.output.events.json"

    save_text(input_file, prompt)

    try:
        print(f"Calling LLM to fix {test_case_name}...")
        response_text, response_events = call_zhipu_llm(
            api_key=api_key,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        save_text(response_file, response_text)
        serialized_events = [serialize_zhipu_event(event) for event in response_events]
        save_text(events_file, json.dumps(serialized_events, ensure_ascii=False, indent=2))

        # Parse fixed code from response
        fixed_code = parse_reply_to_testcase(response_text)
        if fixed_code:
            save_text(fixed_file, fixed_code)
            shutil.copy(fixed_file, output_file)
            return "fixed", True
        else:
            # Failed to parse, copy original
            shutil.copy(test_case_file, output_file)
            return "parse_failed", False
    except Exception as exc:
        print(f"Error fixing {test_case_name}: {exc}")
        # Copy original file
        shutil.copy(test_case_file, output_file)
        return "llm_error", False


def process_directory(
    source_dir: Path,
    log_dir: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    io_dir: Path,
    overwrite: bool = False,
    limit: int = 0,
) -> Dict[str, int]:
    """Process all test cases in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all log files
    log_files = list(log_dir.glob("*.log"))
    if limit > 0:
        log_files = log_files[:limit]

    stats = {
        "success": 0,
        "fixed": 0,
        "already_fixed": 0,
        "copy_failed": 0,
        "parse_failed": 0,
        "llm_error": 0,
        "no_log": 0,
    }

    print(f"Processing {len(log_files)} test cases...")

    for log_file in log_files:
        # Get test case file name from log file name
        test_case_name = log_file.stem + ".py"
        test_case_file = source_dir / test_case_name

        if not test_case_file.exists():
            print(f"Test case file not found: {test_case_file}")
            stats["no_log"] += 1
            continue

        status, was_fixed = process_test_case(
            test_case_file,
            log_file,
            output_dir,
            api_key,
            model,
            temperature,
            max_tokens,
            io_dir,
            overwrite,
        )
        stats[status] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fix failed test cases using LLM.")
    parser.add_argument(
        "--test-results-dir",
        type=Path,
        default=DEFAULT_TEST_RESULTS_DIR,
        help="Directory containing test results (cross_results and single_results)."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Source directory containing original test cases."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for fixed test cases."
    )
    parser.add_argument("--api-key", default=os.getenv("ZHIPUAI_API_KEY", "044f2bc486d14e13aa68259fbd1970c4.hjLhKCmeXaDRt2xf"), help="Zhipu API key.")
    parser.add_argument("--model", default="glm-4.7", help="LLM model.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N test cases per directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fixed files.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create LLM IO directory
    io_dir = args.output_dir / "llm_responses_old" / DEFAULT_IO_SUBDIR
    io_dir.mkdir(parents=True, exist_ok=True)

    # Process cross-framework test cases
    print("\n=== Processing Cross-Framework Test Cases ===")
    cross_source_dir = args.source_dir / "cross_reuse_test_case"
    cross_log_dir = args.test_results_dir / "cross_results"
    cross_output_dir = args.output_dir / "cross_reuse_test_case"
    cross_stats = process_directory(
        cross_source_dir,
        cross_log_dir,
        cross_output_dir,
        args.api_key,
        args.model,
        args.temperature,
        args.max_tokens,
        io_dir,
        args.overwrite,
        args.limit,
    )

    # Process single-framework test cases
    print("\n=== Processing Single-Framework Test Cases ===")
    single_source_dir = args.source_dir / "single_reuse_test_case"
    single_log_dir = args.test_results_dir / "single_results"
    single_output_dir = args.output_dir / "single_reuse_test_case"
    single_stats = process_directory(
        single_source_dir,
        single_log_dir,
        single_output_dir,
        args.api_key,
        args.model,
        args.temperature,
        args.max_tokens,
        io_dir,
        args.overwrite,
        args.limit,
    )

    # Print summary
    print("\n=== Summary ===")
    print("Cross-Framework:")
    for status, count in cross_stats.items():
        print(f"  {status}: {count}")
    print("\nSingle-Framework:")
    for status, count in single_stats.items():
        print(f"  {status}: {count}")

    # Combined summary
    print("\n=== Combined Summary ===")
    total_stats = {}
    for key in cross_stats.keys():
        total_stats[key] = cross_stats[key] + single_stats[key]
    for status, count in total_stats.items():
        print(f"  {status}: {count}")

    print(f"\nFixed test cases saved to: {args.output_dir}")


if __name__ == "__main__":
    main()