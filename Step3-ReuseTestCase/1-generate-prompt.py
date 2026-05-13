#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate prompts for similar APIs based on Step1 issue data, Step0 test cases,
and Step2 per-issue similarity files.

Reads issue data from Step1-CoreAPIIdentification/data/pytorch_core_api_identification,
Step0 test cases from Step0-DataCollection/data/pytorch_test_case/executable and
Step0-DataCollection/data/pytorch_test_case/non_executable, and similarity files from
Step2-CalculateSimilarity/data/similar_api_list.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_STEP1_DIR = REPO_ROOT / "Step1-CoreAPIIdentification" / "data" / "pytorch_core_api_identification"
DEFAULT_TESTCASE_DIR = REPO_ROOT / "Step0-DataCollection" / "data" / "pytorch_test_case"
DEFAULT_SIMILARITY_DIR = REPO_ROOT / "Step2-CalculateSimilarity" / "data" / "similar_api_list"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "prompts"
DEFAULT_PYTORCH_API_SOURCE = REPO_ROOT / "Step2-CalculateSimilarity" / "data" / "pytorch-api-extracted" / "source"
DEFAULT_TENSORFLOW_API_SOURCE = REPO_ROOT / "Step2-CalculateSimilarity" / "data" / "tensorflow-api-extracted" / "source"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def truncate_text(text: str, max_chars: int = 2000, max_lines: int = 40) -> str:
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    short = "\n".join(lines)
    if len(short) > max_chars:
        short = short[:max_chars].rstrip() + "\n..."
    return short.strip()


def normalize_api_name_to_file_name(api_name: str) -> str:
    name = api_name.strip()
    name = name.replace('.', '_').replace('::', '_').replace('/', '_').replace('-', '_')
    if name.startswith('torch_') or name.startswith('tf_'):
        return f"{name}.py"
    if name.startswith('torch'):
        return f"{name}.py"
    if name.startswith('tf'):
        return f"{name}.py"
    return f"{name}.py"


def load_api_definition(api_name: str, source_dirs: List[Path]) -> Optional[str]:
    candidate = normalize_api_name_to_file_name(api_name)
    for source_dir in source_dirs:
        candidate_path = source_dir / candidate
        if candidate_path.exists():
            try:
                text = candidate_path.read_text(encoding='utf-8', errors='ignore')
                return truncate_text(text, max_chars=1200, max_lines=30)
            except Exception:
                continue
        alt_path = source_dir / candidate.lower()
        if alt_path.exists():
            try:
                text = alt_path.read_text(encoding='utf-8', errors='ignore')
                return truncate_text(text, max_chars=1200, max_lines=30)
            except Exception:
                continue
    return None


def is_sufficient_test_code(code: str) -> bool:
    """Heuristic: decide whether a test case string is sufficient."""
    if not code:
        return False
    s = code.strip()
    if len(s) < 30:
        return False
    tokens = ['def ', 'assert ', 'import ', 'torch', 'numpy', 'np', 'pytest']
    for t in tokens:
        if t in s:
            return True
    if '\n' in s and ('(' in s and ')' in s):
        return True
    return False


def build_prompt(issue: Dict[str, Any], test_case: str, similar_api: Dict[str, Any], group: str, main_api: str) -> str:
    issue_title = issue.get('title', '').strip()
    issue_desc = issue.get('bug_description') or issue.get('error_description') or issue.get('repro_output') or ''
    issue_desc = truncate_text(issue_desc, max_chars=1500, max_lines=40)
    issue_id = issue.get('issue_id', '')

    api_name = similar_api.get('api_name') or similar_api.get('name') or 'Unknown API'
    similarity_score = similar_api.get('similarity')
    similarity_note = ''
    reuse_description = ''
    library_scope = ''

    if group == 'pytorch_to_tensorflow':
        similarity_label = 'Cross-library similar API'
        library_scope = 'Cross-library'
        similarity_note = f"The API {api_name} is identified as cross-library similar to the original PyTorch API {main_api}."
        reuse_description = (
            "Please adapt the original PyTorch test case to this TensorFlow-style API, preserving the core bug reproduction logic and verifying the similar API's behavior."
        )
    elif group == 'pytorch_to_pytorch':
        similarity_label = 'Same-library similar API'
        library_scope = 'Same-library'
        similarity_note = f"The API {api_name} is identified as same-library similar to the original PyTorch API {main_api}."
        reuse_description = (
            "Please keep the test case in PyTorch and replace or adapt the original call site to verify the similar PyTorch API."
        )
    else:
        similarity_label = 'Issue-to-API code similarity'
        library_scope = 'Issue-to-API'
        similarity_note = (
            "This similarity is based on code similarity between the issue's repro/fix code and the API implementation or usage pattern. "
            "Please generate a test case that reflects that relationship."
        )
        reuse_description = (
            "Please generate a new test case that preserves the original bug reproduction logic while leveraging the similar API as a candidate for reuse."
        )

    api_info = similar_api.get('description') or ''
    if not api_info:
        source_dirs = [DEFAULT_PYTORCH_API_SOURCE, DEFAULT_TENSORFLOW_API_SOURCE]
        api_def = load_api_definition(api_name, source_dirs)
        api_info = api_def if api_def else f"No extracted definition available for {api_name}."
    api_info = truncate_text(api_info, max_chars=1200, max_lines=30)

    prompt = f"""Issue ID: {issue_id}
Title: {issue_title}
Bug Description:
{issue_desc}

Original API Under Test: {main_api}
Library Scope: {library_scope}
Similarity Type: {similarity_label}
Similar API: {api_name}
"""

    if similarity_score is not None:
        prompt += f"Similarity score: {similarity_score}\n"
    prompt += f"\nSimilar API information:\n{api_info}\n"
    prompt += f"\nOriginal Test Case:\n{truncate_text(test_case, max_chars=3000, max_lines=80)}\n"
    prompt += f"\n{similarity_note}\n{reuse_description}\n"
    prompt += (
        "IMPORTANT:\n"
        "1. Output valid Python test case code.\n"
        "2. Keep the test case runnable and minimal.\n"
        "3. Include any necessary imports and assertions.\n"
        "4. If the API is cross-library, translate semantics appropriately.\n"
    )

    if group == 'issue_to_api':
        prompt += (
            "5. For issue-to-API similarity, focus on how the similar API's code pattern relates to the reported bug and adapt the test accordingly.\n"
        )

    return prompt


def find_test_case_file(issue_id: str, test_case_root: Path) -> Optional[Path]:
    candidate = f"{issue_id}_issue_ori_data.py"
    exec_path = test_case_root / 'executable' / candidate
    if exec_path.exists():
        return exec_path
    non_exec_path = test_case_root / 'non_executable' / candidate
    if non_exec_path.exists():
        return non_exec_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate prompts for similar APIs.")
    parser.add_argument(
        "--step1-dir",
        type=Path,
        default=DEFAULT_STEP1_DIR,
        help="Directory containing Step1 issue JSON files."
    )
    parser.add_argument(
        "--test-case-dir",
        type=Path,
        default=DEFAULT_TESTCASE_DIR,
        help="Root directory containing Step0 test cases (executable/non_executable)."
    )
    parser.add_argument(
        "--similarity-dir",
        type=Path,
        default=DEFAULT_SIMILARITY_DIR,
        help="Directory containing per-issue similarity JSON files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated prompt files."
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N issues.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    issue_files = sorted(args.step1_dir.glob("*_issue_ori_data.json"))
    if args.limit > 0:
        issue_files = issue_files[:args.limit]

    print(f"Processing {len(issue_files)} issues...")

    for issue_file in issue_files:
        issue = load_json(issue_file)
        if not issue:
            continue

        issue_id = issue.get('issue_id') or issue_file.stem.replace('_issue_ori_data', '')
        core_api = None
        if isinstance(issue.get('core_api_identification'), dict):
            core_api = issue['core_api_identification'].get('core_api')
        if not core_api:
            print(f"Skipping {issue_id}: no core_api available")
            continue

        test_case_file = find_test_case_file(issue_id, args.test_case_dir)
        if not test_case_file:
            print(f"Skipping {issue_id}: no test case file found")
            continue

        test_case = test_case_file.read_text(encoding='utf-8', errors='ignore').strip()
        if not is_sufficient_test_code(test_case):
            print(f"Skipping {issue_id}: insufficient test case content")
            continue

        similarity_file = args.similarity_dir / f"{issue_id}.json"
        similarity_data = load_json(similarity_file)
        if not similarity_data:
            print(f"Skipping {issue_id}: missing similarity file")
            continue

        main_api = similarity_data.get('core_api') or core_api
        if not main_api:
            print(f"Skipping {issue_id}: similarity file missing core_api")
            continue

        similar_groups = similarity_data.get('similar_apis', {})
        if not similar_groups:
            print(f"Skipping {issue_id}: no similar APIs in file")
            continue

        for group, api_list in similar_groups.items():
            if not api_list:
                continue
            for idx, similar_api in enumerate(api_list):
                prompt_text = build_prompt(issue, test_case, similar_api, group, main_api)
                prompt_file = args.output_dir / f"{issue_id}_{group}_{idx}.txt"
                save_text(prompt_file, prompt_text)
                print(f"Saved prompt: {prompt_file}")


if __name__ == "__main__":
    main()
