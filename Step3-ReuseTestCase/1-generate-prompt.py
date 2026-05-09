#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate prompts for similar APIs based on issue data and similarity lists.

Reads issue data from Step0-DataCollection/data/pytorch_llm_processed_data_fix_code
and similarity data from Step2-CalculateSimilarity/data/similar_api_list.json,
then generates prompts for each similar API per issue.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ISSUE_DIR = REPO_ROOT / "Step0-DataCollection" / "data" / "pytorch_llm_processed_data_fix_code"
DEFAULT_SIMILARITY_FILE = REPO_ROOT / "Step2-CalculateSimilarity" / "data" / "similar_api_list.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "prompts"


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


def generate_prompt_for_similar_api(issue: Dict[str, Any], similar_api: Dict[str, Any], test_case: str) -> str:
    """
    Generate a prompt for a single similar API.
    """
    main_api = issue.get('core_api') or issue.get('api') or 'Unknown API'
    title = issue.get('title', 'Unknown Title')
    name = similar_api.get('name', 'Unknown API')
    score = similar_api.get('score')
    desc = similar_api.get('description', '')
    similarity_type = similar_api.get('similarity_type', 'similar')

    # Map similarity types to labels
    similarity_labels = {
        'semantic': 'Semantically Similar API',
        'input_output': 'Input-Similar API',
        'code': 'Code-Similar API',
        'pr_code': 'PR Code-Similar API'
    }
    similarity_label = similarity_labels.get(similarity_type, 'Similar API')

    similarity_descs = {
        'semantic': f"identified as semantically similar to {main_api}, sharing comparable functionality or purpose",
        'input_output': f"identified as input-similar to {main_api}, meaning that it accepts similar input parameters",
        'code': f"identified as code-similar to {main_api}, meaning it has a similar implementation structure",
        'pr_code': f"identified as PR code-similar to {main_api}, meaning it has similar code changes from pull requests"
    }
    similarity_desc = similarity_descs.get(similarity_type, f"identified as similar to {main_api}")

    prompt = f"""Bug Report
Bug Description: {title}
Original API Under Test: {main_api}
Original Test Case Reproducing the Bug: {test_case}

{similarity_label}: {name}
"""

    if score is not None:
        prompt += f"\nSimilarity score: {score}\n"
    if desc:
        prompt += f"\nSimilar API Description: {desc}\n"

    prompt += f"\nThe API {name} is {similarity_desc}.\n"
    prompt += f"Please generate a new test case that maintains the structure of the original test case while verifying the {similarity_type.replace('_', ' ')} API {name}.\n"
    prompt += f"\nIMPORTANT INSTRUCTIONS: \n"
    prompt += f"1. ONLY output the Python test case code, nothing else.\n"
    prompt += f"2. Do NOT include any explanations, introductions, or additional text.\n"
    prompt += f"3. Ensure the test case is complete, runnable, and follows the library's testing conventions.\n"
    prompt += f"4. Control the length to ensure it is not truncated - keep it concise but complete.\n"
    prompt += f"5. Make sure all necessary imports are included at the beginning.\n"
    prompt += f"6. Include appropriate assertions to verify the API's functionality.\n"

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Generate prompts for similar APIs.")
    parser.add_argument(
        "--issue-dir",
        type=Path,
        default=DEFAULT_ISSUE_DIR,
        help="Directory containing issue JSON files."
    )
    parser.add_argument(
        "--similarity-file",
        type=Path,
        default=DEFAULT_SIMILARITY_FILE,
        help="JSON file containing similarity lists."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for prompts."
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N issues.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load similarity data
    similarity_data = load_json(args.similarity_file)
    if not similarity_data:
        print(f"Failed to load similarity data from {args.similarity_file}")
        return

    # Get issue files
    issue_files = list(args.issue_dir.glob("*.json"))
    if args.limit > 0:
        issue_files = issue_files[:args.limit]

    print(f"Processing {len(issue_files)} issues...")

    for issue_file in issue_files:
        issue = load_json(issue_file)
        if not issue:
            continue

        issue_id = issue.get('issue_id')
        if not issue_id:
            continue

        test_case = issue.get('best_code') or issue.get('repro_code') or ''
        if not is_sufficient_test_code(test_case):
            print(f"Skipping {issue_id}: insufficient test case")
            continue

        similar_apis = similarity_data.get(issue_id, [])
        if not similar_apis:
            print(f"No similar APIs for {issue_id}")
            continue

        for idx, similar_api in enumerate(similar_apis):
            prompt = generate_prompt_for_similar_api(issue, similar_api, test_case)
            prompt_file = args.output_dir / f"{issue_id}_{idx}.txt"
            save_text(prompt_file, prompt)
            print(f"Saved prompt: {prompt_file}")


if __name__ == "__main__":
    main()