#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate prompts for RQ3 using all replacement APIs.

This script reads existing prompt files from Step3-ReuseTestCase/prompts and
appends an extra requirement to replace multiple source APIs together using
replacement candidates from the RQ3 similarity data.

Default output:
  Step5-Evaluate/RQ3/data/no_coreapi_identification/prompts_all_replacements

Example:
  python Step5-Evaluate/RQ3/3_generate_prompt_all_replacements.py

Then execute prompts with:
  python Step5-Evaluate/RQ3/4_execute_prompt_all_replacements.py
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_STEP1_DIR = REPO_ROOT / "Step1-CoreAPIIdentification" / "data" / "pytorch_core_api_identification"
DEFAULT_SIMILARITY_DIR = SCRIPT_DIR / "data" / "no_coreapi_identification" / "similar_api_list"
DEFAULT_BASE_PROMPT_DIR = REPO_ROOT / "Step3-ReuseTestCase" / "prompts"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "no_coreapi_identification" / "prompts_all_replacements"
DEFAULT_PYTORCH_API_SOURCE = REPO_ROOT / "Step2-CalculateSimilarity" / "data" / "pytorch-api-extracted" / "source"
DEFAULT_TENSORFLOW_API_SOURCE = REPO_ROOT / "Step2-CalculateSimilarity" / "data" / "tensorflow-api-extracted" / "source"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to load {path}: {exc}")
        return None


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Failed to load {path}: {exc}")
        return None


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
    if name.startswith('torch') or name.startswith('tf'):
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


def sanitize_file_name(value: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', value)
    return safe.strip('_') or 'api'


def stable_short_name(*parts: str, max_length: int = 120) -> str:
    raw = '__'.join(part for part in parts if part)
    short = sanitize_file_name(raw)
    if len(short) <= max_length:
        return short
    digest = hashlib.sha1(short.encode('utf-8')).hexdigest()[:10]
    return f"{short[:max_length-12]}__{digest}"


def build_prompt(
    issue: Dict[str, Any],
    test_case: str,
    similar_api: Dict[str, Any],
    group: str,
    main_api: str,
    replacement_api: Optional[str] = None,
) -> str:
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
            'Please adapt the original PyTorch test case to this TensorFlow-style API, preserving the core bug reproduction logic and verifying the similar API\'s behavior.'
        )
    elif group == 'pytorch_to_pytorch':
        similarity_label = 'Same-library similar API'
        library_scope = 'Same-library'
        similarity_note = f"The API {api_name} is identified as same-library similar to the original PyTorch API {main_api}."
        reuse_description = (
            'Please keep the test case in PyTorch and replace or adapt the original call site to verify the similar PyTorch API.'
        )
    else:
        similarity_label = 'Issue-to-API code similarity'
        library_scope = 'Issue-to-API'
        similarity_note = (
            'This similarity is based on code similarity between the issue\'s repro/fix code and the API implementation or usage pattern. '
            'Please generate a test case that reflects that relationship.'
        )
        reuse_description = (
            'Please generate a new test case that preserves the original bug reproduction logic while leveraging the similar API as a candidate for reuse.'
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

    if replacement_api and replacement_api != main_api:
        prompt += f"Replacement API: {replacement_api}\n"

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


def build_multi_api_prompt(
    issue: Dict[str, Any],
    base_prompt: str,
    replacement_set: List[Dict[str, Any]],
    group: str,
    main_api: str,
    rank: int,
) -> str:
    issue_title = issue.get('title', '').strip()
    issue_desc = issue.get('bug_description') or issue.get('error_description') or issue.get('repro_output') or ''
    issue_desc = truncate_text(issue_desc, max_chars=1500, max_lines=40)
    issue_id = issue.get('issue_id', '')
    replacement_lines = []
    for item in replacement_set:
        source_api = item.get('source_api', 'Unknown source')
        replacement_api = item.get('api_name') or item.get('replacement_api') or 'Unknown replacement'
        similarity = item.get('similarity')
        replacement_group = item.get('source_group') or 'unknown'
        replacement_lines.append(
            f"- {source_api} -> {replacement_api} (score={similarity}, source_group={replacement_group})"
        )
    replacement_text = "\n".join(replacement_lines)

    if group == 'pytorch_to_tensorflow':
        similarity_label = 'Cross-library replacement set'
        reuse_description = (
            'Please adapt the original PyTorch test case by replacing each source API with the corresponding cross-library similar API, preserving the original bug reproduction logic and verifying behavior in the target library.'
        )
    elif group == 'pytorch_to_pytorch':
        similarity_label = 'Same-library replacement set'
        reuse_description = (
            'Please modify the original PyTorch test case by replacing each source API with the corresponding same-library similar API, preserving the bug reproduction logic and keeping the test in PyTorch.'
        )
    else:
        similarity_label = 'Replacement set'
        reuse_description = (
            'Please modify the original test case by replacing each listed source API with its corresponding replacement API from this set, preserving the original bug reproduction logic.'
        )

    additional_instruction = (
        "\n\nAdditional requirement:\n"
        "Please modify the prompt below to also replace the following source APIs with their corresponding replacement APIs in one single test case.\n"
        "Keep the original test case logic and translate semantics appropriately.\n"
        "Replacements:\n"
        f"{replacement_text}\n"
    )
    prompt = base_prompt.strip() + additional_instruction
    prompt += (
        "IMPORTANT:\n"
        "1. Output valid Python test case code.\n"
        "2. Keep the test case runnable and minimal.\n"
        "3. Include any necessary imports and assertions.\n"
        "4. Replace all listed source APIs in the original test case with the given APIs.\n"
    )
    return prompt


def resolve_main_api(issue: Dict[str, Any], similarity_data: Dict[str, Any]) -> str:
    main_api = similarity_data.get('core_api') or issue.get('core_api') or ''
    if isinstance(main_api, str) and main_api.upper() == 'NO_CORE_API':
        original_core_api = issue.get('original_core_api') or issue.get('core_api') or ''
        if original_core_api:
            return original_core_api
        replacement_apis = similarity_data.get('replacement_apis') or []
        if replacement_apis:
            return replacement_apis[0]
    return main_api or ''


def collect_replacement_candidates(
    similarity_data: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    replacement_groups = similarity_data.get('similar_apis_by_replacement', {}) or {}
    source_order = similarity_data.get('replacement_apis') or list(replacement_groups.keys())
    result: Dict[str, List[Dict[str, Any]]] = {}

    for source_api in source_order:
        groups = replacement_groups.get(source_api, {}) or {}
        seen: set = set()
        candidates: List[Dict[str, Any]] = []
        for group, api_list in groups.items():
            for item in api_list:
                api_name = str(item.get('api_name') or '').strip()
                if not api_name or api_name in seen:
                    continue
                seen.add(api_name)
                candidate = dict(item)
                candidate['source_api'] = source_api
                candidate['source_group'] = group
                candidates.append(candidate)
        candidates.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        if candidates:
            result[source_api] = candidates
    return result


def build_replacement_sets(
    source_candidates: Dict[str, List[Dict[str, Any]]],
    top_n: int,
) -> List[List[Dict[str, Any]]]:
    if not source_candidates:
        return []
    lengths = [len(cands) for cands in source_candidates.values()]
    max_sets = min(lengths) if lengths else 0
    max_sets = min(max_sets, top_n)
    replacement_sets: List[List[Dict[str, Any]]] = []
    for rank in range(max_sets):
        current_set: List[Dict[str, Any]] = []
        for source_api, candidates in source_candidates.items():
            if rank >= len(candidates):
                break
            current_set.append(candidates[rank])
        if len(current_set) == len(source_candidates):
            replacement_sets.append(current_set)
    return replacement_sets


def read_base_prompt(issue_id: str, base_prompt_dir: Path) -> Optional[str]:
    candidates = list(base_prompt_dir.glob(f"{issue_id}_*.txt"))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0].read_text(encoding='utf-8')


def build_issue_prompts(
    issue: Dict[str, Any],
    similarity_data: Dict[str, Any],
    output_dir: Path,
    base_prompt_dir: Path,
    top_n: int,
) -> int:
    issue_id = str(issue.get('issue_id') or '')
    if not issue_id:
        return 0

    base_prompt = read_base_prompt(issue_id, base_prompt_dir)
    if not base_prompt:
        return 0

    main_api = resolve_main_api(issue, similarity_data)
    source_candidates = collect_replacement_candidates(similarity_data)
    if not source_candidates:
        return 0

    replacement_sets = build_replacement_sets(source_candidates, top_n)
    index = 0
    for rank, replacement_set in enumerate(replacement_sets, start=1):
        prompt_text = build_multi_api_prompt(
            issue,
            base_prompt,
            replacement_set,
            group='mixed_replacement_set',
            main_api=main_api,
            rank=rank,
        )
        safe_name = stable_short_name(issue_id, 'replacement_set', str(rank))
        prompt_file = output_dir / f"{safe_name}.txt"
        save_text(prompt_file, prompt_text)
        index += 1
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ3 prompts using all replacement APIs.")
    parser.add_argument(
        "--step1-dir",
        type=Path,
        default=DEFAULT_STEP1_DIR,
        help="Directory containing Step1 issue JSON files."
    )
    parser.add_argument(
        "--base-prompt-dir",
        type=Path,
        default=DEFAULT_BASE_PROMPT_DIR,
        help="Directory containing existing prompt files."
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
    parser.add_argument("--top-n", type=int, default=10, help="Number of replacement set ranks to generate.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N issues.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    issue_files = sorted(args.step1_dir.glob("*_issue_ori_data.json"))
    if args.limit > 0:
        issue_files = issue_files[:args.limit]

    print(f"Processing {len(issue_files)} issues...")

    total_prompts = 0
    skipped = 0
    for issue_file in issue_files:
        issue = load_json(issue_file)
        if not issue:
            skipped += 1
            continue

        issue_id = issue.get('issue_id') or issue_file.stem.replace('_issue_ori_data', '')
        similarity_file = args.similarity_dir / f"{issue_id}.json"
        similarity_data = load_json(similarity_file)
        if not similarity_data:
            print(f"Skipping {issue_id}: missing similarity file")
            skipped += 1
            continue

        count = build_issue_prompts(issue, similarity_data, args.output_dir, args.base_prompt_dir, args.top_n)
        if count == 0:
            print(f"Skipping {issue_id}: no prompt candidates generated")
            skipped += 1
            continue
        total_prompts += count
        print(f"Generated {count} prompts for {issue_id}")

    print(f"Done. Generated {total_prompts} prompt files in {args.output_dir}")
    print(f"Skipped issues: {skipped}")


if __name__ == '__main__':
    main()
