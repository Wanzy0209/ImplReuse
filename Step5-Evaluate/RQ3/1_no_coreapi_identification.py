#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ3 ablation: remove core API identification.

Instead of using Step1's identified core_api as the query API, this script uses
every API explicitly mentioned in each issue (metadata fields plus simple code
extraction) as replacement query APIs. It then looks up the same Step2 code
similarity files and writes Step3-compatible per-issue similar_api_list JSON.

Default output:
  Step5-Evaluate/RQ3/data/no_coreapi_identification/similar_api_list

Example:
  python Step5-Evaluate/RQ3/1_no_coreapi_identification.py

Then generate prompts with:
  python Step3-ReuseTestCase/1-generate-prompt.py \
    --similarity-dir Step5-Evaluate/RQ3/data/no_coreapi_identification/similar_api_list \
    --output-dir Step5-Evaluate/RQ3/data/no_coreapi_identification/prompts
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_STEP1_DIR = REPO_ROOT / "Step1-CoreAPIIdentification" / "data" / "pytorch_core_api_identification"
DEFAULT_SIMILARITY_DIR = REPO_ROOT / "Step2-CalculateSimilarity" / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "no_coreapi_identification" / "similar_api_list"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        print(f"[WARN] Missing similarity file: {path}")
        return rows
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                print(f"[WARN] Bad JSONL line {line_no} in {path}: {exc}")
    return rows


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return "\n".join(as_text(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(as_text(v) for v in value.values())
    return str(value)


def normalize_api(api: str) -> str:
    api = api.strip().strip("`'\".,;:()[]{}")
    api = re.sub(r"\s+", "", api)
    return api


def collect_issue_apis(issue: Dict[str, Any]) -> List[str]:
    """Collect all API-like tokens from issue metadata and repro code."""
    candidates: Set[str] = set()

    for field in ("apis", "affected_api"):
        value = issue.get(field)
        if isinstance(value, list):
            for item in value:
                api = normalize_api(as_text(item))
                if api:
                    candidates.add(api)
        else:
            api = normalize_api(as_text(value))
            if api:
                candidates.add(api)

    text_fields = [
        "best_code",
        "repro_code",
        "bug_description",
        "error_description",
        "repro_output",
        "title",
    ]
    text = "\n".join(as_text(issue.get(field)) for field in text_fields)
    patterns = [
        r"\btorch(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
        r"\bTensor\.[A-Za-z_][A-Za-z0-9_]*",
        r"\bnn\.[A-Za-z_][A-Za-z0-9_]*",
        r"\btf(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            candidates.add(normalize_api(match))

    cleaned = []
    for api in sorted(candidates):
        if not api or api.lower() in {"none", "numpy", "np", "torch", "tf"}:
            continue
        cleaned.append(api)
    return cleaned


def build_cross_index(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = row.get("pytorch_api")
        target = row.get("tensorflow_api")
        score = row.get("similarity")
        if source and target and score is not None:
            index[source].append({
                "api_name": target,
                "api_type": "tensorflow",
                "similarity": score,
            })
    return sort_index(index)


def build_single_index(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = row.get("pytorch_api_1")
        target = row.get("pytorch_api_2")
        score = row.get("similarity")
        if source and target and score is not None:
            index[source].append({
                "api_name": target,
                "api_type": "pytorch",
                "similarity": score,
            })
    return sort_index(index)


def build_issue_index(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        issue_id = str(row.get("issue_id") or "")
        api_name = row.get("api_name") or row.get("api")
        score = row.get("similarity")
        if issue_id and api_name and score is not None:
            index[issue_id].append({
                "api_name": api_name,
                "api_type": row.get("api_type", "issue"),
                "similarity": score,
            })
    return sort_index(index)


def sort_index(index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    for rows in index.values():
        rows.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
    return index


def dedupe_top(
    items: Iterable[Tuple[str, Dict[str, Any]]],
    top_n: int,
    excluded: Set[str],
) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    result: List[Dict[str, Any]] = []
    for replacement_api, item in items:
        api_name = item.get("api_name")
        if not api_name or api_name in seen or api_name in excluded:
            continue
        copied = dict(item)
        copied["replacement_api"] = replacement_api
        result.append(copied)
        seen.add(api_name)
        if len(result) >= top_n:
            break
    return result


def merged_candidates(
    replacement_apis: List[str],
    index: Dict[str, List[Dict[str, Any]]],
    top_n: int,
    excluded: Set[str],
) -> List[Dict[str, Any]]:
    pool: List[Tuple[str, Dict[str, Any]]] = []
    for replacement_api in replacement_apis:
        for item in index.get(replacement_api, []):
            pool.append((replacement_api, item))
    pool.sort(key=lambda pair: pair[1].get("similarity", 0.0), reverse=True)
    return dedupe_top(pool, top_n, excluded)


def grouped_candidates(
    replacement_apis: List[str],
    cross_index: Dict[str, List[Dict[str, Any]]],
    issue_items: List[Dict[str, Any]],
    single_index: Dict[str, List[Dict[str, Any]]],
    top_n: int,
    original_core_api: str,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for replacement_api in replacement_apis:
        excluded = {replacement_api}
        if original_core_api:
            excluded.add(original_core_api)
        grouped[replacement_api] = {
            "pytorch_to_tensorflow": dedupe_top(
                [(replacement_api, item) for item in cross_index.get(replacement_api, [])],
                top_n,
                excluded,
            ),
            "issue_to_api": issue_items[:top_n],
            "pytorch_to_pytorch": dedupe_top(
                [(replacement_api, item) for item in single_index.get(replacement_api, [])],
                top_n,
                excluded,
            ),
        }
    return grouped


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ3 no-core-API ablation candidates.")
    parser.add_argument("--step1-dir", type=Path, default=DEFAULT_STEP1_DIR)
    parser.add_argument("--similarity-dir", type=Path, default=DEFAULT_SIMILARITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=11)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--split-by-replacement",
        action="store_true",
        help="Write one file per issue/replacement API instead of one merged file per issue.",
    )
    args = parser.parse_args()

    print("Loading Step2 similarity files...")
    cross_index = build_cross_index(load_jsonl(args.similarity_dir / "api_pairwise_similarity.jsonl"))
    issue_index = build_issue_index(load_jsonl(args.similarity_dir / "issue_api_similarity.jsonl"))
    single_index = build_single_index(load_jsonl(args.similarity_dir / "pytorch_pairwise_similarity.jsonl"))

    issue_files = sorted(args.step1_dir.glob("*_issue_ori_data.json"))
    if args.limit > 0:
        issue_files = issue_files[:args.limit]

    written = 0
    skipped = 0
    replacement_total = 0
    for issue_file in issue_files:
        issue = load_json(issue_file)
        if not issue:
            skipped += 1
            continue
        issue_id = str(issue.get("issue_id") or issue_file.stem.replace("_issue_ori_data", ""))
        original_core_api = ""
        if isinstance(issue.get("core_api_identification"), dict):
            original_core_api = normalize_api(as_text(issue["core_api_identification"].get("core_api")))

        replacement_apis = collect_issue_apis(issue)
        replacement_apis = [api for api in replacement_apis if api != original_core_api]
        if not replacement_apis:
            skipped += 1
            continue

        replacement_total += len(replacement_apis)
        excluded = set(replacement_apis)
        if original_core_api:
            excluded.add(original_core_api)

        if args.split_by_replacement:
            for idx, replacement_api in enumerate(replacement_apis):
                result = {
                    "issue_id": issue_id,
                    "core_api": replacement_api,
                    "ablation": "no_coreapi_identification",
                    "original_core_api": original_core_api,
                    "replacement_apis": [replacement_api],
                    "similar_apis": {
                        "pytorch_to_tensorflow": dedupe_top(
                            [(replacement_api, item) for item in cross_index.get(replacement_api, [])],
                            args.top_n,
                            {replacement_api, original_core_api},
                        ),
                        "issue_to_api": issue_index.get(issue_id, [])[:args.top_n],
                        "pytorch_to_pytorch": dedupe_top(
                            [(replacement_api, item) for item in single_index.get(replacement_api, [])],
                            args.top_n,
                            {replacement_api, original_core_api},
                        ),
                    },
                }
                safe_api = re.sub(r"[^A-Za-z0-9_.-]+", "_", replacement_api).replace(".", "_")
                write_json(args.output_dir / f"{issue_id}__{idx}_{safe_api}.json", result)
                written += 1
        else:
            result = {
                "issue_id": issue_id,
                "core_api": "NO_CORE_API",
                "ablation": "no_coreapi_identification",
                "original_core_api": original_core_api,
                "replacement_apis": replacement_apis,
                "similar_apis": {
                    "pytorch_to_tensorflow": merged_candidates(replacement_apis, cross_index, args.top_n, excluded),
                    "issue_to_api": issue_index.get(issue_id, [])[:args.top_n],
                    "pytorch_to_pytorch": merged_candidates(replacement_apis, single_index, args.top_n, excluded),
                },
                "similar_apis_by_replacement": grouped_candidates(
                    replacement_apis,
                    cross_index,
                    issue_index.get(issue_id, []),
                    single_index,
                    args.top_n,
                    original_core_api,
                ),
            }
            write_json(args.output_dir / f"{issue_id}.json", result)
            written += 1

    print(f"Done. Wrote {written} JSON files to {args.output_dir}")
    print(f"Skipped issues: {skipped}")
    print(f"Replacement APIs used: {replacement_total}")


if __name__ == "__main__":
    main()
