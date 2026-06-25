#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ3 ablation: replace code similarity with documentation similarity.

This script keeps Step1 core_api identification, but replaces Step2's code
similarity with TF-IDF cosine similarity computed from API documentation
text/docstrings. It writes the same per-issue similar_api_list JSON format used
by Step3.

Default output:
  Step5-Evaluate/RQ3/data/doc_similarity/similar_api_list

Example:
  python Step5-Evaluate/RQ3/2_no_similarity_calculation.py

Then generate prompts with:
  python Step3-ReuseTestCase/1-generate-prompt.py \
    --similarity-dir Step5-Evaluate/RQ3/data/doc_similarity/similar_api_list \
    --output-dir Step5-Evaluate/RQ3/data/doc_similarity/prompts
"""

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_STEP1_DIR = REPO_ROOT / "Step1-CoreAPIIdentification" / "data" / "pytorch_core_api_identification"
DEFAULT_DOC_SIMILARITY_CSV = SCRIPT_DIR / "data" / "api_documentation_db.csv"
DEFAULT_EQUIVALENT_API_CSV = SCRIPT_DIR / "data" / "equivalent_api_pairs.csv"
DEFAULT_PYTORCH_SOURCE_DIR = REPO_ROOT / "Step2-CalculateSimilarity" / "data" / "pytorch-api-extracted"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "doc_similarity" / "similar_api_list"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


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


def extract_docstring_from_source(source: str) -> str:
    source = source.strip()
    if not source:
        return ""
    try:
        module = ast.parse(source)
    except SyntaxError:
        module = None
    if module is not None:
        module_doc = ast.get_docstring(module) or ""
        docs = [module_doc] if module_doc else []
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                if doc:
                    docs.append(doc)
        if docs:
            return "\n\n".join(docs)

    matches = re.findall(r'(?s)(?:r|u|f|fr|rf)?("""(.*?)"""|\'\'\'(.*?)\'\'\')', source)
    docs = []
    for match in matches:
        doc = match[1] or match[2]
        if doc.strip():
            docs.append(doc.strip())
    return "\n\n".join(docs)


def extract_entry_text(entry: Dict[str, Any], fallback_to_source: bool) -> str:
    direct_fields = [
        "doc",
        "docs",
        "docstring",
        "description",
        "summary",
        "message",
    ]
    parts: List[str] = []
    for field in direct_fields:
        text = as_text(entry.get(field)).strip()
        if text:
            parts.append(text)

    sources: List[str] = []
    if isinstance(entry.get("python_trace"), list):
        for step in entry["python_trace"]:
            if isinstance(step, dict):
                source = as_text(step.get("source")).strip()
                if source:
                    sources.append(source)
    if isinstance(entry.get("source_files"), list):
        for item in entry["source_files"]:
            if isinstance(item, dict):
                source = as_text(item.get("source")).strip()
                if source:
                    sources.append(source)
    source = as_text(entry.get("source")).strip()
    if source:
        sources.append(source)

    for source_text in sources:
        doc = extract_docstring_from_source(source_text)
        if doc:
            parts.append(doc)

    if not parts and fallback_to_source:
        parts.extend(sources)

    text = "\n\n".join(part for part in parts if part.strip())
    return re.sub(r"\s+", " ", text).strip()


def load_api_docs(source_dir: Path, fallback_to_source: bool) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    aggregated = source_dir / "api_sources.json"
    if aggregated.exists():
        data = load_json(aggregated)
        if isinstance(data, dict):
            for api_name, entry in data.items():
                if isinstance(entry, dict):
                    text = extract_entry_text(entry, fallback_to_source)
                    if text:
                        docs[api_name] = text
            return docs

    for path in sorted(source_dir.glob("*.json")):
        if path.name == "api_sources.json":
            continue
        entry = load_json(path)
        if not isinstance(entry, dict):
            continue
        api_name = entry.get("api") or path.stem
        text = extract_entry_text(entry, fallback_to_source)
        if text:
            docs[str(api_name)] = text
    return docs


def issue_doc(issue: Dict[str, Any]) -> str:
    fields = [
        "title",
        "bug_description",
        "error_description",
        "repro_output",
        "affected_api",
        "apis",
        "trigger_conditions",
        "environment",
    ]
    return re.sub(r"\s+", " ", "\n".join(as_text(issue.get(field)) for field in fields)).strip()


def normalize_api(api: str) -> str:
    api = api.strip().strip("`'\".,;:()[]{}")
    if api.startswith("pytorch."):
        api = api[len("pytorch."):]
    if api.startswith("tensorflow."):
        api = api[len("tensorflow."):]
    return api


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", text.lower())


def cosine_from_counters(a: Counter, b: Counter, idf: Dict[str, float]) -> float:
    common = set(a) & set(b)
    numerator = sum((a[token] * idf.get(token, 1.0)) * (b[token] * idf.get(token, 1.0)) for token in common)
    norm_a = math.sqrt(sum((count * idf.get(token, 1.0)) ** 2 for token, count in a.items()))
    norm_b = math.sqrt(sum((count * idf.get(token, 1.0)) ** 2 for token, count in b.items()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(numerator / (norm_a * norm_b))


class DocSimilarityIndex:
    """Reusable TF-IDF index for one target API documentation collection."""

    def __init__(self, target_docs: Dict[str, str], api_type: str):
        self.api_type = api_type
        self.names = sorted(target_docs)
        self.docs = [target_docs[name] for name in self.names]
        self.vectorizer = None
        self.matrix = None
        self.counters: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception:
            self._build_fallback()
            return

        self.vectorizer = TfidfVectorizer(
            max_features=32768,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def _build_fallback(self) -> None:
        self.counters = [Counter(tokenize(text)) for text in self.docs]
        df: Counter = Counter()
        for counter in self.counters:
            for token in counter:
                df[token] += 1
        total_docs = len(self.counters)
        self.idf = {token: math.log((1 + total_docs) / (1 + freq)) + 1.0 for token, freq in df.items()}

    def scores(self, query: str) -> List[float]:
        if not query.strip() or not self.names:
            return [0.0 for _ in self.names]
        if self.vectorizer is not None and self.matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vec = self.vectorizer.transform([query])
            return [float(score) for score in cosine_similarity(query_vec, self.matrix).ravel()]

        query_counter = Counter(tokenize(query))
        return [cosine_from_counters(query_counter, counter, self.idf) for counter in self.counters]


def fallback_tfidf_similarity(query: str, names: Sequence[str], docs: Sequence[str]) -> List[float]:
    corpus = [query] + list(docs)
    counters = [Counter(tokenize(text)) for text in corpus]
    df: Counter = Counter()
    for counter in counters:
        for token in counter:
            df[token] += 1
    total_docs = len(counters)
    idf = {token: math.log((1 + total_docs) / (1 + freq)) + 1.0 for token, freq in df.items()}
    return [cosine_from_counters(counters[0], counters[i + 1], idf) for i in range(len(names))]


def tfidf_similarity(query: str, names: Sequence[str], docs: Sequence[str]) -> List[float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return fallback_tfidf_similarity(query, names, docs)

    if not query.strip() or not docs:
        return [0.0 for _ in docs]
    vectorizer = TfidfVectorizer(
        max_features=32768,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform([query] + list(docs))
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    return [float(score) for score in scores]


def top_similar(
    query_text: str,
    target_index: DocSimilarityIndex,
    top_n: int,
    excluded: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    excluded_set = {api for api in excluded if api}
    scores = target_index.scores(query_text)
    rows = [
        {
            "api_name": name,
            "api_type": target_index.api_type,
            "similarity": score,
            "similarity_method": "doc_tfidf",
        }
        for name, score in zip(target_index.names, scores)
        if name not in excluded_set
    ]
    rows.sort(key=lambda item: item["similarity"], reverse=True)
    return rows[:top_n]


def api_aliases(api: str) -> set:
    api = normalize_api(api)
    aliases = {api}
    if api.startswith("torch."):
        aliases.add(api[len("torch."):])
    elif api:
        aliases.add(f"torch.{api}")
    return {item for item in aliases if item}


def infer_api_type(api: str) -> str:
    if api.startswith("torch") or api.startswith("nn.") or api.startswith("Tensor."):
        return "pytorch"
    if api.startswith("tf") or api.startswith("tensorflow"):
        return "tensorflow"
    return "tensorflow" if "tensorflow" in api else "pytorch"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        print(f"[WARN] Missing CSV file: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as fin:
        return list(csv.DictReader(fin))


def sort_candidate_index(index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    for rows in index.values():
        rows.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
    return index


def add_indexed_candidate(index: Dict[str, List[Dict[str, Any]]], source_api: str, candidate: Dict[str, Any]) -> None:
    if not source_api or not candidate.get("api_name"):
        return
    for alias in api_aliases(source_api):
        index.setdefault(alias, []).append(candidate)


def load_doc_similarity_csv(path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], Dict[str, str]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    pytorch_docs: Dict[str, str] = {}
    tensorflow_docs: Dict[str, str] = {}
    seen_pairs = set()
    for row in read_csv_rows(path):
        pytorch_api = normalize_api(row.get("pytorch_api", ""))
        tf_api = normalize_api(row.get("tf_api", ""))
        if not pytorch_api or not tf_api:
            continue
        try:
            score = float(row.get("similarity", ""))
        except ValueError:
            score = 0.0
        add_indexed_candidate(index, pytorch_api, {
            "api_name": tf_api,
            "api_type": "tensorflow",
            "similarity": score,
            "similarity_method": "doc_similarity_csv",
            "pytorch_doc": row.get("pytorch_doc", ""),
            "target_doc_snippet": row.get("tf_doc", ""),
        })
        pytorch_doc = as_text(row.get("pytorch_doc", "")).strip()
        tensorflow_doc = as_text(row.get("tf_doc", "")).strip()
        if pytorch_doc:
            pytorch_docs.setdefault(pytorch_api, pytorch_doc)
        if tensorflow_doc:
            tensorflow_docs.setdefault(tf_api, tensorflow_doc)
    sort_candidate_index(index)
    return index, pytorch_docs, tensorflow_docs


def load_equivalent_api_csv(path: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    seen = set()
    for row in read_csv_rows(path):
        source_api = normalize_api(row.get("source_api", ""))
        target_api = normalize_api(row.get("target_api", ""))
        if not source_api or not target_api:
            continue
        target_type = infer_api_type(target_api)
        group = "pytorch_to_pytorch" if target_type == "pytorch" else "pytorch_to_tensorflow"
        key = (source_api, target_api, group)
        if key in seen:
            continue
        seen.add(key)
        candidate = {
            "api_name": target_api,
            "api_type": target_type,
            "similarity": 1.0,
            "similarity_method": "equivalent_api_csv",
            "source_doc_snippet": row.get("source_doc_snippet", ""),
            "target_doc_snippet": row.get("target_doc_snippet", ""),
        }
        for alias in api_aliases(source_api):
            index.setdefault(alias, {
                "pytorch_to_tensorflow": [],
                "pytorch_to_pytorch": [],
            })[group].append(candidate)
    for groups in index.values():
        for rows in groups.values():
            rows.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
    return index


def merge_csv_candidates(
    primary: Iterable[Dict[str, Any]],
    secondary: Iterable[Dict[str, Any]],
    top_n: int,
    excluded: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    excluded_set = {normalize_api(api) for api in excluded if api}
    seen = set()
    result: List[Dict[str, Any]] = []
    for item in list(primary) + list(secondary):
        api_name = normalize_api(as_text(item.get("api_name")))
        if not api_name or api_name in seen or api_name in excluded_set:
            continue
        copied = dict(item)
        copied["api_name"] = api_name
        result.append(copied)
        seen.add(api_name)
        if len(result) >= top_n:
            break
    return result


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ3 documentation-similarity ablation candidates using CSV similarity data.")
    parser.add_argument("--step1-dir", type=Path, default=DEFAULT_STEP1_DIR)
    parser.add_argument("--doc-similarity-csv", type=Path, default=DEFAULT_DOC_SIMILARITY_CSV)
    parser.add_argument("--equivalent-api-csv", type=Path, default=DEFAULT_EQUIVALENT_API_CSV)
    parser.add_argument("--pytorch-source-dir", type=Path, default=DEFAULT_PYTORCH_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=11)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--no-source-fallback",
        action="store_true",
        help="Use only CSV documentation text and do not fall back to extracted source/docs when docs are missing.",
    )
    args = parser.parse_args()

    fallback_to_source = not args.no_source_fallback
    print("Loading CSV-based API documentation similarity data...")
    csv_index, pytorch_docs, tensorflow_docs = load_doc_similarity_csv(args.doc_similarity_csv)
    equivalent_index = load_equivalent_api_csv(args.equivalent_api_csv)

    if fallback_to_source:
        print("Loading extracted API documentation text as fallback...")
        fallback_docs = load_api_docs(args.pytorch_source_dir, fallback_to_source=True)
        for api_name, text in fallback_docs.items():
            pytorch_docs.setdefault(api_name, text)

    print(f"Loaded {len(pytorch_docs)} PyTorch API docs from CSV")
    print(f"Loaded {len(tensorflow_docs)} TensorFlow API docs from CSV")
    print("Building reusable document similarity index for PyTorch API docs...")
    pytorch_index = DocSimilarityIndex(pytorch_docs, "pytorch")

    issue_files = sorted(args.step1_dir.glob("*_issue_ori_data.json"))
    if args.limit > 0:
        issue_files = issue_files[:args.limit]

    written = 0
    skipped = 0
    for issue_file in issue_files:
        issue = load_json(issue_file)
        if not issue:
            skipped += 1
            continue
        issue_id = str(issue.get("issue_id") or issue_file.stem.replace("_issue_ori_data", ""))
        core_api = ""
        if isinstance(issue.get("core_api_identification"), dict):
            core_api = normalize_api(as_text(issue["core_api_identification"].get("core_api")))
        if not core_api or core_api.lower() == "none":
            skipped += 1
            continue

        core_doc = pytorch_docs.get(core_api)
        if not core_doc:
            if fallback_to_source:
                core_doc = load_api_docs(args.pytorch_source_dir, True).get(core_api)
            if not core_doc:
                print(f"[SKIP] {issue_id}: no documentation text for core API {core_api}")
                skipped += 1
                continue

        issue_text = issue_doc(issue)
        cross_candidates = merge_csv_candidates(
            csv_index.get(core_api, []),
            equivalent_index.get(core_api, {}).get("pytorch_to_tensorflow", []),
            args.top_n,
            excluded=[core_api],
        )
        result = {
            "issue_id": issue_id,
            "core_api": core_api,
            "ablation": "doc_similarity_instead_of_code_similarity",
            "similar_apis": {
                "pytorch_to_tensorflow": cross_candidates,
                "issue_to_api": top_similar(
                    issue_text,
                    pytorch_index,
                    args.top_n,
                    excluded=[core_api],
                ),
                "pytorch_to_pytorch": top_similar(
                    core_doc,
                    pytorch_index,
                    args.top_n,
                    excluded=[core_api],
                ),
            },
        }
        write_json(args.output_dir / f"{issue_id}.json", result)
        written += 1

    print(f"Done. Wrote {written} JSON files to {args.output_dir}")
    print(f"Skipped issues: {skipped}")


if __name__ == "__main__":
    main()
