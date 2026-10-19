# -*- coding: utf-8 -*-
"""
Single-library baseline script.

This script is a single-framework version of the CrossProbe-style pipeline.
It scans source code, extracts framework API calls, looks up documentation
for the same framework, and generates prompt-based outputs without doing
cross-framework API matching.
"""

import argparse
import ast
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pandas as pd
from tqdm import tqdm

FRAMEWORK_MODULE_ALIASES = {
    "pytorch": ["torch", "pytorch"],
    "tensorflow": ["tensorflow", "tf"],
}


class APIExtractor(ast.NodeVisitor):
    def __init__(self, framework: str):
        self.framework = framework
        self.alias_map = {}
        self.api_calls = []
        self.module_names = FRAMEWORK_MODULE_ALIASES.get(framework, [framework])

    def visit_Import(self, node):
        for alias in node.names:
            fullname = alias.name
            asname = alias.asname or alias.name
            self.alias_map[asname] = fullname

    def visit_ImportFrom(self, node):
        if node.module is None:
            return
        for alias in node.names:
            fullname = f"{node.module}.{alias.name}"
            asname = alias.asname or alias.name
            self.alias_map[asname] = fullname

    def visit_Call(self, node):
        api_path = self._get_full_path(node.func)
        if api_path and self._is_framework_api(api_path):
            self.api_calls.append(api_path)
        self.generic_visit(node)

    def _is_framework_api(self, api_path: str):
        for module_name in self.module_names:
            if api_path == module_name or api_path.startswith(f"{module_name}."):
                return True
        return False

    def _get_full_path(self, node):
        if isinstance(node, ast.Attribute):
            prefix = self._get_full_path(node.value)
            if prefix:
                return f"{prefix}.{node.attr}"
            return node.attr
        if isinstance(node, ast.Name):
            return self.alias_map.get(node.id, node.id)
        return ""


def extract_apis(code: str, framework: str):
    try:
        tree = ast.parse(code)
    except Exception as e:
        print(f"[extract_apis] ast.parse failed: {e}")
        return []
    extractor = APIExtractor(framework)
    extractor.visit(tree)
    return list(set(extractor.api_calls))


def load_doc_db(doc_db_path: str, framework: str):
    df = pd.read_csv(doc_db_path)
    if f"{framework}_api" in df.columns and f"{framework}_doc" in df.columns:
        return df, f"{framework}_api", f"{framework}_doc"
    if "api" in df.columns and "doc" in df.columns:
        return df, "api", "doc"
    raise ValueError(
        f"Unsupported documentation table for framework '{framework}'. Expected columns '{framework}_api'/'{framework}_doc' or 'api'/'doc'."
    )


def get_api_doc(api: str, doc_db, api_col: str, doc_col: str):
    if not api:
        return ""
    row = doc_db[doc_db[api_col] == api]
    if row.empty:
        return ""
    return row[doc_col].iloc[0]


def preprocess_text(text):
    text = text.replace("```", "")
    text = " ".join(text.split())
    return text[:2000]


def build_prompt(code: str, framework: str, doc_db, api_col: str, doc_col: str):
    apis = extract_apis(code, framework)
    context = []
    api_records = []
    for api in apis:
        doc = get_api_doc(api, doc_db, api_col, doc_col)
        doc_snippet = preprocess_text(doc)[:500]
        api_records.append({
            "api": api,
            "doc_snippet": doc_snippet,
        })
        context.append(
            f"API ({framework}): {api}\n"
            f"Documentation: {doc_snippet}\n"
            "----------------------------------------"
        )
    context_str = "\n".join(context)
    prompt = f"""
Documentation Context:
{context_str}

Source Code to Process:
```python
{code}
```

Requirements:
1. Output only valid {framework} code
2. Preserve comments and code structure
3. Add conversion comments where non-trivial
4. Include necessary imports
"""
    return prompt, api_records


def mock_translate(code: str, framework: str):
    return f"# SINGLE_LIBRARY_BASELINE for {framework}\n# original source preserved below\n{code}"


def call_zhipu_llm(api_key, prompt, model="glm-4.6", temperature=0.0, max_tokens=8192):
    try:
        from zai import ZhipuAiClient
    except Exception as e:
        raise RuntimeError("zhipu SDK not installed or import failed: %s" % e)
    try:
        client = ZhipuAiClient(api_key=api_key)
        messages = [
            {"role": "system", "content": "You are an expert Python developer specializing in deep learning framework code."},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def process_directory(source_dir: str, target_dir: str, framework: str, doc_db, api_col: str, doc_col: str, api_key=None, use_real_llm=False):
    os.makedirs(target_dir, exist_ok=True)
    prompt_records = []
    for root, _, files in os.walk(source_dir):
        for file in tqdm(files, desc=f"Processing {framework}"):
            if not file.endswith(".py"):
                continue
            source_path = os.path.join(root, file)
            rel_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(target_dir, rel_path)
            with open(source_path, "r", encoding="utf-8") as f:
                code = f.read()
            prompt, api_records = build_prompt(code, framework, doc_db, api_col, doc_col)
            prompt_records.append({
                "source_path": rel_path,
                "framework": framework,
                "apis": [item["api"] for item in api_records],
                "prompt": prompt,
            })
            if use_real_llm:
                translated = call_zhipu_llm(api_key=api_key, prompt=prompt)
                if translated is None:
                    translated = mock_translate(code, framework)
            else:
                translated = mock_translate(code, framework)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(translated)
    return prompt_records


def save_prompt_records(path, prompt_records):
    if not prompt_records:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompt_records, f, ensure_ascii=False, indent=2)


def format_file(file):
    with open(file, "r+", encoding="utf-8") as f:
        contents = ""
        for line in f.readlines():
            if line.startswith(" ") and not line.startswith("  "):
                new_line = line[1:]
                contents += new_line
            else:
                contents += line
        f.seek(0)
        f.truncate()
        f.write(contents)


def check_and_run_tests(test_dir):
    valid = []
    invalid = []
    for f in os.listdir(test_dir):
        file = os.path.join(test_dir, f)
        if not file.endswith(".py"):
            continue
        format_file(file)
        try:
            subprocess.run(["pylint", "-E", file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            valid.append(file)
        except subprocess.CalledProcessError:
            invalid.append(file)
            print(f"Invalid code {file}")
    print(f"Valid: {len(valid)}")
    print(f"Invalid: {len(invalid)}")
    success = []
    failed = []
    for file in valid:
        try:
            subprocess.run(["python", file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            success.append(file)
        except subprocess.CalledProcessError:
            failed.append(file)
            print(f"Error running {file}")
    print(f"Success: {len(success)}")
    print(f"Failed: {len(failed)}")


def main():
    parser = argparse.ArgumentParser(description="Single-library baseline pipeline")
    parser.add_argument("--doc_db_path", type=str, default="api_documentation_db.csv")
    parser.add_argument("--source_dir", type=str, default="test_case")
    parser.add_argument("--target_dir", type=str, default="results_single")
    parser.add_argument("--framework", type=str, default="pytorch", choices=["pytorch", "tensorflow"])
    parser.add_argument("--api_key", type=str, default=os.getenv("ZHIPUAI_API_KEY"))
    parser.add_argument("--prompt_info_path", type=str, default="prompt_records_single.json")
    parser.add_argument("--use_real_llm", action="store_true", help="Use real Zhipu API instead of mock translation")
    args = parser.parse_args()

    doc_db, api_col, doc_col = load_doc_db(args.doc_db_path, args.framework)
    prompt_records = process_directory(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        framework=args.framework,
        doc_db=doc_db,
        api_col=api_col,
        doc_col=doc_col,
        api_key=args.api_key,
        use_real_llm=args.use_real_llm,
    )
    save_prompt_records(args.prompt_info_path, prompt_records)
    check_and_run_tests(args.target_dir)


if __name__ == "__main__":
    main()
