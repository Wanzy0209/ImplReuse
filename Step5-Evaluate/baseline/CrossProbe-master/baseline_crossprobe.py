# -------------------- 依赖安装与导入 --------------------
# 推荐在命令行先运行: pip install transformers torch 'numpy<2' pandas tqdm openai pylint torchvision
import os
import ast
import textwrap
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# -------------------- API抽取工具 --------------------
class APIVisitor(ast.NodeVisitor):
    def __init__(self):
        self.apis = []
        self.current_class = None
    def visit_FunctionDef(self, node):
        if node.name.startswith("_"):
            return
        func_doc = ast.get_docstring(node)
        if func_doc is None:
            return
        prefix = self.current_class + "." if self.current_class else ""
        self.apis.append({
            "name": prefix + node.name,
            "doc": func_doc.strip(),
            "type": "method" if self.current_class else "function",
        })
        self.generic_visit(node)

def extract_api_info(source_path):
    with open(source_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return []
    visitor = APIVisitor()
    visitor.visit(tree)
    return visitor.apis

def scan_project(project_root, project_name):
    all_apis = []
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith(".py") and not file.startswith("_"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_root)
                module_path = rel_path.replace("/", ".").replace("\\", ".")[:-3]
                for api in extract_api_info(file_path):
                    full_name = (
                        f"{module_path}.{api['name']}"
                        if module_path != "."
                        else api["name"]
                    )
                    all_apis.append({
                        "project": project_name,
                        "name": full_name,
                        "content": textwrap.dedent(api["doc"]).strip(),
                        "type": api["type"],
                    })
    return all_apis

# -------------------- 文档加载与预处理 --------------------
tf_docs = scan_project("data/tensorflow-2.17.0", "tensorflow")
torch_docs = scan_project("data/pytorch-2.4.0", "pytorch")
print(f"Loaded {len(tf_docs)} TensorFlow APIs")
print(f"Loaded {len(torch_docs)} PyTorch APIs")

def preprocess_text(text):
    text = text.replace("```", "")
    text = " ".join(text.split())
    return text[:2000]
for doc in tf_docs + torch_docs:
    doc["processed_text"] = preprocess_text(doc["content"])

# -------------------- BERT嵌入生成 --------------------

# ----------- CodeBERT本地模型加载 -------------
local_model_dir = Path(__file__).parent / 'codebert-base'
print(f"[INFO] Using CodeBERT model from: {local_model_dir}")
try:
    from transformers import RobertaTokenizer, RobertaModel
    import torch

    # Load CodeBERT model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(local_model_dir)
    model = RobertaModel.from_pretrained(local_model_dir)

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size = 64 if device.type == "cuda" else 16
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except ImportError as e:
    # Handle TypeIs import error and other import issues
    if "TypeIs" in str(e):
        raise RuntimeError(f"CodeBERT model loading failed due to typing_extensions version issue: {e}. Please update typing_extensions with 'pip install --upgrade typing_extensions'")
    else:
        raise RuntimeError(f"CodeBERT model loading failed due to import error: {e}")
except Exception as e:
    raise RuntimeError(f"CodeBERT model loading failed: {e}")

def get_bert_embeddings(texts):
    embeddings = []
    with tqdm(total=len(texts), desc="Generating BERT embeddings") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state.mean(dim=1)
            batch_embeddings = hidden_states.to("cpu").numpy()
            embeddings.append(batch_embeddings)
            pbar.update(len(batch))
    return np.vstack(embeddings)

tf_embeddings = get_bert_embeddings([doc["processed_text"] for doc in tf_docs])
torch_embeddings = get_bert_embeddings([doc["processed_text"] for doc in torch_docs])
print(f"TensorFlow embedding matrix shape: {tf_embeddings.shape}")
print(f"PyTorch embedding matrix shape: {torch_embeddings.shape}")

# -------------------- 代码转换主流程 --------------------

FRAMEWORK_MODULE_ALIASES = {
    'pytorch': ['torch', 'pytorch'],
    'tensorflow': ['tensorflow', 'tf'],
}

def load_doc_db(doc_db_path):
    return pd.read_csv(doc_db_path)


def get_api_columns(framework: str):
    if framework == 'pytorch':
        return 'pytorch_api', 'pytorch_doc'
    if framework == 'tensorflow':
        return 'tf_api', 'tf_doc'
    raise ValueError(f'Unsupported framework: {framework}')


def _short_api_name(api: str):
    if not api:
        return ''
    return api.split('.')[-1]


def _pick_best_match(rows):
    if rows.empty:
        return None
    if 'similarity' in rows.columns:
        rows = rows.sort_values('similarity', ascending=False)
    return rows.iloc[0]


def find_top_match(api: str, source_framework: str, target_framework: str, doc_db):
    source_col, _ = get_api_columns(source_framework)
    target_col, _ = get_api_columns(target_framework)
    if source_framework == target_framework:
        return api
    exact = doc_db[doc_db[source_col] == api]
    if not exact.empty:
        return exact[target_col].iloc[0]

    basename = _short_api_name(api)
    if basename:
        suffix_mask = doc_db[source_col].astype(str).str.endswith('.' + basename)
        suffix_candidates = doc_db[suffix_mask]
        if not suffix_candidates.empty:
            best = _pick_best_match(suffix_candidates)
            return best[target_col]

    contains_candidates = doc_db[doc_db[source_col].astype(str).str.contains(api, na=False, regex=False)]
    if not contains_candidates.empty:
        best = _pick_best_match(contains_candidates)
        return best[target_col]

    if basename:
        basename_candidates = doc_db[doc_db[source_col].astype(str).str.contains(basename, na=False, regex=False)]
        if not basename_candidates.empty:
            best = _pick_best_match(basename_candidates)
            return best[target_col]

    return ''


def get_api_doc(api: str, framework: str, doc_db):
    if not api:
        return ''
    api_col, doc_col = get_api_columns(framework)
    row = doc_db[doc_db[api_col] == api]
    if row.empty:
        return ''
    return row[doc_col].iloc[0]

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
        return ''

def extract_apis(code: str, framework: str):
    try:
        tree = ast.parse(code)
    except Exception as e:
        print(f"[extract_apis] ast.parse 失败: {e}")
        return []
    extractor = APIExtractor(framework)
    extractor.visit(tree)
    return list(set(extractor.api_calls))

def save_equivalent_api_pairs(path, api_pairs):
    if not api_pairs:
        return
    df = pd.DataFrame(api_pairs)
    df.to_csv(path, index=False, encoding='utf-8')


def save_prompt_records(path, prompt_records):
    if not prompt_records:
        return
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(prompt_records, f, ensure_ascii=False, indent=2)


def build_prompt(code, source_framework, target_framework, doc_db):
    apis = extract_apis(code, source_framework)
    context = []
    api_pairs = []
    for api in apis:
        source_doc = get_api_doc(api, source_framework, doc_db)
        target_api = find_top_match(api, source_framework, target_framework, doc_db)
        target_doc = get_api_doc(target_api, target_framework, doc_db) if target_api else ''
        source_snippet = source_doc[:500]
        target_snippet = target_doc[:500]
        api_pairs.append({
            'source_api': api,
            'target_api': target_api,
            'source_doc_snippet': source_snippet,
            'target_doc_snippet': target_snippet,
        })
        context.append(
            f"Source API ({'PyTorch' if source_framework == 'pytorch' else 'TensorFlow'}): {api}\n"
            f"Documentation: {source_snippet}\n\n"
            f"Target API ({target_framework}): {target_api}\n"
            f"Documentation: {target_snippet}\n"
            "----------------------------------------"
        )
    context_str = "\n".join(context)
    prompt = f"""
Documentation Context:
{context_str}

Source Code to Translate:
```python
{code}
```

Requirements:
1. Output only valid {target_framework} code
2. Preserve comments and code structure
3. Add conversion comments where non-trivial
4. Include necessary imports
"""
    return prompt, api_pairs


def mock_translate(code, target_framework):
    return f"# MOCK_TRANSLATION to {target_framework}\n# original source preserved below\n{code}"


def call_zhipu_llm(api_key, prompt, model='glm-4.6', temperature=0.0, max_tokens=8192):
    """调用智谱AI LLM API"""
    try:
        from zai import ZhipuAiClient
    except Exception as e:
        raise RuntimeError('zhipu SDK not installed or import failed: %s' % e)
    try:
        client = ZhipuAiClient(api_key=api_key)
        messages = [
            {"role": "system", "content": "You are an expert Python developer specializing in fixing test cases for deep learning libraries."},
            {"role": "user", "content": prompt}
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def process_directory(source_dir: str, target_dir: str, source_framework: str, target_framework: str, doc_db, api_key=None, use_real_llm=False):
    os.makedirs(target_dir, exist_ok=True)
    prompt_records = []
    equivalent_api_pairs = []
    for root, _, files in os.walk(source_dir):
        for file in tqdm(files, desc=f"Transferring from {source_framework}"):
            if file.endswith(".py"):
                source_path = os.path.join(root, file)
                rel_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, rel_path)
                with open(source_path, "r", encoding="utf-8") as f:
                    code = f.read()
                prompt, api_pairs = build_prompt(code, source_framework, target_framework, doc_db)
                prompt_records.append({
                    'source_path': rel_path,
                    'source_framework': source_framework,
                    'target_framework': target_framework,
                    'source_apis': [pair['source_api'] for pair in api_pairs],
                    'target_apis': [pair['target_api'] for pair in api_pairs],
                    'prompt': prompt,
                })
                equivalent_api_pairs.extend(api_pairs)
                if use_real_llm:
                    translated = call_zhipu_llm(api_key=api_key, prompt=prompt)
                    if translated is None:
                        translated = mock_translate(code, target_framework)
                else:
                    translated = mock_translate(code, target_framework)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(translated)
    return prompt_records, equivalent_api_pairs

# -------------------- 代码格式检查与运行 --------------------
import subprocess

def format_file(file):
    with open(file, "r+") as f:
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
            subprocess.run([
                "pylint", "-E", file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
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
            subprocess.run([
                "python", file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            success.append(file)
        except subprocess.CalledProcessError:
            failed.append(file)
            print(f"Error running {file}")
    print(f"Success: {len(success)}")
    print(f"Failed: {len(failed)}")

# -------------------- 主流程入口 --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CrossProbe Code Transfer Pipeline")
    parser.add_argument('--doc_db_path', type=str, default="api_documentation_db.csv")
    parser.add_argument('--source_dir', type=str, default='filtered_test_cases')
    parser.add_argument('--target_dir', type=str, default='results')
    parser.add_argument('--source_framework', type=str, default='pytorch')
    parser.add_argument('--target_framework', type=str, default='tensorflow')
    parser.add_argument('--api_key', type=str, default='044f2bc486d14e13aa68259fbd1970c4.hjLhKCmeXaDRt2xf')
    parser.add_argument('--equivalent_api_path', type=str, default='equivalent_api_pairs.csv')
    parser.add_argument('--prompt_info_path', type=str, default='prompt_records.json')
    parser.add_argument('--use_real_llm', action='store_true', help='Use real Zhipu API instead of mock translation', default=False)
    args = parser.parse_args()

    # 加载API对齐数据库
    doc_db = load_doc_db(args.doc_db_path)

    # 代码转换
    prompt_records, equivalent_api_pairs = process_directory(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        source_framework=args.source_framework,
        target_framework=args.target_framework,
        doc_db=doc_db,
        api_key=args.api_key,
        use_real_llm=args.use_real_llm
    )

    # 存储等价 API 对与 prompt 信息
    save_equivalent_api_pairs(args.equivalent_api_path, equivalent_api_pairs)
    save_prompt_records(args.prompt_info_path, prompt_records)

    # 格式检查与运行
    check_and_run_tests(args.target_dir)
