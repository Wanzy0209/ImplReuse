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

# -------------------- 嵌入生成与相似度计算 --------------------

def initialize_embeddings(skip_embedding=False):
    if skip_embedding:
        print("[INFO] Using TF-IDF embedding for API similarity calculation")
        return generate_tfidf_embeddings()
    
    local_model_dir = Path(__file__).parent / 'codebert-base'
    print(f"[INFO] Using CodeBERT model from: {local_model_dir}")
    try:
        from transformers import RobertaTokenizer, RobertaModel
        import torch

        tokenizer = RobertaTokenizer.from_pretrained(local_model_dir)
        model = RobertaModel.from_pretrained(local_model_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch_size = 64 if device.type == "cuda" else 16
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        return tf_embeddings, torch_embeddings
    except Exception as e:
        print(f"[WARNING] CodeBERT model loading failed: {e}")
        print("[INFO] Falling back to TF-IDF embedding")
        return generate_tfidf_embeddings()


def generate_tfidf_embeddings():
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tf_texts = [doc["processed_text"] for doc in tf_docs]
    torch_texts = [doc["processed_text"] for doc in torch_docs]
    
    all_texts = tf_texts + torch_texts
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        lowercase=True
    )
    vectorizer.fit(all_texts)
    
    tf_embeddings = vectorizer.transform(tf_texts).toarray()
    torch_embeddings = vectorizer.transform(torch_texts).toarray()
    
    print(f"TensorFlow TF-IDF embedding shape: {tf_embeddings.shape}")
    print(f"PyTorch TF-IDF embedding shape: {torch_embeddings.shape}")
    return tf_embeddings, torch_embeddings


def compute_similarity_matrix(source_embeddings, target_embeddings):
    source_norm = source_embeddings / (np.linalg.norm(source_embeddings, axis=1, keepdims=True) + 1e-8)
    target_norm = target_embeddings / (np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = source_norm @ target_norm.T
    return similarity_matrix

# -------------------- 代码转换主流程 --------------------

FRAMEWORK_MODULE_ALIASES = {
    'pytorch': ['torch', 'pytorch'],
    'tensorflow': ['tensorflow', 'tf'],
}

def load_doc_db(doc_db_path=None):
    tf_api_map = {doc["name"]: doc for doc in tf_docs}
    torch_api_map = {doc["name"]: doc for doc in torch_docs}
    
    rows = []
    for tf_doc in tf_docs:
        for torch_doc in torch_docs:
            rows.append({
                'tf_api': tf_doc["name"],
                'tf_doc': tf_doc["content"],
                'pytorch_api': torch_doc["name"],
                'pytorch_doc': torch_doc["content"],
                'similarity': 0.0
            })
    
    df = pd.DataFrame(rows)
    print(f"[INFO] Created full cross-framework API database with {len(df)} rows")
    return df


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


def _pick_best_match(rows, top_k=1):
    if rows.empty:
        return rows
    if 'similarity' in rows.columns:
        rows = rows.sort_values('similarity', ascending=False)
    return rows.head(top_k)


tf_api_names = [doc["name"] for doc in tf_docs]
torch_api_names = [doc["name"] for doc in torch_docs]
tf_api_map = {doc["name"]: i for i, doc in enumerate(tf_docs)}
torch_api_map = {doc["name"]: i for i, doc in enumerate(torch_docs)}

_similarity_matrix = None

def init_similarity_matrix(source_embeddings, target_embeddings):
    global _similarity_matrix
    print("[INFO] Computing full cross-framework similarity matrix...")
    _similarity_matrix = compute_similarity_matrix(source_embeddings, target_embeddings)
    print(f"[INFO] Similarity matrix shape: {_similarity_matrix.shape}")


def find_top_match(api: str, source_framework: str, target_framework: str, doc_db=None, top_k=1):
    if source_framework == target_framework:
        return [api] * top_k

    global _similarity_matrix
    
    if source_framework == 'pytorch':
        source_map, target_map, target_names = torch_api_map, tf_api_map, tf_api_names
    else:
        source_map, target_map, target_names = tf_api_map, torch_api_map, torch_api_names

    api_idx = None
    
    if api in source_map:
        api_idx = source_map[api]
    else:
        basename = _short_api_name(api)
        candidates = [name for name in source_map if name.endswith('.' + basename) or basename in name]
        if candidates:
            candidate_scores = []
            for candidate in candidates:
                parts_api = api.split('.')
                parts_candidate = candidate.split('.')
                score = sum(1 for p in parts_api if p in parts_candidate)
                candidate_scores.append((score, candidate))
            candidate_scores.sort(reverse=True, key=lambda x: x[0])
            best_candidate = candidate_scores[0][1]
            api_idx = source_map[best_candidate]
            print(f"[INFO] API '{api}' not found, using best match: '{best_candidate}'")
    
    if api_idx is None:
        return [''] * top_k

    if _similarity_matrix is None:
        print("[ERROR] Similarity matrix not initialized!")
        return [''] * top_k

    if source_framework == 'pytorch':
        similarities = _similarity_matrix[api_idx]
    else:
        similarities = _similarity_matrix[:, api_idx]
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [target_names[idx] for idx in top_indices]
    
    return results


def get_api_doc(api: str, framework: str, doc_db=None):
    if not api:
        return ''
    
    if framework == 'pytorch':
        for doc in torch_docs:
            if doc["name"] == api:
                return doc["content"]
    else:
        for doc in tf_docs:
            if doc["name"] == api:
                return doc["content"]
    return ''

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


def build_prompt(code, source_framework, target_framework, doc_db, top_k=1):
    apis = extract_apis(code, source_framework)
    all_prompts = []
    all_api_pairs_list = []
    
    for api in apis:
        source_doc = get_api_doc(api, source_framework, doc_db)
        target_apis = find_top_match(api, source_framework, target_framework, doc_db, top_k=top_k)
        
        api_matches = []
        for idx, target_api in enumerate(target_apis):
            target_doc = get_api_doc(target_api, target_framework, doc_db) if target_api else ''
            source_snippet = source_doc[:500]
            target_snippet = target_doc[:500]
            api_matches.append({
                'source_api': api,
                'target_api': target_api,
                'source_doc_snippet': source_snippet,
                'target_doc_snippet': target_snippet,
                'rank': idx + 1,
            })
        all_api_pairs_list.append(api_matches)
    
    for k_idx in range(top_k):
        context = []
        api_pairs = []
        for api_matches in all_api_pairs_list:
            if k_idx < len(api_matches):
                match = api_matches[k_idx]
            else:
                match = api_matches[-1] if api_matches else None
            
            if match:
                api_pairs.append(match)
                context.append(
                    f"Source API ({'PyTorch' if source_framework == 'pytorch' else 'TensorFlow'}): {match['source_api']}\n"
                    f"Documentation: {match['source_doc_snippet']}\n\n"
                    f"Target API ({target_framework}): {match['target_api']}\n"
                    f"Documentation: {match['target_doc_snippet']}\n"
                    f"Matching Rank: {match['rank']}\n"
                    "----------------------------------------"
                )
        
        context_str = "\n".join(context)
        prompt = f"""
Documentation Context (Top-{k_idx+1} matching APIs):
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
        all_prompts.append((prompt, api_pairs))
    
    return all_prompts


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

def process_directory(source_dir: str, target_dir: str, source_framework: str, target_framework: str, doc_db, api_key=None, use_real_llm=False, top_k=1):
    os.makedirs(target_dir, exist_ok=True)
    prompt_records = []
    equivalent_api_pairs = []
    for root, _, files in os.walk(source_dir):
        for file in tqdm(files, desc=f"Transferring from {source_framework}"):
            if file.endswith(".py"):
                source_path = os.path.join(root, file)
                rel_path = os.path.relpath(source_path, source_dir)
                with open(source_path, "r", encoding="utf-8") as f:
                    code = f.read()
                all_prompts = build_prompt(code, source_framework, target_framework, doc_db, top_k=top_k)
                
                for k_idx, (prompt, api_pairs) in enumerate(all_prompts):
                    if top_k > 1:
                        base_name, ext = os.path.splitext(rel_path)
                        ranked_rel_path = f"{base_name}_top{k_idx+1}{ext}"
                        ranked_target_path = os.path.join(target_dir, ranked_rel_path)
                    else:
                        ranked_rel_path = rel_path
                        ranked_target_path = os.path.join(target_dir, rel_path)
                    
                    prompt_records.append({
                        'source_path': rel_path,
                        'rank': k_idx + 1,
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
                    os.makedirs(os.path.dirname(ranked_target_path), exist_ok=True)
                    with open(ranked_target_path, "w", encoding="utf-8") as f:
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
    parser.add_argument('--api_key', type=str, default=os.getenv("ZHIPUAI_API_KEY"))
    parser.add_argument('--equivalent_api_path', type=str, default='equivalent_api_pairs.csv')
    parser.add_argument('--prompt_info_path', type=str, default='prompt_records.json')
    parser.add_argument('--use_real_llm', action='store_true', help='Use real Zhipu API instead of mock translation', default=True)
    parser.add_argument('--top_k', type=int, default=10, help='Number of top matching APIs to generate (default: 1)')
    parser.add_argument('--use_tfidf', action='store_true', help='Use TF-IDF instead of CodeBERT for similarity', default=True)
    args = parser.parse_args()

    # 初始化嵌入（使用 TF-IDF 或 CodeBERT）
    tf_embeddings, torch_embeddings = initialize_embeddings(args.use_tfidf)

    # 初始化跨框架相似度矩阵（叉乘计算）
    init_similarity_matrix(torch_embeddings, tf_embeddings)

    # 加载API对齐数据库（现在是动态生成的完整叉乘数据库）
    doc_db = load_doc_db()

    # 代码转换
    prompt_records, equivalent_api_pairs = process_directory(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        source_framework=args.source_framework,
        target_framework=args.target_framework,
        doc_db=doc_db,
        api_key=args.api_key,
        use_real_llm=args.use_real_llm,
        top_k=args.top_k
    )

    # 存储等价 API 对与 prompt 信息
    save_equivalent_api_pairs(args.equivalent_api_path, equivalent_api_pairs)
    save_prompt_records(args.prompt_info_path, prompt_records)

    # 格式检查与运行
    check_and_run_tests(args.target_dir)
