#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate similarity between issue sub_fix_code and framework API codes.

This script reads:
- Issue data with sub_fix_code from: Step0-DataCollection/data/pytorch_llm_processed_data_fix_code
- PyTorch API source data from: data/pytorch-api-extracted
- TensorFlow API source data from: data/tensorflow-api-extracted

It computes a similarity score for every sub_fix_code against all PyTorch and
TensorFlow APIs, writing results as JSONL with x*(y+z) pairs where:
- x = number of issues with sub_fix_code
- y = number of PyTorch APIs
- z = number of TensorFlow APIs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def load_api_sources(source_dir: Path) -> Dict[str, str]:
    api_texts: Dict[str, str] = {}
    aggregated = source_dir / 'api_sources.json'
    if aggregated.exists():
        print(f'    [INFO] Loading aggregated API sources from {aggregated}')
        data = read_json(aggregated)
        if isinstance(data, dict):
            print(f'    [INFO] Processing {len(data)} API entries...')
            for i, (api_name, obj) in enumerate(data.items(), 1):
                if i % 100 == 0:
                    print(f'    [PROGRESS] Loading API {i}/{len(data)}')
                code_text = extract_code_text(obj)
                if code_text:
                    api_texts[api_name] = code_text
            print(f'    [INFO] Loaded {len(api_texts)} APIs from aggregated file')
            return api_texts

    json_files = sorted(source_dir.glob('*.json'))
    json_files = [f for f in json_files if f.name != 'api_sources.json']
    print(f'    [INFO] Loading {len(json_files)} individual JSON files...')
    for i, file_path in enumerate(json_files, 1):
        if i % 50 == 0:
            print(f'    [PROGRESS] Loading file {i}/{len(json_files)}: {file_path.name}')
        obj = read_json(file_path)
        if not isinstance(obj, dict):
            continue
        api_name = obj.get('api') or file_path.stem
        code_text = extract_code_text(obj)
        if code_text:
            api_texts[api_name] = code_text
    print(f'    [INFO] Loaded {len(api_texts)} APIs from individual files')
    return api_texts


def load_issue_fix_codes(issue_dir: Path) -> Dict[str, str]:
    fix_codes: Dict[str, str] = {}
    json_files = sorted(issue_dir.glob('*.json'))
    print(f'    [INFO] Loading {len(json_files)} issue JSON files...')
    for i, file_path in enumerate(json_files, 1):
        if i % 50 == 0:
            print(f'    [PROGRESS] Loading issue file {i}/{len(json_files)}: {file_path.name}')
        obj = read_json(file_path)
        if not isinstance(obj, dict):
            continue
        sub_fix_code = obj.get('sub_fix_code')
        if isinstance(sub_fix_code, list) and sub_fix_code:
            combined_code = '\n'.join(sub_fix_code)
            issue_id = obj.get('issue_id') or file_path.stem
            if combined_code.strip():
                fix_codes[issue_id] = combined_code
        elif isinstance(sub_fix_code, str) and sub_fix_code.strip():
            issue_id = obj.get('issue_id') or file_path.stem
            fix_codes[issue_id] = sub_fix_code.strip()
    print(f'    [INFO] Loaded {len(fix_codes)} issues with sub_fix_code')
    return fix_codes


def extract_code_text(entry: Dict) -> str:
    if not isinstance(entry, dict):
        return ''
    parts: List[str] = []
    if isinstance(entry.get('python_trace'), list) and entry['python_trace']:
        for step in entry['python_trace']:
            if isinstance(step, dict):
                source = step.get('source')
                if isinstance(source, str) and source.strip():
                    parts.append(source.strip())
    if not parts and isinstance(entry.get('source_files'), list):
        for item in entry['source_files']:
            if isinstance(item, dict):
                source = item.get('source')
                if isinstance(source, str) and source.strip():
                    parts.append(source.strip())
    if not parts and isinstance(entry.get('source'), str) and entry['source'].strip():
        parts.append(entry['source'].strip())
    if not parts:
        return ''
    return '\n\n'.join(parts)


def normalize_code(code: str) -> str:
    return ' '.join(code.split())


def build_code_embeddings(texts: List[str], local_model_dir: Optional[str] = None) -> Tuple[np.ndarray, str]:
    """Build code embeddings using CodeBERT when available, else fall back to TF-IDF."""
    try:
        from transformers import RobertaTokenizer, RobertaModel
        import torch
    except Exception as e:
        print(f'  [INFO] transformers library not available: {str(e)[:50]}')
        tokenizer = None
        model = None
    else:
        try:
            print(f'  [INFO] Loading CodeBERT model from: {local_model_dir or "microsoft/codebert-base"}')
            tokenizer = RobertaTokenizer.from_pretrained(local_model_dir or 'microsoft/codebert-base')
            model = RobertaModel.from_pretrained(local_model_dir or 'microsoft/codebert-base')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'  [INFO] Using device: {device}')
            model.to(device)
            model.eval()
            batch_size = 16
            total_batches = (len(texts) + batch_size - 1) // batch_size
            print(f'  [INFO] Total texts: {len(texts)}, batch_size: {batch_size}, total_batches: {total_batches}')
            embeddings: List[np.ndarray] = []
            for i in range(0, len(texts), batch_size):
                batch_start = i + 1
                batch_end = min(i + batch_size, len(texts))
                batch = texts[i:i + batch_size]
                print(f'  [PROGRESS] Processing batch {batch_start}-{batch_end}/{len(texts)} (batch {i//batch_size + 1}/{total_batches})')
                print(f'    - Tokenizing {len(batch)} texts...')
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
                print(f'    - Moving inputs to device...')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                print(f'    - Running model inference...')
                with torch.no_grad():
                    outputs = model(**inputs)
                    last = outputs.last_hidden_state
                    mask = inputs['attention_mask'].unsqueeze(-1)
                    emb = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    embeddings.append(emb.cpu().numpy())
                print(f'    - Done, embedding shape: {embeddings[-1].shape}')
            if embeddings:
                result = np.vstack(embeddings)
                print(f'  [INFO] CodeBERT embeddings completed, shape: {result.shape}')
                return result, f'codebert:{local_model_dir or "microsoft/codebert-base"}'
        except Exception as e:
            print(f'  [ERROR] CodeBERT failed: {str(e)[:100]}')
            pass

    print('  [INFO] Falling back to TF-IDF')
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        raise RuntimeError('Neither transformers nor sklearn are available for similarity computation.') from e

    print('  [INFO] Fitting TF-IDF vectorizer...')
    vectorizer = TfidfVectorizer(max_features=16384, ngram_range=(1, 2), token_pattern=r'\b\w+\b')
    matrix = vectorizer.fit_transform(texts)
    print(f'  [INFO] TF-IDF completed, shape: {matrix.shape}')
    if hasattr(matrix, 'toarray'):
        return matrix.toarray(), 'tfidf'
    return matrix, 'tfidf'


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def compute_pairwise_jsonl(
    issue_names: List[str],
    issue_emb: np.ndarray,
    pt_names: List[str],
    pt_emb: np.ndarray,
    tf_names: List[str],
    tf_emb: np.ndarray,
    out_path: Path,
    progress_interval: int = 1
) -> int:
    total_pairs = len(issue_names) * (len(pt_names) + len(tf_names))
    with out_path.open('w', encoding='utf-8') as fout:
        written = 0
        for i, issue_id in enumerate(issue_names):
            pt_scores = np.dot(issue_emb[i:i + 1], pt_emb.T).ravel()
            for j, pt_name in enumerate(pt_names):
                item = {
                    'issue_id': issue_id,
                    'api_name': pt_name,
                    'api_type': 'pytorch',
                    'similarity': float(pt_scores[j])
                }
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                written += 1
            
            tf_scores = np.dot(issue_emb[i:i + 1], tf_emb.T).ravel()
            for j, tf_name in enumerate(tf_names):
                item = {
                    'issue_id': issue_id,
                    'api_name': tf_name,
                    'api_type': 'tensorflow',
                    'similarity': float(tf_scores[j])
                }
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                written += 1
            
            if (i + 1) % progress_interval == 0 or i == len(issue_names) - 1:
                print(f'[{i+1}/{len(issue_names)}] Issue "{issue_id}" compared against {len(pt_names)} PyTorch APIs and {len(tf_names)} TensorFlow APIs, total written {written}/{total_pairs}')
    return total_pairs


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate similarity between issue sub_fix_code and framework API codes.')
    parser.add_argument('--issue-dir', default='Step0-DataCollection/data/pytorch_llm_processed_data_fix_code', help='Path to issue data directory with sub_fix_code')
    parser.add_argument('--pytorch-dir', default='data/pytorch-api-extracted', help='Path to PyTorch extracted API directory')
    parser.add_argument('--tensorflow-dir', default='data/tensorflow-api-extracted', help='Path to TensorFlow extracted API directory')
    parser.add_argument('--output', default='data/issue_api_similarity.jsonl', help='Output JSONL file path')
    parser.add_argument('--model-dir', default='./codebert-base', help='Local CodeBERT directory or pretrained name')
    parser.add_argument('--progress-interval', type=int, default=10, help='How often to log progress (issues)')
    return parser.parse_args()


def main():
    args = parse_args()
    issue_dir = Path(args.issue_dir)
    pt_dir = Path(args.pytorch_dir)
    tf_dir = Path(args.tensorflow_dir)
    out_path = Path(args.output)

    if not issue_dir.exists():
        raise FileNotFoundError('issue-dir must exist.')
    if not pt_dir.exists() or not tf_dir.exists():
        raise FileNotFoundError('Both pytorch-dir and tensorflow-dir must exist.')

    print('Loading issue fix codes from', issue_dir)
    issue_sources = load_issue_fix_codes(issue_dir)
    print('Loading PyTorch API sources from', pt_dir)
    pt_sources = load_api_sources(pt_dir)
    print('Loading TensorFlow API sources from', tf_dir)
    tf_sources = load_api_sources(tf_dir)
    print(f'Loaded {len(issue_sources)} issues, {len(pt_sources)} PyTorch APIs and {len(tf_sources)} TensorFlow APIs')

    if not issue_sources:
        raise RuntimeError('No issues with sub_fix_code found.')
    if not pt_sources or not tf_sources:
        raise RuntimeError('No API sources found in one of the directories.')

    issue_names = sorted(issue_sources.keys())
    pt_names = sorted(pt_sources.keys())
    tf_names = sorted(tf_sources.keys())
    
    issue_texts = [normalize_code(issue_sources[name]) for name in issue_names]
    pt_texts = [normalize_code(pt_sources[name]) for name in pt_names]
    tf_texts = [normalize_code(tf_sources[name]) for name in tf_names]

    print('Building issue fix code embeddings...')
    issue_emb, method = build_code_embeddings(issue_texts, args.model_dir)
    print(f'Issue embeddings built with method: {method}')
    print('Building PyTorch code embeddings...')
    pt_emb, method_pt = build_code_embeddings(pt_texts, args.model_dir)
    print(f'PyTorch embeddings built with method: {method_pt}')
    print('Building TensorFlow code embeddings...')
    tf_emb, method_tf = build_code_embeddings(tf_texts, args.model_dir)
    print(f'TensorFlow embeddings built with method: {method_tf}')

    issue_emb = normalize_embeddings(issue_emb)
    pt_emb = normalize_embeddings(pt_emb)
    tf_emb = normalize_embeddings(tf_emb)

    print('Computing similarity between issue fix codes and all APIs...')
    total_pairs = compute_pairwise_jsonl(issue_names, issue_emb, pt_names, pt_emb, tf_names, tf_emb, out_path, progress_interval=args.progress_interval)
    print(f'Done. Wrote {total_pairs} similarity pairs to {out_path}')


if __name__ == '__main__':
    main()
