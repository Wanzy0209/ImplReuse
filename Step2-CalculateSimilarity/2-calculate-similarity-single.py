#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate pairwise code similarity between extracted PyTorch APIs.

This script reads extracted API source data from:
- data/pytorch-api-extracted

It computes a similarity score for every PyTorch/PyTorch API pair (including
self-comparison) and writes the result as JSONL, one pair per line. The script
reports progress while building embeddings and while writing the m*m pairs.
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
    pt_names: List[str],
    pt_emb: np.ndarray,
    out_path: Path,
    progress_interval: int = 1
) -> int:
    total_pairs = len(pt_names) * len(pt_names)
    with out_path.open('w', encoding='utf-8') as fout:
        written = 0
        for i, pt_name1 in enumerate(pt_names):
            scores = np.dot(pt_emb[i:i + 1], pt_emb.T).ravel()
            for j, pt_name2 in enumerate(pt_names):
                item = {
                    'pytorch_api_1': pt_name1,
                    'pytorch_api_2': pt_name2,
                    'similarity': float(scores[j])
                }
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                written += 1
            if (i + 1) % progress_interval == 0 or i == len(pt_names) - 1:
                print(f'[{i+1}/{len(pt_names)}] PyTorch API "{pt_name1}" compared against {len(pt_names)} PyTorch APIs, total written {written}/{total_pairs}')
    return total_pairs


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate pairwise code similarity for PyTorch APIs.')
    parser.add_argument('--pytorch-dir', default='data/pytorch-api-extracted', help='Path to PyTorch extracted API directory')
    parser.add_argument('--output', default='data/pytorch_pairwise_similarity.jsonl', help='Output JSONL file path')
    parser.add_argument('--model-dir', default='./codebert-base', help='Local CodeBERT directory or pretrained name')
    parser.add_argument('--progress-interval', type=int, default=10, help='How often to log progress (PyTorch APIs)')
    return parser.parse_args()


def main():
    args = parse_args()
    pt_dir = Path(args.pytorch_dir)
    out_path = Path(args.output)

    if not pt_dir.exists():
        raise FileNotFoundError('pytorch-dir must exist.')

    print('Loading PyTorch API sources from', pt_dir)
    pt_sources = load_api_sources(pt_dir)
    print(f'Loaded {len(pt_sources)} PyTorch APIs')

    if not pt_sources:
        raise RuntimeError('No API sources found in pytorch-dir.')

    pt_names = sorted(pt_sources.keys())
    pt_texts = [normalize_code(pt_sources[name]) for name in pt_names]

    print('Building PyTorch code embeddings...')
    pt_emb, method = build_code_embeddings(pt_texts, args.model_dir)
    print(f'PyTorch embeddings built with method: {method}')

    pt_emb = normalize_embeddings(pt_emb)

    print('Computing pairwise similarity for all PyTorch API pairs...')
    total_pairs = compute_pairwise_jsonl(pt_names, pt_emb, out_path, progress_interval=args.progress_interval)
    print(f'Done. Wrote {total_pairs} PyTorch API pairs to {out_path}')


if __name__ == '__main__':
    main()
