#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Get top-10 similar APIs for each core_api from three similarity files."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def load_jsonl(path: Path) -> List[dict]:
    """Load all lines from a JSONL file."""
    data = []
    if not path.exists():
        return data
    try:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return data


def get_top_similar_pytorch_to_tensorflow(core_api: str, data: List[dict], top_n: int = 10) -> List[dict]:
    """Get top-N similar TensorFlow APIs for a given PyTorch core_api."""
    matches = []
    for item in data:
        if item.get('pytorch_api') == core_api:
            matches.append({
                'api_name': item['tensorflow_api'],
                'api_type': 'tensorflow',
                'similarity': item['similarity']
            })
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return matches[:top_n]


def get_top_similar_issue_api(core_api: str, data: List[dict], top_n: int = 10) -> List[dict]:
    """Get top-N similar APIs for a given core_api from issue_api_similarity."""
    matches = []
    for item in data:
        if item.get('api_name') == core_api:
            matches.append({
                'api_name': item['issue_id'],
                'api_type': 'issue',
                'similarity': item['similarity']
            })
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return matches[:top_n]


def get_top_similar_pytorch_to_pytorch(core_api: str, data: List[dict], top_n: int = 10) -> List[dict]:
    """Get top-N similar PyTorch APIs for a given PyTorch core_api."""
    matches = []
    for item in data:
        if item.get('pytorch_api_1') == core_api:
            matches.append({
                'api_name': item['pytorch_api_2'],
                'api_type': 'pytorch',
                'similarity': item['similarity']
            })
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return matches[:top_n]


def main():
    parser = argparse.ArgumentParser(description='Get top-10 similar APIs for each core_api.')
    parser.add_argument('--step1-dir', default='../Step1-CoreAPIIdentification/data/pytorch_core_api_identification',
                        help='Path to Step1 JSON files')
    parser.add_argument('--similarity-dir', default='data',
                        help='Path to similarity JSONL files')
    parser.add_argument('--output', default='data/similar_api_list.json',
                        help='Output JSON file path')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top similar APIs to return')
    args = parser.parse_args()

    step1_dir = Path(args.step1_dir)
    similarity_dir = Path(args.similarity_dir)
    output_path = Path(args.output)

    print('Loading similarity data...')
    
    api_pairwise_data = load_jsonl(similarity_dir / 'api_pairwise_similarity.jsonl')
    print(f'  Loaded {len(api_pairwise_data)} PyTorch-TensorFlow pairs')
    
    issue_api_data = load_jsonl(similarity_dir / 'issue_api_similarity.jsonl')
    print(f'  Loaded {len(issue_api_data)} issue-API pairs')
    
    pytorch_pairwise_data = load_jsonl(similarity_dir / 'pytorch_pairwise_similarity.jsonl')
    print(f'  Loaded {len(pytorch_pairwise_data)} PyTorch-PyTorch pairs')

    print('Loading core API data...')
    json_files = sorted(step1_dir.glob('*_issue_ori_data.json'))
    print(f'  Found {len(json_files)} issue JSON files')

    results = []
    
    for json_file in json_files:
        data = read_json(json_file)
        if not data:
            continue
        
        issue_id = data.get('issue_id') or json_file.stem.replace('_issue_ori_data', '')
        core_api_info = data.get('core_api_identification', {})
        core_api = core_api_info.get('core_api')
        
        if not core_api or core_api.lower() == 'none':
            print(f'  [SKIP] {issue_id}: No core_api found')
            continue
        
        print(f'  [PROCESSING] {issue_id}: {core_api}')
        
        result = {
            'issue_id': issue_id,
            'core_api': core_api,
            'similar_apis': {
                'pytorch_to_tensorflow': get_top_similar_pytorch_to_tensorflow(core_api, api_pairwise_data, args.top_n),
                'issue_to_api': get_top_similar_issue_api(core_api, issue_api_data, args.top_n),
                'pytorch_to_pytorch': get_top_similar_pytorch_to_pytorch(core_api, pytorch_pairwise_data, args.top_n)
            }
        }
        
        results.append(result)

    print(f'\nWriting results to {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f'Done. Processed {len(results)} issues.')


if __name__ == '__main__':
    main()