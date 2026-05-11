#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse LLM output files to extract core API and update JSON files."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def clean_core_api(api: str) -> str:
    """Clean extracted API string to remove duplicates and artifacts."""
    if not api:
        return api
    
    original_api = api
    
    api = api.strip().strip(".`").strip()
    
    api = re.sub(r'\s+', ' ', api)
    
    api = re.sub(r'\*+', '', api)
    
    api = api.strip()
    
    if api.lower() in ['none', 'nonenone'] or 'none.none' in api.lower() or api.endswith('.non') or api.endswith('.Non'):
        return 'None'
    
    api = api.replace('`', '')
    
    text_prefixes = ['the api is', 'the answer is', 'result:', 'final:', 'output:', 'answer:', 'api:', 'core api:', 'this is', 'it is']
    for prefix in text_prefixes:
        if api.lower().startswith(prefix):
            api = api[len(prefix):].strip()
    
    doubled_match = re.match(r'^(.+)\1+$', api)
    if doubled_match:
        base = doubled_match.group(1)
        if len(base) > len(api) // 2 and len(base) >= 3:
            api = base
    
    overlap_patterns = [
        (r'torch\.([a-z]+)(torch\.[a-z]+)', ['compile', 'jit', 'export', 'autocast', 'dynamo', 'add', 'min']),
        (r'torch\.([a-z]+\.[a-z]+)(torch\.[a-z]+\.[a-z]+)', ['scaled_dot_product_attention', 'fused_all_gather_scaled_matmul', 'jit.script', 'onnx.export', 'export.export', 'checkpoint.checkpoint']),
        (r'([a-z]+torch\.[a-z]+\.[a-z]+)\1', ['jit.script']),
    ]
    for pattern, api_parts in overlap_patterns:
        overlap_match = re.search(pattern, api)
        if overlap_match:
            for apipart in api_parts:
                if apipart in api:
                    api = f'torch.{apipart}' if not apipart.startswith('torch.') else apipart
                    break
            break
    
    if re.match(r'^torch\.[a-z]+\.([a-z]+)\1', api):
        match = re.match(r'^torch\.[a-z]+\.([a-z]+)', api)
        if match:
            api = 'torch.' + match.group(1)
    
    torch_specific_fix = {
        'torch.compiletorch.compile': 'torch.compile',
        'torch.jittorch.jit.script': 'torch.jit.script',
        'torch.onnx.exporttorch.onnx.export': 'torch.onnx.export',
        'torch.export.exporttorch.export.export': 'torch.export.export',
        'torch.optim.LBFGStorch.optim.LBFGS': 'torch.optim.LBFGS',
        'torch.addtorch.add': 'torch.add',
        'torch.nn.Bilineartorch.nn.Bilinear': 'torch.nn.Bilinear',
        'torch.autocasttorch.autocast': 'torch.autocast',
        'torch.sparse_coo_tensortorch.sparse_coo_tensor': 'torch.sparse_coo_tensor',
        'torch.ops.load_librarytorch.ops.load_library': 'torch.ops.load_library',
        'torch.utils.checkpoint.checkpointtorch.utils.checkpoint.checkpoint': 'torch.utils.checkpoint.checkpoint',
    }
    if api in torch_specific_fix:
        api = torch_specific_fix[api]
    
    torch_matches = re.findall(r'torch\.[a-zA-Z0-9_.]+', api)
    if torch_matches:
        valid_apis = set()
        for m in torch_matches:
            if m.startswith('torch.'):
                valid_apis.add(m.strip())
        if valid_apis:
            best = max(valid_apis, key=len)
            return best.strip()
    
    common_tensor_methods = ['copy_', 'resize_', 'view', 'squeeze', 'unsqueeze', 'transpose', 'permute', 'expand', 'repeat', 'contiguous', 'reshape', 'chunk', 'split', 'index', 'masked_fill', 'to', 'cpu', 'cuda', 'numpy']
    for method in common_tensor_methods:
        pattern = rf'Tensor\.{method}'
        if re.search(pattern, api):
            return f'Tensor.{method}'.strip()
    
    tensor_matches = re.findall(r'Tensor\.[a-zA-Z0-9_]+', api)
    if tensor_matches:
        return tensor_matches[0].strip()
    
    nn_matches = re.findall(r'nn\.[a-zA-Z0-9_]+', api)
    if nn_matches:
        return nn_matches[0].strip()
    
    if api.count('.') >= 3:
        half_len = len(api) // 2
        first_half = api[:half_len]
        second_half = api[half_len:]
        if first_half == second_half and len(first_half) > 5:
            return first_half.strip()
    
    parts = api.split('.')
    cleaned_parts = []
    prev = None
    for part in parts:
        if part and part != prev:
            cleaned_parts.append(part)
            prev = part
    api = '.'.join(cleaned_parts)
    
    match = re.match(r'^(torch\.[a-zA-Z0-9_.]+?)\1+$', api)
    if match:
        api = match.group(1)
    
    api = api.strip()
    api = api.strip('.')
    api = api.strip()
    
    return api


def extract_core_api_from_output(output_text: str) -> Tuple[Optional[str], bool]:
    """Extract core API from LLM output text.
    
    Returns:
        Tuple of (core_api, parse_error)
    """
    text = output_text.strip()
    if not text:
        return None, True
    
    lines = text.splitlines()
    if not lines:
        return None, True
    
    last_line = lines[-1].strip()
    second_last_line = lines[-2].strip() if len(lines) >= 2 else ""
    
    patterns = [
        r"Final (?:decision|answer|string):\s*`?([^`\n]+)`?\.?([a-zA-Z0-9_.]+)",
        r"Final answer:\s*`?([^`\n]+)`?\.?([a-zA-Z0-9_.]+)",
        r"Core API:\s*`?([^`\n]+)`?",
        r"torch\.[a-zA-Z0-9_.]+",
        r"None",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, last_line, re.IGNORECASE)
        if match:
            if "None" in pattern and match.group(0).lower() == "none":
                return "None", False
            if match.lastindex and match.lastindex >= 1:
                result = match.group(1) if match.group(1) else match.group(0)
            else:
                result = match.group(0)
            result = clean_core_api(result)
            if result.lower() != "none" and result:
                return result, False
    
    for pattern in patterns:
        match = re.search(pattern, second_last_line, re.IGNORECASE)
        if match:
            if "None" in pattern and match.group(0).lower() == "none":
                return "None", False
            if match.lastindex and match.lastindex >= 1:
                result = match.group(1) if match.group(1) else match.group(0)
            else:
                result = match.group(0)
            result = clean_core_api(result)
            if result.lower() != "none" and result:
                return result, False
    
    api_pattern = r"torch\.[a-zA-Z0-9_.]+"
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        match = re.search(api_pattern, line)
        if match:
            return clean_core_api(match.group(0)), False
        
        if line.lower() == "none":
            return "None", False
        
        words = line.split()
        for word in reversed(words):
            word = word.strip(".,:`'\"*")
            if word and (word.startswith("torch.") or word.startswith("Tensor.") or word.startswith("nn.")):
                return clean_core_api(word), False
    
    return None, True


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_output_file(output_path: Path, json_path: Path) -> Tuple[Optional[str], bool]:
    """Process a single output file and update the corresponding JSON.
    
    Returns:
        Tuple of (extracted_core_api, parse_error)
    """
    if not output_path.exists():
        print(f"  [SKIP] Output file not found: {output_path}")
        return None, True
    
    output_text = output_path.read_text(encoding='utf-8', errors='ignore')
    core_api, parse_error = extract_core_api_from_output(output_text)
    
    if json_path.exists():
        data = read_json(json_path)
        if data is not None:
            if "core_api_identification" not in data:
                data["core_api_identification"] = {}
            data["core_api_identification"]["core_api"] = core_api
            data["core_api_identification"]["parse_error"] = parse_error
            save_json(json_path, data)
            return core_api, parse_error
    
    return core_api, parse_error


def main():
    parser = argparse.ArgumentParser(description='Parse LLM output files to extract core API.')
    parser.add_argument('--llm-io-dir', default='data/pytorch_core_api_identification/llm_io', 
                        help='Directory containing LLM output files')
    parser.add_argument('--json-dir', default='data/pytorch_core_api_identification', 
                        help='Directory containing JSON files to update')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite existing core_api values')
    args = parser.parse_args()
    
    llm_io_dir = Path(args.llm_io_dir)
    json_dir = Path(args.json_dir)
    
    if not llm_io_dir.exists():
        print(f"Error: LLM IO directory not found: {llm_io_dir}")
        return
    
    output_files = sorted(llm_io_dir.glob("*_issue_ori_data.output.txt"))
    print(f"Found {len(output_files)} output files")
    
    success_count = 0
    error_count = 0
    
    for output_file in output_files:
        base_name = output_file.stem.replace('.output', '')
        json_file = json_dir / f"{base_name}.json"
        
        core_api, parse_error = process_output_file(output_file, json_file)
        if core_api is not None:
            status = "ERROR" if parse_error else "OK"
            print(f"  [{status}] {base_name}: {core_api}")
            if parse_error:
                error_count += 1
            else:
                success_count += 1
        else:
            print(f"  [FAIL] {base_name}: Could not extract core_api")
            error_count += 1
    
    print(f"\nDone. Success: {success_count}, Errors: {error_count}")


if __name__ == '__main__':
    main()