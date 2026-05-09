#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Link PR fix code to issue data.

This script reads JSON files from data/pytorch_llm_processed_data,
extracts pull_urls, fetches the corresponding PR code from GitHub web page,
parses the HTML diff to extract added and deleted code lines, and saves
enriched JSON files to data/pytorch_llm_processed_data_fix_code/ with
add_fix_code and sub_fix_code fields.

Fetched PR code is cached to avoid repeated web requests.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List

requests = None
try:
    import requests
except ImportError:
    pass

BeautifulSoup = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    pass


# Cache for PR code to avoid repeated fetching
PR_CODE_CACHE = {}
CACHE_DIR = Path(__file__).parent / "pr_code_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(pr_url: str) -> str:
    """Generate a cache key from PR URL."""
    return hashlib.md5(pr_url.encode()).hexdigest()


def load_from_cache(pr_url: str) -> Optional[Dict[str, str]]:
    """Load PR code from cache (memory or disk)."""
    cache_key = get_cache_key(pr_url)
    
    # Check in-memory cache first
    if cache_key in PR_CODE_CACHE:
        print(f"  [CACHE-HIT] {pr_url}")
        return PR_CODE_CACHE[cache_key]
    
    # Check disk cache
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with cache_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                PR_CODE_CACHE[cache_key] = data
                print(f"  [CACHE-LOAD] {pr_url}")
                return data
        except Exception as e:
            print(f"  [CACHE-ERR] Failed to load cache for {pr_url}: {e}")
    
    return None


def save_to_cache(pr_url: str, pr_code: Dict[str, List[str]]):
    """Save PR code to cache (memory and disk)."""
    cache_key = get_cache_key(pr_url)
    PR_CODE_CACHE[cache_key] = pr_code
    
    if pr_code:
        cache_file = CACHE_DIR / f"{cache_key}.json"
        try:
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump(pr_code, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [CACHE-SAVE-ERR] {e}")


def fetch_pr_files(pr_url: str, retry_count: int = 3, retry_delay: int = 60, token: str = '') -> Dict[str, List[str]]:
    """
    Fetch PR diff and extract added/deleted code.
    Returns dict with "add_fix_code" and "sub_fix_code" lists.
    """
    if requests is None:
        print(f"  [ERR] requests library not available. Install with: pip install requests")
        return {"add_fix_code": [], "sub_fix_code": []}
    
    cleaned_url = pr_url.strip().strip('`"')
    
    # Check cache first
    cached = load_from_cache(cleaned_url)
    if cached is not None:
        return cached
    
    result = {"add_fix_code": [], "sub_fix_code": []}
    try:
        m = re.match(r'https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)', cleaned_url)
        if not m:
            m = re.match(r'([^/]+)/([^/]+)/pull/(\d+)', cleaned_url)
            if not m:
                print(f"  [PARSE-ERR] Invalid PR URL format: {cleaned_url}")
                return result
        
        owner, repo, pr_number = m.groups()
        
        # Try GitHub API first (patch format)
        api_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/vnd.github.v3.patch'
        }
        if token:
            headers['Authorization'] = f'token {token}'
        
        print(f"  [FETCH] {owner}/{repo}/pull/{pr_number}")
        print(f"  [URL] {api_url}")
        
        resp = requests.get(api_url, headers=headers, timeout=15)
        
        if resp.status_code == 403:
            for attempt in range(retry_count):
                print(f"  [API-RATE-LIMIT] Status 403, waiting {retry_delay}s before retry ({attempt + 1}/{retry_count})...")
                time.sleep(retry_delay)
                resp = requests.get(api_url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    break
                elif resp.status_code == 403:
                    continue
                else:
                    break
            
            if resp.status_code != 200:
                print(f"  [API-ERR] Status {resp.status_code}, falling back to HTML...")
                full_pr_url = f'https://github.com/{owner}/{repo}/pull/{pr_number}/files'
                headers_html = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                resp = requests.get(full_pr_url, headers=headers_html, timeout=15)
                
                if resp.status_code == 403:
                    for attempt in range(retry_count):
                        print(f"  [HTML-RATE-LIMIT] Status 403, waiting {retry_delay}s before retry ({attempt + 1}/{retry_count})...")
                        time.sleep(retry_delay)
                        resp = requests.get(full_pr_url, headers=headers_html, timeout=15)
                        if resp.status_code == 200:
                            break
                        elif resp.status_code == 403:
                            continue
                        else:
                            break
                
                if resp.status_code != 200:
                    print(f"  [HTML-ERR] Status {resp.status_code}")
                    return result
                
                if BeautifulSoup is None:
                    print(f"  [ERR] BeautifulSoup not available")
                    return result
                
                soup = BeautifulSoup(resp.text, 'html.parser')
                add_lines = []
                sub_lines = []
                
                for span in soup.find_all('span'):
                    classes = span.get('class', [])
                    if isinstance(classes, list):
                        class_str = ' '.join(classes)
                        if 'blob-code-addition' in class_str or 'text-green-500' in class_str:
                            code_text = span.get_text(strip=False) or ''
                            if code_text.startswith('+'):
                                code_text = code_text[1:]
                            if code_text.strip():
                                add_lines.append(code_text)
                        elif 'blob-code-deletion' in class_str or 'text-red-500' in class_str:
                            code_text = span.get_text(strip=False) or ''
                            if code_text.startswith('-'):
                                code_text = code_text[1:]
                            if code_text.strip():
                                sub_lines.append(code_text)
                
                if not add_lines and not sub_lines:
                    for tr in soup.find_all('tr'):
                        tr_classes = tr.get('class', [])
                        if isinstance(tr_classes, list):
                            if 'blob-code-addition' in tr_classes:
                                code_span = tr.find('span', class_='blob-code-inner')
                                if code_span:
                                    code_text = code_span.get_text(strip=False) or ''
                                    if code_text.startswith('+'):
                                        code_text = code_text[1:]
                                    if code_text.strip():
                                        add_lines.append(code_text)
                            elif 'blob-code-deletion' in tr_classes:
                                code_span = tr.find('span', class_='blob-code-inner')
                                if code_span:
                                    code_text = code_span.get_text(strip=False) or ''
                                    if code_text.startswith('-'):
                                        code_text = code_text[1:]
                                    if code_text.strip():
                                        sub_lines.append(code_text)
                
                result["add_fix_code"] = add_lines
                result["sub_fix_code"] = sub_lines
                print(f"  [HTML-SUCCESS] {len(add_lines)} added line(s), {len(sub_lines)} deleted line(s)")
                save_to_cache(cleaned_url, result)
                return result
        
        if resp.status_code == 200:
            patch_content = resp.text
            add_lines = []
            sub_lines = []
            
            for line in patch_content.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    add_lines.append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    sub_lines.append(line[1:])
            
            result["add_fix_code"] = add_lines
            result["sub_fix_code"] = sub_lines
            print(f"  [SUCCESS] {len(add_lines)} added line(s), {len(sub_lines)} deleted line(s)")
        else:
            print(f"  [API-ERR] Status {resp.status_code}")
    
    except requests.RequestException as e:
        print(f"  [NET-ERR] {e}")
    except Exception as e:
        print(f"  [ERR] {e}")
    
    save_to_cache(cleaned_url, result)
    return result


def process_issue_file(input_path: Path, output_path: Path, token: str = ''):
    """
    Read issue JSON, fetch PR code for all pull_urls, add add_fix_code and sub_fix_code fields, save to output.
    """
    try:
        with input_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERR] Failed to read {input_path}: {e}")
        return False
    
    # Extract pull_urls
    pull_urls = data.get('pull_urls', [])
    if isinstance(pull_urls, str):
        pull_urls = [pull_urls] if pull_urls else []
    
    # Fetch PR code for each URL and aggregate
    all_add_lines = []
    all_sub_lines = []
    
    for url in pull_urls:
        if not url or not isinstance(url, str):
            continue
        pr_code = fetch_pr_files(url, token=token)
        if pr_code:
            all_add_lines.extend(pr_code.get("add_fix_code", []))
            all_sub_lines.extend(pr_code.get("sub_fix_code", []))
    
    # Add aggregated fix_code fields to data
    data['add_fix_code'] = all_add_lines
    data['sub_fix_code'] = all_sub_lines
    
    # Save to output
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[ERR] Failed to write {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Fetch PR fix code and link to issue data.'
    )
    parser.add_argument(
        '--input-dir',
        default='data/pytorch_llm_processed_data',
        help='Input directory containing issue JSON files'
    )
    parser.add_argument(
        '--output-dir',
        default='data/pytorch_llm_processed_data_fix_code',
        help='Output directory for enriched JSON files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Process only first N files (0 for all)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay in seconds between requests to avoid rate limiting (default: 0.5)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default='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',
        help='GitHub personal access token for authenticated requests (optional)'
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find all JSON files
    json_files = sorted([p for p in input_dir.glob('*.json') if p.is_file()])
    total = len(json_files)
    
    if args.limit > 0:
        json_files = json_files[:args.limit]
        print(f"Processing first {len(json_files)} of {total} files")
    else:
        print(f"Processing all {total} files")
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, input_path in enumerate(json_files, start=1):
        output_path = output_dir / input_path.name
        
        print(f"[{idx:4d}/{len(json_files):4d}] {input_path.name}")
        if process_issue_file(input_path, output_path, token=args.token):
            success_count += 1
        else:
            fail_count += 1
        
        if idx < len(json_files) and args.delay > 0:
            time.sleep(args.delay)
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(json_files)}")
    print(f"  Output:  {output_dir}")
    print(f"  Cache:   {CACHE_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
