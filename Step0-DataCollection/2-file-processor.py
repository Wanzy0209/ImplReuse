import os
import re
import json
import argparse
from bs4 import BeautifulSoup

PRIORITY_FRAMEWORKS = ('torch', 'tensorflow', 'tf', 'np', 'numpy')

def save_json(output_dir, base_name, data):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, base_name + '.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def extract_title(html):
    soup = BeautifulSoup(html, 'html.parser')
    t = soup.find(attrs={'data-testid': 'issue-title'})
    if t and t.get_text(strip=True):
        return t.get_text(strip=True)
    h1 = soup.find('h1')
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    title_tag = soup.find('title')
    return title_tag.get_text(strip=True) if title_tag else ''


def extract_code_snippets(html):
    soup = BeautifulSoup(html, 'html.parser')
    snippets = []
    for pre in soup.find_all('pre'):
        text = pre.get_text()
        if text and len(text) > 20:
            snippets.append(text.strip())
    for code in soup.find_all('code'):
        text = code.get_text()
        if text and len(text) > 30 and text.strip() not in snippets:
            snippets.append(text.strip())
    return snippets


def pick_best_code(snippets):
    if not snippets:
        return ''
    for s in snippets:
        low = s.lower()
        if any(fr in low for fr in PRIORITY_FRAMEWORKS):
            return s
    return max(snippets, key=len)


def extract_bug_description(html):
    soup = BeautifulSoup(html, 'html.parser')
    script = soup.find('script', type='application/ld+json')
    if not script or not script.string:
        return ''
    try:
        data = json.loads(script.string.strip())
        if isinstance(data, list):
            data = data[0] if data else {}
        if 'articleBody' in data:
            bug_desc = data['articleBody'].strip()
            return bug_desc[:4000]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return ''


def extract_apis(text):
    found = set()
    for kw in PRIORITY_FRAMEWORKS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text, re.I):
            found.add(kw)
    for m in re.finditer(r'\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w+)+)\s*\(', text):
        found.add(m.group(1))
    return list(found)[:6]


def extract_pull_urls(html):
    pull_urls = set()
    pull_pattern = r'https?://github\.com/[^/]+/[^/]+/pull/\d+'
    for match in re.finditer(pull_pattern, html):
        pull_urls.add(match.group(0))
    soup = BeautifulSoup(html, 'html.parser')
    for a in soup.find_all('a', href=True):
        href = a['href']
        if re.match(pull_pattern, href):
            pull_urls.add(href)
    return list(pull_urls)[:10]


def process_file(path, output_dir, overwrite=False):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()
    title = extract_title(html)

    issue_id_match = re.search(r'(\d+)', os.path.basename(path))
    issue_id = issue_id_match.group(1) if issue_id_match else ''
    snippets = extract_code_snippets(html)
    best_code = pick_best_code(snippets)
    bug_description = extract_bug_description(html)
    apis = extract_apis(html + '\n' + best_code)
    pull_urls = extract_pull_urls(html)

    issue_json = {
        'title': title,
        'issue_id': issue_id,
        'bug_description': bug_description,
        'apis': apis,
        'best_code': best_code,
        'pull_urls': pull_urls
    }

    out_base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_dir, out_base + '.json')
    if os.path.exists(out_path) and not overwrite:
        print(f"Skip existing {out_base}.json")
        return issue_json

    saved = save_json(os.path.join(os.path.abspath(os.path.dirname(__file__)), output_dir), out_base, issue_json)
    print(f"Saved Json -> {saved}")

    return issue_json


def main():
    parser = argparse.ArgumentParser(description='Process GitHub issue HTML files and extract information.')
    parser.add_argument('--input-dir', default='./data/pytorch_issue',
                        help='Input HTML file or directory containing HTML files')
    parser.add_argument('--output-dir', default='./data/pytorch_processed_data',
                        help='Output directory for JSON files')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing JSON files')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))

    if os.path.isdir(input_path):
        html_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.html')])
        for html_file in html_files:
            process_file(html_file, output_dir, overwrite=args.overwrite)
        print(f"Processed {len(html_files)} files.")
    elif os.path.isfile(input_path):
        process_file(input_path, output_dir, overwrite=args.overwrite)
    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == '__main__':
    main()
