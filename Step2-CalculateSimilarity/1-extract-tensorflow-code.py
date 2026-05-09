import ast
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


DEFAULT_REPO_ROOT = Path(__file__).resolve().parent / 'data' / 'tensorflow-code'
DEFAULT_API_DEF = Path(__file__).resolve().parent / 'data' / 'tensorflow_API_def.txt'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'data' / 'tensorflow-api-extracted'
CACHE_FILE = Path(__file__).resolve().parent / 'tensorflow_extraction_cache.json'
CPP_EXTENSIONS = ('.cpp', '.c', '.h', '.cuh', '.cu', '.cc')


def read_api_list(path: str) -> List[str]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"API definition file not found: {path}")
    apis = []
    with path_obj.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            api_name = line.split('(')[0].strip()
            if api_name:
                apis.append(api_name)
    return apis


def load_cache(path: Path = CACHE_FILE) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_cache(cache: Dict[str, Dict], path: Path = CACHE_FILE):
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')


class PythonFileInfo:
    def __init__(self, path: Path, module_name: str, source: str, tree: ast.AST):
        self.path = path
        self.module_name = module_name
        self.source = source
        self.tree = tree
        self.imports = self._collect_imports()
        self.defs = self._collect_defs()
        self.class_methods = self._collect_class_methods()
        self.aliases = self._collect_aliases()
        self.alias_nodes = self._collect_alias_nodes()

    def _collect_imports(self) -> Dict[str, str]:
        imports: Dict[str, str] = {}
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    imports[alias.asname or alias.name] = full_name
        return imports

    def _collect_defs(self) -> Dict[str, ast.AST]:
        defs: Dict[str, ast.AST] = {}
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[node.name] = node
            elif isinstance(node, ast.ClassDef):
                defs[node.name] = node
        return defs

    def _collect_aliases(self) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                alias_name = node.targets[0].id
                if isinstance(node.value, ast.Call):
                    func_name = get_call_name(node.value.func)
                    # TensorFlow may use different alias patterns, e.g., tf.function or direct assignments
                    if func_name and node.value.args:
                        arg = node.value.args[0]
                        target_name = get_call_name(arg)
                        if target_name:
                            aliases[alias_name] = target_name
                    else:
                        target_name = get_call_name(node.value.func)
                        if target_name:
                            aliases[alias_name] = target_name
                elif isinstance(node.value, (ast.Name, ast.Attribute)):
                    target_name = get_call_name(node.value)
                    if target_name:
                        aliases[alias_name] = target_name
        return aliases

    def _collect_alias_nodes(self) -> Dict[str, ast.AST]:
        alias_nodes: Dict[str, ast.AST] = {}
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                alias_nodes[node.targets[0].id] = node
        return alias_nodes

    def _collect_class_methods(self) -> Dict[str, ast.AST]:
        methods: Dict[str, ast.AST] = {}
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods[f"{node.name}.{child.name}"] = child
        return methods


def module_name_from_path(path: Path, repo_root: Path) -> str:
    rel = path.relative_to(repo_root).with_suffix('')
    parts = rel.parts
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def build_python_index(repo_root: Path) -> Tuple[Dict[str, PythonFileInfo], Dict[str, Set[str]]]:
    module_index: Dict[str, PythonFileInfo] = {}
    simple_index: Dict[str, Set[str]] = {}

    for root, _, files in os.walk(repo_root):
        root_path = Path(root)
        for filename in files:
            if not filename.endswith('.py'):
                continue
            file_path = root_path / filename
            try:
                source = file_path.read_text(encoding='utf-8', errors='ignore')
                tree = ast.parse(source)
            except Exception:
                continue
            module_name = module_name_from_path(file_path, repo_root)
            file_info = PythonFileInfo(file_path, module_name, source, tree)
            module_index[module_name] = file_info
            for name in file_info.defs:
                qualified = f"{module_name}.{name}" if module_name else name
                simple_index.setdefault(name, set()).add(qualified)
            for name in file_info.class_methods:
                qualified = f"{module_name}.{name}" if module_name else name
                simple_index.setdefault(name.split('.')[-1], set()).add(qualified)
    return module_index, simple_index


def get_source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment:
        return segment
    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
        lines = source.splitlines()
        return '\n'.join(lines[node.lineno - 1:node.end_lineno])
    return ''


def normalize_api_name(api_name: str) -> Tuple[str, str]:
    if '(' in api_name:
        api_name = api_name.split('(')[0].strip()
    if api_name.count('.') < 1:
        return '', api_name
    parts = api_name.split('.')
    return '.'.join(parts[:-1]), parts[-1]


def get_call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: List[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return '.'.join(reversed(parts))
    return None


def collect_calls(func_node: ast.AST) -> Set[str]:
    class CallCollector(ast.NodeVisitor):
        def __init__(self):
            self.calls: Set[str] = set()

        def visit_Call(self, node: ast.Call):
            call_name = get_call_name(node.func)
            if call_name:
                self.calls.add(call_name)
            self.generic_visit(node)

    collector = CallCollector()
    collector.visit(func_node)
    return collector.calls


def resolve_candidate_names(call_name: str, file_info: PythonFileInfo, module_index: Dict[str, PythonFileInfo], simple_index: Dict[str, Set[str]]) -> List[str]:
    candidates: List[str] = []
    if call_name.startswith('self.'):
        method_name = call_name.split('.', 1)[1]
        for full_name in file_info.class_methods:
            if full_name.endswith(f'.{method_name}'):
                candidates.append(f"{file_info.module_name}.{full_name}")
        return candidates

    if '.' in call_name:
        parts = call_name.split('.')
        if parts[0] in file_info.imports:
            resolved = file_info.imports[parts[0]]
            candidates.append('.'.join([resolved] + parts[1:]))
        else:
            candidates.append(call_name)
    else:
        local_candidate = f"{file_info.module_name}.{call_name}" if file_info.module_name else call_name
        if local_candidate in module_index or local_candidate in simple_index.get(call_name, set()):
            candidates.append(local_candidate)
        candidates.extend(sorted(simple_index.get(call_name, [])))
    return list(dict.fromkeys(candidates))


def is_cpp_leaf(call_name: str) -> bool:
    # TensorFlow C++ prefixes
    return any(call_name.startswith(prefix) for prefix in [
        'tensorflow::', 'tf::', 'ops::', 'kernel::', 'gen_', 'c_api::', 'pywrap_', 'swig::',
        'gen_nn_ops.', 'gen_math_ops.', 'gen_array_ops.', 'gen_linalg_ops.', 'gen_random_ops.',
        'gen_string_ops.', 'gen_sparse_ops.', 'gen_rnn_ops.', 'gen_image_ops.', 'gen_audio_ops.',
        'gen_summary_ops.', 'gen_control_flow_ops.', 'gen_functional_ops.', 'gen_data_flow_ops.',
        'gen_io_ops.', 'gen_logging_ops.', 'gen_lookup_ops.', 'gen_manip_ops.', 'gen_state_ops.',
        'gen_training_ops.', 'gen_user_ops.'
    ])


def find_python_definition(qualified_name: str, module_index: Dict[str, PythonFileInfo]) -> Optional[Tuple[PythonFileInfo, ast.AST]]:
    if qualified_name in module_index:
        return None
    parts = qualified_name.split('.')
    for module_len in range(len(parts) - 1, 0, -1):
        module_name = '.'.join(parts[:module_len])
        function_name = '.'.join(parts[module_len:])
        file_info = module_index.get(module_name)
        if not file_info:
            continue
        if function_name in file_info.defs:
            return file_info, file_info.defs[function_name]
        if function_name in file_info.class_methods:
            return file_info, file_info.class_methods[function_name]
    return None


def search_python_definition(api_module: str, api_func: str, module_index: Dict[str, PythonFileInfo], simple_index: Dict[str, Set[str]]) -> Optional[Tuple[str, PythonFileInfo, ast.AST]]:
    def resolve_alias(file_info: PythonFileInfo, alias_name: str):
        alias_target = file_info.aliases.get(alias_name)
        if not alias_target:
            return None
        resolved = resolve_candidate_names(alias_target, file_info, module_index, simple_index)
        for target in resolved:
            resolved_def = search_python_definition(*normalize_api_name(target), module_index, simple_index)
            if resolved_def:
                return resolved_def
        if alias_name in file_info.alias_nodes:
            return file_info.module_name, file_info, file_info.alias_nodes[alias_name]
        return None

    if api_module:
        module_candidates = [api_module, api_module + '.' + api_func]
        for candidate in module_candidates:
            if candidate in module_index:
                file_info = module_index[candidate]
                if api_func in file_info.defs:
                    return candidate, file_info, file_info.defs[api_func]
                if api_func in file_info.class_methods:
                    return candidate, file_info, file_info.class_methods[api_func]
                alias_definition = resolve_alias(file_info, api_func)
                if alias_definition:
                    return alias_definition
        file_info = module_index.get(api_module)
        if file_info and api_func in file_info.defs:
            return api_module, file_info, file_info.defs[api_func]
        if file_info:
            alias_definition = resolve_alias(file_info, api_func)
            if alias_definition:
                return alias_definition
    possible = simple_index.get(api_func, set())
    for qualified in sorted(possible):
        parts = qualified.rsplit('.', 1)
        if len(parts) == 2:
            module_name, name = parts
            file_info = module_index.get(module_name)
            if file_info and name in file_info.defs:
                return module_name, file_info, file_info.defs[name]
    return None


def c_source_for_function(api_name: str, repo_root: Path) -> str:
    patterns = [
        re.compile(rf'\b{re.escape(api_name)}\s*\(', re.MULTILINE),
        re.compile(rf'\b{re.escape(api_name)}\s*<', re.MULTILINE),
        re.compile(rf'\b{api_name.capitalize()}\s*\(', re.MULTILINE),  # Conv2d
        re.compile(rf'\bREGISTER_OP.*{re.escape(api_name)}\b', re.MULTILINE),
        re.compile(rf'\bREGISTER_KERNEL_BUILDER.*{re.escape(api_name)}\b', re.MULTILINE),
    ]
    for root, _, files in os.walk(repo_root):
        for file_name in files:
            if not file_name.endswith(CPP_EXTENSIONS):
                continue
            file_path = Path(root) / file_name
            try:
                text = file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            if not any(pattern.search(text) for pattern in patterns):
                continue
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                start = match.start()
                end = _extract_brace_block(text, start)
                if end:
                    return text[start:end]
    return ''


def _extract_brace_block(text: str, start: int) -> Optional[int]:
    brace_depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if ch == '\\' and not escape:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        if escape:
            escape = False
        if in_string:
            continue
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0:
                return i + 1
    return None


def aggregate_call_chain(api_name: str, entry_module: str, entry_node: ast.AST, entry_file: PythonFileInfo, module_index: Dict[str, PythonFileInfo], simple_index: Dict[str, Set[str]], repo_root: Path, max_depth: int) -> Dict:
    visited: Set[str] = set()
    trace: List[Dict] = []

    def recurse(qualified_name: str, file_info: PythonFileInfo, node: ast.AST, depth: int):
        if depth > max_depth:
            return
        if qualified_name in visited:
            return
        visited.add(qualified_name)
        source = get_source_segment(file_info.source, node).strip()
        calls = sorted(collect_calls(node))
        step = {
            'qualified_name': qualified_name,
            'module': file_info.module_name,
            'source': source,
            'calls': calls,
            'child_functions': [],
            'cpp_leaves': {}
        }
        trace.append(step)

        for call_name in calls:
            resolved = resolve_candidate_names(call_name, file_info, module_index, simple_index)
            if not resolved:
                if is_cpp_leaf(call_name):
                    cpp_source = c_source_for_function(call_name.split('.')[-1], repo_root)
                    if cpp_source:
                        step['cpp_leaves'][call_name] = cpp_source.strip()
                continue
            for target in resolved:
                if target in visited:
                    continue
                resolved_def = search_python_definition(*normalize_api_name(target), module_index, simple_index)
                if resolved_def:
                    _, target_file, target_node = resolved_def
                    step['child_functions'].append(target)
                    recurse(target, target_file, target_node, depth + 1)
                elif is_cpp_leaf(target):
                    leaf_name = target.split('.')[-1]
                    cpp_source = c_source_for_function(leaf_name, repo_root)
                    if cpp_source:
                        step['cpp_leaves'][target] = cpp_source.strip()
        return

    entry_name = f"{entry_module}.{getattr(entry_node, 'name', api_name)}" if entry_module else getattr(entry_node, 'name', api_name)
    recurse(entry_name, entry_file, entry_node, 1)
    return {
        'entry': entry_name,
        'trace': trace,
        'leaf_cpp': {step['qualified_name']: step['cpp_leaves'] for step in trace if step['cpp_leaves']}
    }


def safe_filename(name: str) -> str:
    safe = re.sub(r'[^0-9a-zA-Z_]+', '_', name)
    return safe.strip('_') or 'api'


def save_extracted_source(api_name: str, trace: Dict, output_dir: Path):
    source_dir = output_dir / 'source'
    source_dir.mkdir(parents=True, exist_ok=True)
    file_name = safe_filename(api_name) + '.py'
    content_lines: List[str] = [f"# Extracted call chain for {api_name}", '']
    for step in trace['trace']:
        content_lines.append(f"# --- {step['qualified_name']} ---")
        content_lines.append(step['source'])
        content_lines.append('')
    extracted_path = source_dir / file_name
    extracted_path.write_text('\n'.join(content_lines), encoding='utf-8')
    return extracted_path


def extract_api_sources(api_file: str, repo_root: str, output_dir: str, max_depth: int, overwrite: bool):
    repo_root_path = Path(repo_root).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    apis = read_api_list(api_file)
    print(f"\n{'='*80}")
    print(f"Extracting {len(apis)} APIs from TensorFlow source code")
    print(f"Repository: {repo_root_path}")
    print(f"Output directory: {output_path}")
    print(f"Max recursion depth: {max_depth}")
    print(f"{'='*80}\n")
    
    module_index, simple_index = build_python_index(repo_root_path)
    print(f"Built index with {len(module_index)} TensorFlow modules\n")
    
    cache = load_cache()
    result: Dict[str, Dict] = {}
    found_count = 0
    not_found_count = 0

    for idx, api in enumerate(apis, start=1):
        print(f"[{idx:3d}/{len(apis):3d}] Processing: {api:<60}", end=" ", flush=True)
        
        if api in cache and not overwrite:
            result[api] = cache[api]
            status = cache[api].get('status', 'unknown')
            if status == 'found':
                found_count += 1
                print("✓ [cached]")
            else:
                not_found_count += 1
                print("✗ [cached]")
            continue
        
        api_module, api_func = normalize_api_name(api)
        entry_def = search_python_definition(api_module, api_func, module_index, simple_index)
        entry_info: Dict[str, Optional[str]] = {
            'api': api,
            'status': 'not_found',
            'entry_module': api_module,
            'entry_function': api_func,
            'python_trace': [],
            'cpp_trace': {},
            'source_file': None
        }
        
        if entry_def:
            found_module, file_info, node = entry_def
            trace = aggregate_call_chain(api, found_module, node, file_info, module_index, simple_index, repo_root_path, max_depth)
            entry_info['status'] = 'found'
            entry_info['python_trace'] = trace['trace']
            entry_info['cpp_trace'] = trace['leaf_cpp']
            entry_info['source_file'] = str(save_extracted_source(api, trace, output_path))
            found_count += 1
            print(f"✓ Found ({len(trace['trace'])} function layers)")
        else:
            not_found_count += 1
            entry_info['status'] = 'not_found'
            entry_info['message'] = 'Python entry definition not found; consider searching C++ sources directly.'
            print("✗ Not found")
        
        result[api] = entry_info
        cache[api] = entry_info
        save_cache(cache)

    output_json = output_path / 'api_sources.json'
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    
    print(f"\n{'='*80}")
    print(f"Extraction completed:")
    print(f"  ✓ Found: {found_count}")
    print(f"  ✗ Not found: {not_found_count}")
    print(f"  Total: {len(result)}")
    print(f"Output written to: {output_json}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Extract TensorFlow API source with static call graph analysis from a local TensorFlow checkout.')
    parser.add_argument('--api-file', default=str(DEFAULT_API_DEF), help='API definition list file')
    parser.add_argument('--repo-root', default=str(DEFAULT_REPO_ROOT), help='本地 TensorFlow 源码根目录')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='输出目录')
    parser.add_argument('--max-depth', type=int, default=5, help='递归调用分析最大深度')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已有缓存和输出')
    args = parser.parse_args()

    extract_api_sources(args.api_file, args.repo_root, args.output_dir, args.max_depth, args.overwrite)


if __name__ == '__main__':
    main()