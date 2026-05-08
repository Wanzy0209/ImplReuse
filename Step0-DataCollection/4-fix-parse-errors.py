import os
import time
import re
import json
import argparse
from zai import ZhipuAiClient


def save_json(output_dir, base_name, data):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, base_name + '.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def save_text(output_dir, base_name, suffix, text):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{base_name}.{suffix}.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_prompt(prompt_dir, base_name):
    prompt_path = os.path.join(prompt_dir, base_name + '.txt')
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    return None


def call_zhipu_llm(api_key, prompt, model='glm-4.7', temperature=0.0, max_tokens=8192):
    client = ZhipuAiClient(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in analyzing GitHub issues."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        thinking={
            "type": "enabled"
        }
    )
    
    full_content = ""
    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            full_content += chunk.choices[0].delta.reasoning_content
        if chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content
    return full_content


def fix_parse_error(json_file, api_key, prompt_dir, overwrite=False):
    # 读取有错误的JSON文件
    issue_json = read_json(json_file)
    
    # 检查是否有parse_error
    if not issue_json.get('parse_error', False):
        print(f"Skip {os.path.basename(json_file)} - no parse error")
        return False
    
    out_base = os.path.splitext(os.path.basename(json_file))[0]
    output_dir = os.path.dirname(json_file)
    
    # 从prompt目录读取已生成的prompt
    prompt = read_prompt(prompt_dir, out_base)
    if not prompt:
        print(f"Warning: Prompt not found for {out_base}, skipping...")
        return False

    print(f"Fixing parse error for {os.path.basename(json_file)} ...")
    response_text = call_zhipu_llm(api_key, prompt)

    io_dir = os.path.join(output_dir, 'llm_io')
    input_file = save_text(io_dir, out_base, 'input', prompt)
    output_file = save_text(io_dir, out_base, 'output', response_text)
    print(f"Saved LLM IO -> {input_file}, {output_file}")

    try:
        # 查找 ```json 代码块中的 JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1).strip()
            parsed = json.loads(json_str)
        else:
            # 如果没有找到 ```json 代码块，尝试直接解析整个响应
            parsed = json.loads(response_text)

        issue_json['_meta'] = {
            'file': os.path.basename(json_file),
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'glm-4.7',
            'llm_input_file': os.path.relpath(input_file),
            'llm_output_file': os.path.relpath(output_file),
            'parse_error': bool(parsed.get('parse_error'))
        }
        issue_json['llm_full_response'] = response_text
        issue_json['title'] = parsed.get('title', issue_json.get('title', ''))
        issue_json['error_description'] = parsed.get('error_description', '')
        issue_json['repro_code'] = parsed.get('repro_code', '')
        issue_json['repro_output'] = parsed.get('repro_output', '')
        issue_json['affected_api'] = parsed.get('affected_api', '')
        issue_json['trigger_conditions'] = parsed.get('trigger_conditions', '')
        issue_json['environment'] = parsed.get('environment', '')
        issue_json['labels'] = parsed.get('labels', [])
        # 移除parse_error字段，因为已经修复
        if 'parse_error' in issue_json:
            del issue_json['parse_error']
        if 'raw_response_file' in issue_json:
            del issue_json['raw_response_file']
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        # 尝试提取 ```json 代码块中的 JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                parsed = json.loads(json_str)
            except Exception:
                parsed = {'parse_error': True, 'raw_response_file': os.path.relpath(output_file)}
        else:
            parsed = {'parse_error': True, 'raw_response_file': os.path.relpath(output_file)}
        issue_json['llm_full_response'] = response_text
        issue_json['parse_error'] = parsed.get('parse_error', True)
        issue_json['raw_response_file'] = parsed.get('raw_response_file', os.path.relpath(output_file))

    saved = save_json(output_dir, out_base, issue_json)
    print(f"Saved -> {saved}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Fix parse errors in LLM processed JSON files.')
    parser.add_argument('--input-dir', default='./data/pytorch_llm_processed_data',
                        help='Input directory containing JSON files with parse errors')
    parser.add_argument('--prompt-dir', default='./data/pytorch_prompts',
                        help='Directory containing pre-generated prompt files')
    parser.add_argument('--api-key', default='044f2bc486d14e13aa68259fbd1970c4.hjLhKCmeXaDRt2xf',
                        help='Zhipu API key (or use ZHIPUAI_API_KEY env var)')
    args = parser.parse_args()

    if not args.api_key:
        print("Error: API key is required. Set ZHIPUAI_API_KEY environment variable or use --api-key.")
        return

    input_path = os.path.abspath(args.input_dir)
    prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.prompt_dir))

    if not os.path.exists(prompt_dir):
        print(f"Error: Prompt directory does not exist: {prompt_dir}")
        return

    if os.path.isdir(input_path):
        json_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.json')])
        print(f"Found {len(json_files)} JSON files in {input_path}")
        
        fixed_count = 0
        skipped_count = 0
        
        for i, json_file in enumerate(json_files, 1):
            try:
                issue_json = read_json(json_file)
                if issue_json.get('parse_error', False):
                    print(f"[{i}/{len(json_files)}] Fixing {os.path.basename(json_file)}...")
                    success = fix_parse_error(json_file, args.api_key, prompt_dir)
                    if success:
                        fixed_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"[{i}/{len(json_files)}] Error processing {json_file}: {e}")
                skipped_count += 1
        
        print(f"\nFixing complete:")
        print(f"- Total files: {len(json_files)}")
        print(f"- Fixed: {fixed_count}")
        print(f"- Skipped: {skipped_count}")
    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == '__main__':
    main()