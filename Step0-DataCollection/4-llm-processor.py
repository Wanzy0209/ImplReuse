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


def llm_process(json_file, api_key, output_dir, issue_json, prompt_dir, overwrite=False):
    # 使用JSON文件名作为基础名称
    out_base = os.path.splitext(os.path.basename(json_file))[0]
    out_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_dir, out_base + '.json')
    
    if os.path.exists(out_path) and not overwrite:
        print(f"Skip existing {out_base}.json")
        return read_json(out_path)

    # 从prompt目录读取已生成的prompt
    prompt = read_prompt(prompt_dir, out_base)
    if not prompt:
        print(f"Warning: Prompt not found for {out_base}, skipping...")
        return issue_json

    print(f"Calling LLM for {os.path.basename(json_file)} ...")
    response_text = call_zhipu_llm(api_key, prompt)

    io_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_dir, 'llm_io')
    input_file = save_text(io_dir, out_base, 'input', prompt)
    output_file = save_text(io_dir, out_base, 'output', response_text)
    print(f"Saved LLM IO -> {input_file}, {output_file}")

    try:
        # 从响应中提取最后一个 JSON 对象
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
        issue_json['title'] = parsed['title']
        issue_json['error_description'] = parsed['error_description']
        issue_json['repro_code'] = parsed['repro_code']
        issue_json['repro_output'] = parsed['title']
        issue_json['affected_api'] = parsed.get('affected_api', '')
        issue_json['trigger_conditions'] = parsed.get('trigger_conditions', '')
        issue_json['environment'] = parsed.get('environment', '')
        issue_json['labels'] = parsed.get('labels', [])
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
        issue_json['parse_error'] = parsed.get('parse_error', True)
        issue_json['raw_response_file'] = parsed.get('raw_response_file', os.path.relpath(output_file))

    saved = save_json(os.path.join(os.path.abspath(os.path.dirname(__file__)), output_dir), out_base, issue_json)
    print(f"Saved -> {saved}")
    return issue_json


def main():
    parser = argparse.ArgumentParser(description='Process GitHub issue JSON files with LLM.')
    parser.add_argument('--input-dir', default='./data/pytorch_processed_data',
                        help='Input JSON file or directory containing JSON files')
    parser.add_argument('--output-dir', default='./data/pytorch_llm_processed_data',
                        help='Output directory for processed JSON files')
    parser.add_argument('--prompt-dir', default='./data/pytorch_prompts',
                        help='Directory containing pre-generated prompt files')
    parser.add_argument('--api-key', default='044f2bc486d14e13aa68259fbd1970c4.hjLhKCmeXaDRt2xf',
                        help='Zhipu API key (or use ZHIPUAI_API_KEY env var)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing JSON files')
    args = parser.parse_args()

    if not args.api_key:
        print("Error: API key is required. Set ZHIPUAI_API_KEY environment variable or use --api-key.")
        return

    input_path = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
    prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.prompt_dir))

    if not os.path.exists(prompt_dir):
        print(f"Error: Prompt directory does not exist: {prompt_dir}")
        return

    if os.path.isdir(input_path):
        json_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.json')])
        print(f"Found {len(json_files)} JSON files in {input_path}")
        
        processed_count = 0
        skipped_count = 0
        
        for i, json_file in enumerate(json_files, 1):
            try:
                print(f"[{i}/{len(json_files)}] Processing {os.path.basename(json_file)}...")
                issue_json = read_json(json_file)
                # 直接传递JSON文件路径给llm_process
                result = llm_process(json_file, args.api_key, output_dir, issue_json, prompt_dir, overwrite=args.overwrite)
                if result != issue_json:
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"[{i}/{len(json_files)}] Error processing {json_file}: {e}")
                skipped_count += 1
        
        print(f"\nProcessing complete:")
        print(f"- Total files: {len(json_files)}")
        print(f"- Processed: {processed_count}")
        print(f"- Skipped: {skipped_count}")
        
    elif os.path.isfile(input_path):
        try:
            print(f"Processing {os.path.basename(input_path)}...")
            issue_json = read_json(input_path)
            llm_process(input_path, args.api_key, output_dir, issue_json, prompt_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == '__main__':
    main()