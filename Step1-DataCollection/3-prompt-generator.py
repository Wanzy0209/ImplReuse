import os
import json
import argparse


def build_prompt(payload):
    prompt = f"""
    You are an assistant that extracts structured information from a GitHub issue.
    You MUST only extract information explicitly stated in the input.
    Do NOT infer, speculate, or generate explanations.
    If a field is missing, use empty string.

    Input fields:
    - title: {payload['title']!s}
    - issue_id: {payload['issue_id']!s}
    - bug_description: {payload['bug_description']!s}

    Return JSON only (no explanation, no extra text) in this exact schema:
    {{
      "title": "<original title>",
      "error_description": "<natural language bug description>",

      "repro_code": "<minimal reproducible code if exists, otherwise empty string>",
      "repro_output": "<observed incorrect output if exists>",

      "trigger_conditions": {{
        "backend": "<e.g., MPS>",
        "dtype": "<data types mentioned or unknown>",
        "input_shape": "<shapes or dimensional constraints if mentioned>",
        "other_conditions": "<any additional explicit conditions>"
      }},

      "affected_api": "<API/function directly involved, e.g., F.linear>",

      "environment": {{
        "pytorch_version": "<version or unknown>",
        "python_version": "<version or unknown>",
        "os": "<os info or unknown>",
        "hardware": "<cpu/gpu info or unknown>"
      }},

      "labels": ["<label1>", "<label2>"]
    }}

    Requirements:
    - Extract only what is explicitly present.
    - Keep original wording for code and key descriptions when possible.
    - Do not summarize away critical technical details.
    - Do not include any explanation outside JSON.
    """
    return prompt.strip()


def read_json(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Generate LLM prompt from processed GitHub issue JSON files.')
    parser.add_argument('--input-dir', default='./data/pytorch_processed_data',
                        help='Input JSON file or directory containing JSON files')
    parser.add_argument('--output-dir', default='./data/pytorch_prompts',
                        help='Output directory for prompt files')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        json_files = sorted([f for f in os.listdir(input_path) if f.endswith('.json')])
        print(f"Found {len(json_files)} JSON files in {input_path}")

        for i, json_file in enumerate(json_files, 1):
            try:
                full_path = os.path.join(input_path, json_file)
                issue_json = read_json(full_path)
                prompt = build_prompt(issue_json)

                out_base = os.path.splitext(json_file)[0]
                out_path = os.path.join(output_dir, out_base + '.txt')

                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                print(f"[{i}/{len(json_files)}] Saved: {out_path}")
            except Exception as e:
                print(f"[{i}/{len(json_files)}] Error processing {json_file}: {e}")

        print(f"\nDone! Generated {len(json_files)} prompts in {output_dir}")

    elif os.path.isfile(input_path):
        try:
            issue_json = read_json(input_path)
            prompt = build_prompt(issue_json)

            out_base = os.path.splitext(os.path.basename(input_path))[0]
            out_path = os.path.join(output_dir, out_base + '.txt')

            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            print(f"Prompt saved to: {out_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == '__main__':
    main()