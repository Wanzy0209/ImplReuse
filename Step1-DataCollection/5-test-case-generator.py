import os
import json
import argparse


def generate_test_case(path, output_dir, issue_json, overwrite=False):
    out_base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_dir, out_base + '.py')
    if os.path.exists(out_path) and not overwrite:
        print(f"Skip existing {out_base}.py")
        return None
    os.makedirs(output_dir, exist_ok=True)
    if issue_json.get("best_code") is not None:
        with open(out_path, 'w', encoding='utf-8') as w_file:
            w_file.write(issue_json["best_code"])
        print(f"Saved test case -> {out_path}")


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Generate test cases from processed GitHub issue JSON files.')
    parser.add_argument('--input-dir', default='./data/pytorch_test_case',
                        help='Input JSON file or directory containing JSON files')
    parser.add_argument('--output-dir', default='./data/pytorch_test_case',
                        help='Output directory for test case Python files')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing Python files')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))

    if os.path.isdir(input_path):
        json_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.json')])
        for json_file in json_files:
            try:
                issue_json = read_json(json_file)
                generate_test_case(json_file, output_dir, issue_json, overwrite=args.overwrite)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        print(f"Processed {len(json_files)} files.")
    elif os.path.isfile(input_path):
        issue_json = read_json(input_path)
        generate_test_case(input_path, output_dir, issue_json, overwrite=args.overwrite)
    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == '__main__':
    main()