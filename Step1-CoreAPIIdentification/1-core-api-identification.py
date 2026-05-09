import argparse
import csv
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DEFAULT_INPUT_DIR = os.path.join(
    REPO_ROOT, "Step0-DataCollection", "data", "pytorch_llm_processed_data"
)
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "pytorch_core_api_identification")
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "core_api_identification_results.csv")


PROMPT_TEMPLATE = """Bug Report:
Title: {title}
Issue ID: {issue_id}
Original bug description: {bug_description}
Structured error description: {error_description}
Minimal reproducible example: {repro_code}
Trigger conditions: {trigger_conditions}
Environment: {environment}

Task:
Analyze the title, descriptions, reproducible code, trigger conditions, and environment.
Infer which single API is most responsible for triggering the bug.

Constraints:
- Return only one API name, and nothing else.
- Do not return JSON, Markdown, explanation, punctuation, or quotes.
- If no responsible API can be identified, return None.
- Prefer the API whose behavior change, compilation path, backend dispatch, RNG state semantics, or execution semantics best explains the observed bug.
"""


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        item = as_text(value)
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def split_api_text(value: Any) -> List[str]:
    text = as_text(value)
    if not text:
        return []
    parts = re.split(r"[,;\n]+", text)
    return unique_preserve_order(parts)


def extract_candidate_apis(issue: Dict[str, Any]) -> List[str]:
    apis = issue.get("apis", [])
    if isinstance(apis, list):
        candidates = unique_preserve_order(as_text(api) for api in apis)
    else:
        candidates = split_api_text(apis)

    if not candidates:
        candidates = split_api_text(issue.get("affected_api", ""))

    return candidates


def build_core_api_prompt(issue: Dict[str, Any]) -> Tuple[str, List[str]]:
    candidates = extract_candidate_apis(issue)
    prompt = PROMPT_TEMPLATE.format(
        title=as_text(issue.get("title")),
        issue_id=as_text(issue.get("issue_id")),
        bug_description=as_text(issue.get("bug_description")),
        error_description=as_text(issue.get("error_description")),
        repro_code=as_text(issue.get("repro_code")) or as_text(issue.get("best_code")),
        trigger_conditions=json.dumps(
            issue.get("trigger_conditions", {}), ensure_ascii=False
        ),
        environment=json.dumps(issue.get("environment", {}), ensure_ascii=False),
    )
    return prompt, candidates


def extract_zhipu_chunk_content(chunk: Any) -> Optional[str]:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta")
            if isinstance(delta, dict):
                if delta.get("content") is not None:
                    return delta["content"]
                if delta.get("reasoning_content") is not None:
                    return delta["reasoning_content"]
            message = choice.get("message")
            if isinstance(message, dict):
                if message.get("content") is not None:
                    return message["content"]
                if message.get("reasoning_content") is not None:
                    return message["reasoning_content"]
            if isinstance(choice, dict):
                if choice.get("text") is not None:
                    return choice["text"]
                if choice.get("reasoning_content") is not None:
                    return choice["reasoning_content"]
        return None

    try:
        choice = chunk.choices[0]
    except Exception:
        return None

    delta = getattr(choice, "delta", None)
    if delta is not None:
        content = getattr(delta, "content", None)
        if content is not None:
            return content
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content is not None:
            return reasoning_content
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return content
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is not None:
            return reasoning_content
    text = getattr(choice, "text", None)
    if text is not None:
        return text
    return getattr(choice, "reasoning_content", None)


def serialize_zhipu_event(event: Any) -> Any:
    if isinstance(event, dict):
        return event
    try:
        return json.loads(json.dumps(event, ensure_ascii=False))
    except Exception:
        return repr(event)


def call_zhipu_llm(
    api_key: str,
    prompt: str,
    model: str = "glm-4.7",
    temperature: float = 0.0,
    max_tokens: int = 32768,
) -> Tuple[str, List[Any]]:
    try:
        from zai import ZhipuAiClient
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'zai'. Install it before running LLM inference."
        ) from exc

    client = ZhipuAiClient(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": "You identify the single core PyTorch API from a bug report. Reply with only one API name, or None.",
        },
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        thinking={"type": "enabled"},
    )

    chunks: List[str] = []
    raw_events: List[Any] = []
    for chunk in response:
        raw_events.append(chunk)
        content = extract_zhipu_chunk_content(chunk)
        if content is not None:
            chunks.append(content)

    return "".join(chunks).strip(), raw_events


def parse_json_response(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def normalize_selected_api(selected_api: Any, candidates: List[str]) -> Tuple[str, bool]:
    selected = as_text(selected_api)
    if not selected or selected.lower() == "none":
        return "None", False

    for candidate in candidates:
        if selected == candidate:
            return candidate, True

    selected_key = selected.lower()
    for candidate in candidates:
        if selected_key == candidate.lower():
            return candidate, True

    return selected, False


def parse_core_api_response(response_text: str, candidates: List[str]) -> Tuple[str, bool]:
    text = response_text.strip()
    fence_match = re.search(r"```(?:text|json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    if text.startswith("{"):
        parsed = parse_json_response(text)
        text = as_text(parsed.get("core_api") or parsed.get("api"))

    text = text.strip().strip("\"'`").strip()
    if "\n" in text:
        text = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if text.endswith("."):
        text = text[:-1].strip()

    return normalize_selected_api(text, candidates)


def build_result_without_llm(prompt: str, candidates: List[str]) -> Dict[str, Any]:
    return {
        "core_api": "None",
        "parse_error": False,
        "prompt": prompt,
    }


def build_csv_row(issue: Dict[str, Any], source_file: str, output_file: str) -> Dict[str, str]:
    result = issue.get("core_api_identification", {})
    meta = issue.get("_meta", {}).get("core_api_identification", {})
    return {
        "issue_id": as_text(issue.get("issue_id")),
        "title": as_text(issue.get("title")),
        "core_api": as_text(result.get("core_api")),
        "parse_error": as_text(result.get("parse_error")),
        "source_file": os.path.relpath(source_file, REPO_ROOT) if source_file else "",
        "output_file": os.path.relpath(output_file, REPO_ROOT),
        "prompt_file": as_text(meta.get("prompt_file")),
        "raw_response_file": as_text(result.get("raw_response_file"))
        or as_text(meta.get("response_file")),
    }


def read_existing_result(out_path: str) -> Dict[str, str]:
    issue = read_json(out_path)
    return build_csv_row(issue, "", out_path)


def process_file(
    json_file: str,
    output_dir: str,
    api_key: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    overwrite: bool,
    prompt_only: bool,
) -> Dict[str, str]:
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    out_path = os.path.join(output_dir, base_name + ".json")
    io_dir = os.path.join(output_dir, "llm_io")
    prompt_path = os.path.join(io_dir, base_name + ".input.txt")
    response_path = os.path.join(io_dir, base_name + ".output.txt")
    response_log_path = os.path.join(io_dir, base_name + ".output.events.json")

    if os.path.exists(out_path) and not overwrite:
        print(f"Skip existing: {out_path}")
        return read_existing_result(out_path)

    issue = read_json(json_file)
    prompt, candidates = build_core_api_prompt(issue)
    save_text(prompt_path, prompt)

    if prompt_only or not api_key:
        llm_result = build_result_without_llm(prompt, candidates)
        response_text = ""
    else:
        if os.path.exists(response_path) and not overwrite:
            print(f"Reuse saved LLM output: {response_path}")
            with open(response_path, "r", encoding="utf-8", errors="ignore") as f:
                response_text = f.read().strip()
        else:
            print(f"Calling LLM for {os.path.basename(json_file)} ...")
            response_text, response_events = call_zhipu_llm(
                api_key=api_key,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            save_text(response_path, response_text)
            serialized_events = [serialize_zhipu_event(event) for event in response_events]
            save_text(response_log_path, json.dumps(serialized_events, ensure_ascii=False, indent=2))

        try:
            core_api, _ = parse_core_api_response(response_text, candidates)
            llm_result = {
                "core_api": core_api,
                "parse_error": False,
                "raw_response_file": os.path.relpath(response_path, REPO_ROOT),
                "raw_response_events_file": os.path.relpath(response_log_path, REPO_ROOT),
            }
        except Exception as exc:
            llm_result = {
                "core_api": "None",
                "parse_error": True,
                "parse_error_message": as_text(exc),
                "raw_response_file": os.path.relpath(response_path, REPO_ROOT),
                "raw_response_events_file": os.path.relpath(response_log_path, REPO_ROOT),
            }

    issue["core_api_identification"] = llm_result
    issue.setdefault("_meta", {})
    issue["_meta"]["core_api_identification"] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model if api_key and not prompt_only else "",
        "prompt_file": os.path.relpath(prompt_path, REPO_ROOT),
        "response_file": os.path.relpath(response_path, REPO_ROOT)
        if response_text
        else "",
        "prompt_only": bool(prompt_only or not api_key),
    }

    save_json(out_path, issue)
    print(f"Saved: {out_path}")
    return build_csv_row(issue, json_file, out_path)


def save_csv(csv_path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "issue_id",
        "title",
        "core_api",
        "parse_error",
        "source_file",
        "output_file",
        "prompt_file",
        "raw_response_file",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_json_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.endswith(".json")
        )
    if os.path.isfile(input_path) and input_path.endswith(".json"):
        return [input_path]
    raise FileNotFoundError(f"Input path does not exist or is not JSON: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify core APIs from processed PyTorch bug reports with an LLM."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Input JSON file or directory. Defaults to Step0 PyTorch LLM processed data.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for enriched core API identification JSON files.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ZHIPUAI_API_KEY", "044f2bc486d14e13aa68259fbd1970c4.hjLhKCmeXaDRt2xf"),
        help="Zhipu API key. If omitted, prompts are generated without LLM inference.",
    )
    parser.add_argument("--model", default="glm-4.7", help="LLM model name.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=0, help="Process only N files.")
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help="CSV file used to store one predicted core API per issue.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Only generate prompts and output skeleton JSON; do not call the LLM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    json_files = iter_json_files(input_path)
    if args.limit > 0:
        json_files = json_files[: args.limit]

    if not args.api_key or args.prompt_only:
        print("LLM inference disabled; generating prompts and skeleton results.")

    print(f"Found {len(json_files)} JSON file(s).")
    processed = 0
    failed = 0
    csv_rows = []
    for index, json_file in enumerate(json_files, 1):
        try:
            print(f"[{index}/{len(json_files)}] Processing {os.path.basename(json_file)}")
            row = process_file(
                json_file=json_file,
                output_dir=output_dir,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                overwrite=args.overwrite,
                prompt_only=args.prompt_only,
            )
            csv_rows.append(row)
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"Error processing {json_file}: {exc}")

    csv_path = os.path.abspath(args.csv_path)
    save_csv(csv_path, csv_rows)
    print(f"Done. Processed: {processed}; failed: {failed}; output: {output_dir}")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
