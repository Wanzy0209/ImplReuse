import argparse
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


PROMPT_TEMPLATE = """Bug Report:
Title: {title}
Detailed bug description: {bug_description}
Minimal reproducible example: {repro_code}
Candidate APIs: {candidate_apis}

Task:
Summarize the observed incorrect behavior.
Infer the most likely root cause.
Identify the single API most responsible for triggering the bug.

Return JSON only, without Markdown fences or extra explanation, using this schema:
{{
  "error_summary": "<concise summary of the incorrect behavior>",
  "likely_root_cause": "<most likely root cause inferred from the report and code>",
  "core_api": "<one API selected from Candidate APIs>",
  "confidence": <number from 0.0 to 1.0>,
  "evidence": "<brief reason for selecting this API>"
}}

Constraints:
- Select exactly one API from Candidate APIs.
- If Candidate APIs is empty, set core_api to an empty string and explain why in evidence.
- Do not invent APIs that are not in Candidate APIs.
- Prefer the API whose behavior change, compilation path, backend dispatch, or execution semantics best explains the observed bug.
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
    parts = re.split(r"[,;，；\n]+", text)
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
    candidate_text = ", ".join(candidates)
    bug_description = as_text(issue.get("error_description")) or as_text(
        issue.get("bug_description")
    )
    repro_code = as_text(issue.get("repro_code")) or as_text(issue.get("best_code"))

    prompt = PROMPT_TEMPLATE.format(
        title=as_text(issue.get("title")),
        bug_description=bug_description,
        repro_code=repro_code,
        candidate_apis=candidate_text,
    )
    return prompt, candidates


def call_zhipu_llm(
    api_key: str,
    prompt: str,
    model: str = "glm-4.7",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
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
            "content": "You are an assistant specialized in structured analysis of PyTorch bug reports.",
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

    chunks = []
    for chunk in response:
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            chunks.append(delta.content)
    return "".join(chunks).strip()


def parse_llm_json(response_text: str) -> Dict[str, Any]:
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
    if not selected:
        return "", not candidates

    for candidate in candidates:
        if selected == candidate:
            return candidate, True

    selected_key = selected.lower()
    for candidate in candidates:
        if selected_key == candidate.lower():
            return candidate, True

    return selected, False


def build_result_without_llm(prompt: str, candidates: List[str]) -> Dict[str, Any]:
    return {
        "error_summary": "",
        "likely_root_cause": "",
        "core_api": "",
        "confidence": 0.0,
        "evidence": "LLM inference was not executed; prompt was generated only.",
        "candidate_apis": candidates,
        "candidate_match": False,
        "parse_error": False,
        "prompt": prompt,
    }


def process_file(
    json_file: str,
    output_dir: str,
    api_key: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    overwrite: bool,
    prompt_only: bool,
) -> str:
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    out_path = os.path.join(output_dir, base_name + ".json")
    io_dir = os.path.join(output_dir, "llm_io")
    prompt_path = os.path.join(io_dir, base_name + ".input.txt")
    response_path = os.path.join(io_dir, base_name + ".output.txt")

    if os.path.exists(out_path) and not overwrite:
        print(f"Skip existing: {out_path}")
        return out_path

    issue = read_json(json_file)
    prompt, candidates = build_core_api_prompt(issue)
    save_text(prompt_path, prompt)

    if prompt_only or not api_key:
        llm_result = build_result_without_llm(prompt, candidates)
        response_text = ""
    else:
        print(f"Calling LLM for {os.path.basename(json_file)} ...")
        response_text = call_zhipu_llm(
            api_key=api_key,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        save_text(response_path, response_text)
        try:
            parsed = parse_llm_json(response_text)
            core_api, candidate_match = normalize_selected_api(
                parsed.get("core_api"), candidates
            )
            llm_result = {
                "error_summary": as_text(parsed.get("error_summary")),
                "likely_root_cause": as_text(parsed.get("likely_root_cause")),
                "core_api": core_api,
                "confidence": parsed.get("confidence", 0.0),
                "evidence": as_text(parsed.get("evidence")),
                "candidate_apis": candidates,
                "candidate_match": candidate_match,
                "parse_error": False,
            }
        except Exception as exc:
            llm_result = {
                "error_summary": "",
                "likely_root_cause": "",
                "core_api": "",
                "confidence": 0.0,
                "evidence": f"Failed to parse LLM response: {exc}",
                "candidate_apis": candidates,
                "candidate_match": False,
                "parse_error": True,
                "raw_response_file": os.path.relpath(response_path, REPO_ROOT),
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
    return out_path


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
        default=os.getenv("ZHIPUAI_API_KEY", ""),
        help="Zhipu API key. If omitted, prompts are generated without LLM inference.",
    )
    parser.add_argument("--model", default="glm-4.7", help="LLM model name.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="Process only N files.")
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
    for index, json_file in enumerate(json_files, 1):
        try:
            print(f"[{index}/{len(json_files)}] Processing {os.path.basename(json_file)}")
            process_file(
                json_file=json_file,
                output_dir=output_dir,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                overwrite=args.overwrite,
                prompt_only=args.prompt_only,
            )
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"Error processing {json_file}: {exc}")

    print(f"Done. Processed: {processed}; failed: {failed}; output: {output_dir}")


if __name__ == "__main__":
    main()
