#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execute RQ3 prompt files and save LLM responses.

This script reads prompt text files and saves each response plus raw event data
into the configured output directory.

Default output:
  Step5-Evaluate/RQ3/data/no_coreapi_identification/llm_responses_all_replacements

Example:
  python Step5-Evaluate/RQ3/3_execute_prompt_all_replacements.py
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_DIR = SCRIPT_DIR / "data" / "no_coreapi_identification" / "prompts_all_replacements"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "no_coreapi_identification" / "llm_responses_all_replacements"
DEFAULT_IO_SUBDIR = "llm_io"


def extract_zhipu_chunk_content(chunk: Any) -> Optional[str]:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta")
            if isinstance(delta, dict) and delta.get("content") is not None:
                return delta["content"]
            message = choice.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return message["content"]
            if isinstance(choice, dict) and choice.get("text") is not None:
                return choice["text"]
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
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return content
    return getattr(choice, "text", None)


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
    max_tokens: int = 8192,
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
            "content": "You are a helpful assistant specialized in generating test cases for similar APIs based on bug reports.",
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


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def load_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding='utf-8')
    except Exception as exc:
        print(f"Failed to load {path}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Execute prompts and save LLM responses for RQ3 all replacements.")
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=DEFAULT_PROMPT_DIR,
        help="Directory containing prompt files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for LLM responses."
    )
    parser.add_argument("--api-key", default=os.getenv("ZHIPUAI_API_KEY", "044f2bc486d14e13aa68259fbd1970c4.hjLhKCmeXaDRt2xf"), help="Zhipu API key.")
    parser.add_argument("--model", default="glm-4.7", help="LLM model.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N prompts.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing response files.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    io_dir = args.output_dir / DEFAULT_IO_SUBDIR
    io_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = list(args.prompt_dir.glob("*.txt"))
    if args.limit > 0:
        prompt_files = prompt_files[:args.limit]

    print(f"Processing {len(prompt_files)} prompts...")

    for prompt_file in prompt_files:
        prompt_text = load_text(prompt_file)
        if prompt_text is None:
            print(f"Skipping {prompt_file}: failed to read")
            continue

        base_name = prompt_file.stem
        input_file = io_dir / f"{base_name}.input.txt"
        response_file = io_dir / f"{base_name}.output.txt"
        events_file = io_dir / f"{base_name}.output.events.json"

        save_text(input_file, prompt_text)
        if response_file.exists() and not args.overwrite:
            print(f"Skipping {base_name}: response exists")
            continue

        print(f"Calling LLM for {base_name}...")
        try:
            response_text, response_events = call_zhipu_llm(
                api_key=args.api_key,
                prompt=prompt_text,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            save_text(response_file, response_text)
            serialized_events = [serialize_zhipu_event(event) for event in response_events]
            save_text(events_file, json.dumps(serialized_events, ensure_ascii=False, indent=2))
            print(f"Saved response: {response_file}")
        except Exception as exc:
            print(f"Error for {base_name}: {exc}")


if __name__ == "__main__":
    main()
