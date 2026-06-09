#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


INPUT_PATH = Path("keywords.txt")
OUTPUT_PATH = Path("keywords_train_messages.jsonl")


def normalize_text(text: str) -> str:
    replacements = {
        "\u3000": " ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "，": ",",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def split_system_prompt_and_body(raw: str) -> tuple[str, str]:
    marker = "用户输入："
    if marker in raw:
        system_prompt, body = raw.split(marker, 1)
        return system_prompt.strip(), body.strip()
    raise ValueError("未找到 '用户输入：'，无法拆分 system prompt 和样本正文。")


def extract_samples(body: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"^\s*(\d+)[\.\．、]\s*(.*?)\n\s*(\[[\s\S]*?\])(?=\n\s*\d+[\.\．、]|\Z)",
        re.MULTILINE,
    )
    matches = pattern.findall(body)
    if not matches:
        raise ValueError("没有识别到任何样本，请检查 keywords.txt 格式。")
    return [(complaint.strip(), labels.strip()) for _, complaint, labels in matches]


def try_parse_labels(label_text: str) -> list[str]:
    candidate = normalize_text(label_text)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    variants = [
        candidate,
        candidate.replace('""', '"'),
        candidate.replace(",,", ","),
        candidate.replace("[,", "["),
        candidate.replace(",]", "]"),
    ]

    last_error = None
    for item in variants:
        try:
            value = ast.literal_eval(item)
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                return [x.strip() for x in value if x.strip()]
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    # Fallback: if the array syntax is messy, extract quoted string fragments directly.
    fragments = re.findall(r'"([^"]+)"|\'([^\']+)\'', candidate)
    merged = [a or b for a, b in fragments]
    merged = [x.strip(" ,") for x in merged if x.strip(" ,")]
    if merged:
        return merged

    raise ValueError(f"标签解析失败: {label_text}") from last_error


def build_records(system_prompt: str, samples: list[tuple[str, str]]) -> list[dict]:
    records = []
    for idx, (complaint, labels_text) in enumerate(samples, start=1):
        labels = try_parse_labels(labels_text)
        records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": complaint},
                    {"role": "assistant", "content": json.dumps(labels, ensure_ascii=False)},
                ],
                "meta": {"source": "keywords.txt", "index": idx},
            }
        )
    return records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 keywords.txt 转为 LoRA 微调可用的 JSONL 格式。")
    parser.add_argument("--input", default=str(INPUT_PATH), help=f"输入文件，默认 {INPUT_PATH}")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help=f"输出 JSONL，默认 {OUTPUT_PATH}")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    raw = input_path.read_text(encoding="utf-8")
    system_prompt, body = split_system_prompt_and_body(raw)
    samples = extract_samples(body)
    records = build_records(system_prompt, samples)
    write_jsonl(records, output_path)

    print(f"System prompt loaded: {len(system_prompt)} chars")
    print(f"Parsed samples: {len(records)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
