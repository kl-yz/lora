#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


INPUT_PATH = Path("keywords_train_messages.jsonl")
OUTPUT_PATH = Path("keywords_train_messages.clean.jsonl")


REPLACEMENTS = {
    "「": '"',
    "」": '"',
    "＋": "+",
    "・": "·",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "\u3000": " ",
}


def normalize_text(text: str) -> str:
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def normalize_assistant_json(text: str) -> str:
    text = normalize_text(text)
    text = text.replace('"系统故障 联系检查"', '\\"系统故障 联系检查\\"')
    return text


def clean_row(row: dict) -> dict:
    cleaned = dict(row)
    messages = []
    for message in row["messages"]:
        content = message["content"]
        if message["role"] == "assistant":
            content = normalize_assistant_json(content)
        else:
            content = normalize_text(content)
        messages.append(
            {
                "role": message["role"],
                "content": content,
            }
        )
    cleaned["messages"] = messages
    if "meta" in row:
        cleaned["meta"] = row["meta"]
    return cleaned


def main() -> None:
    rows = []
    for line in INPUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(clean_row(row))

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Loaded rows: {len(rows)}")
    print(f"Saved clean dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
