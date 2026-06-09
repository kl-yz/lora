#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


BASE_MODEL = "./Qwen3.5-2B"
ADAPTER_PATH = Path("outputs/qwen35-2b-keyword-lora-v4/adapter")
PROMPT_CONFIG = ADAPTER_PATH / "prompt_config.json"
MAX_SEQ_LENGTH = 1024


def load_system_prompt() -> str:
    if PROMPT_CONFIG.exists():
        config = json.loads(PROMPT_CONFIG.read_text(encoding="utf-8"))
        return config["system_prompt"]

    raise FileNotFoundError(f"找不到 v4 prompt 配置：{PROMPT_CONFIG}")


def text_tokenizer(processor_or_tokenizer):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def main():
    import unsloth  # noqa: F401
    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    if len(sys.argv) < 2:
        print('Usage: python infer.py "用户投诉：车辆无法启动，仪表黑屏。"')
        raise SystemExit(1)

    user_text = sys.argv[1]
    system_prompt = load_system_prompt()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        local_files_only=True,
        use_exact_model_name=True,
    )

    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = text_tokenizer(tokenizer)(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=text_tokenizer(tokenizer).eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    result = text_tokenizer(tokenizer).decode(new_tokens, skip_special_tokens=True).strip()
    print(result)


if __name__ == "__main__":
    main()
