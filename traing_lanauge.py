#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_FUSED_CE_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"


DATA_PATH = Path("keywords_train_messages.clean.jsonl")
BASE_MODEL = "./Qwen3.5-2B"
OUTPUT_DIR = "outputs/qwen35-2b-keyword-lora-v4"
MAX_SEQ_LENGTH = 1024
RANDOM_STATE = 3407
EVAL_RATIO = 0.1

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the v4 keyword-extraction LoRA adapter.")
    parser.add_argument("--prepare_only", action="store_true", help="只构造并检查数据，不加载模型训练。")
    return parser.parse_args()


def read_data(data_path: Path) -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []

    with data_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            messages = row["messages"]
            if len(messages) < 3 or messages[-1].get("role") != "assistant":
                raise ValueError(f"{data_path} 第 {line_no} 行必须包含 system/user/assistant messages。")

            answer = messages[-1]["content"]
            try:
                parsed = json.loads(answer)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{data_path} 第 {line_no} 行的 assistant 内容不是合法 JSON 数组:\n{answer}"
                ) from exc

            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise ValueError(f"{data_path} 第 {line_no} 行的 assistant 内容必须是 list[str]:\n{answer}")

            normalized_messages = [dict(msg) for msg in messages[:-1]]
            normalized_messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(parsed, ensure_ascii=False),
                }
            )
            rows.append(normalized_messages)

    return rows


def chat(tokenizer, messages, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def text_tokenizer(processor_or_tokenizer):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def encode_text(tokenizer, text):
    return text_tokenizer(tokenizer)(text, add_special_tokens=False)["input_ids"]


def build_records(tokenizer, data_path: Path):
    rows = read_data(data_path)
    records = []
    truncated = 0

    for messages in rows:
        prompt_messages = messages[:-1]
        full_messages = messages

        prompt_ids = encode_text(tokenizer, chat(tokenizer, prompt_messages, add_generation_prompt=True))
        full_ids = encode_text(tokenizer, chat(tokenizer, full_messages, add_generation_prompt=False))

        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template 前缀不一致，无法可靠 mask prompt tokens。")

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if len(full_ids) > MAX_SEQ_LENGTH:
            truncated += 1

        input_ids = full_ids[:MAX_SEQ_LENGTH]
        records.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels[:MAX_SEQ_LENGTH],
            }
        )

    print(f"Loaded {len(records)} examples from {data_path}")
    print(f"Truncated examples: {truncated}")
    return records


def build_datasets(tokenizer, data_path: Path):
    from datasets import Dataset

    dataset = Dataset.from_list(build_records(tokenizer, data_path))
    split = dataset.train_test_split(test_size=EVAL_RATIO, seed=RANDOM_STATE, shuffle=True)
    print(f"Train examples: {len(split['train'])}; eval examples: {len(split['test'])}")
    return split["train"], split["test"]


def load_system_prompt_from_data(data_path: Path) -> str:
    first_line = data_path.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(first_line)
    return row["messages"][0]["content"]


def save_prompt_config(adapter_dir: Path, data_path: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "version": "v4",
        "data_path": str(data_path),
        "system_prompt": load_system_prompt_from_data(data_path),
        "max_seq_length": MAX_SEQ_LENGTH,
        "eval_ratio": EVAL_RATIO,
        "random_state": RANDOM_STATE,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
        },
    }
    (adapter_dir / "prompt_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.prepare_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=True)
        build_datasets(tokenizer, DATA_PATH)
        save_prompt_config(Path(OUTPUT_DIR) / "adapter", DATA_PATH)
        print(f"Prepared v4 prompt config at {Path(OUTPUT_DIR) / 'adapter' / 'prompt_config.json'}")
        return

    import unsloth  # noqa: F401
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    use_cuda = torch.cuda.is_available()
    bf16_supported = bool(use_cuda and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16_supported else torch.float16 if use_cuda else torch.float32

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print("Warning: CUDA is not available. Training a 2B model on CPU will be very slow.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=dtype,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=use_cuda,
        local_files_only=True,
        use_exact_model_name=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_a",
            "in_proj_b",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_STATE,
    )

    train_dataset, eval_dataset = build_datasets(tokenizer, DATA_PATH)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(text_tokenizer(tokenizer), label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=6,
            learning_rate=3e-5,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=1,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=10,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_8bit" if use_cuda else "adamw_torch",
            fp16=bool(use_cuda and not bf16_supported),
            bf16=bf16_supported,
            seed=RANDOM_STATE,
            data_seed=RANDOM_STATE,
            report_to="none",
            remove_unused_columns=False,
        ),
    )

    trainer.train()

    adapter_dir = Path(OUTPUT_DIR) / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    save_prompt_config(adapter_dir, DATA_PATH)
    print(f"Saved v4 adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
