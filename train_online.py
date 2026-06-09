#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_FUSED_CE_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"


DATA_PATH = Path("train_messages.jsonl")
BASE_MODEL = "unsloth/Qwen3.5-2B"
OUTPUT_DIR = "outputs/qwen35-2b-keyword-lora-online"
MAX_SEQ_LENGTH = 2048


def read_data():
    rows = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            messages = row["messages"]
            answer = messages[-1]["content"]
            json.loads(answer)
            rows.append(messages)
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


def build_dataset(tokenizer):
    from datasets import Dataset

    rows = read_data()
    records = []

    for messages in rows:
        prompt_messages = messages[:-1]
        full_messages = messages

        prompt_ids = encode_text(tokenizer, chat(tokenizer, prompt_messages, add_generation_prompt=True))
        full_ids = encode_text(tokenizer, chat(tokenizer, full_messages, add_generation_prompt=False))

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        records.append(
            {
                "input_ids": full_ids[:MAX_SEQ_LENGTH],
                "attention_mask": [1] * min(len(full_ids), MAX_SEQ_LENGTH),
                "labels": labels[:MAX_SEQ_LENGTH],
            }
        )

    print(f"Loaded {len(records)} examples")
    return Dataset.from_list(records)


def main():
    import unsloth  # noqa: F401
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        local_files_only=False,
        use_exact_model_name=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
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
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    use_cuda = torch.cuda.is_available()
    trainer = Trainer(
        model=model,
        train_dataset=build_dataset(tokenizer),
        data_collator=DataCollatorForSeq2Seq(text_tokenizer(tokenizer), label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=5e-5,
            warmup_steps=2,
            logging_steps=1,
            save_strategy="no",
            optim="adamw_8bit",
            fp16=use_cuda,
            bf16=False,
            report_to="none",
            remove_unused_columns=False,
        ),
    )

    trainer.train()
    model.save_pretrained(f"{OUTPUT_DIR}/adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")
    print(f"Saved adapter to {OUTPUT_DIR}/adapter")


if __name__ == "__main__":
    main()
