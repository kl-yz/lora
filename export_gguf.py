#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


BASE_MODEL = Path("./Qwen3.5-2B")
ADAPTER_PATH = Path("outputs/qwen35-2b-keyword-lora-v4/adapter")
GGUF_OUTPUT_DIR = Path("outputs/qwen35-2b-keyword-lora-v4/gguf")
MERGED_MODEL_DIR = Path("outputs/qwen35-2b-keyword-lora-v4/merged_hf")
LLAMA_CPP_DIR = Path("llama.cpp-local")
GGUF_FILENAME = "qwen3.5-2B-Q4_K_M.gguf"
CHAT_TEMPLATE_FILENAME = "chat_template.jinja"
F16_FILENAME = "qwen3.5-2B-F16.gguf"
MAX_SEQ_LENGTH = 1024
QUANTIZATION = "Q4_K_M"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def merge_adapter() -> tuple[object, object]:
    import unsloth  # noqa: F401
    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(BASE_MODEL),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        local_files_only=True,
        use_exact_model_name=True,
    )

    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    if not hasattr(model, "peft_config") or not model.peft_config:
        raise RuntimeError(f"Adapter load failed: {ADAPTER_PATH}")

    merged_model = model.merge_and_unload()
    return merged_model, tokenizer


def save_merged_model() -> None:
    ensure_exists(BASE_MODEL, "base model directory")
    ensure_exists(ADAPTER_PATH, "adapter directory")

    merged_model, tokenizer = merge_adapter()
    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged HF model to {MERGED_MODEL_DIR}")
    merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)

    chat_template_src = BASE_MODEL / CHAT_TEMPLATE_FILENAME
    if chat_template_src.exists():
        shutil.copy2(chat_template_src, MERGED_MODEL_DIR / CHAT_TEMPLATE_FILENAME)


def convert_to_f16_gguf() -> Path:
    converter = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    ensure_exists(converter, "GGUF converter")
    ensure_exists(MERGED_MODEL_DIR, "merged HF model directory")

    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    f16_path = GGUF_OUTPUT_DIR / F16_FILENAME

    run(
        [
            "python",
            str(converter),
            str(MERGED_MODEL_DIR),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
            "--no-mtp",
        ]
    )
    return f16_path


def quantize_gguf(f16_path: Path) -> Path:
    quantizer = LLAMA_CPP_DIR / "llama-quantize"
    ensure_exists(quantizer, "GGUF quantizer")
    ensure_exists(f16_path, "F16 GGUF file")

    final_path = GGUF_OUTPUT_DIR / GGUF_FILENAME
    run([str(quantizer), str(f16_path), str(final_path), QUANTIZATION])
    return final_path


def copy_chat_template() -> None:
    chat_template_src = BASE_MODEL / CHAT_TEMPLATE_FILENAME
    chat_template_dst = GGUF_OUTPUT_DIR / CHAT_TEMPLATE_FILENAME
    if chat_template_src.exists():
        shutil.copy2(chat_template_src, chat_template_dst)
        print(f"Copied chat template to {chat_template_dst}")


def main() -> None:
    save_merged_model()
    f16_path = convert_to_f16_gguf()
    final_path = quantize_gguf(f16_path)
    copy_chat_template()
    print("Export completed.")
    print(f"Merged HF model: {MERGED_MODEL_DIR}")
    print(f"F16 GGUF: {f16_path}")
    print(f"Quantized GGUF: {final_path}")


if __name__ == "__main__":
    main()
