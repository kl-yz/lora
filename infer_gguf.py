#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = ROOT / "outputs/qwen35-2b-keyword-lora-v4/gguf/qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_MMPROJ = ROOT / "Qwen3.5-2B_gguf/Qwen3.5-2B.F16-mmproj.gguf"
DEFAULT_CHAT_TEMPLATE = ROOT / "outputs/qwen35-2b-keyword-lora-v4/gguf/chat_template.jinja"
DEFAULT_SYSTEM_PROMPT_FILE = ROOT / "outputs/qwen35-2b-keyword-lora-v4/adapter/prompt_config.json"
DEFAULT_LLAMA_CLI = ROOT / "llama.cpp-local/llama-cli"
DEFAULT_LLAMA_MTMD_CLI = ROOT / "llama.cpp-local/llama-mtmd-cli"

JSON_ARRAY_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
}

FALLBACK_SYSTEM_PROMPT = (
    "你是一名资深动力电池与 BMS 售后诊断工程师。请严格根据用户投诉内容提取关键字。"
    "只提取用户明确描述的故障现象、故障码、报警提示、功能失效、操作异常、状态异常和部件故障。"
    "忽略里程、正常使用情况、停放历史等非故障信息。"
    "如果没有可提取的故障现象，返回空数组：[]。"
    "只输出 JSON array (str)，不要输出任何解释。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run keyword extraction with the exported Qwen3.5 GGUF model."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="用户投诉文本。不传时会从 stdin 读取。",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"GGUF 模型路径，默认：{DEFAULT_MODEL.relative_to(ROOT)}",
    )
    parser.add_argument(
        "--llama_cli",
        default=str(DEFAULT_LLAMA_CLI),
        help=f"llama.cpp 推理程序路径，默认：{DEFAULT_LLAMA_CLI.relative_to(ROOT)}",
    )
    parser.add_argument(
        "--chat_template",
        default=str(DEFAULT_CHAT_TEMPLATE),
        help=f"chat template 路径，默认：{DEFAULT_CHAT_TEMPLATE.relative_to(ROOT)}",
    )
    parser.add_argument(
        "--system_prompt_file",
        default=str(DEFAULT_SYSTEM_PROMPT_FILE),
        help=f"从该文件读取 system prompt，默认：{DEFAULT_SYSTEM_PROMPT_FILE.relative_to(ROOT)}",
    )
    parser.add_argument(
        "--use_mmproj",
        action="store_true",
        help="改用 llama-mtmd-cli 并加载 mmproj。纯文本一般不用开。",
    )
    parser.add_argument(
        "--mtmd_cli",
        default=str(DEFAULT_LLAMA_MTMD_CLI),
        help=f"llama-mtmd-cli 路径，默认：{DEFAULT_LLAMA_MTMD_CLI.relative_to(ROOT)}",
    )
    parser.add_argument(
        "--mmproj",
        default=str(DEFAULT_MMPROJ),
        help=f"mmproj 路径，默认：{DEFAULT_MMPROJ.relative_to(ROOT)}",
    )
    parser.add_argument("--ctx_size", type=int, default=4096, help="上下文长度。")
    parser.add_argument("--n_predict", type=int, default=256, help="最多生成 token 数。")
    parser.add_argument("--threads", type=int, default=0, help="CPU 线程数，0 表示 llama.cpp 自动选择。")
    parser.add_argument("--gpu_layers", default="auto", help="GPU offload 层数，默认 auto。CPU 跑可设为 0。")
    parser.add_argument("--temp", type=float, default=0.2, help="温度。")
    parser.add_argument("--top_k", type=int, default=40, help="top-k。")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p。")
    parser.add_argument("--repeat_last_n", type=int, default=256, help="重复惩罚检查窗口。")
    parser.add_argument("--repeat_penalty", type=float, default=1.12, help="重复惩罚系数。")
    parser.add_argument("--dry_multiplier", type=float, default=0.0, help="DRY 抗重复强度，0 表示关闭。")
    parser.add_argument("--timeout", type=int, default=300, help="推理超时时间（秒）。")
    parser.add_argument(
        "--json_schema",
        action="store_true",
        help="开启 llama.cpp JSON schema 约束。当前本地 llama.cpp 可能会 sampler 初始化失败，默认关闭。",
    )
    parser.add_argument(
        "--no_json_schema",
        dest="json_schema",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="直接打印 llama.cpp 原始输出，不做 JSON 数组提取和校验。",
    )
    parser.add_argument(
        "--print_cmd",
        action="store_true",
        help="只打印将要执行的 llama.cpp 命令，不真正推理。",
    )
    return parser.parse_args()


def read_user_text(parts: list[str]) -> str:
    if parts:
        return " ".join(parts).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("请传入用户投诉文本，或通过 stdin 输入。")


def load_system_prompt(path: Path) -> str:
    if not path.exists():
        return FALLBACK_SYSTEM_PROMPT

    text = path.read_text(encoding="utf-8").strip()
    if path.suffix == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return text or FALLBACK_SYSTEM_PROMPT
        if isinstance(value, dict) and isinstance(value.get("system_prompt"), str):
            return value["system_prompt"]
        return text or FALLBACK_SYSTEM_PROMPT

    if "用户输入：" in text:
        return text.split("用户输入：", 1)[0].strip()
    if "\n1." in text:
        return text.split("\n1.", 1)[0].strip()
    return text or FALLBACK_SYSTEM_PROMPT


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    model = Path(args.model).expanduser().resolve()
    chat_template = Path(args.chat_template).expanduser().resolve()
    llama_cli = Path(args.llama_cli).expanduser().resolve()
    mmproj = None

    if args.use_mmproj:
        llama_cli = Path(args.mtmd_cli).expanduser().resolve()
        mmproj = Path(args.mmproj).expanduser().resolve()
        if not mmproj.exists():
            raise SystemExit(f"找不到 mmproj 文件：{mmproj}")

    if not model.exists():
        raise SystemExit(f"找不到 GGUF 模型：{model}")
    if not llama_cli.exists():
        raise SystemExit(f"找不到 llama.cpp 推理程序：{llama_cli}")
    if not chat_template.exists():
        raise SystemExit(f"找不到 chat template：{chat_template}")

    return model, llama_cli, mmproj


def add_common_sampling_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--ctx-size",
            str(args.ctx_size),
            "--predict",
            str(args.n_predict),
            "--temp",
            str(args.temp),
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--gpu-layers",
            str(args.gpu_layers),
            "--repeat-last-n",
            str(args.repeat_last_n),
            "--repeat-penalty",
            str(args.repeat_penalty),
            "--dry-multiplier",
            str(args.dry_multiplier),
            "--no-warmup",
            "--offline",
            "--log-disable",
        ]
    )
    if args.threads > 0:
        cmd.extend(["--threads", str(args.threads)])
    if args.json_schema:
        cmd.extend(["--json-schema", json.dumps(JSON_ARRAY_SCHEMA, ensure_ascii=False, separators=(",", ":"))])


def build_command(args: argparse.Namespace, user_text: str) -> list[str]:
    model, llama_cli, mmproj = validate_paths(args)
    system_prompt = load_system_prompt(Path(args.system_prompt_file).expanduser())

    if args.use_mmproj:
        cmd = [
            str(llama_cli),
            "-m",
            str(model),
            "--mmproj",
            str(mmproj),
            "--system-prompt",
            system_prompt,
            "--prompt",
            user_text,
        ]
        add_common_sampling_args(cmd, args)
        return cmd

    cmd = [
        str(llama_cli),
        "-m",
        str(model),
        "--chat-template-file",
        str(Path(args.chat_template).expanduser().resolve()),
        "--jinja",
        "--chat-template-kwargs",
        json.dumps({"enable_thinking": False, "add_vision_id": False}, ensure_ascii=False, separators=(",", ":")),
        "--system-prompt",
        system_prompt,
        "--prompt",
        user_text,
        "--single-turn",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--no-display-prompt",
        "--no-show-timings",
        "--simple-io",
    ]

    add_common_sampling_args(cmd, args)

    return cmd


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def find_json_array(text: str, start: int) -> str:
    in_string = False
    escaped = False
    quote = ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue

        if char in {"'", '"'}:
            in_string = True
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("模型输出里的 JSON 数组没有闭合。")


def parse_array(text: str) -> list[str]:
    clean = strip_ansi(text).strip()
    for match in re.finditer(r"\[", clean):
        try:
            candidate = find_json_array(clean, match.start())
        except ValueError:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
            except Exception:
                continue

        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value

    raise ValueError("模型输出里没有找到有效的 JSON string array。")


def run(cmd: list[str], timeout: int) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"推理超时（{timeout} 秒）。可以调大 --timeout，或减小 --n_predict。") from exc

    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        sys.stderr.write(completed.stdout)
        raise SystemExit(f"llama.cpp 推理失败，退出码：{completed.returncode}")

    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    user_text = read_user_text(args.text)
    cmd = build_command(args, user_text)

    if args.print_cmd:
        print(" ".join(json.dumps(part, ensure_ascii=False) for part in cmd))
        return

    output = run(cmd, args.timeout)
    if args.raw:
        print(output)
        return

    try:
        array = parse_array(output)
    except Exception as exc:
        print(output)
        raise SystemExit(f"无法解析为 JSON string array：{exc}") from exc

    print(json.dumps(array, ensure_ascii=False))


if __name__ == "__main__":
    main()
