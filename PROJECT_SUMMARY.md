# 项目总结

## 1. 项目目标

这个项目的目标，是把本地的 `Qwen3.5-2B` 微调成一个“车辆投诉故障现象抽取模型”。

目标行为是：

- 输入：一段中文用户投诉文本
- 输出：一个 JSON 字符串数组，例如 `["无法启动", "仪表黑屏", "无法挂挡"]`
- 约束：只提取用户明确描述的故障现象，不补充推断，不输出解释

当前项目最终被拆成了 3 条独立链路：

- `traing_lanauge.py`：训练
- `infer.py`：Python 推理
- `export_gguf.py`：导出 GGUF 给 `llama.cpp` / 类似工具使用


## 2. 项目演进过程

### 阶段 1：原始数据来自 `data.txt`

项目一开始使用的是手工整理的 `data.txt`。

这个文件里包含：

- 一段很长的 system 指令
- 按编号排列的样例
- 每条样例对应一段用户投诉
- 每条样例对应一个目标 JSON 数组

这种格式对人来说比较直观，但它不是标准微调格式。

它的问题在于：

- 需要额外写自定义解析逻辑
- 编号样例格式比较脆弱
- 指令文本、备注文本、训练内容混在一起
- 后续如果换框架，不方便复用


### 阶段 2：转换成标准 chat JSONL

为了让训练更标准、更稳定，后面把数据统一转换成了 `train_messages.jsonl`。

现在每一行是一个标准样本，结构类似：

```json
{"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"..."},
  {"role":"assistant","content":"[\"故障1\", \"故障2\"]"}
]}
```

这样做的好处是：

- 数据格式标准化
- 训练输入结构明确
- assistant 输出严格保持为 JSON
- system prompt 可以同时复用于训练和推理


## 3. 遇到过的主要问题

### 问题 A：本地模型下载与离线加载

最开始训练脚本默认尝试从 Hugging Face 在线拉模型。

当时遇到的典型报错有：

- `Network is unreachable`
- 缺少 `config.json`
- 本地模型目录和线上模型名混淆

解决方式：

- 把基座模型下载到本地 `./Qwen3.5-2B`
- 后续所有训练、推理、导出脚本都改成优先使用本地路径

现在统一使用：

```text
./Qwen3.5-2B
```


### 问题 B：`data.txt` 不是标准训练格式

虽然 `data.txt` 能被解析，但它并不是“标准微调数据格式”。

解决方式：

- 生成标准格式的 `train_messages.jsonl`
- 训练脚本改成直接读取 `train_messages.jsonl`
- `data.txt` 保留为原始可编辑样例来源


### 问题 C：模型学会了 system prompt，而不是学会答案

中间一度出现了非常典型的失败现象：模型输出的不是故障数组，而是 system prompt 的碎片，比如：

- `全文逐显告告`
- `停车联系服务店`
- 以及其他规则文本残片

根因：

- 一开始训练时，模型是在学整段序列
- system 和 user token 也参与了 loss
- 样本只有 20 条，特别容易过拟合

解决方式：

- 构造训练数据时，把 system/user 部分的 label 全部置为 `-100`
- 只让 assistant 的 JSON 输出参与 loss
- 让模型只学习“答案”，不学习长提示词本身

当前训练逻辑是：

- prompt token：`-100`
- assistant token：正常训练


### 问题 D：Qwen3.5 的 tokenizer / processor 混淆

这个 `Qwen3.5-2B` 实际上更接近多模态风格的 checkpoint，所以 Unsloth 返回的对象并不总是纯文本 tokenizer，更像一个 processor。

出现过的典型报错：

- 文本 prompt 被错误传入图像处理逻辑
- 最终报 `Incorrect image source`

解决方式：

- 增加了一个辅助函数：

```python
def text_tokenizer(processor_or_tokenizer):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
```

- 后续所有纯文本 tokenization 都只走内部真正的 text tokenizer


### 问题 E：Unsloth 精度配置冲突

训练过程中出现过多次精度相关冲突。

典型报错：

- `Can only load in 4bit or 8bit or 16bit, not a combination`
- bf16 / fp16 不匹配

解决方式：

- 显式关闭 4bit 和 8bit
- 显式使用 16bit 加载
- 训练参数不再依赖默认自动推断

后面项目收敛到更保守的配置：

- `dtype=torch.float16`
- `load_in_16bit=True`
- `bf16=False`
- CUDA 环境下训练使用 `fp16=True`


### 问题 F：Qwen3.5 的 linear attention CUDA 不稳定

本地这个 Qwen3.5 模型在结构上用了大量 `linear_attention`。

出现过的典型报错：

- `torch.AcceleratorError: CUDA error: unknown error`
- 报错点在 `torch_chunk_gated_delta_rule`

可能原因：

- 缺少 fast path 依赖
- 于是回退到 PyTorch fallback 实现
- 当前 CUDA / Torch / Unsloth 组合下，这条 fallback 路径不稳定

缓解方式：

- 固定为更保守的 float16 路线
- 把 `lora_dropout` 调成 `0`
- 关闭 compile 相关优化

这也是为什么训练脚本后来被不断简化，优先选择“能稳定跑完”的方案。


### 问题 G：`SFTTrainer` checkpoint 保存时 pickle 失败

后面训练不再死在前向过程，而是死在 checkpoint 保存阶段。

典型报错：

- `PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>`

根因：

- `SFTTrainer`
- Unsloth 编译后的 trainer
- `trl` 的 `SFTConfig`

这三者在保存 checkpoint 元数据时发生了兼容性问题。

解决方式：

- 训练脚本改成更朴素的 `transformers.Trainer`
- 中途 checkpoint 保存关闭

```python
save_strategy="no"
```

- 最终 adapter 在训练结束后手工保存


### 问题 H：WSL 网络与代理问题

在导出 GGUF 时，Unsloth 仍然会访问 GitHub，因此网络问题再次暴露出来。

当时的现象是：

- `github.com` 超时
- `huggingface.co` 超时
- `curl` 访问百度正常
- 用 `127.0.0.1:7897` 走代理失败

根因：

- WSL 不会自动继承 Windows 代理
- WSL 内的 `127.0.0.1` 不是 Windows 主机代理地址

解决方式：

- 通过下面命令找出 Windows 主机在 WSL 里的地址：

```bash
ip route | grep default
```

- 得到主机地址：

```text
172.26.0.1
```

- 在 WSL 中配置代理：

```bash
export http_proxy=http://172.26.0.1:7897
export https_proxy=http://172.26.0.1:7897
export HTTP_PROXY=http://172.26.0.1:7897
export HTTPS_PROXY=http://172.26.0.1:7897
```

之后：

- `github.com` 可访问
- `huggingface.co` 可访问


## 4. 当前项目架构

### 4.1 训练

文件：

- [traing_lanauge.py](/home/project/Lora/traing_lanauge.py:1)

作用：

- 读取 chat JSONL 数据
- 构造只训练 assistant 的 masked labels
- 加载本地基座模型
- 挂载 LoRA
- 使用 `transformers.Trainer` 做训练
- 保存最终 adapter

关键设计：

- 标准数据源：`train_messages.jsonl`
- 只让 assistant 输出参与 loss
- 不保存中途 checkpoint
- 使用更保守的 fp16 路线

关键常量：

- `DATA_PATH = Path("train_messages.jsonl")`
- `BASE_MODEL = "./Qwen3.5-2B"`
- `OUTPUT_DIR = "outputs/qwen35-2b-keyword-lora-v2"`


### 4.2 推理

文件：

- [infer.py](/home/project/Lora/infer.py:1)

作用：

- 加载基座模型 + LoRA adapter
- 从 `train_messages.jsonl` 第一行读取 system prompt
- 接收一条命令行输入的投诉文本
- 生成模型输出

当前推理路径：

```text
base model + adapter -> apply_chat_template -> generate -> print result
```

用法：

```bash
python /home/project/Lora/infer.py "用户投诉：车辆停放一晚后无法启动，仪表黑屏，无法挂挡。"
```


### 4.3 GGUF 导出

文件：

- [export_gguf.py](/home/project/Lora/export_gguf.py:1)

作用：

- 把导出逻辑和训练逻辑分开
- 加载基座模型 + adapter
- patch Unsloth，让它使用本地 `llama.cpp` 工具
- 导出量化后的 GGUF 文件

关键本地依赖：

- `llama.cpp-local/llama-quantize`
- `llama.cpp-local/convert_hf_to_gguf.py`

当前目标输出：

- `outputs/qwen35-2b-keyword-lora-v2/gguf/qwen3.5-2B-Q4_K_M.gguf`


## 5. 当前目录结构

核心文件：

- [data.txt](/home/project/Lora/data.txt:1)：原始人工可读样例
- [train_messages.jsonl](/home/project/Lora/train_messages.jsonl:1)：标准训练集
- [traing_lanauge.py](/home/project/Lora/traing_lanauge.py:1)：训练脚本
- [infer.py](/home/project/Lora/infer.py:1)：adapter 推理脚本
- [export_gguf.py](/home/project/Lora/export_gguf.py:1)：GGUF 导出脚本

模型资产：

- `Qwen3.5-2B/`：本地基座模型
- `outputs/qwen35-2b-keyword-lora-v2/adapter/`：当前 LoRA adapter
- `outputs/qwen35-2b-keyword-lora-v2/gguf/`：导出的 GGUF 文件
- `llama.cpp-local/`：本地 GGUF 转换工具链


## 6. 当前已经能工作的部分

目前已经可用的部分：

- 本地基座模型加载
- 标准 JSONL 训练数据
- assistant-only loss masking
- Python 方式的 base model + LoRA adapter 推理
- WSL 通过代理访问 GitHub / Hugging Face
- 独立的 GGUF 导出链路

当前已经存在的可用产物：

- `outputs/qwen35-2b-keyword-lora-v2/adapter`
- `outputs/qwen35-2b-keyword-lora-v2/gguf/qwen3.5-2B-Q4_K_M.gguf`


## 7. 当前仍然存在的风险

这个项目已经可用，但仍然有几个现实限制：

- 数据集太小，当前只有 20 条
- 这个 Qwen3.5 不是最简单的纯文本模型
- 训练稳定性依赖当前 CUDA / Unsloth / Transformers 版本组合
- GGUF 导出仍然可能受网络或上游 Unsloth 行为影响

如果你想继续提高效果，最重要的下一步不是继续改代码，而是扩充和清洗数据。


## 8. 推荐的后续步骤

1. 在 `data.txt` 里继续增加更多样例
2. 重新生成 `train_messages.jsonl`
3. 重新训练 adapter
4. 用 `infer.py` 在未见过的新投诉上做验证
5. 确认效果满意后，再刷新 GGUF 导出


## 9. 常用命令

训练：

```bash
python /home/project/Lora/traing_lanauge.py
```

推理：

```bash
python /home/project/Lora/infer.py "用户投诉：车辆停放一晚后无法启动，仪表黑屏，无法挂挡。"
```

导出 GGUF：

```bash
python /home/project/Lora/export_gguf.py
```

WSL 代理：

```bash
export http_proxy=http://172.26.0.1:7897
export https_proxy=http://172.26.0.1:7897
export HTTP_PROXY=http://172.26.0.1:7897
export HTTPS_PROXY=http://172.26.0.1:7897
```
