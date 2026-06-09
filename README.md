# Qwen3.5-2B LoRA 微调项目说明

这个项目的目标是：基于本地 `Qwen3.5-2B` 模型做 LoRA 微调，让模型能够从用户的车辆投诉文本中提取故障关键词，并输出一个标准的 JSON 字符串数组。

现在这个仓库已经拆成了三条清晰的链路：

- `traing_lanauge.py`：训练 LoRA adapter
- `infer.py`：用基座模型 + adapter 做原生推理
- `infer_gguf.py` / `export_gguf.py`：导出并使用 GGUF 模型

如果你只是想先跑通项目，建议顺序是：

1. 准备和检查训练数据
2. 跑 `traing_lanauge.py` 完成微调
3. 跑 `infer.py` 验证微调结果
4. 满意后再跑 `export_gguf.py`
5. 最后用 `infer_gguf.py` 或 `llama.cpp` 推理

## 一、项目目录树

下面这棵树不是把所有缓存文件都展开，而是把你真正需要理解和维护的部分列出来：

```text
Lora/
├── README.md
├── PROJECT_SUMMARY.md
├── requirements.txt
├── data.txt
├── train_messages.jsonl
├── traing_lanauge.py
├── infer.py
├── export_gguf.py
├── infer_gguf.py
├── Qwen3.5-2B/
├── Qwen3.5-2B_gguf/
├── llama.cpp-local/
├── outputs/
│   ├── preview/
│   │   └── train_messages.jsonl
│   ├── qwen35-2b-keyword-lora/
│   └── qwen35-2b-keyword-lora-v2/
│       ├── adapter/
│       ├── checkpoint-3/
│       └── gguf/
└── unsloth_compiled_cache/
```

## 二、根目录文件说明

### `README.md`

就是你现在看到的这个文件。它负责解释整个项目怎么使用、怎么理解目录结构、关键脚本分别是干什么的。

### `PROJECT_SUMMARY.md`

这是项目过程总结文档。它更像“项目复盘”和“问题记录”，重点写的是：

- 项目从怎么开始
- 中途踩过哪些坑
- 每个问题的根因是什么
- 我们最后怎么修好

如果你以后忘了为什么当时那样改，这个文件最有用。

### `requirements.txt`

Python 依赖列表。通常用于安装训练、推理、导出所需的包，比如：

- `torch`
- `transformers`
- `peft`
- `datasets`
- `trl`
- `unsloth`

它负责定义这个项目的 Python 运行环境。

### `data.txt`

这是最早的人类可读原始样例文件。它更像“草稿数据”或“原始标注文本”，方便人工编辑和查看。

它的特点是：

- 可读性强
- 适合手工补样本
- 不一定是标准训练格式

训练本身现在不直接依赖它，而是依赖整理好的 JSONL 文件。

### `train_messages.jsonl`

这是当前真正用于训练的标准数据格式文件，也是最重要的数据文件之一。

格式是每行一个 JSON，例如：

```json
{"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"..."},
  {"role":"assistant","content":"[...]"}
]}
```

这个格式的好处是：

- 跟 chat 模型训练格式一致
- 结构稳定
- 适合直接套 tokenizer 的 chat template

现在训练脚本默认就是读取它。

### `traing_lanauge.py`

这是当前项目的训练脚本，负责把本地 `Qwen3.5-2B` 微调成你的故障关键词提取模型。

它现在的职责比较单纯：

- 读取 `train_messages.jsonl`
- 构造 chat prompt
- 只让 assistant 的 JSON 答案参与 loss
- 把 system / user token 全部 mask 为 `-100`
- 加载本地基座模型
- 挂 LoRA
- 开始训练
- 保存最终 adapter

这个脚本之所以重要，是因为我们前面已经把很多不稳定因素剥掉了，比如：

- 不再把训练、导出、测试混在一个大脚本里
- 不再依赖复杂命令行参数
- 不再让 system prompt 被一起训练

你可以把它理解成“专门负责训练”的入口。

这个文件现在是“本地模型版”，默认使用：

```python
BASE_MODEL = "./Qwen3.5-2B"
```

也就是要求你已经把基座模型下载到项目目录里。

### `train_online.py`

这是新增的“在线下载模型版”训练脚本。

它和 `traing_lanauge.py` 的训练逻辑基本一致，但区别在于：

- `traing_lanauge.py`：从本地目录加载基座模型
- `train_online.py`：从 Hugging Face 在线拉取基座模型

它默认使用：

```python
BASE_MODEL = "unsloth/Qwen3.5-2B"
```

并且：

```python
local_files_only = False
```

所以前提是你的当前环境能够访问 Hugging Face。

### `infer.py`

这是原生推理脚本，用于直接加载：

- 基座模型 `Qwen3.5-2B`
- LoRA adapter `outputs/qwen35-2b-keyword-lora-v2/adapter`

然后做一次文本推理。

它的作用是验证微调是否有效。一般建议你在导出 GGUF 之前，先用这个脚本检查模型输出是否正常。

也就是说，它是“训练结果验收脚本”。

### `export_gguf.py`

这是 GGUF 导出脚本，专门负责把：

- 基座模型
- LoRA adapter

转换成可以被 `llama.cpp` 使用的 GGUF 文件。

它现在还负责一件很重要的事：把导出的辅助文件一起整理好，比如：

- `chat_template.jinja`
- 最终统一命名后的 `qwen3.5-2B-Q4_K_M.gguf`

这个脚本是“部署转换脚本”。

### `infer_gguf.py`

这是 GGUF 推理脚本，专门给导出的 GGUF 模型用。

它会调用本地 `llama.cpp-local/llama-cli`，并自动帮你处理：

- GGUF 模型路径
- chat template
- system prompt
- 采样参数
- 输出清洗

你可以把它理解成“最终部署版推理入口”。

## 三、模型目录说明

### `Qwen3.5-2B/`

这是本地基座模型目录，也就是训练时真正加载的基础模型。

这个目录通常包含：

- `config.json`：模型结构配置
- `tokenizer.json` / `tokenizer_config.json`：分词器配置
- `chat_template.jinja`：聊天模板
- `model.safetensors*`：模型权重
- `processor_config.json` / `preprocessor_config.json`：处理器配置

这个目录非常关键，因为：

- 训练时要从这里加载基座模型
- 推理时也要从这里加载 tokenizer / processor
- 导出 GGUF 时也要参考这里的模板和配置

### `Qwen3.5-2B_gguf/`

这是一次单独导出的 GGUF 目录，更像“原始导出产物目录”。

里面现在有：

- `Qwen3.5-2B.Q4_K_M.gguf`
- `Qwen3.5-2B.F16-mmproj.gguf`

其中：

- `Q4_K_M.gguf` 是主模型
- `F16-mmproj.gguf` 是多模态投影文件，主要给图片输入用

如果你只做文本关键词提取，一般主要关注主模型即可。

## 四、推理工具目录说明

### `llama.cpp-local/`

这是你本地准备好的 `llama.cpp` 工具链目录，用来完成 GGUF 推理和量化转换。

你目前最常用的文件是：

- `llama-cli`：纯文本 GGUF 推理
- `llama-mtmd-cli`：多模态推理
- `llama-quantize`：量化工具
- `convert_hf_to_gguf.py`：HF 模型转 GGUF 脚本

这个目录相当于“本地部署工具箱”。

你平时不用去改它的源码，但会经常用到它的可执行文件。

## 五、输出目录说明

### `outputs/preview/`

这里放的是预处理或预览阶段的中间文件。

当前最典型的是：

- `outputs/preview/train_messages.jsonl`

它通常用于检查原始样本被转换成标准训练格式后的样子。

### `outputs/qwen35-2b-keyword-lora/`

这是较早的一版训练输出目录，可以看成旧实验结果。

里面有：

- `adapter/`
- `checkpoint-24/`
- `gguf/`

这部分不是现在的主线产物，但保留下来可以帮助你回溯之前的问题。

### `outputs/qwen35-2b-keyword-lora-v2/`

这是当前主要使用的新一版训练输出目录，也是你现在最该关注的结果目录。

它下面主要有三个子目录：

#### `outputs/qwen35-2b-keyword-lora-v2/adapter/`

这是最终可用的 LoRA adapter 目录。

里面常见的文件有：

- `adapter_model.safetensors`：LoRA 权重
- `adapter_config.json`：LoRA 配置
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `processor_config.json`

这个目录的作用是：

- 给 `infer.py` 直接加载
- 作为 GGUF 导出的输入之一

如果你问“我微调后的模型到底在哪”，对于 LoRA 版本来说，答案基本就是这个目录。

#### `outputs/qwen35-2b-keyword-lora-v2/checkpoint-3/`

这是训练过程中的中间 checkpoint。

它的作用主要是：

- 保留训练中途的一份权重
- 出问题时用于排查
- 某些情况下可以拿来恢复

但从使用角度说，它不是你最终优先使用的目录。一般优先用 `adapter/`。

#### `outputs/qwen35-2b-keyword-lora-v2/gguf/`

这是当前整理好的 GGUF 最终目录。

里面现在有：

- `qwen3.5-2B-Q4_K_M.gguf`
- `chat_template.jinja`

这个目录就是给 `infer_gguf.py` 和 `llama.cpp` 用的。  
如果你想把模型拿去做轻量部署，本质上主要就是用这里的内容。

## 六、缓存目录说明

### `unsloth_compiled_cache/`

这是 Unsloth 自动生成的编译缓存目录。

它里面会出现大量：

- 编译后的 Python 模块
- trainer 包装代码
- CUDA / Triton 相关缓存

这个目录通常不是手工维护对象。你可以把它理解成运行时生成的“加速缓存”。

平时知道它存在就行，除非遇到很奇怪的缓存兼容问题，否则不需要手动改里面的文件。

## 七、当前项目的主流程

如果按实际使用顺序来看，这个项目现在的结构可以理解成下面这条链：

### 1. 原始数据

`data.txt`

用于人工整理和维护样本。

### 2. 标准训练数据

`train_messages.jsonl`

用于正式喂给模型训练。

### 3. 微调训练

`traing_lanauge.py`

输出：

- `outputs/qwen35-2b-keyword-lora-v2/adapter/`

### 4. 原生推理验证

`infer.py`

用于验证 adapter 是否真的学到了任务。

### 5. GGUF 导出

`export_gguf.py`

输出：

- `outputs/qwen35-2b-keyword-lora-v2/gguf/qwen3.5-2B-Q4_K_M.gguf`

### 6. GGUF 推理

`infer_gguf.py`

或直接调用 `llama.cpp-local/llama-cli`

## 八、你现在最需要关心的文件

虽然目录看起来很多，但你平时真正最需要关心的，其实就这几个：

- `train_messages.jsonl`：训练数据
- `traing_lanauge.py`：训练脚本
- `train_online.py`：在线下载基座模型的训练脚本
- `infer.py`：原生推理验证
- `export_gguf.py`：导出 GGUF
- `infer_gguf.py`：GGUF 推理
- `outputs/qwen35-2b-keyword-lora-v2/adapter/`：LoRA 成果
- `outputs/qwen35-2b-keyword-lora-v2/gguf/`：GGUF 成果

其他目录大多属于：

- 基座模型文件
- 工具链
- 历史实验结果
- 自动缓存

## 九、常用命令

### 训练

```bash
python /home/project/Lora/traing_lanauge.py
```

### 在线下载 base model 后训练

```bash
python /home/project/Lora/train_online.py
```

### 原生推理

```bash
python /home/project/Lora/infer.py "用户投诉：车辆停放一晚后无法启动，仪表黑屏，无法挂挡。"
```

### 导出 GGUF
/home/project/Lora/llama.cpp-local/llama-cli \
  -m /home/project/Lora/outputs/qwen35-2b-keyword-lora-v3/gguf/new_qwen3.5-2B-Q4_K_M.gguf \
  --prompt "你好，请用一句话介绍你自己。" \
  --single-turn \
  --reasoning off \
  --reasoning-budget 0 \
  --ctx-size 4096 \
  --predict 256 \
  --temp 0.7 \
  --top-p 0.9 \
  --gpu-layers 0 \
  --no-display-prompt \
  --no-show-timings \
  --simple-io \
  --offline \
  --log-disable

```bash
python /home/project/Lora/export_gguf.py
```

### GGUF 推理

```bash
python /home/project/Lora/infer_gguf.py "用户投诉：车辆停放一晚后无法启动，仪表黑屏，无法挂挡。" --gpu_layers 0
```

## 十、补充说明

这个项目之所以看起来文件不少，是因为它实际上同时覆盖了三类工作：

- 模型训练
- 模型验证
- 部署格式转换

如果只是做实验，你主要看训练和原生推理。  
如果是准备部署，再去看 GGUF 那一支。

后面如果你愿意，我们还可以继续把 README 再压缩出一版“极简上手版”，专门写成三部分：

- 训练怎么跑
- 推理怎么跑
- GGUF 怎么跑

这样你以后回来看会更省心。
# lora
