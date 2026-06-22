# Qwen2.5-1.5B QLoRA 微调实践

基于 Qwen2.5-1.5B-Instruct 大语言模型的 QLoRA 微调项目，包含完整的训练→推理→可视化→Chat UI 流水线。在 8GB 显存的消费级 GPU 上，通过 4-bit 量化 + LoRA 低秩适配，仅训练 0.59% 参数即可完成指令微调。

## 技术栈

| 层级 | 技术 |
|------|------|
| 基座模型 | Qwen2.5-1.5B-Instruct (1.54B params, 32K context) |
| 微调方法 | QLoRA (4-bit NF4 量化 + LoRA r=8 α=16) |
| 训练框架 | PyTorch 自定义训练循环（避免 Windows 兼容性问题） |
| 推理加速 | BitsAndBytes 4-bit, Flash Attention |
| 可视化 | Matplotlib（支持中文） |
| 对话界面 | Flask + 原生 HTML/CSS/JS |
| 模型下载 | ModelScope（国内镜像）+ HuggingFace 回退 |

## 项目结构

```
qlora-finetune/
├── scripts/
│   └── download_model.py       # 模型下载（ModelScope/HuggingFace）
├── training/
│   ├── finetune_qwen.py        # v1: TRL SFTTrainer 方案
│   ├── finetune_final.py       # v2: 绕过 datasets 库的简化版
│   └── finetune_qwen25.py      # v3: 纯 PyTorch 自定义训练循环 ★
├── inference/
│   ├── inference_test.py       # 基础推理对比（微调前 vs 后）
│   └── inference_compare.py    # 详细对比 + 困惑度分析
├── visualization/
│   └── visualize.py            # 训练曲线、困惑度等可视化
├── chat_ui/
│   ├── server.py               # Flask 对话服务端
│   ├── screenshot.py           # Playwright 自动截图
│   └── templates/
│       └── index.html           # 对话 Web 界面
├── output/                     # 训练产物（图表、截图、JSON）
│   ├── training_curves.png
│   ├── perplexity_analysis.png
│   ├── inference_comparison.png
│   ├── training_history.json
│   └── ...
└── requirements.txt
```

## 快速开始

### 1. 环境准备

```bash
# 创建 Conda 环境
conda create -n llm-finetune python=3.10 -y
conda activate llm-finetune

# 安装 PyTorch（根据你的 CUDA 版本选择）
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# Playwright 浏览器（截图需要）
playwright install chromium
```

### 2. 下载模型

```bash
python scripts/download_model.py
```

模型将下载到 `models/Qwen2.5-1.5B-Instruct/`，约 2.9GB。

### 3. 微调训练

```bash
# 推荐：v3 自定义训练循环（Windows/Linux 通用）
python training/finetune_qwen25.py
```

训练参数（在脚本中配置）：
- LoRA rank: 8, alpha: 16
- 学习率: 2e-4，3 epochs
- Batch size: 2，梯度累积: 4 步
- 混合精度: FP16，最大长度: 256 tokens

在 RTX 4060 Laptop (8GB) 上约 30 秒完成。

### 4. 推理对比

```bash
python inference/inference_compare.py
```

对比基座模型与微调模型在代码生成、知识问答、逻辑推理等 5 类任务上的表现，输出对比图表和困惑度分析。

### 5. 启动对话界面

```bash
python chat_ui/server.py
```

浏览器打开 http://127.0.0.1:5000，支持多轮对话、Temperature/Max Tokens 调节、System Prompt 自定义。

## 版本迭代

本项目经历了 3 个版本的训练脚本迭代，记录了在 Windows 环境下进行 LLM 微调时遇到的问题和解决过程：

### v1 — TRL SFTTrainer（`finetune_qwen.py`）

使用 HuggingFace TRL 库的 SFTTrainer 进行训练，这是最"标准"的做法。

**遇到的问题**：在 Windows 上出现段错误（segfault），原因在于 `datasets` 库的多进程数据加载与 Windows 的 spawn 机制冲突。

### v2 — 绕过 datasets 库（`finetune_final.py`）

自定义 `SimpleDataset` 类替代 HuggingFace datasets，但仍使用 SFTTrainer 的训练循环。

**遇到的问题**：SFTTrainer 内部仍依赖 DataCollator 等组件，在某些 Windows 配置下仍有稳定性问题。

### v3 — 纯 PyTorch 自定义训练循环（`finetune_qwen25.py`）★

完全放弃 TRL/SFTTrainer，使用 PyTorch 原生 `Dataset` + `DataLoader` + 手动训练循环。

**优势**：
- 完全避免所有已知的 Windows 兼容性问题
- 对训练过程有完全的控制（梯度累积、混合精度、checkpoint 保存）
- 代码简洁（245 行），易于理解和修改
- Windows/Linux 通用

## 数据集

使用手工构建的 12 条中文指令数据，覆盖以下类别：

| 类别 | 示例 |
|------|------|
| 代码生成 | Fibonacci 数列、二分查找 |
| 知识问答 | 深度学习框架对比、迁移学习 |
| 生活建议 | 烹饪技巧、健康饮食 |
| 算法题解 | 排序算法、梯度下降 |
| 技术解释 | QLoRA 原理、Attention 机制 |

数据格式为 Qwen2.5 的 ChatML 模板（`<|im_start|>` / `<|im_end|>`）。

## 训练结果

| 指标 | 初始 | Epoch 1 | Epoch 2 | Epoch 3 |
|------|------|---------|---------|---------|
| Train Loss | 6.14 | 5.27 | 3.78 | 3.10 |
| Eval Loss | 6.74 | 5.51 | 5.02 | 4.49 |

- 可训练参数: 9.2M / 1.55B (0.59%)
- 显存占用: ~1.2GB（4-bit 量化后）
- 训练时间: ~30 秒（RTX 4060 Laptop 8GB）

## 硬件要求

- NVIDIA GPU: 8GB+ VRAM（4-bit 量化）
- 如使用 FP16 全精度: 建议 6GB+ VRAM
- 纯 CPU 推理: 可行但较慢（~8GB RAM）

## License

本项目代码采用 MIT License。Qwen2.5 模型权重遵循其原始协议（Apache 2.0）。
