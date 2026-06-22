# Qwen2.5 本地大模型对话助手

## 项目简介

基于 Qwen2.5-1.5B-Instruct + QLoRA 微调的本地大模型对话应用。无需云端 API，完全在本地 GPU 上运行，支持多轮对话、参数调节。

## 技术栈

| 层级 | 技术 |
|------|------|
| 模型 | Qwen2.5-1.5B-Instruct + QLoRA LoRA 适配器 |
| 后端 | Python Flask |
| 前端 | HTML + CSS + JavaScript（原生，无框架） |
| 推理 | PyTorch + Transformers + PEFT + BitsAndBytes (4-bit量化) |

## 项目结构

```
chat_ui/
├── server.py              # Flask 后端，加载模型并提供 /chat API
├── templates/
│   └── index.html         # Web 对话界面
├── README.md              # 本文件
└── screenshot.py          # Playwright 截图脚本
```

## 功能特性

- **本地模型推理**：完全离线运行，无需 API Key
- **QLoRA 微调权重**：加载自定义微调的 LoRA 适配器
- **多轮对话**：自动维护对话历史，支持上下文理解
- **System Prompt**：可自定义系统提示词
- **参数调节**：Temperature（0.1-1.5）和 Max Tokens（64-1024）滑块
- **快捷提问**：预设示例问题一键发送
- **代码渲染**：回复中的代码块自动语法高亮

## 运行方法

### 环境要求

- Windows 11 + NVIDIA GPU (8GB+ VRAM)
- Conda 环境 `dl-toolkit`（含 PyTorch 2.6.0、Transformers、PEFT、BitsAndBytes、Flask）
- 已下载 Qwen2.5-1.5B-Instruct 模型
- 已完成 QLoRA 微调（LoRA 适配器在 `output/qwen2.5-lora/final_lora/`）

### 启动

```bash
conda activate dl-toolkit
cd chat_ui
python server.py
```

浏览器打开 http://127.0.0.1:5000

### API 接口

**POST /chat**
```json
{
  "messages": [
    {"role": "user", "content": "你好，请介绍一下你自己"}
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

**GET /health** — 返回模型状态和显存占用

## System Prompt 设计

应用使用以下系统提示词引导模型行为：

```
你是一个有用的AI助手，请准确、简洁地回答用户的问题。
```

该提示词通过 ChatML 格式注入每条对话，确保模型遵循指令风格。

## 上下文管理

- 前端维护完整对话历史（`messages` 数组）
- 每次请求将全部历史发送给后端
- 后端在服务端拼接 System Prompt
- 模型基于完整上下文生成回复
- "清空对话"按钮重置上下文

## 截图

见 `../output/chat_ui_response1.png` 等文件。
