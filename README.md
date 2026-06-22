# zlx-llm-portfolio

大语言模型（LLM）技术学习与实战项目集，涵盖模型微调、量化、RAG、Agent 等方向。

## 项目列表

### [QLoRA 微调实践](./qlora-finetune/)

Qwen2.5-1.5B-Instruct 大语言模型的 QLoRA 微调项目。包含完整的训练→推理→可视化→Chat UI 流水线，在 8GB 消费级 GPU 上通过 4-bit 量化 + LoRA 低秩适配完成指令微调。

**技术栈**: PyTorch, Transformers, PEFT, BitsAndBytes, Flask, Matplotlib

### [本地 RAG 知识库问答](./rag-knowledge-qa/)

基于 LangChain 的本地文档检索增强生成（RAG）系统。支持多格式文档加载、向量化存储、语义检索和 LLM 问答。

**技术栈**: LangChain, ChromaDB, HuggingFace Embeddings, Flask

## 技术路线

```
模型微调 ──→ QLoRA/PEFT 高效微调
模型量化 ──→ 4-bit/8-bit 量化推理
RAG ──────→ 文档检索增强生成
Agent ────→ 工具调用与自主决策（规划中）
```

## 环境

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU (8GB+ VRAM recommended)
