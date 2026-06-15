# 本地 RAG 知识库问答

基于 **LangChain + Ollama + Chroma** 的本地知识库问答系统，全离线运行，无需任何外部 API。

## 技术栈

| 层 | 技术 | 用途 |
|---|------|------|
| LLM | Ollama (qwen2.5 / llama3 等) | 本地大模型，生成回答 |
| Embedding | sentence-transformers (bge-large-zh-v1.5) | 本地向量化，中英文支持 |
| 向量库 | Chroma | 本地持久化存储和相似度检索 |
| 框架 | LangChain | 链组装，统一接口 |
| API | FastAPI（进阶） | Web 服务化 |

## 项目结构

```
rag-knowledge-qa/
├── config/                 # 配置中心
│   └── settings.py         # 所有可调参数
├── data/                   # 数据目录
│   ├── documents/          # 放入要检索的文档
│   ├── vector_store/       # Chroma 持久化文件
│   └── processed/          # 预处理中间产物
├── src/                    # 核心代码
│   ├── document_loader/    # ① 文档加载（PDF/TXT/MD）
│   ├── text_splitter/      # ② 文本分割
│   ├── embedding/          # ③ 向量化
│   ├── vector_store/       # ④ 向量存储与检索
│   ├── retriever/          # ⑤ 检索策略
│   ├── llm/                # ⑥ Ollama LLM 封装
│   ├── chain/              # ⑦ LangChain 链组装
│   └── utils/              # 工具（日志等）
├── scripts/                # 可执行脚本
│   ├── ingest.py           # 文档摄入
│   ├── query.py            # 命令行问答
│   └── start_api.py        # 启动 Web 服务
├── api/                    # FastAPI Web 层（进阶）
├── tests/                  # 单元测试
└── notebooks/              # Jupyter 学习笔记
```

## 学习路线

### 第一阶段：理解原理（Jupyter Notebooks）
按顺序运行 `notebooks/` 中的实验，逐步理解每个环节：
1. `01_document_loading.ipynb` — 探索不同类型文档的加载方式
2. `02_chunking_strategy.ipynb` — 对比不同分割策略的效果
3. `03_embedding_retrieval.ipynb` — 理解向量化和相似度检索
4. `04_qa_chain.ipynb` — 组装完整问答链

### 第二阶段：命令行工具
- `python scripts/ingest.py` — 将文档入库
- `python scripts/query.py` — 交互式问答

### 第三阶段：Web 服务化
- `python scripts/start_api.py` — 启动 REST API
- 用 curl / Postman 测试 `/upload` 和 `/query` 接口

### 第四阶段：工程化
- 编写 `tests/` 下的单元测试
- 完善错误处理和日志

## 前置准备

1. 安装 [Ollama](https://ollama.com/) 并拉取模型：`ollama pull qwen2.5`
2. 安装 Python 依赖：`pip install -r requirements.txt`
3. 将文档放入 `data/documents/` 目录
4. 运行 `python scripts/ingest.py` 建立向量库
5. 运行 `python scripts/query.py` 开始问答

## 面试要点

这个项目可以帮助你在面试中展示以下能力：
- **RAG 全链路理解**：加载 → 分割 → 向量化 → 检索 → 生成
- **Chunking 策略**：文本分割对检索质量的影响
- **检索策略**：相似度检索 vs MMR，top_k 调参
- **LangChain 链类型**：stuff / refine / map_reduce 的适用场景
- **本地部署**：Ollama + Chroma 离线方案的优势
