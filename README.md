# Modular RAG MCP Server

一个可插拔、可观测的**模块化 RAG（检索增强生成）**服务框架，通过 **MCP（Model Context Protocol）** 对外暴露工具接口，支持 Copilot、Claude 等 AI 助手直接调用知识库检索与问答。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| **数据摄取** | PDF → Markdown → 分块 → 增强（元数据/图片描述）→ 双路向量化 → 写入向量库，支持增量更新与多模态 |
| **混合检索** | Dense（向量）+ Sparse（BM25）+ RRF 融合，可选 Rerank（Cross-Encoder / LLM / 云 API） |
| **MCP 服务** | 标准 MCP 协议，stdio 传输，暴露 `query_knowledge_hub`、`list_collections`、`get_document_summary` 等工具 |
| **管理面板** | Streamlit 六页：总览、数据浏览、摄取管理、摄取追踪、查询追踪、评估面板 |
| **评估体系** | Ragas|


---

## 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd MODULAR-RAG-MCP-SERVER

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 核心依赖
pip install -e .

# 可选：PDF、Dashboard、Rerank、评估
pip install -e ".[pdf,dashboard,rerank,eval]"
```

---

## 配置

主配置：`config/settings.yaml`。默认文本与视觉模型使用 Xiaomi MiMo，通过 `MIMO_API_KEY` 配置；Embedding 使用本地 Ollama，无需 API Key。

主要配置项：

- **llm**：推理模型（如 `openai` / `dashscope` / `ollama`）
- **embedding**：向量模型（如 `openai` / `dashscope` / `ollama`）
- **vision_llm**：多模态图片描述（可选）
- **vector_store**：向量库（默认 Chroma，`persist_directory`、`collection_name`）
- **retrieval**：`dense_top_k`、`sparse_top_k`、`fusion_top_k`、`rrf_k`
- **rerank**：是否启用、后端（`dashscope` / `cross_encoder` / `llm` 等）

使用前请复制或修改 `config/settings.yaml`，并确保不将真实 API Key 提交到仓库。

---

## 使用方式

### 1. 摄取文档

```bash
# 单文件
python scripts/ingest.py --path documents/report.pdf --collection my_docs

# 目录
python scripts/ingest.py --path documents/ --collection my_docs

# 强制重新处理（忽略历史）
python scripts/ingest.py --path documents/report.pdf --collection my_docs --force

# 自定义配置
python scripts/ingest.py --path documents/ --collection my_docs --config config/settings_financial.yaml
```

### 2. 命令行查询

```bash
python scripts/query.py --query "你的问题" --collection my_docs
python scripts/query.py --query "你的问题" --verbose   # 显示召回与重排细节
python scripts/query.py --query "你的问题" --no-rerank # 关闭 Rerank
```

### 3. 启动 MCP 服务（供 Copilot / Claude 等连接）

```bash
# 在项目根目录执行
python -m src.mcp_server.server
```

MCP 使用 stdio 传输，由 Cursor、Claude Desktop 等通过配置的 MCP Server 命令调用，无需单独开端口。

### 4. 启动管理面板

```bash
python scripts/start_dashboard.py
# 或指定端口
python scripts/start_dashboard.py --port 8502
```

浏览器访问 `http://localhost:8501`，可进行总览、数据浏览、摄取管理、追踪查看、评估任务等操作。

### 5. 运行评估

```bash
python scripts/evaluate.py
# 具体参数见脚本帮助与 Golden Test Set 配置
```

---

## MCP 暴露的工具

| 工具 | 说明 |
|------|------|
| **query_knowledge_hub** | 对指定集合进行混合检索（Dense + Sparse + RRF + 可选 Rerank），返回答案与引用片段 |
| **list_collections** | 列出当前向量库中的集合名称及简要统计 |
| **get_document_summary** | 根据文档标识获取文档摘要或概要信息 |

---

## 项目结构（简要）

```
├── config/
│   └── settings.yaml          # 主配置
├── main.py                    # 占位入口（实际 MCP 入口为 src.mcp_server.server）
├── scripts/
│   ├── ingest.py              # 摄取 CLI
│   ├── query.py               # 查询 CLI
│   ├── evaluate.py            # 评估运行
│   └── start_dashboard.py     # 面板启动
├── src/
│   ├── core/                  # 配置、类型、检索、响应、追踪
│   ├── ingestion/             # 摄取流水线、存储、Transform
│   ├── libs/                  # Embedding / LLM / Loader / Splitter / VectorStore / Reranker 等可插拔实现
│   ├── mcp_server/            # MCP 协议与工具实现
│   └── observability/         # 日志、Dashboard、评估
└── tests/
```

---

## 测试

```bash
# 单元测试
pytest tests/ -v

# 排除需外部服务的测试
pytest tests/ -v -m "not llm"
```

---

## 更多说明

- 本项目采用可插拔设计：LLM、Embedding、Rerank、Loader、Splitter、VectorStore 等均可通过配置或实现抽象接口替换，便于扩展与二次开发。
- MiMo、Ollama Embedding、MarkItDown 和本地 Langfuse 的配置步骤见 [本地模型与可观测性配置](docs/local-model-and-observability-setup.md)。

---

## License

MIT

