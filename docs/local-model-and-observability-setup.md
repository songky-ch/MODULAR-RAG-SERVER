# MiMo、Ollama、MarkItDown 与本地 Langfuse 配置

## 1. MiMo

在 Xiaomi MiMo API 开放平台的 `控制台 -> API Keys` 创建按量付费 Key；Token Plan 用户在 Token Plan 页面创建专属 Key，并复制该页面给出的 OpenAI 兼容 Base URL。两种 Key 不可混用。

```bash
export MIMO_API_KEY="你的 MiMo Key"
export MIMO_BASE_URL="Token Plan 页面显示的 OpenAI 兼容 Base URL"
```

默认配置使用：

- `mimo-v2.5-pro`：文本生成、Chunk 增强、元数据增强、LLM Rerank。
- `mimo-v2.5`：图片理解与图片描述。

一把 Key 可供两个模型共用。如果需要拆分 Key，可设置：

```bash
export MIMO_LLM_API_KEY="文本模型 Key"
export MIMO_VISION_API_KEY="视觉模型 Key"
```

专用变量优先于 `MIMO_API_KEY`。模型通过 `config/settings.yaml` 中的 `llm.model` 和 `vision_llm.model` 分别选择。MiMo 当前没有供本项目使用的 Embedding API，因此 Embedding 使用本地 Ollama。

## 2. Ollama Embedding

安装并启动 Ollama：

```bash
brew install ollama
brew services start ollama
ollama pull qwen3-embedding
```

检查服务和模型：

```bash
curl http://localhost:11434/api/tags
ollama list
```

项目配置已经设为：

```yaml
embedding:
  provider: "ollama"
  model: "qwen3-embedding"
  dimensions: 4096
  base_url: "http://localhost:11434"
```

首次更换 Embedding 模型后，必须删除旧 Collection 并重新摄取文档，不能把不同模型或不同维度的向量混在同一个 Collection 中。

如 `qwen3-embedding` 在本机资源占用过高，可改成其他 Ollama Embedding 模型，但必须同步修改 `model` 和实际向量维度，并重新摄取数据。

## 3. MarkItDown PDF 解析

安装项目及 PDF 可选依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[pdf,rerank,dashboard,eval,observability,dev]"
```

项目配置已经设为：

```yaml
pdf_loader:
  backend: "markitdown"
```

验证单个 PDF：

```bash
python scripts/ingest.py --path tests/fixtures/sample_documents/simple.pdf --collection local_test
python scripts/query.py --query "文档主要内容是什么" --collection local_test --verbose
```

MarkItDown 适合文本型 PDF。扫描件、复杂跨页表格和高保真版面解析效果可能弱于在线 OCR；这不会影响项目主流程，但会影响进入 RAG 的原始文本质量。

## 4. 本地 Langfuse

本项目不复制 Langfuse 的基础设施配置，直接使用 Langfuse 官方仓库当前版本的 Docker Compose，避免本地副本随上游架构升级失效。

```bash
mkdir -p deploy/langfuse
git clone --depth 1 https://github.com/langfuse/langfuse.git deploy/langfuse/.runtime
cd deploy/langfuse/.runtime
docker compose up -d
docker compose ps
```

浏览器访问 `http://localhost:3000`，创建本地账户和项目，然后在项目设置中创建 Public Key 与 Secret Key：

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_BASE_URL=http://localhost:3000
export LANGFUSE_PUBLIC_KEY="本地项目 Public Key"
export LANGFUSE_SECRET_KEY="本地项目 Secret Key"
```

重新运行摄取或查询命令后，MiMo 的 Chunk 增强、元数据增强、图片描述和 LLM Rerank 调用会发送到本地 Langfuse。原有 `logs/traces.jsonl` 和 Streamlit Trace 页面继续保留。

停止服务：

```bash
cd deploy/langfuse/.runtime
docker compose down
```

删除数据属于不可逆操作，只有明确不再需要本地追踪数据时才执行 `docker compose down -v`。
