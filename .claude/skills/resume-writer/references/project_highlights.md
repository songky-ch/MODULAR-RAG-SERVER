# 项目技术亮点清单（Modular RAG MCP Server）

> 从 DEV_SPEC 与源码提炼，供简历编写时按需选取。每个亮点附带"简历话术方向"和"可量化角度"。

---

## 亮点 1：多阶段混合检索架构（Hybrid Search + Rerank）

**技术要点**：
- 设计并实现"粗排召回 → 精排重排"两段式检索架构
- 粗排阶段并行执行 Dense Retrieval（语义向量，Cosine Similarity）+ Sparse Retrieval（BM25 关键词匹配）
- 通过 RRF（Reciprocal Rank Fusion）算法融合双路结果，平衡查准率与查全率
- 精排阶段支持四种可插拔 Reranker：LLM Rerank（结构化 Prompt）/ Cross-Encoder 本地模型 / API Reranker（如 Jina）/ DashScope TextReRank；支持 None 直通
- 精排失败时自动回退至融合排名（Graceful Fallback），保障系统可用性

**简历话术方向**：
- "设计并实现了 Hybrid Search 混合检索引擎，结合 BM25 稀疏检索与 Dense Embedding 稠密检索，通过 RRF 融合算法实现查准率与查全率的平衡"
- "引入可插拔 Rerank 模块（LLM/Cross-Encoder/API/DashScope），在不牺牲响应速度的前提下将 Top-K 检索精准度提升 XX%"

**可量化角度**：RAGAS 的 Answer Relevancy、Context Precision、Faithfulness，Hit Rate@K、MRR、NDCG、Rerank 前后 Top-1 命中率变化、端到端查询延迟等

---

## 亮点 2：全链路可插拔架构 + LangChain 集成（Factory + 配置驱动）

**技术要点**：
- 为 LLM / Embedding / Splitter / VectorStore / Reranker / Evaluator 六大组件定义统一抽象接口（Base 类）
- 采用工厂模式（Factory Pattern）+ YAML 配置驱动，实现"改配置不改代码"的组件切换
- **LangChain 集成**：LLM/Embedding 通过 LCLLMFactory 统一创建 LangChain `BaseChatModel` / `Embeddings`；VectorStore 使用 `LangChainChromaStore` 封装 `langchain_chroma.Chroma`；自研 Trace 通过 `LangChainTraceCallback` 桥接 LangChain 回调，将 LLM/Embedding 调用纳入全链路追踪
- LLM Provider 支持 Azure OpenAI / OpenAI / Ollama / DashScope（通义）等，通过 LangChain `init_chat_model` 与各 Provider 配置类（OpenAI/Ollama/DashScope）映射
- Embedding 支持 OpenAI / DashScope / Ollama 等，由 EmbeddingFactory 调用 LCLLMFactory 创建 LangChain Embeddings 后经 `LangChainEmbeddingAdapter` 适配为项目内 `BaseEmbedding` 接口
- 向量数据库接口预留扩展（当前默认 Chroma 基于 LangChain Chroma 实现）
- Vision 与 Text 共用同一 `BaseChatModel`，通过消息级多模态能力支持图像处理

**简历话术方向**：
- "设计了全链路可插拔架构，基于抽象接口 + 工厂模式 + 配置驱动，实现 LLM/Embedding/VectorStore 等 6 大核心组件的零代码热切换；引入 LangChain 统一 LLM/Embedding/VectorStore 实现，并通过自研 Callback 将 LangChain 调用纳入全链路追踪"
- "架构支持 Azure OpenAI、Ollama、DashScope 等多种 Provider 无缝切换，满足企业合规与成本优化需求"

**可量化角度**：支持 N 种 LLM Provider、N 种 Embedding 后端、配置切换零代码修改

---

## 亮点 3：智能数据摄取流水线（Ingestion Pipeline）

**技术要点**：
- **六阶段流水线**：File Integrity（SHA256 前置跳过）→ Load → Split → Transform → Encode → Storage；支持 `force` 强制重跑，未变更文件可跳过 Load 及后续阶段，降低 API 与算力消耗
- **Loader**：支持 MarkItDown（PdfLoader，PDF → canonical Markdown，保留结构并提取图片）与 GLM OCR（GlmOcrPdfLoader，适用扫描件/复杂版式）
- **Splitter**：配置化选择 RecursiveSplitter（LangChain `RecursiveCharacterTextSplitter`）或 FinancialReportSplitter（财报结构感知、单遍线性扫描、段落/表格/小节上下文栈、中文分隔符，内部仍用 LangChain 切分）
- **Transform 四子步**：ChunkRefiner（LLM/规则双模式智能重组与去噪）→ MetadataEnricher（Title/Summary/Tags 语义元数据）→ SectionTableMetadataTransform（无外部依赖的段落/表格启发式元数据，与 FinancialReportSplitter 搭配）→ ImageCaptioner（Vision LLM 图片描述，实现"搜文出图"）
- 双路向量化：Dense（经 BatchProcessor）+ Sparse（BM25）并行编码；Storage 阶段写 Chroma、BM25 索引与 ImageStorage 索引，chunk_id 与 doc_hash 一致性与幂等 Upsert 有保障

**简历话术方向**：
- "设计并实现六阶段智能数据摄取流水线：SHA256 前置跳过、多 Loader（MarkItDown/GLM OCR）、可配置 Splitter（通用递归切分/财报结构感知）、四步 Transform（Refine + Enrich + Section/Table 元数据 + 图片描述）、双路向量化与多存储幂等写入"
- "实现基于 SHA256 的增量摄取与可选跳过 Load，避免重复处理与重复调用，降低 API 与算力成本"

**可量化角度**：处理文档数、生成 Chunk 数、增量摄取跳过率、LLM/规则增强覆盖率、端到端摄取耗时

---

## 亮点 4：MCP 协议集成（Model Context Protocol）

**技术要点**：
- 遵循 MCP 标准（JSON-RPC 2.0 + Stdio Transport）实现知识检索 Server
- 暴露 3 个标准 Tool：query_knowledge_hub / list_collections / get_document_summary
- 支持 GitHub Copilot、Claude Desktop 等主流 MCP Client 即插即用
- 返回格式支持 TextContent + ImageContent 多模态内容，含结构化 Citation 引用
- Stdio Transport 零配置、零网络依赖，天然适合私有知识库场景

**简历话术方向**：
- "基于 MCP（Model Context Protocol）标准实现知识检索 Server，支持 GitHub Copilot / Claude Desktop 等 AI Agent 直接调用私有知识库"
- "实现引用透明的结构化响应（Citation），支持文本 + 图像多模态返回，增强 AI 输出的可信度"

**可量化角度**：支持 N 种 MCP Client、工具调用成功率、端到端响应延迟

---

## 亮点 5：多模态图像处理（Image-to-Text）

**技术要点**：
- 采用 Image-to-Text 策略，复用纯文本 RAG 链路实现多模态检索
- Loader 阶段自动提取 PDF 图片并插入占位符标记
- Transform 阶段调用 Vision LLM（GPT-4o）生成结构化图片描述（Caption）
- 描述文本注入 Chunk 正文，被 Embedding 覆盖后可通过自然语言检索图片
- 检索命中后动态读取原始图片、编码 Base64 返回 MCP Client

**简历话术方向**：
- "设计 Image-to-Text 多模态处理方案，利用 Vision LLM 将文档图片转化为语义描述并嵌入检索链路，实现'搜文出图'能力"
- "无需引入 CLIP 等多模态向量库，复用纯文本 RAG 架构即可支持图像检索，降低架构复杂度"

**可量化角度**：处理图片数、图片描述平均长度、图片相关查询命中率

---

## 亮点 6：全链路可观测性与可视化管理平台

**技术要点**：
- 设计双链路追踪体系：Ingestion Trace（6 阶段：Integrity/Load/Split/Transform/Encode/Storage）+ Query Trace（多阶段含 Rerank）
- TraceContext 显式调用模式，低侵入记录各阶段耗时、候选数量、分数分布；LangChain 调用的 LLM/Embedding 通过 LangChainTraceCallback 写入同一 Trace，实现全链路统一观测
- JSON Lines 结构化日志持久化，零外部依赖（无 LangSmith/LangFuse）
- 基于 Streamlit 构建六页面管理平台：
  - 系统总览（组件配置 + 数据资产统计）
  - 数据浏览器（文档/Chunk/图片详情查看）
  - Ingestion 管理（文件上传、实时进度条、文档删除）
  - Ingestion 追踪（阶段耗时瀑布图）
  - Query 追踪（Dense/Sparse 对比、Rerank 前后变化）
  - 评估面板（Ragas 指标、历史趋势）
- Dashboard 基于 Trace 中 method/provider 字段动态渲染，更换组件后自动适配

**简历话术方向**：
- "构建全链路白盒化追踪体系（Ingestion + Query 双链路），LangChain 调用经自研 Callback 纳入同一 Trace，每次检索与摄取过程透明可回溯"
- "基于 Streamlit 实现六页面可视化管理平台，涵盖数据浏览、摄取管理、追踪分析、评估面板，实现 RAG 系统的全生命周期管理"

**可量化角度**：追踪覆盖阶段数、Dashboard 页面数、追踪日志条数、问题定位效率提升

---

## 亮点 7：自动化评估体系

**技术要点**：
- 可插拔评估框架：Ragas（Faithfulness/Answer Relevancy/Context Precision）+ 自定义指标（Hit Rate/MRR）
- CompositeEvaluator 支持多评估器并行执行与结果汇总
- EvalRunner 基于 Golden Test Set 进行回归评估
- 评估历史持久化，支持策略调整前后的量化对比
- 评估面板可视化展示指标趋势

**简历话术方向**：
- "建立基于 Ragas + 自定义指标的自动化评估闭环，拒绝'凭感觉调优'，每次策略调整都有量化分数支撑"
- "集成 Golden Test Set 回归测试，确保检索质量基线稳定（Hit Rate@K ≥ 90%, MRR ≥ 0.8）"

**可量化角度**：评估指标数、测试集规模、Hit Rate/MRR/Faithfulness 具体数值

---

## 亮点 8：文档生命周期管理（DocumentManager）

**技术要点**：
- DocumentManager 独立于 Pipeline，负责跨 4 个存储的协调操作
- 支持文档列表、详情查看、协调删除（Chroma + BM25 + ImageStorage + FileIntegrity 四路同步）
- Pipeline 支持 on_progress(stage_name, current, total) 回调（共 6 阶段），Dashboard 实时展示各阶段进度条
- 幂等 Upsert 设计：chunk_id 与向量存储 ID 一致，doc_hash 与 FileIntegrity 一致，保障重复摄取可跳过或覆盖一致

**简历话术方向**：
- "实现跨存储协调的文档生命周期管理，支持 Chroma/BM25/图片/处理记录四路同步删除，保障数据一致性"

**可量化角度**：管理文档数、跨存储操作成功率、删除操作耗时

---

## 亮点 9：工程化实践

**技术要点**：
- TDD 开发：1198+ 单元测试 + 30 E2E 测试全绿
- 9 个开发阶段、68 个子任务全部完成
- 分层测试金字塔：Unit → Integration → E2E
- SQLite 轻量持久化（ingestion_history + image_index + BM25 索引），零外部数据库依赖
- 配置驱动的零代码组件切换
- Prompt 模板外置（config/prompts/），支持独立迭代

**简历话术方向**：
- "遵循 TDD 开发范式，累计编写 1200+ 自动化测试用例，覆盖单元/集成/E2E 三层"
- "采用 SQLite Local-First 持久化方案，零外部数据库依赖，pip install 即可运行"

**可量化角度**：测试用例数、代码覆盖率、开发阶段数、子任务完成率

---

## 亮点 10：Agent 扩展性（面向 Agent 方向的延伸叙事）

**技术要点**：
- MCP Server 天然支持 Agent 调用（Tool Calling 范式）
- 系统可作为知识检索 Agent 嵌入 Multi-Agent 体系
- 支持构建自定义 Agent Client（ReAct / Chain of Thought 模式）
- 可快速适配不同业务场景（替换数据源、调整检索策略、定制 Prompt）

**简历话术方向**（适用于偏 Agent 方向的岗位）：
- "基于 MCP 协议构建知识检索 Agent，支持 Tool Calling / ReAct 模式，可嵌入 Multi-Agent 协作系统"
- "设计通用化知识检索框架，支持快速适配不同业务场景（替换数据源 + 调整检索策略 + 定制 Prompt），作为 Agent 生态的知识中枢"

**可量化角度**：支持的 Agent Client 数量、业务场景适配数
