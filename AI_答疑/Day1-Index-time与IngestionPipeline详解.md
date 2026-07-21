# Day1: Index-time 与 IngestionPipeline 详解

## 1. 本文定位

本文对应 `docs/技术学习课程/01-Day1-RAG系统心智模型.md` 中的 Index-time 部分，重点讲解：

- `src/ingestion/pipeline.py` 的整体执行流程；
- 每个阶段输入什么、输出什么；
- 流程中关键函数和组件的逻辑；
- 这些实现背后的工程思维；
- 当前源码实际表现出的能力边界。

Day1 需要建立的核心认识是：RAG 不是只有用户提问时才开始工作。完整 RAG 系统包含两条链路：

```text
Index-time：提前处理知识，为未来查询建立索引
Query-time：根据当前问题检索证据并组织返回结果
```

本文只讨论 Index-time。

---

## 2. Index-time 解决什么问题

假设知识库中有数百份 PDF。用户提问时，系统不可能再临时完成以下全部工作：

```text
读取所有 PDF
→ 恢复正文和图片
→ 切分文本
→ 调用 Embedding
→ 统计关键词
→ 建立索引
→ 再执行检索
```

因此，系统需要提前把面向人阅读的原始文件，加工成面向检索系统使用的数据：

```text
原始 PDF
→ Document
→ Chunk[]
→ 清洗和元数据增强后的 Chunk[]
→ Dense 向量 + Sparse 词项统计
→ ChromaDB + BM25 + 图片索引
```

这条知识准备链就是 Index-time。

它的目标不是简单保存文件，而是回答：

> 为了让未来的查询快速找到正确材料，现在应该如何加工这份文档？

例如，原文中有一句：

```text
差旅住宿标准为每人每天 500 元。
```

经过 Index-time 后，它会获得两类主要检索入口：

- Dense 向量：帮助“住宿报销限额是多少”这类不同表达进行语义匹配；
- Sparse 词项：帮助“500 元”“差旅住宿”等关键词进行精确匹配。

---

## 3. `pipeline.py` 的角色

`src/ingestion/pipeline.py` 是文档入库流程的编排器。

它不亲自实现 PDF 解析、文本切分、Embedding 或 BM25 算法，而是负责：

1. 根据配置创建所需组件；
2. 按正确顺序调用组件；
3. 把上一步输出交给下一步；
4. 记录各阶段耗时、数量和中间结果；
5. 统一处理成功、跳过和失败结果；
6. 在流程结束后释放资源。

可以把它理解成总导演：

```text
IngestionPipeline
├── SQLiteIntegrityChecker：判断文件是否需要处理
├── PdfLoader / GlmOcrPdfLoader：解析 PDF
├── DocumentChunker：把 Document 转换成 Chunk[]
├── ChunkRefiner：清洗和优化 Chunk 正文
├── MetadataEnricher：提取标题、摘要和标签
├── SectionTableMetadataTransform：识别章节和表格
├── ImageCaptioner：将图片转换成文字描述
├── DenseEncoder：生成语义向量
├── SparseEncoder：生成 BM25 词项统计
├── VectorUpserter：写入向量库
├── BM25Indexer：构建倒排索引
└── ImageStorage：注册图片索引
```

主调用关系是：

```text
run_pipeline()
→ load_settings()
→ IngestionPipeline(settings)
→ pipeline.run(file_path)
→ pipeline.close()
```

---

## 4. 核心数据契约

理解流水线之前，要先理解数据在流程中如何改变。

### 4.1 `Document`

`Document` 表示文件已经完成解析，但还没有切分：

```python
Document(
    id="doc_abc123",
    text="# 差旅制度\n住宿标准……\n[IMAGE: image_001]",
    metadata={
        "source_path": "travel-policy.pdf",
        "doc_type": "pdf",
        "title": "差旅制度",
        "page_count": 20,
        "images": [...],
    },
)
```

它将不同 PDF 加载后端的结果统一成同一种业务对象。后续组件只依赖 `Document`，不需要了解 PDF 是通过本地解析还是 OCR 服务得到的。

### 4.2 `Chunk`

`Chunk` 表示可以独立参与检索的文本片段：

```python
Chunk(
    id="doc_abc123_0001_9f31a5c8",
    text="住宿标准为每人每天 500 元。",
    metadata={
        "source_path": "travel-policy.pdf",
        "chunk_index": 1,
        "source_ref": "doc_abc123",
    },
)
```

Chunk 不只是短文本，它还必须保留来源、顺序、父文档、图片和其他元数据，否则后续无法过滤、引用和排错。

### 4.3 编码结果

同一个 Chunk 会产生两种表示：

```text
Chunk.text
├── Dense vector：语义表示
└── Sparse stats：词频、文档长度、唯一词数
```

三者必须保持严格的顺序对应关系：

```text
chunks[0]         描述第一个 Chunk
dense_vectors[0] 描述第一个 Chunk
sparse_stats[0]  描述第一个 Chunk
```

---

## 5. `PipelineResult`：流水线结果报告

`PipelineResult` 统一描述一次执行的结果，主要字段包括：

```python
success
file_path
doc_id
chunk_count
image_count
vector_ids
error
stages
```

其中 `stages` 保存各阶段统计：

```python
{
    "integrity": {...},
    "loading": {...},
    "chunking": {...},
    "transform": {...},
    "encoding": {...},
    "storage": {...},
}
```

它体现了一个重要思路：系统不能只返回“成功或失败”，还应保留每一步实际产生了什么。这样才能判断问题发生在解析、切分、转换、编码还是存储阶段。

`to_dict()` 将结果转换成可序列化字典，并只返回 `vector_ids_count`，而不是直接放入完整 ID 列表。

---

## 6. `IngestionPipeline.__init__()`：组装全部组件

初始化阶段不处理具体文件，只根据配置准备流水线。

### 6.1 保存运行参数

```python
self.settings = settings
self.collection = collection
self.force = force
```

- `settings`：项目配置；
- `collection`：文档所属的知识集合；
- `force`：是否强制重新处理已成功入库的文件。

`collection` 相当于逻辑命名空间，例如 `contracts`、`financial_reports`。它会影响向量集合、BM25 目录和图片目录。

### 6.2 初始化完整性检查器

```python
self.integrity_checker = SQLiteIntegrityChecker(
    db_path="data/db/ingestion_history.db"
)
```

SQLite 中记录文件哈希、路径、处理状态、collection、错误和处理时间。该状态必须跨进程保存，因此不能只放在 Python 内存中。

### 6.3 选择 PDF 加载器

```python
if loader_backend == "glm_ocr":
    self.loader = GlmOcrPdfLoader(...)
else:
    self.loader = PdfLoader(...)
```

这里采用配置驱动和统一接口设计。底层实现不同，但流水线都通过以下调用获得 `Document`：

```python
document = self.loader.load(file_path)
```

### 6.4 初始化转换、编码和存储组件

转换组件负责把 Chunk 加工得更适合检索；编码组件负责生成 Dense 和 Sparse 表示；存储组件负责持久化三类索引。

`EmbeddingFactory.create(settings)` 和 `VectorStoreFactory` 相关实现，使流水线可以根据配置选择具体服务，而不把某个 Embedding 或向量数据库写死在编排器中。

---

## 7. `run()` 的六阶段主流程

`run()` 是 Index-time 的核心入口：

```python
pipeline.run(file_path, trace=None, on_progress=None)
```

输入是一份文件，输出是 `PipelineResult`。

完整流程为：

```text
Stage 1：文件完整性检查
Stage 2：文档加载
Stage 3：文档切分
Stage 4：内容转换
Stage 5：Dense + Sparse 编码
Stage 6：向量、BM25 和图片索引存储
```

### 7.1 `_notify()`

`run()` 内部定义 `_notify(stage_name, step)`，在提供 `on_progress` 回调时通知调用方当前阶段和总阶段数。

例如管理页面可以据此显示：

```text
split 3/6
```

它不参与数据加工，只负责向外报告进度。

---

## 8. Stage 1：文件完整性检查

核心逻辑：

```python
file_hash = self.integrity_checker.compute_sha256(str(file_path))

if not self.force and self.integrity_checker.should_skip(file_hash):
    return PipelineResult(...)
```

### 8.1 为什么使用 SHA256

系统根据文件内容而不是文件名判断是否发生变化：

```text
文件名相同、内容变化
→ SHA256 变化
→ 重新处理

文件路径变化、内容完全相同
→ SHA256 不变
→ 可以识别为相同内容
```

`compute_sha256()` 每次读取 64 KB，而不是一次将大文件全部载入内存。

### 8.2 跳过逻辑

```text
历史状态为 success，并且 force=False
→ 跳过

历史状态为 failed
→ 允许重试

force=True
→ 无条件重新处理
```

这使入库操作具备基本幂等性。

### 8.3 `file_hash` 的后续用途

它会作为：

- `PipelineResult.doc_id`；
- 每个 Chunk 的 `metadata.doc_hash`；
- 入库历史、向量数据和图片索引之间的关联键。

---

## 9. Stage 2：文档加载

核心调用：

```python
document = self.loader.load(str(file_path))
```

数据变化：

```text
PDF 路径
→ Document(id, text, metadata)
```

加载器负责恢复正文、基础结构和图片信息。图片会保存到磁盘，并在正文中留下占位符：

```text
[IMAGE: image_001]
```

占位符保留图片在正文中的语义位置，后续视觉模型才能将描述准确追加到引用该图片的 Chunk。

本阶段会记录文档 ID、文本长度、图片数量和文本预览。如果提供 `TraceContext`，还会记录加载后正文和加载后端，便于判断知识是否在解析阶段已经丢失。

---

## 10. Stage 3：文档切分

核心调用：

```python
chunks = self.chunker.split_document(document)
```

数据变化：

```text
Document
→ 文本片段 str[]
→ 带 ID 和元数据的 Chunk[]
```

### 10.1 为什么必须分块

整份几十页 PDF 如果只生成一个向量，会混合太多主题，并造成定位、引用和上下文长度问题。

Chunk 的目标是成为可以独立召回、独立理解、独立追踪的知识单元。

### 10.2 `DocumentChunker.split_document()`

主要步骤：

1. 检查 `document.text` 是否为空；
2. 调用底层 splitter 得到文本片段；
3. 为每个片段生成确定性 ID；
4. 继承和扩展文档元数据；
5. 创建符合统一契约的 `Chunk`。

### 10.3 `_generate_chunk_id()`

ID 格式：

```text
{doc_id}_{chunk_index:04d}_{content_hash前8位}
```

例如：

```text
doc_123_0002_9f31a5c8
```

同样的父文档、顺序和正文会生成同样 ID，兼顾唯一性、确定性和可读性。

### 10.4 `_inherit_metadata()`

它复制文档元数据，并增加：

```text
chunk_index：Chunk 顺序
source_ref：父 Document ID
image_refs：正文引用的图片 ID
images：当前 Chunk 引用图片的完整元数据
page_num：尝试从引用图片推断页码
```

它不会把文档的全部图片复制到每个 Chunk，而是通过 `[IMAGE: id]` 占位符，只关联当前 Chunk 真正引用的图片。

### 10.5 注入 `doc_hash`

切分完成后，主流程执行：

```python
for chunk in chunks:
    chunk.metadata["doc_hash"] = file_hash
```

这样入库历史、向量记录和图片记录可以通过相同文件哈希关联。

---

## 11. Stage 4：内容转换流水线

Stage 4 包含四个连续子阶段：

```text
4a. Chunk Refinement
4b. Metadata Enrichment
4c. Section/Table Metadata Enrichment
4d. Image Captioning
```

整体思路是先清洗正文，再为正文增加结构和检索辅助信息。

### 11.1 Chunk Refinement

核心调用：

```python
chunks = self.chunk_refiner.transform(chunks, trace)
```

规则清洗始终执行，主要包括：

- 临时提取并保护 Markdown 代码块；
- 删除页眉页脚分隔线；
- 删除 HTML 注释；
- 删除 HTML 标签但保留正文；
- 合并多余空格和换行；
- 清理每行尾部空白；
- 恢复代码块。

先保护代码块，是为了避免全局清洗破坏代码内部的空格、标签和换行。

如果配置启用 LLM，则在规则结果基础上进一步优化：

```text
规则清洗
→ LLM 优化成功：refined_by="llm"
→ LLM 优化失败：保留规则结果，refined_by="rule"
```

启用 LLM 时使用线程池并行处理不同 Chunk，但通过原始索引将结果放回对应位置，因此输出顺序保持不变。

### 11.2 Metadata Enrichment

核心调用：

```python
chunks = self.metadata_enricher.transform(chunks, trace)
```

它主要增加：

```text
title
summary
tags
enriched_by
```

规则标题提取优先级：

```text
Markdown 标题
→ 较短的第一行
→ 第一条句子
→ 前 100 个字符
```

规则摘要取前几句话并限制总长度。规则标签主要识别英文专有名词、camelCase、snake_case 和 Markdown 强调内容。

如果启用 LLM，则要求模型返回 Title、Summary、Tags，再用正则解析。LLM 失败时使用规则结果，并记录 `enrich_fallback_reason`。

单个 Chunk 增强异常时，组件会保留正文并生成最小元数据，避免单条数据异常立即丢失整批内容。

### 11.3 Section/Table Metadata

核心调用：

```python
chunks = self.section_table_metadata.transform(chunks, trace)
```

它使用纯规则识别：

```text
is_table_chunk
table_title
section_title
```

表格主要根据 Markdown `|` 行比例判断；章节标题识别 `# 标题`、`一、标题`、`（一）标题`、`1. 标题` 等形式。

这些字段让未来检索、过滤、展示和引用不仅依赖正文，还能利用文档结构。

### 11.4 Image Captioning

核心调用：

```python
chunks = self.image_captioner.transform(chunks, trace)
```

视觉模型未启用时，组件原样返回 Chunk，不阻断入库。

启用后执行：

```text
收集图片元数据
→ 找出正文真正引用的唯一图片
→ 并行调用视觉模型生成描述
→ 缓存图片描述
→ 将描述写回 Chunk.text 和 metadata
```

例如：

```text
[IMAGE: image_001]
(Description: 图中显示 2025 年收入同比增长 20%)
```

Embedding 和 BM25 只能直接处理文本，因此图片描述的作用是将视觉语义转换成可检索文本。

---

## 12. Stage 5：Dense + Sparse 双路编码

核心调用：

```python
batch_result = self.batch_processor.process(chunks, trace)
dense_vectors = batch_result.dense_vectors
sparse_stats = batch_result.sparse_stats
```

### 12.1 `BatchProcessor`

它根据 `batch_size` 将 Chunk 分批：

```text
chunks[0:100]
chunks[100:200]
……
```

每批依次执行 Dense 和 Sparse 编码，并收集耗时、成功数量和失败数量。

虽然类注释提到 Parallel Encodings，但当前实现中同一批次的 Dense 和 Sparse 编码是顺序调用的。

### 12.2 `DenseEncoder.encode()`

数据变化：

```text
Chunk[]
→ text[]
→ Embedding 服务
→ List[List[float]]
```

主要逻辑：

1. 提取所有 `chunk.text`；
2. 拒绝空文本；
3. 按批次调用 `embedding.embed()`；
4. 验证返回向量数等于输入文本数；
5. 验证所有向量维度一致；
6. 按输入顺序合并结果。

数量和维度校验能够防止向量与 Chunk 错位。

Dense 表示主要解决“表达不同但语义相近”的匹配问题。

### 12.3 `SparseEncoder.encode()`

每个 Chunk 会生成：

```python
{
    "chunk_id": "...",
    "term_frequencies": {"machine": 2, "learning": 1},
    "doc_length": 3,
    "unique_terms": 2,
}
```

英文和数字使用正则提取并转换为小写；连续中文采用 2-gram：

```text
中国市场
→ 中国、国市、市场
```

这种方法不依赖外部分词库，也能为中文建立基础的关键词匹配能力。

Sparse 表示尤其适合错误码、API 名称、型号、金额和明确术语等精确匹配场景。

---

## 13. Stage 6：持久化三类索引

Stage 6 将内存中的加工结果写成 Query-time 可使用的持久数据。

### 13.1 写入向量库

```python
vector_ids = self.vector_upserter.upsert(chunks, dense_vectors, trace)
```

`VectorUpserter` 首先验证 Chunk 数量和向量数量一致，然后根据以下内容生成最终存储 ID：

```text
source_path 哈希 + chunk_index + 最终正文哈希
```

Stage 3 以后正文可能经过清洗、LLM 优化或图片描述追加，所以最终 `vector_id` 与最初的业务层 `Chunk.id` 不一定相同。

向量库记录包含：

```python
{
    "id": chunk_id,
    "vector": dense_vector,
    "metadata": {
        ...原有元数据,
        "text": chunk.text,
        "chunk_id": chunk_id,
        "doc_hash": file_hash,
    },
}
```

向量负责计算相似度，metadata 负责返回正文、来源、标题、页码和其他引用信息。

### 13.2 构建 BM25 索引

写入前，主流程先统一 Sparse 和向量库的最终 ID：

```python
for stat, vid in zip(sparse_stats, vector_ids):
    stat["chunk_id"] = vid
```

这是 Dense 和 Sparse 结果能够进行融合的关键。如果两边 ID 不一致，Query-time 会把同一个 Chunk 当成两条不同记录。

`BM25Indexer.build()` 主要执行：

1. 计算文档数量 $N$；
2. 计算平均 Chunk 长度；
3. 计算每个词的文档频率 DF；
4. 计算 IDF；
5. 建立词项到 Chunk 的倒排表；
6. 将索引持久化到 JSON 文件。

IDF 公式为：

$$
IDF(t)=\log\frac{N-df(t)+0.5}{df(t)+0.5}
$$

直观理解：只在少数 Chunk 中出现的词区分度更高，几乎所有 Chunk 都出现的词区分度更低。

倒排结构类似：

```python
"报销": {
    "idf": 1.2,
    "df": 3,
    "postings": [
        {"chunk_id": "...", "tf": 2, "doc_length": 100}
    ],
}
```

它能够快速回答：给定一个词，哪些 Chunk 包含它？

### 13.3 注册图片索引

图片文件已经在加载阶段保存，本阶段只建立索引关系：

```text
image_id
→ 图片路径
→ collection
→ doc_hash
→ page_num
```

这样 Query-time 返回图片引用时，可以根据图片 ID 定位真实图片和原始页码。

---

## 14. 成功、失败和资源释放

### 14.1 为什么最后才标记成功

只有加载、切分、转换、编码和三类存储都完成后，才调用：

```python
self.integrity_checker.mark_success(...)
```

因此：

```text
解析成功但向量写入失败
≠ 入库成功

向量写入成功但 BM25 构建失败
≠ 入库成功
```

### 14.2 统一异常处理

六阶段处于同一个 `try` 中。出现异常时，流水线记录失败状态，并返回包含错误和已完成阶段信息的 `PipelineResult`。

这使调用方可以判断失败前执行到了哪个阶段。

### 14.3 `close()` 与 `finally`

`run_pipeline()` 使用：

```python
try:
    return pipeline.run(file_path)
finally:
    pipeline.close()
```

无论成功、失败还是提前返回，都会执行资源释放。

---

## 15. 用一份 PDF 串起数据变化

假设输入：

```text
travel-policy.pdf
```

### 15.1 完整性检查

```text
travel-policy.pdf
→ SHA256: abc123...
```

### 15.2 文档加载

```python
Document(
    id="doc_abc123",
    text="# 差旅制度\n住宿标准……",
    metadata={"source_path": "travel-policy.pdf", "images": [...]},
)
```

### 15.3 文档切分

```python
[
    Chunk(text="# 差旅制度\n适用范围……", metadata={"chunk_index": 0}),
    Chunk(text="住宿标准为每日 500 元……", metadata={"chunk_index": 1}),
]
```

### 15.4 内容转换

```python
Chunk(
    text="住宿标准为每日 500 元。",
    metadata={
        "title": "住宿标准",
        "summary": "住宿标准为每日 500 元。",
        "section_title": "住宿标准",
        "refined_by": "rule",
        "enriched_by": "rule",
    },
)
```

### 15.5 双路编码

```text
Dense：[0.13, -0.27, 0.51, ...]

Sparse：
住宿 → 1
标准 → 1
每日 → 1
500 → 1
```

### 15.6 持久化

```text
ChromaDB：保存向量、正文和来源元数据
BM25：保存词项到 Chunk ID 的倒排关系
ImageStorage：保存图片 ID 到路径、文档和页码的关系
```

未来用户查询“出差住宿最多报多少”时，Query-time 才会使用这些预先建立的数据执行 Dense 和 Sparse 召回。

---

## 16. 这段代码体现的核心逻辑思维

### 16.1 用数据契约隔离阶段

```text
文件
→ Document
→ Chunk[]
→ vectors / term_stats
→ 存储记录
```

每个阶段只依赖约定好的输入和输出，不需要了解所有上下游实现。

### 16.2 Dense 和 Sparse 相互补充

```text
Dense：解决表达不同但意思相近
Sparse：解决编号、术语和字符需要精确匹配
```

因此两种索引不是重复建设，而是为混合检索提供互补候选。

### 16.3 ID 是跨系统对齐的核心

系统中存在：

```text
file_hash
Document.id
Chunk.id
vector_id
image_id
```

它们分别标识文件、解析文档、业务 Chunk、最终存储记录和图片。BM25 与向量库必须使用相同最终 Chunk ID，才能在 Query-time 正确融合。

### 16.4 元数据不是附属信息

正文回答“内容是什么”，元数据回答：

```text
来自哪里？
属于哪篇文档？
位于第几个 Chunk？
是否是表格？
属于哪个章节？
关联什么图片？
由规则还是 LLM 加工？
```

过滤、引用、排错和数据管理都依赖这些信息。

### 16.5 可观测性属于主流程

`TraceContext` 可以记录加载正文、切分结果、清洗前后对比、Dense 维度、Sparse 高频词、存储映射和各阶段耗时。

这让系统能够回答“错误发生在哪一层”，而不是只根据最终检索结果猜测。

### 16.6 可选智能组件不应轻易阻断基础链路

ChunkRefiner、MetadataEnricher 和 ImageCaptioner 都允许在 LLM 不可用时使用规则结果或原始内容继续处理。

这体现了：基础摄取能力与辅助智能增强应尽量解耦。

---

## 17. 当前源码的实际行为

阅读源码时要区分注释描述的设计和实际调用关系形成的行为。

### 17.1 进度回调的调用时机

`on_progress` 的参数说明称其在阶段完成时调用，但当前 `_notify()` 实际位于各阶段主要工作开始之前。

### 17.2 编码批次失败后的结果

`BatchProcessor` 在某批编码失败后会继续处理后续批次。失败批次不会产生向量，最终向量数量可能少于 Chunk 数量，随后 `VectorUpserter` 的长度校验会使整次入库失败。

### 17.3 BM25 的构建方式

当前 `pipeline.run()` 将本次文件产生的 `sparse_stats` 直接交给 `BM25Indexer.build()`。`build()` 会根据传入数据重新构造并保存索引。

因此，从当前调用链看，同一 collection 连续摄取多份文件时，后一次构建使用的是本次文件数据，而不是显式读取旧索引并追加。

### 17.4 文件哈希计算前的异常

如果异常发生在 `file_hash` 成功赋值之前，异常处理首先调用 `mark_failed(file_hash, ...)`，此时 `file_hash` 可能尚不存在，可能掩盖原始异常。

### 17.5 资源关闭范围

`IngestionPipeline.close()` 当前显式关闭 `image_storage`。其他组件主要依赖自身生命周期或内部关闭逻辑。

这些现象不改变主流程的学习价值，但有助于建立源码阅读习惯：不仅看类注释，还要追踪真实执行顺序、输入输出和异常路径。

---

## 18. 最终总结

`src/ingestion/pipeline.py` 的本质可以概括为：

> 它通过统一数据契约，把 PDF 依次转换成可追踪、可清洗、可进行语义检索和关键词检索、可持久化并且可观测的知识单元。

需要记住的主线是：

```text
文件哈希判断是否需要处理
→ 加载器把 PDF 统一成 Document
→ 切分器把 Document 变成 Chunk[]
→ 转换组件优化正文并增强元数据
→ 编码器生成 Dense 和 Sparse 表示
→ 存储组件建立向量、BM25 和图片索引
→ 成功后记录入库历史
```

Index-time 的质量决定 Query-time 的上限：

- 解析阶段丢失的内容，查询阶段无法找回；
- 切分后不完整的知识，检索后仍可能无法回答；
- 缺失的来源和页码，最终无法形成可信引用；
- Dense 与 Sparse ID 未对齐，混合检索无法正确融合；
- 索引没有正确持久化，未来查询就没有可靠候选。

因此，理解 Index-time 不是背诵六个 Stage，而是能够沿着数据流回答：

1. 当前阶段输入是什么？
2. 当前阶段输出是什么？
3. 增加了哪些信息？
4. 丢失了哪些信息？
5. 这些结果将如何影响未来检索？
