# Day1: Query-time 与查询链路详解

## 1. 本文定位

本文对应 `docs/技术学习课程/01-Day1-RAG系统心智模型.md` 中的 Query-time 部分，结合当前项目源码，重点讲解：

- Query-time 从 MCP 请求到响应返回的完整调用链；
- 查询处理、Dense 召回、Sparse 召回、RRF 融合和 Rerank 的区别；
- 每个阶段的输入、输出和关键函数逻辑；
- 引用、图片和 MCP 内容块如何组装；
- 当前项目 Query-time 的真实能力边界；
- 如何根据中间结果定位检索问题。

本文主要追踪以下源码：

```text
src/mcp_server/tools/query_knowledge_hub.py
src/core/query_engine/query_processor.py
src/core/query_engine/dense_retriever.py
src/core/query_engine/sparse_retriever.py
src/core/query_engine/fusion.py
src/core/query_engine/hybrid_search.py
src/core/query_engine/reranker.py
src/libs/reranker/llm_reranker.py
src/core/response/citation_generator.py
src/core/response/multimodal_assembler.py
src/core/response/response_builder.py
src/core/types.py
```

---

## 2. Query-time 解决什么问题

Index-time 提前把原始文档加工成可以检索的数据：

```text
PDF
→ Document
→ Chunk[]
→ Dense 向量 + Sparse 词项
→ ChromaDB + BM25 + 图片索引
```

Query-time 则在用户问题到来时，利用这些索引寻找证据：

```text
用户问题
→ 查询处理
→ Dense + Sparse 候选召回
→ RRF 融合
→ 可选 Rerank
→ 选择最终结果
→ 引用和图片组装
→ MCP 响应
```

它的核心问题是：

> 面对当前问题，系统应该从知识库中找出哪些证据，并以什么顺序返回？

Day1 给出的完整 RAG Query-time 理论链路还包括：

```text
最终上下文
→ 组装问答 Prompt
→ LLM 生成答案
→ 忠实性检查
→ 答案与引用
```

但当前项目主链只完成到“检索结果、引用和 MCP 格式化”，没有完成基于证据生成综合答案的闭环。后文会用实际调用关系证明这一点。

---

## 3. 当前项目的真实调用链

从 MCP 客户端调用开始，真实顺序是：

```text
MCP Client
→ query_knowledge_hub_handler()
→ QueryKnowledgeHubTool.execute()
→ _ensure_initialized()
→ _perform_search()
→ HybridSearch.search()
   ├── QueryProcessor.process()
   ├── DenseRetriever.retrieve()
   ├── SparseRetriever.retrieve()
   ├── RRFFusion.fuse()
   └── metadata post-filter
→ _apply_rerank()
→ CoreReranker.rerank()
→ ResponseBuilder.build()
   ├── CitationGenerator.generate()
   └── MultimodalAssembler.assemble()
→ MCPToolResponse.to_mcp_content()
→ types.CallToolResult
→ MCP Client
```

这条链可以分为七层：

```text
协议入口层：校验 MCP 参数并返回 CallToolResult
工具编排层：初始化组件、调用搜索、重排和响应构建
查询处理层：提取关键词和过滤条件
召回层：Dense 与 Sparse 找候选
排序层：RRF 融合与可选 Rerank
响应层：生成检索片段、引用和图片内容块
观测层：记录每一阶段结果与耗时
```

---

## 4. Query-time 的统一数据契约

### 4.1 `ProcessedQuery`

原始用户问题经过查询处理后，转换为：

```python
ProcessedQuery(
    original_query="如何配置 Azure OpenAI？",
    keywords=["如何", "配置", "Azure", "OpenAI"],
    filters={},
    expanded_terms=[],
)
```

它把同一个查询拆成两种用途：

- `original_query`：交给 Dense 检索器做语义向量化；
- `keywords`：交给 Sparse 检索器查询 BM25；
- `filters`：限定文档范围；
- `expanded_terms`：为未来查询扩展预留，当前主流程没有填充。

### 4.2 `RetrievalResult`

Dense、Sparse、Fusion 和 Rerank 最终都使用同一种结果对象：

```python
RetrievalResult(
    chunk_id="a1b2c3d4_0001_9f31a5c8",
    score=0.85,
    text="Azure OpenAI 的配置步骤……",
    metadata={
        "source_path": "docs/azure-guide.pdf",
        "chunk_index": 1,
        "title": "Azure 配置",
        "page_num": 3,
    },
)
```

统一契约的价值是：上游检索方法可以不同，下游 Fusion、Rerank 和 ResponseBuilder 不需要分别适配每一种结果格式。

需要注意，`score` 的含义会随阶段变化：

```text
Dense 阶段：向量库返回的相似度
Sparse 阶段：BM25 分数
Fusion 阶段：RRF 分数
Rerank 阶段：Reranker 给出的新分数
```

所以不能脱离阶段，把所有 `score` 都理解成同一种“相关度概率”。

### 4.3 `MCPToolResponse`

ResponseBuilder 最终生成：

```python
MCPToolResponse(
    content="Markdown 检索结果",
    citations=[...],
    metadata={...},
    is_empty=False,
    image_contents=[...],
)
```

随后再转换成 MCP 的文本和图片内容块。

---

## 5. MCP 工具入口

入口文件为：

```text
src/mcp_server/tools/query_knowledge_hub.py
```

### 5.1 工具定义

工具名称：

```text
query_knowledge_hub
```

输入参数：

```text
query：必填，用户查询
top_k：可选，最终最多返回多少条，Schema 范围为 1 到 20
collection：可选，查询哪个知识集合
```

`register_tool()` 将名称、描述、输入 Schema 和 Handler 注册给 MCP 协议处理器。

### 5.2 `query_knowledge_hub_handler()`

Handler 的职责主要是协议适配：

```python
tool = get_tool_instance()
response = await tool.execute(...)
content_blocks = response.to_mcp_content()
return types.CallToolResult(content=content_blocks, ...)
```

它不直接执行检索算法，而是：

1. 获取模块级工具实例；
2. 调用 `execute()`；
3. 把业务响应转换成 MCP 内容块；
4. 将参数错误和内部异常转换成 `isError=True` 的 MCP 结果。

### 5.3 模块级单例

```python
_tool_instance: Optional[QueryKnowledgeHubTool] = None
```

`get_tool_instance()` 第一次调用时创建实例，后续重复使用。这样 Embedding、Reranker 和搜索组件不需要每次请求都从头创建。

---

## 6. `QueryKnowledgeHubTool`：查询总编排器

`QueryKnowledgeHubTool` 在 Query-time 中的角色，与 `IngestionPipeline` 在 Index-time 中的角色相似：它负责组织组件，不亲自实现全部算法。

### 6.1 初始化状态

构造函数保存：

```text
settings
config
hybrid_search
reranker
embedding_client
response_builder
initialized
current_collection
```

其中：

- Embedding Client 和 Reranker 可以跨查询复用；
- Vector Store、Retriever 和 HybridSearch 与 collection 绑定；
- collection 变化时，需要重新组装对应搜索组件。

### 6.2 `settings` 属性

```python
@property
def settings(self):
    if self._settings is None:
        self._settings = load_settings()
```

配置采用延迟加载，只有真正需要时才读取。

---

## 7. `_ensure_initialized()`：按 collection 组装查询组件

这个函数的核心思想是“分层缓存”。

### 7.1 同 collection 快速返回

```python
if self._initialized and self._current_collection == collection:
    return
```

如果组件已经为当前 collection 初始化，就直接复用。

### 7.2 完全缓存的组件

```text
Embedding Client
CoreReranker
Settings
```

这些组件不依赖当前 collection，因此只创建一次。

### 7.3 collection 变化时重建的组件

```text
Vector Store
DenseRetriever
SparseRetriever
QueryProcessor
HybridSearch
```

向量库工厂接收：

```python
collection_name=collection
```

这样 Dense 查询会落到指定 Chroma collection。

BM25Indexer 使用：

```text
data/db/bm25/{collection}
```

SparseRetriever 的默认 collection 也被设置为同一个值。

### 7.4 为什么 BM25 每次查询重新加载

`SparseRetriever._ensure_index_loaded()` 每次检索都调用：

```python
self.bm25_indexer.load(collection=collection)
```

原因是 BM25 JSON 文件可能被其他进程更新。重新读取磁盘可以让已缓存的 SparseRetriever 看到较新的索引内容。

### 7.5 为什么使用 `asyncio.to_thread()`

`execute()` 是异步 MCP 方法，但组件初始化、Embedding API、ChromaDB、BM25 和 Rerank 都包含同步阻塞操作。

代码使用：

```python
await asyncio.to_thread(...)
```

把这些操作放到工作线程，避免阻塞 MCP 的异步事件循环和 stdio 传输。

---

## 8. `execute()`：工具层完整流程

`execute()` 的实际步骤是：

```text
校验 query
→ 计算 effective_top_k
→ 计算 effective_collection
→ 创建 query TraceContext
→ 初始化组件
→ 执行 Hybrid Search
→ 可选 Rerank
→ ResponseBuilder.build()
→ 保存 final_results 到 Trace
→ TraceCollector.collect()
→ 返回 MCPToolResponse
```

### 8.1 查询参数处理

空查询会直接抛出 `ValueError`。

`top_k` 会限制到工具配置的最大值：

```python
effective_top_k = min(
    top_k or default_top_k,
    max_top_k,
)
```

没有传 collection 时使用工具默认值 `default`。

### 8.2 创建查询 Trace

```python
trace = TraceContext(trace_type="query")
```

基础元数据包括：

```text
query
top_k
collection
source="mcp"
```

后续查询处理、Dense、Sparse、Fusion、Rerank 和最终结果都会记录到同一个 Trace 中。

### 8.3 `_perform_search()` 为什么扩大候选集

如果启用 Rerank：

```python
initial_top_k = top_k * 2
```

例如最终需要 5 条，Hybrid Search 会先返回最多 10 条候选，再交给更精确但更昂贵的 Reranker 重新判断，最后截取 5 条。

背后的思路是：

```text
召回阶段：先保证正确答案尽量进入候选集
重排阶段：再提高前几名的准确性
```

### 8.4 `execute()` 的终点

搜索和重排结束后直接调用：

```python
response = self._response_builder.build(
    results=results,
    query=query,
    collection=effective_collection,
)
```

这里没有将结果拼成问答 Prompt，也没有调用回答模型生成综合答案。

---

## 9. QueryProcessor：把一个问题拆成多路检索输入

核心调用：

```python
processed_query = query_processor.process(query)
```

完整流程：

```text
原始查询
→ _normalize()
→ _extract_filters()
→ _tokenize()
→ _filter_keywords()
→ ProcessedQuery
```

### 9.1 `_normalize()`

```python
normalized = " ".join(query.split())
```

它主要去除首尾空白，并把多个空白字符合并成单个空格。

### 9.2 `_extract_filters()`

支持在查询文本中写：

```text
collection:docs Azure 配置
type:pdf 报销制度
source:policy.pdf 住宿标准
tag:财务,制度 报销
```

正则识别 `key:value`，并转换为：

```python
filters = {
    "collection": "docs",
    "doc_type": "pdf",
    "source_path": "policy.pdf",
    "tags": ["财务", "制度"],
}
```

过滤语法会从用于检索的正文中删除，避免 `collection:docs` 本身参与语义和关键词检索。

未识别的 key 会作为通用精确匹配过滤条件保留。

### 9.3 `_tokenize()`

英文、数字、下划线和连字符使用正则提取；连续中文使用 2-gram：

```text
如何配置
→ 如何、何配、配置
```

这与 Index-time 的 SparseEncoder 中文分词策略保持一致。

如果建索引时使用 2-gram，而查询时使用完全不同的分词方式，同一段中文可能无法共享词项。

### 9.4 `_filter_keywords()`

它依次执行：

```text
大小写不敏感去重
→ 去除中英文停用词
→ 检查最小长度
→ 保留原始大小写
→ 限制最多 20 个关键词
```

输出关键词交给 BM25；原始完整查询仍交给 Dense 检索。

---

## 10. HybridSearch：混合检索核心编排器

`HybridSearch.search()` 的完整逻辑：

```text
校验 query
→ QueryProcessor.process()
→ 合并过滤条件
→ 并行运行 Dense 和 Sparse
→ 处理单路失败
→ RRF 融合
→ 元数据后置过滤
→ 截取 top_k
→ RetrievalResult[]
```

### 10.1 配置

当前配置为：

```yaml
retrieval:
  dense_top_k: 20
  sparse_top_k: 20
  fusion_top_k: 10
  rrf_k: 60
```

含义是每一路最多取 20 个候选，默认融合输出 10 个，RRF 平滑常数为 60。

当工具层显式向 `search()` 传入 `top_k` 时，它会覆盖 HybridSearch 的默认 `fusion_top_k`，但 Dense 与 Sparse 各自的候选数量仍读取配置中的 20。

### 10.2 合并过滤条件

```python
merged = query_filters.copy()
merged.update(explicit_filters)
```

显式参数优先于从查询文本中提取的过滤条件。

当前 `QueryKnowledgeHubTool._perform_search()` 传入 `filters=None`，所以主 MCP 链中主要使用 QueryProcessor 从查询文本里提取的过滤条件。

### 10.3 是否运行某一路

Dense 运行条件：

```text
enable_dense=True
并且 DenseRetriever 存在
```

Sparse 运行条件还要求：

```text
processed_query.keywords 非空
```

如果查询处理后没有关键词，Sparse 不运行，但 Dense 仍可使用完整原始问题进行语义检索。

### 10.4 并行召回

当两路都可用并且 `parallel_retrieval=True` 时，使用两个线程同时执行：

```text
Thread 1：DenseRetriever.retrieve()
Thread 2：SparseRetriever.retrieve()
```

这两路相互独立，可以并发，从而降低整体查询延迟。

### 10.5 Graceful Degradation

HybridSearch 对两路失败分别处理：

```text
Dense 失败、Sparse 成功
→ 只返回 Sparse 结果

Sparse 失败、Dense 成功
→ 只返回 Dense 结果

两路都失败
→ 抛出 RuntimeError

两路都成功但为空
→ 返回空列表
```

这体现了混合检索除了提升召回率，也能提供一定可用性冗余。

---

## 11. DenseRetriever：语义召回

核心流程：

```text
原始查询字符串
→ Embedding
→ 查询向量
→ Vector Store 相似度搜索
→ RetrievalResult[]
```

### 11.1 为什么使用原始查询

DenseRetriever 接收 `processed_query.original_query`，而不是关键词列表。

例如：

```text
用户问题：住宿费用最高可以报销多少？
文档内容：差旅住宿标准为每人每天 500 元。
```

词面不完全相同，但 Embedding 可以尝试捕捉整体语义接近。

### 11.2 `retrieve()` 的步骤

1. 校验 query 是非空字符串；
2. 校验 Embedding Client 和 Vector Store 已注入；
3. 调用 `embedding_client.embed([query])`；
4. 取第一条查询向量；
5. 调用 `vector_store.query()`；
6. 将后端字典统一转换成 `RetrievalResult`。

### 11.3 Index-time 与 Query-time 模型必须一致

查询向量必须与库中 Chunk 向量处于同一向量空间。

如果 Index-time 和 Query-time 使用不同且不兼容的 Embedding 模型，即使向量维度碰巧一致，距离也不再具有可靠语义。

### 11.4 `_transform_results()`

向量库原始结果需要包含：

```text
id
score
text
metadata
```

单条结果格式异常时会记录 warning 并跳过，不让一条坏记录阻断全部 Dense 结果。

---

## 12. SparseRetriever：BM25 关键词召回

SparseRetriever 的数据流与 Dense 不同：

```text
关键词列表
→ BM25 倒排索引
→ chunk_id + BM25 score
→ Vector Store.get_by_ids()
→ 补回正文和元数据
→ RetrievalResult[]
```

### 12.1 为什么还要访问向量库

BM25 索引主要保存词项、Chunk ID、词频和文档长度，并不承担完整正文和元数据存储。

因此 BM25 先找 ID，再通过统一 ID 从向量库取回正文和来源信息。

这再次说明 Index-time 中 Dense 与 Sparse ID 对齐的重要性。

### 12.2 `retrieve()` 的步骤

1. 校验关键词列表和依赖；
2. 确定 top_k 和 collection；
3. 每次从磁盘重新加载指定 BM25 索引；
4. 调用 `bm25_indexer.query()`；
5. 提取所有 `chunk_id`；
6. 调用 `vector_store.get_by_ids()`；
7. 将 BM25 分数与正文、元数据合并。

### 12.3 Sparse 的优势

Sparse 更适合：

```text
ERR-1042
API 名称
产品型号
法规编号
金额和日期
文档中的固定术语
```

这些内容即使语义向量表现不稳定，也能通过字符重合精确命中。

### 12.4 没有 BM25 索引时

`_ensure_index_loaded()` 返回 False，SparseRetriever 返回空列表，而不是直接让整个查询失败。

Dense 路仍然可以继续工作。

---

## 13. 为什么不能直接比较 Dense 和 BM25 分数

Dense 分数和 BM25 分数来自不同计算体系：

```text
Dense：向量相似度或距离转换值
BM25：TF、IDF 和文档长度归一化后的词项分数
```

例如：

```text
Dense 第一名：0.82
BM25 第一名：7.35
```

不能因为 7.35 大于 0.82，就认为 BM25 第一名一定更相关。

这就是项目使用 RRF 的原因：不比较原始分数，只比较每条结果在各自列表中的名次。

---

## 14. RRFFusion：多路结果融合

RRF 公式：

$$
RRF(d)=\sum_i\frac{1}{k+rank_i(d)}
$$

其中：

- $d$：某个 Chunk；
- $i$：第几路检索结果；
- $rank_i(d)$：该 Chunk 在第 $i$ 路中的排名，从 1 开始；
- $k$：平滑常数，当前为 60。

### 14.1 示例

假设：

```text
Dense 排名：A、B、C
Sparse 排名：B、D、A
```

那么：

```text
A = 1/(60+1) + 1/(60+3)
B = 1/(60+2) + 1/(60+1)
C = 1/(60+3)
D = 1/(60+2)
```

A 和 B 同时得到两路贡献，通常会高于只在一路出现的 C、D。

### 14.2 `fuse()` 的实现步骤

1. 过滤空排名列表；
2. 遍历每一路结果及其排名；
3. 按 `chunk_id` 累加 RRF 贡献；
4. 保存每个 Chunk 第一次出现时的正文和元数据；
5. 创建新的 `RetrievalResult`；
6. 按 RRF 分数降序排序；
7. 分数相同时按 `chunk_id` 排序，保证结果稳定；
8. 截取 top_k。

### 14.3 RRF 的逻辑价值

RRF 不要求知道 Dense 和 BM25 分数如何归一化，只利用排名位置和多路共识。

它更信任：

> 同一个 Chunk 同时被多种检索方式排在前面。

### 14.4 只有一路结果时

HybridSearch 不执行真正融合，直接保留该路原顺序并截取 top_k。

### 14.5 没有 Fusion 组件时

代码使用 Dense、Sparse 轮流交错的回退策略，并通过 `chunk_id` 去重。

---

## 15. 元数据过滤

HybridSearch 会在融合后再执行一次过滤，以弥补底层存储过滤能力差异。

支持逻辑包括：

```text
collection：精确匹配 collection 或 source_collection
doc_type：精确匹配
tags：结果标签与查询标签有交集
source_path：路径包含匹配
其他字段：精确匹配
```

所有过滤条件之间是 AND 关系：一个结果必须满足全部条件。

过滤发生在融合结果已经截取到 `effective_top_k` 之后的阶段附近，过滤掉结果后不会再次向 Dense 或 Sparse 补取候选，因此最终数量可能少于 top_k。

---

## 16. Rerank：对小候选集做更精细判断

召回与重排的目标不同：

```text
召回：正确 Chunk 尽量不要漏
重排：真正能回答问题的 Chunk 尽量排在前面
```

当前配置：

```yaml
rerank:
  enabled: true
  provider: "llm"
  model: "mimo-v2.5-pro"
  top_k: 5
```

### 16.1 `CoreReranker`

CoreReranker 是业务层适配器：

```text
RetrievalResult[]
→ candidate dict[]
→ 具体 Reranker Backend
→ reranked candidate[]
→ RetrievalResult[]
```

它负责：

- 根据配置通过 Factory 创建 LLM、CrossEncoder、API 或 None Reranker；
- 转换统一结果格式；
- 保留原分数并写入新分数；
- Rerank 失败时回退到原顺序。

### 16.2 候选格式转换

输入 Reranker 的候选：

```python
{
    "id": result.chunk_id,
    "text": result.text,
    "score": result.score,
    "metadata": result.metadata,
}
```

Rerank 成功后，新结果 metadata 增加：

```text
original_score
rerank_score
reranked=True
```

### 16.3 LLMReranker 做什么

LLM Reranker 把 Query 和所有候选 Passage 放入重排 Prompt，要求模型返回 JSON：

```json
[
  {"passage_id": "chunk_b", "score": 0.95},
  {"passage_id": "chunk_a", "score": 0.72}
]
```

代码验证：

- 输出必须是 JSON 数组；
- 每项必须包含 `passage_id`；
- 每项必须包含数值型 `score`。

然后根据 score 降序排序并映射回原候选。

### 16.4 LLM Rerank 不等于 LLM 生成答案

LLM 在这里的任务是：

```text
判断哪些 Passage 与 Query 更相关
```

它的输出是候选 ID 和分数，不是面向用户的综合答案。

所以项目虽然可能在 Rerank 阶段调用 LLM，也不能据此判断它已经完成 Retrieval-Augmented Generation 的 Generation 部分。

### 16.5 Rerank 失败回退

如果后端调用或 JSON 解析失败，并且 `fallback_on_error=True`：

```text
保留 Fusion 原顺序
→ 截取最终 top_k
→ metadata 标记 rerank_fallback=True
```

工具层 `_apply_rerank()` 还有一层异常保护：CoreReranker 整体抛错时，也会返回原始排序的前 top_k。

---

## 17. CitationGenerator：将结果转换成引用

每条 RetrievalResult 会生成一条 Citation：

```python
Citation(
    index=1,
    chunk_id="...",
    source="docs/guide.pdf",
    page=3,
    score=0.91,
    text_snippet="……",
    metadata={...},
)
```

生成逻辑：

1. 按最终结果顺序，从 1 开始编号；
2. 从 `source_path` 提取来源；
3. 从 `page` 或 `page_num` 提取页码；
4. 清理文本空白并截取摘要；
5. 选择 title、section、chunk_index、doc_type 等额外元数据。

引用编号只是最终结果顺序编号，不代表引用已经与某个“答案事实主张”完成对齐。

当前系统返回的是检索片段本身，因此引用可以指向每条片段；如果未来加入综合答案，还需要验证答案中的每个结论是否真的由相应引用支持。

---

## 18. MultimodalAssembler：把图片加入响应

它从 RetrievalResult 中寻找图片引用：

```text
优先读取 metadata.images
→ 没有结构化图片时解析正文 [IMAGE: id]
```

然后尝试解析图片路径：

```text
metadata 中的显式 path
→ ImageStorage 查询
→ data/images/{collection}/{image_id}.{ext}
```

找到文件后：

```text
读取二进制
→ 判断 MIME 类型
→ Base64 编码
→ MCP ImageContent
```

多条结果引用同一图片时，会根据图片数据前缀哈希去重。

图片缺失或读取失败不会阻断文本响应。

---

## 19. ResponseBuilder：格式化结果，不生成答案

核心调用：

```python
response_builder.build(results, query, collection)
```

执行步骤：

```text
结果为空？
├── 是：构建“未找到相关结果”响应
└── 否：
    ├── CitationGenerator.generate()
    ├── _build_markdown_content()
    ├── _build_metadata()
    ├── MultimodalAssembler.assemble()
    └── MCPToolResponse
```

### 19.1 Markdown 内容

Markdown 中会展示：

```text
查询内容
结果数量
每条结果的相关度
来源
页码
最多 300 字符的正文片段
引用来源列表
```

默认正文最多展示 5 条结果。即使 citations 中包含更多结果，正文部分也可能只显示前 5 条，并提示还有多少条未显示。

### 19.2 空结果

没有结果时返回：

```text
未找到相关结果
查询内容
collection
更换关键词、检查摄取状态、扩大范围等提示
```

### 19.3 MCP 内容块

`MCPToolResponse.to_mcp_content()` 生成：

```text
第一个 TextContent：人类可读 Markdown
中间 ImageContent：可选图片
最后一个 TextContent：引用和 metadata 的 JSON
```

这样 MCP 客户端既能展示文本和图片，也能读取结构化引用数据。

### 19.4 为什么这不是答案生成

`ResponseBuilder` 的内容来自：

```text
遍历 RetrievalResult
→ 截取原文
→ 添加来源和编号
```

它没有执行：

```text
将 Query 和证据组装成回答 Prompt
→ 调用回答 LLM
→ 生成综合答案
→ 检查答案是否忠实于证据
```

因此它是检索响应构建器，不是问答生成器。

---

## 20. Query Trace：如何观察中间过程

一次查询创建一个 `TraceContext(trace_type="query")`。

主要阶段包括：

```text
initialization
query_processing
dense_retrieval
sparse_retrieval
fusion
rerank
```

并在 metadata 中保存最终结果。

Trace 可以回答：

```text
Query 被拆成了哪些关键词？
Dense 找到了哪些 Chunk？
Sparse 找到了哪些 Chunk？
Fusion 后排名如何变化？
Rerank 是否改变了前几名？
最终返回了哪些来源？
每个阶段耗时多少？
```

这对应 Day1 的核心方法：不要把所有失败都称为“大模型幻觉”，要先定位正确知识在哪一层丢失。

---

## 21. 用一个查询串起完整数据变化

假设用户调用：

```text
query="住宿费用最高可以报销多少？"
top_k=5
collection="policies"
```

### 21.1 工具层

```text
effective_top_k=5
effective_collection="policies"
```

### 21.2 QueryProcessor

```python
ProcessedQuery(
    original_query="住宿费用最高可以报销多少？",
    keywords=["住宿", "宿费", "费用", "用最", "最高", "高可", "可以", "以报", "报销", "销多", "多少"],
    filters={},
)
```

停用词过滤后，实际关键词可能更少。

### 21.3 Dense

完整问题被转换成查询向量，可能找到：

```text
1. 差旅住宿标准为每人每天 500 元
2. 出差人员住宿费按照城市等级执行
3. 交通费用报销规则
```

### 21.4 Sparse

BM25 使用“住宿”“费用”“最高”“报销”等词项，可能找到：

```text
1. 差旅住宿标准为每人每天 500 元
2. 住宿费用不得超过规定限额
3. 报销申请流程
```

### 21.5 RRF

第一条在两路中都靠前，因此获得两路排名贡献，通常排在融合结果前面。

### 21.6 Rerank

Reranker 比较问题与候选正文，判断“明确给出金额”的 Chunk 比“只描述报销流程”的 Chunk 更能回答当前问题。

### 21.7 ResponseBuilder

最终返回类似：

```text
结果 1
来源：travel-policy.pdf
页码：5
片段：差旅住宿标准为每人每天 500 元……

结果 2
……

引用来源
[1] travel-policy.pdf (p.5)
```

它不会自动生成一句“住宿费用最高可报销 500 元/天”的综合答案，而是把支持材料交给 MCP 客户端或上层 AI 使用。

---

## 22. 当前项目的真实能力边界

根据实际调用关系，当前主链已经完成：

```text
MCP 参数校验
查询预处理
Dense 语义召回
Sparse/BM25 关键词召回
RRF 融合
单路失败回退
可选 LLM/CrossEncoder/API 重排
元数据过滤
检索片段格式化
结构化引用
可选图片内容块
查询 Trace
```

当前主链没有完成：

```text
检索上下文选择后的问答 Prompt 组装
面向用户的 LLM 综合答案生成
证据不足时的回答约束
答案事实与引用逐条对齐
生成答案忠实性校验
```

所以最准确的定位是：

> 当前项目是模块化、可观测、可通过 MCP 调用的 RAG 检索基础设施。它可以为上层 AI 提供相关 Chunk、引用和图片，但主查询工具本身不是完整的 RAG 问答生成闭环。

---

## 23. 如何按层定位 Query-time 问题

### 23.1 正确 Chunk 在 Dense 和 Sparse 中都不存在

先检查：

```text
Index-time 是否摄取了正确文档
Chunk 是否包含完整答案
Embedding 是否兼容
BM25 分词是否与查询一致
召回 top_k 是否过小
```

### 23.2 Dense 没找到错误码，Sparse 找到了

这是两路检索互补的正常表现。错误码、型号和专有标识更适合 Sparse 精确匹配。

### 23.3 正确 Chunk 在某一路出现，但 Fusion 后靠后

检查：

```text
它在 Dense 和 Sparse 中各排第几？
是否只有一路命中？
其他 Chunk 是否得到两路共识？
RRF k 是否影响排名差异？
```

### 23.4 Fusion 排名正确，Rerank 后变差

检查：

```text
Reranker 输入是否包含足够上下文
Rerank Prompt 是否符合任务
模型是否正确返回所有 passage_id
Rerank score 和最终顺序
是否发生 fallback
```

### 23.5 最终片段正确，但没有综合答案

这不是生成失败，而是当前主链没有实现回答生成阶段。

### 23.6 结果正确，但来源或页码错误

检查 Index-time 是否正确保留 `source_path`、`page` 或 `page_num`，以及 Chunk 与原文的映射是否可靠。

---

## 24. 当前源码需要特别辨认的实际行为

以下结论来自当前代码的具体调用方式。

### 24.1 Search 异常可能表现为空结果

`QueryKnowledgeHubTool._perform_search()` 捕获 HybridSearch 异常后返回空列表。因此部分搜索错误会被 ResponseBuilder 表现成“未找到相关结果”，而不是工具错误响应。

### 24.2 `top_k` 在不同层含义不同

```text
工具 top_k：用户最终希望得到的数量
initial_top_k：启用 Rerank 时通常为工具 top_k 的两倍
dense_top_k：Dense 初始候选数，当前配置 20
sparse_top_k：Sparse 初始候选数，当前配置 20
```

这些值共同形成“宽召回、窄输出”的漏斗。

### 24.3 RRF score 被展示为百分比

Fusion 后 `RetrievalResult.score` 是 RRF 累积分数。ResponseBuilder 使用百分比格式展示 `citation.score`。

RRF 分数不是校准后的相关性概率，所以界面中的百分比应理解为当前排序分数的格式化展示，而不是“答案正确概率”。

### 24.4 Sparse 合并依赖 ID 查询顺序

`SparseRetriever._merge_results()` 使用 `zip(bm25_results, records)` 配对，因此它依赖 `vector_store.get_by_ids()` 按请求 ID 顺序返回对应记录，或对缺失记录保留占位关系。

### 24.5 collection 参数与查询内过滤语法不同

MCP 的 `collection` 参数用于选择哪套 Chroma 和 BM25 组件；查询文本中的 `collection:xxx` 会进入 QueryProcessor filters，并参与检索和融合后元数据过滤。两者不是同一条控制路径。

### 24.6 图片描述 metadata 的形态差异

Index-time 的 ImageCaptioner 将 `image_captions` 写成列表结构，而 MultimodalAssembler 在为 ImageReference 附加 caption 时主要按字典结构读取。图片描述通常已经写入 Chunk 正文，但结构化 caption 附加需要注意两端数据形态。

### 24.7 引用不等于事实支持验证

CitationGenerator 按检索结果生成来源和片段，没有验证某个综合答案中的事实主张，因为当前主链本身没有综合答案。

---

## 25. Query-time 最重要的逻辑思维

### 25.1 召回、融合、重排必须分开理解

```text
召回：从大规模知识库中找候选
融合：合并不同召回器的排名
重排：在小候选集中做更精细判断
```

它们解决的问题不同，不能统称为“搜索”。

### 25.2 不要跨阶段直接比较 score

Dense、BM25、RRF 和 Rerank 的分数语义不同。评价时首先要确认当前分数来自哪个阶段。

### 25.3 正确答案必须先进入候选集

Reranker 只能重新排列已经召回的候选，不能找回完全没有进入候选集的 Chunk。

所以：

```text
召回决定上限
重排改善前排质量
```

### 25.4 Query-time 上限受 Index-time 限制

```text
PDF 解析丢表格
→ Dense 和 Sparse 都看不到表格

规则被切成两个不完整 Chunk
→ 检索到任何一半都可能无法回答

来源和页码未保存
→ CitationGenerator 无法生成可信引用

Dense 与 Sparse ID 不一致
→ RRF 无法识别两路中的同一 Chunk
```

### 25.5 使用 LLM 不代表已经生成答案

必须看 LLM 的输入输出职责：

```text
LLM 输入 Query + Passages，输出 passage_id + score
→ 这是 Rerank

LLM 输入 Query + Context + 回答约束，输出面向用户的答案
→ 这才是 Generation
```

### 25.6 引用必须与系统输出类型一起评估

当前系统返回原始检索片段，引用与片段是一一对应的。如果未来返回综合答案，还需要新增事实主张与证据之间的忠实性验证。

---

## 26. 最终总结

当前项目 Query-time 可以概括为：

```text
MCP Query
→ 参数校验和组件初始化
→ QueryProcessor 生成原始查询、关键词和过滤条件
→ Dense 使用完整问题执行语义召回
→ Sparse 使用关键词执行 BM25 召回
→ RRF 使用排名而不是原始分数进行融合
→ Reranker 对较小候选集重新判断相关性
→ CitationGenerator 和 MultimodalAssembler 生成引用与图片
→ ResponseBuilder 返回检索片段和结构化 MCP 内容
```

最需要记住的三个结论：

1. Dense 与 Sparse 负责“尽量找全”，RRF 负责融合多路共识，Rerank 负责“前面尽量准确”；
2. Rerank 中调用 LLM，不等于系统已经完成基于证据的答案生成；
3. 当前 `query_knowledge_hub` 的最终输出是检索结果、引用和可选图片，不是经过忠实性校验的综合问答答案。

阅读 Query-time 源码时，应始终沿着以下问题追踪：

```text
原始 Query 被如何处理？
每一路召回分别找到了什么？
正确 Chunk 在哪一层进入或退出候选集？
Fusion 为什么改变了排序？
Rerank 是否保留了正确证据？
最终返回的是证据，还是基于证据生成的答案？
引用是否真的支持系统输出？
```
