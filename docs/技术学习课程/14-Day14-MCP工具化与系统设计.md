# Day 14: MCP 工具化与系统综合设计

## 今日定位

内部检索能力要被 Agent 稳定调用, 需要清晰的工具描述、JSON Schema、协议错误和结构化结果。MCP 工具化是把技术能力变成系统边界的最后一步。

## 学习目标

- 理解 MCP Tool 的发现与调用模型;
- 掌握 JSON Schema、协议层与业务层分离;
- 理解异步入口与阻塞 I/O 隔离;
- 能把两周知识组合成可调用、可追踪、可评估的工具。

## 1. 技术背景

普通 Python 函数只有进程内调用者知道参数和语义。Agent 需要先发现工具:

```text
工具叫什么?
解决什么问题?
参数有哪些?
哪些必填?
结果和错误是什么?
```

MCP 使用标准协议暴露 Tool Schema 与调用入口, 让模型客户端不需要了解服务器内部实现。

## 2. 核心原理

### 2.1 Tool Schema

一个工具定义至少包含:

```text
name
description
inputSchema:
  type: object
  properties
  required
```

Schema 不只是校验器, 也是 Agent 选择工具和生成参数的说明书。描述模糊会直接降低调用正确率。

### 2.2 协议与业务分层

```text
MCP JSON-RPC
  → Protocol Handler
  → Tool Handler
  → Core Service
  → Infrastructure
```

- Protocol Handler 负责注册、发现、校验和协议错误;
- Tool Handler 负责业务参数默认值和响应转换;
- Core Service 负责检索策略;
- Infrastructure 负责模型、索引和存储。

协议层不应包含 RRF 或 Rerank 业务算法。

### 2.3 异步边界

MCP Server 运行在异步事件循环中。同步的模型 SDK 或本地检索若直接执行会阻塞整个循环。项目使用:

```text
await asyncio.to_thread(sync_function, ...)
```

把阻塞工作放到线程中, 保持服务器继续处理其他请求。它不会让单次计算变快, 只是避免阻塞事件循环。

### 2.4 结构化结果与引用

Agent 需要的不只是文本, 还包括来源、Chunk ID、分数、降级状态和多模态内容。结构化输出便于:

- 展示引用;
- 二次推理;
- 质量审计;
- 自动化测试。

## 3. 项目案例分析

### 3.1 源码阅读顺序

1. `src/mcp_server/protocol_handler.py`;
2. `src/mcp_server/server.py`;
3. `src/mcp_server/tools/query_knowledge_hub.py`;
4. `src/mcp_server/tools/list_collections.py`;
5. `src/mcp_server/tools/get_document_summary.py`;
6. `src/core/response/response_builder.py`;
7. `src/core/response/citation_generator.py`。

### 3.2 协议注册

`ProtocolHandler.register_tool()` 保存 name、description、input_schema 与 handler。`get_tool_schemas()` 为 `tools/list` 生成 MCP Tool 定义, `execute_tool()` 根据名称找到 Handler 并转换返回值或错误。

`create_mcp_server()` 把:

```text
server.list_tools
server.call_tool
```

绑定到 ProtocolHandler, 从而将协议生命周期与工具实现分离。

### 3.3 Query 工具主链

```text
query_knowledge_hub_handler
  → QueryKnowledgeHubTool.execute
  → lazy initialization by collection
  → asyncio.to_thread(_perform_search)
  → optional asyncio.to_thread(_apply_rerank)
  → ResponseBuilder
  → citations / text / image content
  → collect Trace
```

按 collection 懒初始化并缓存查询组件, 避免每次请求重复构造索引和模型对象。若依赖对象不是线程安全的, 缓存与并发仍需额外保护。

### 3.4 错误边界

Protocol Handler 将未知工具、无效参数和内部异常转换为协议可理解的结果, 同时避免泄漏内部堆栈。业务工具负责把检索失败转为面向调用者的错误内容, 两层错误语义不能混淆。

### 3.5 项目真实能力

`ResponseBuilder` 主要封装检索片段、引用和可能的图片内容。它不是一个完整的答案生成器。因此这个 MCP 工具更准确的定位是“知识检索工具”, 而不是已经完成事实综合的问答 Agent。

## 4. 两周知识综合

```text
数据契约
  → 分块和稳定 ID
  → Dense + BM25 双索引
  → 多路召回
  → RRF
  → Rerank
  → 降级状态
  → Trace
  → Golden Set 评估
  → MCP Tool Schema 与结构化响应
```

这条链体现三类核心能力:

- 算法能力: Chunking、Embedding、BM25、RRF、Rerank;
- 架构能力: Pipeline、Adapter、依赖倒置、幂等、故障隔离;
- 质量能力: Trace、离线评估、协议契约。

## 5. 通用工程经验

- Tool 名称和描述应表达单一、明确能力;
- Schema 尽量收紧类型、范围和必填字段;
- 协议错误、业务无结果和内部故障必须区分;
- 阻塞调用移出事件循环, 同时设置合理超时;
- 响应携带来源和降级信息, 让 Agent 能判断可信度。

## 6. 综合实践

把 Day 13 的检索器封装为独立工具:

```text
search_knowledge(
  query: string,
  top_k: integer,
  category?: string
)
```

返回:

```text
results[
  {
    chunk_id,
    text,
    score,
    source,
    retrieval_paths
  }
]
degraded
fallback_reasons
trace_id
```

验收要求:

1. 提供严格 JSON Schema;
2. 未知参数和非法 `top_k` 返回参数错误;
3. 阻塞检索不阻塞事件循环;
4. 单路召回失败仍可返回降级结果;
5. 每次调用生成 Trace;
6. 用 Golden Set 验证工具输出中的 Chunk ID;
7. 明确说明它返回证据, 不伪装成生成答案。

## 7. 结业验收问题

1. Tool Schema 为什么会影响 Agent 调用质量?
2. `asyncio.to_thread` 解决什么问题, 不解决什么问题?
3. 协议层和业务层分别处理哪些错误?
4. 为什么检索工具必须保留引用和降级状态?
5. 如果新增一个向量数据库, 哪些层应变化, 哪些层不应变化?
6. 如果答案质量下降, 如何沿 Trace 和评估数据定位阶段?

## 8. 结业成果

完成 Day 1 至 Day 14 后, 你应能:

- 阅读类似 RAG/搜索项目的核心源码;
- 解释从文档到可检索数据的完整技术链;
- 设计 Dense、Sparse、Fusion 和 Rerank 分层;
- 构建可插拔、可重试、可降级、可追踪的工程系统;
- 用离线评估证明修改带来的真实质量变化;
- 将内部能力封装为 Agent 可稳定调用的工具接口。
