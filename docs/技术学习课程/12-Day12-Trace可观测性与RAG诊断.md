# Day 12: Trace 可观测性与 RAG 诊断

## 今日定位

RAG 返回错误答案时, “结果不对”只是现象。Trace 要把一次请求拆成可观测阶段, 让工程师判断问题发生在查询处理、召回、融合、精排还是响应封装。

## 学习目标

- 区分 Logs、Metrics 和 Traces;
- 理解 Trace ID、Stage、耗时和快照;
- 掌握 RAG 分阶段诊断方法;
- 能为多阶段程序加入最小 Trace。

## 1. 技术背景

一次查询可能出现:

- 查询关键词提取错误;
- Dense 没召回正确 Chunk;
- Sparse 索引缺数据;
- RRF 把相关结果排低;
- Reranker 降级;
- 响应层丢失引用。

只有最终日志“查询失败”无法定位。分布式追踪的思想是用同一个关联 ID 串起一次请求中的全部阶段和数据摘要。

## 2. 核心原理

### 2.1 三类可观测信号

- Logs: 离散事件, 适合错误上下文;
- Metrics: 聚合数值, 适合趋势、告警和 SLO;
- Traces: 单次请求的因果路径, 适合定位慢和错在哪一段。

三者互补。Trace 定位某次异常请求, Metric 判断是否普遍退化, Log 查看具体错误。

### 2.2 Trace 数据模型

```text
trace_id
trace_type
start_time / end_time
metadata
stages:
  stage_name:
    start/end or duration
    input snapshot
    output snapshot
    status/error
```

阶段记录应保留诊断所需最小数据, 不能无控制地保存完整用户文档。

### 2.3 RAG 诊断顺序

```text
1. 正确 Chunk 是否存在于索引?
2. 是否进入单路召回候选?
3. 融合后排名如何变化?
4. Rerank 后是否被提升或压低?
5. 响应是否保留正文和来源?
```

先确认最早偏离预期的阶段, 不要直接调整最后一个参数。

## 3. 项目案例分析

### 3.1 核心文件

- `src/core/trace/trace_context.py`;
- `src/core/trace/trace_collector.py`;
- `src/core/trace/langchain_callback.py`;
- `src/core/query_engine/hybrid_search.py`;
- `src/mcp_server/tools/query_knowledge_hub.py`;
- `src/observability/` 下的日志、服务和 Dashboard 代码。

### 3.2 TraceContext

`TraceContext` 默认用 UUID 创建 `trace_id`, `trace_type` 区分 `query` 与 `ingestion`。核心方法:

- `record_stage()`: 写入阶段数据;
- `finish()`: 结束生命周期;
- `elapsed_ms()`: 计算阶段或总耗时;
- `to_dict()`: 转为可存储和展示的结构;
- `get_stage_data()`: 读取某阶段快照。

### 3.3 查询链阶段

`QueryKnowledgeHubTool.execute()` 创建 Query Trace, 写入 query、top_k、collection 和 source。随后:

```text
initialization
→ query_processing
→ dense_retrieval
→ sparse_retrieval
→ fusion
→ rerank
→ final_results
```

无论成功或异常, Trace 都应 finish 并交给 `TraceCollector`。这保证失败请求也有诊断证据。

### 3.4 当前记录方式的理解

项目的 `record_stage()` 更接近“阶段快照记录”, 不是完整 OpenTelemetry Span 树。它足以支撑单进程教学和 Dashboard, 但跨进程传播、上下游 Span 关系和采样策略仍需更成熟的追踪系统。

## 4. 通用工程经验

- 在入口创建 Trace ID, 沿调用链显式传递;
- 阶段名保持稳定, 否则 Dashboard 和指标无法聚合;
- 记录候选 ID、排名和数量, 不必记录全部正文;
- 对 query、文档、密钥等敏感信息截断或脱敏;
- 成功、降级和失败都要完成 Trace 生命周期。

## 5. 实践练习

给 Day 11 的双数据源程序增加 `TraceContext`:

```text
request_validation
source_a
source_b
merge
response
```

每阶段记录:

- `duration_ms`;
- `input_count`;
- `output_count`;
- `status`;
- `error_type`;
- `degraded`。

模拟三种故障, 写一个命令行诊断器回答:

1. 最慢阶段是什么?
2. 第一个失败阶段是什么?
3. 最终结果是否降级?
4. 哪个来源贡献了最终数据?

## 6. 验收问题

1. Log、Metric、Trace 各自回答什么问题?
2. 为什么失败请求也必须调用 `finish()`?
3. RAG 结果错误时应按什么顺序排查?
4. 为什么不应在 Trace 中保存完整敏感正文?
5. 项目 Trace 与完整分布式 Span 系统有什么差别?

## 7. 今日学习成果

完成本课后, 你应能设计阶段化 Trace, 用候选快照和延迟数据定位 RAG 质量与性能问题, 而不是盲目调参。
