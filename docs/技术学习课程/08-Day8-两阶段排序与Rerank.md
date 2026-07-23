# Day 8: 两阶段排序与 Rerank

## 今日定位

召回器要在大规模语料中快速找候选, 精排器要在少量候选中做更细的相关性判断。两阶段架构用计算成本换取更好的最终排序。

## 学习目标

- 区分 Bi-Encoder 召回与 Cross-Encoder 精排;
- 理解 Top-K 候选和 Top-N 输出的成本关系;
- 掌握 Reranker 适配、结果映射和失败回退;
- 能设计一个小型两阶段排序实验。

## 1. 技术背景

Bi-Encoder 独立编码查询和文档, 文档向量可以预计算, 因而适合大规模召回。但它无法直接建模查询词与候选文本之间的逐 Token 交互。

Cross-Encoder 将查询与候选拼成一对:

```text
[query, candidate] → model → relevance score
```

判断更细, 但每个查询都要对每个候选推理, 无法对全库逐项执行。

## 2. 核心原理

### 2.1 两阶段漏斗

```text
百万级语料
  → 快速召回 Top-50
  → Rerank Top-50
  → 返回 Top-5
```

最终质量上限受召回限制。如果正确答案未进入 Top-50, Reranker 无法找回它。

### 2.2 三类 Reranker

- Cross-Encoder: 本地或托管判别模型, 稳定且专注相关性;
- API Reranker: 供应商提供的排序服务, 接入简单但受网络和配额影响;
- LLM Reranker: 用提示词让生成模型排序, 灵活但成本、延迟和确定性较差。

### 2.3 身份保持

精排后分数和顺序会变化, 但候选 ID、原文、元数据和来源必须保持。Reranker 不应无意改写检索事实。

## 3. 项目案例分析

### 3.1 源码阅读顺序

1. `src/libs/reranker/base_reranker.py`;
2. `src/libs/reranker/cross_encoder_reranker.py`;
3. `src/libs/reranker/api_reranker.py`;
4. `src/libs/reranker/llm_reranker.py`;
5. `src/libs/reranker/dashscope_reranker.py`;
6. `src/libs/reranker/reranker_factory.py`;
7. `src/core/query_engine/reranker.py`。

### 3.2 Core 与 Backend 的分工

`CoreReranker` 负责:

- 从配置判断是否启用;
- 将 `RetrievalResult` 转为后端候选格式;
- 调用具体 Reranker;
- 把结果映射回领域对象;
- 处理异常和回退;
- 写入 Trace。

`libs.reranker` 负责具体模型或 API 调用。

### 3.3 调用关系

```text
RetrievalResult[]
  → CoreReranker._results_to_candidates
  → BaseReranker.rerank(query, candidates, top_n)
  → candidate[] with relevance score
  → CoreReranker._candidates_to_results
  → RerankResult
```

映射时以 `chunk_id` 查回原始结果, 保留文本和元数据。后端失败且 `fallback_on_error=True` 时, 返回原始顺序, 元数据写入 `rerank_fallback=True`, 同时设置 `fallback_reason`。

### 3.4 实现边界

回退后的结果“可用”不代表已经精排。调用方必须检查 `used_fallback`, 监控也应把回退率作为质量风险指标。

## 4. 通用工程经验

- 先提高 Recall@候选数, 再优化 Reranker;
- 候选数越大, 精排成本近似线性增加;
- Rerank score 不应冒充原召回分数;
- 失败回退要保留原序, 避免制造随机顺序;
- 评估时分别报告召回前后指标, 才能定位收益来源。

## 5. 实践练习

构造 20 个候选句子:

1. 用关键词重合或向量相似度召回 Top-10;
2. 对每个候选计算联合特征:
   - 查询词覆盖率;
   - 连续短语命中;
   - 否定词冲突;
3. 重新排序并取 Top-3;
4. 比较召回排序和精排排序的 MRR;
5. 模拟精排函数抛错, 验证原顺序回退和状态标记。

重点不是训练模型, 而是理解候选漏斗、接口转换和故障语义。

## 6. 验收问题

1. 为什么不能对全库运行 Cross-Encoder?
2. Reranker 能否补救召回阶段完全漏掉的答案?
3. Top-K 和 Top-N 分别表示什么?
4. 为什么精排后仍要保留原始候选 ID 和元数据?
5. 回退结果为什么必须显式标记?

## 7. 今日学习成果

完成本课后, 你应能设计召回—精排两阶段系统, 评估它的成本与收益, 并实现具有身份保持和回退语义的 Rerank 编排层。
