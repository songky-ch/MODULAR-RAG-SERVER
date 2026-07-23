# Day 4: Embedding 与向量空间

## 今日定位

Embedding 把离散文本映射为连续向量, 使“意思接近但用词不同”的文本可以被检索。本课重点不是调用模型, 而是理解向量检索成立的条件和工程契约。

## 学习目标

- 理解文本向量、相似度和 Top-K 最近邻;
- 区分文档编码与查询编码;
- 掌握维度、归一化、批处理和模型一致性约束;
- 能判断向量检索适用和失效的场景。

## 1. 技术背景

关键词检索依赖字面重合。当用户问“怎样申请退款”, 文档写的是“取消订单后的资金退回流程”, 两者可能没有足够的共同词。Embedding 模型通过训练把语义相似文本放到向量空间中的邻近位置。

向量检索解决的是语义召回, 不是事实判断。相似只表示表达相关, 不保证内容正确、最新或足以回答问题。

## 2. 核心原理

### 2.1 向量与余弦相似度

文本 $x$ 经过编码器得到 $d$ 维向量:

$$
\mathbf{e}_x=f_\theta(x)\in\mathbb{R}^d
$$

余弦相似度衡量方向接近程度:

$$
\cos(\mathbf{a},\mathbf{b})=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|\|\mathbf{b}\|}
$$

若向量已做 L2 归一化, 余弦相似度等价于点积。

### 2.2 索引时与查询时必须同空间

```text
索引时: Chunk 文本 → 同一个 Embedding 模型 → 文档向量
查询时: Query 文本 → 同一个 Embedding 模型 → 查询向量
```

模型、维度或预处理方式改变后, 旧向量和新向量不再可比较。模型升级通常要求重建全部向量索引。

### 2.3 Bi-Encoder 的效率来源

文档向量可以离线计算并保存。在线请求只编码一次查询, 再做最近邻搜索。这种独立编码称为 Bi-Encoder。它很快, 但编码查询时看不到具体候选文档, 因而细粒度判断弱于 Cross-Encoder。

### 2.4 批处理

Embedding API 通常支持批量输入。批量过小浪费网络和模型吞吐, 批量过大可能触发 Token、内存或限流约束。批处理必须保持输入与输出顺序和数量一致。

## 3. 项目案例分析

### 3.1 源码阅读顺序

1. `src/libs/embedding/base_embedding.py`: Provider 统一接口;
2. `src/libs/embedding/embedding_factory.py`: 配置到实现的创建逻辑;
3. `src/ingestion/embedding/dense_encoder.py`: Chunk 批量编码;
4. `src/ingestion/storage/vector_upserter.py`: 文本、向量、元数据一并写入;
5. `src/core/query_engine/dense_retriever.py`: 查询编码与向量检索;
6. `src/libs/vector_store/base_vector_store.py`: 向量库契约。

### 3.2 调用关系

```text
索引链:
List[Chunk]
  → DenseEncoder.encode
  → BaseEmbedding.embed_documents
  → List[List[float]]
  → VectorUpserter.upsert

查询链:
query
  → DenseRetriever
  → BaseEmbedding.embed_query
  → VectorStore.query
  → List[RetrievalResult]
```

### 3.3 关键契约

`DenseEncoder.encode()` 负责按批次抽取 Chunk 文本并调用 Embedding 后端。输出向量的数量和顺序必须与输入 Chunk 一致, 否则后续 Upsert 会把向量绑定到错误文本。

`BaseEmbedding` 把外部模型差异隔离在 `libs` 层。核心查询代码依赖统一语义, 不应知道 DashScope、Azure 或其他 Provider 的请求格式。

### 3.4 实现边界

向量库返回的 score 语义可能是相似度, 也可能是距离。跨后端时不能只看字段名, 必须确认“越大越相关”还是“越小越相关”。统一接口应该把这种差异转换为稳定契约。

## 4. 通用工程经验

- 在索引元数据中记录模型名称、版本和向量维度;
- 文档和查询必须使用兼容编码方式;
- Provider 替换后做小规模语义回归, 不只检查接口成功;
- 批量编码前后验证数量, 必要时验证维度;
- 精确编号、人名缩写、罕见术语往往是 Dense 的弱项, 应与稀疏检索互补。

## 5. 实践练习

准备十句话, 包含:

- 两组同义表达;
- 两组关键词相同但语义不同的表达;
- 两条编号或专有名词查询。

用任意 Embedding 模型编码, 计算 $10\times10$ 余弦相似度矩阵。然后回答:

1. 同义表达是否互为近邻?
2. 关键词相同是否一定相似?
3. 编号查询是否能准确找到对应文本?
4. 对所有向量乘以不同正数后, 余弦排名是否变化?

再实现三个断言:

```text
len(vectors) == len(texts)
all(len(v) == expected_dimension)
query_vector_dimension == document_vector_dimension
```

## 6. 验收问题

1. 为什么模型升级后通常必须重建索引?
2. 余弦相似度与点积何时等价?
3. Bi-Encoder 为什么快, 又为什么不够精细?
4. 为什么向量相似不能证明事实正确?
5. Embedding 批处理最重要的数据契约是什么?

## 7. 今日学习成果

完成本课后, 你应能解释向量检索的数学基础、在线离线分工及模型一致性要求, 并能识别 Dense Retrieval 的能力边界。
