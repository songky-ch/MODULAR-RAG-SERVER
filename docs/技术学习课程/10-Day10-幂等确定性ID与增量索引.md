# Day 10: 幂等、确定性 ID 与增量索引

## 今日定位

摄取任务会重试、重跑和被中断。一个可靠索引系统必须保证同一输入重复处理不会无限制造副本, 内容变化后又能正确更新。

## 学习目标

- 理解幂等、内容寻址、业务主键和 Upsert;
- 掌握稳定 Document/Chunk ID 的设计;
- 理解成功标记时机与部分写入风险;
- 能实现最小增量索引器。

## 1. 技术背景

现实任务可能因网络超时、进程重启或人工重跑而重复执行。如果每次都随机生成 ID 并 Insert:

- 向量库产生重复 Chunk;
- 搜索结果重复;
- BM25 文档频率被污染;
- 无法准确删除旧版本。

幂等要求操作执行一次和执行多次的最终可观察状态一致。

## 2. 核心原理

### 2.1 内容哈希

SHA-256 将文件内容映射为固定摘要:

$$
h=SHA256(bytes)
$$

相同内容得到相同哈希, 内容改变几乎必然得到不同哈希。它适合判断“这一版本是否已成功处理”, 但文件路径移动而内容不变时仍会被视为相同内容, 是否合理取决于业务身份定义。

### 2.2 确定性 ID

项目在两个阶段生成可重复 ID:

```text
DocumentChunker:
{document_id}_{index}_{text_hash}

VectorUpserter:
{source_path_hash}_{index}_{text_hash}
```

同一来源、位置和内容产生相同 ID。内容变化后 ID 变化, 使新旧版本可区分。

### 2.3 Upsert

Upsert 的语义是:

```text
主键不存在 → Insert
主键已存在 → Update/Replace
```

它是幂等写入的重要组成部分, 但只靠 Upsert 不会自动删除“本次已不存在的旧 Chunk”。增量索引还要处理旧版本清理。

### 2.4 成功标记

正确顺序应接近:

```text
计算哈希
→ 判断是否已成功处理
→ 解析、分块、编码
→ 写入所有必要索引
→ 只有全部关键步骤成功后 mark_success
```

若在写入前标记成功, 中途失败后重试会被错误跳过。

## 3. 项目案例分析

### 3.1 源码阅读顺序

1. `src/libs/loader/file_integrity.py`;
2. `src/ingestion/chunking/document_chunker.py`;
3. `src/ingestion/storage/vector_upserter.py`;
4. `src/ingestion/pipeline.py` 中 `mark_success`;
5. `src/ingestion/storage/bm25_indexer.py` 中 `remove_document`;
6. `src/ingestion/document_manager.py`。

### 3.2 完整关系

```text
file bytes
  → compute_sha256
  → should_skip
  → Document / Chunk stable identity
  → vector upsert + BM25 build/update
  → mark_success(file_hash, path, collection)
```

`SQLiteIntegrityChecker` 持久化成功或失败状态。`mark_success` 使用可重复写入语义; 测试还验证多次调用不会制造重复记录。

### 3.3 双索引一致性

项目同时维护向量索引和 BM25 索引。一次摄取若只成功写入其中一个, 查询两路看到的数据集合就不一致。这不是数据库事务能自动解决的跨存储问题。

可选工程策略:

- 先写临时版本, 全部完成后切换版本指针;
- 保存阶段状态并让重试从失败阶段继续;
- 使用 Outbox/任务日志驱动最终一致;
- 重建小型本地 BM25 索引, 避免复杂增量合并。

### 3.4 重要边界

确定性 ID 解决重复写入, 不自动解决垃圾回收。文档缩短时, 旧版多出来的 Chunk ID 仍需显式删除。

## 4. 通用工程经验

- 先定义业务身份, 再选择哈希字段;
- 成功状态只能在所有关键副作用完成后提交;
- 幂等键应贯穿日志、索引、评估和引用;
- 删除与更新是增量索引的一等场景;
- 跨存储一致性优先采用可恢复工作流, 不要假装存在原子事务。

## 5. 实践练习

实现目录文本索引器, 持久化:

```text
file_path
content_hash
status
chunk_ids
updated_at
```

要求:

1. 同一文件连续运行三次, 索引记录数量不变;
2. 修改一段内容, 只更新受影响文件;
3. 删除一个段落, 旧 Chunk 不再可查询;
4. 模拟第二个索引写入失败, 不得写成功标记;
5. 下次运行能够恢复并最终一致。

## 6. 验收问题

1. Insert、Upsert 和幂等是什么关系?
2. 为什么随机 UUID 不适合可重复索引?
3. 成功标记提前会造成什么故障?
4. 确定性 ID 为什么不能自动清理旧 Chunk?
5. 双索引部分成功时有哪些恢复策略?

## 7. 今日学习成果

完成本课后, 你应能为可重试数据任务设计身份、状态和提交边界, 并能识别“看似幂等、实际残留旧数据”的陷阱。
