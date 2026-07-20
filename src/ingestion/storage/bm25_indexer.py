"""用于构建和查询倒排索引的 BM25 索引器。

This module implements the BM25 indexing component, responsible for:
- Computing IDF (Inverse Document Frequency) scores
- Building inverted index structures
- Persisting and loading indexes from disk
- Supporting incremental updates

Design Principles:
- Idempotent: Rebuild produces same results for same input
- Observable: Accepts TraceContext for future integration
- Persistent: Indexes saved to data/db/bm25/ directory
- Deterministic: Same corpus produces same IDF scores
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class BM25Indexer:
    """构建和查询 BM25 倒排索引。
    
    This indexer receives term statistics from SparseEncoder and constructs
    a queryable BM25 index with IDF scores and posting lists.
    
    Index Structure:
        {
            "metadata": {
                "num_docs": int,
                "avg_doc_length": float,
                "total_terms": int
            },
            "index": {
                "term": {
                    "idf": float,
                    "df": int,  # document frequency
                    "postings": [
                        {"chunk_id": str, "tf": int, "doc_length": int},
                        ...
                    ]
                },
                ...
            }
        }
    
    BM25 IDF Formula:
        IDF(term) = log((N - df + 0.5) / (df + 0.5))
        
        Where:
        - N = total number of documents
        - df = document frequency (number of docs containing term)
    
    Example:
        >>> indexer = BM25Indexer(index_dir="data/db/bm25")
        >>> 
        >>> # Build index from SparseEncoder output
        >>> term_stats = [
        ...     {"chunk_id": "1", "term_frequencies": {"hello": 2, "world": 1}, "doc_length": 3},
        ...     {"chunk_id": "2", "term_frequencies": {"hello": 1, "python": 1}, "doc_length": 2}
        ... ]
        >>> indexer.build(term_stats)
        >>> 
        >>> # Query the index
        >>> results = indexer.query(["hello"], top_k=2)
        >>> len(results) <= 2  # True
    """
    
    def __init__(
        self,
        index_dir: str = "data/db/bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """初始化 BM25Indexer。
        
        Args:
            index_dir: Directory to store index files (default: data/db/bm25)
            k1: BM25 term frequency saturation parameter (default: 1.5)
            b: BM25 length normalization parameter (default: 0.75)
        
        Raises:
            ValueError: If k1 or b are out of valid ranges
        """
        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b}")
        
        self.index_dir = Path(index_dir)
        self.k1 = k1
        self.b = b
        
        # 内存索引结构
        self._index: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {}
        
    def build(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> None:
        """根据词项统计构建 BM25 索引。
        
        This method:
        1. Calculates corpus-level statistics (N, avg_doc_length, DF)
        2. Computes IDF for each term
        3. Builds inverted index with posting lists
        4. Persists to disk
        
        Args:
            term_stats: List of statistics from SparseEncoder.encode()
                Each item should have: chunk_id, term_frequencies, doc_length
            collection: Collection name for organizing indexes (default: "default")
            trace: Optional TraceContext for observability
        
        Raises:
            ValueError: If term_stats is empty or has invalid structure
        
        Example:
            >>> term_stats = [
            ...     {
            ...         "chunk_id": "doc1_chunk0",
            ...         "term_frequencies": {"machine": 2, "learning": 1},
            ...         "doc_length": 3
            ...     }
            ... ]
            >>> indexer.build(term_stats, collection="my_docs")
        """
        if not term_stats:
            raise ValueError("Cannot build index from empty term_stats")
        
        # 校验数据结构
        self._validate_term_stats(term_stats)
        
        # 步骤 1: 计算语料库级统计信息
        num_docs = len(term_stats)
        total_length = sum(stat["doc_length"] for stat in term_stats)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        
        # 计算每个词项的文档频率（DF）
        doc_freq: Dict[str, int] = {}
        for stat in term_stats:
            for term in stat["term_frequencies"].keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        # 步骤 2: 使用 IDF 构建倒排索引
        index: Dict[str, Dict[str, Any]] = {}
        
        for term, df in doc_freq.items():
            # 使用 BM25 公式计算 IDF
            idf = self._calculate_idf(num_docs, df)
            
            # 为当前词项构建倒排记录列表
            postings = []
            for stat in term_stats:
                tf = stat["term_frequencies"].get(term, 0)
                if tf > 0:  # Only include docs that contain this term
                    postings.append({
                        "chunk_id": stat["chunk_id"],
                        "tf": tf,
                        "doc_length": stat["doc_length"]
                    })
            
            index[term] = {
                "idf": idf,
                "df": df,
                "postings": postings
            }
        
        # 步骤 3: 存储元数据
        self._metadata = {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "total_terms": len(index),
            "collection": collection,
        }
        
        self._index = index
        
        # 步骤 4: 持久化到磁盘
        self._save(collection)
    
    def load(
        self,
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> bool:
        """从磁盘加载索引。
        
        Args:
            collection: Collection name to load
            trace: Optional TraceContext for observability
        
        Returns:
            True if index loaded successfully, False if not found
        
        Raises:
            ValueError: If index file is corrupted
        """
        index_path = self._get_index_path(collection)
        
        if not index_path.exists():
            return False
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 校验数据结构
            if "metadata" not in data or "index" not in data:
                raise ValueError(f"Invalid index file structure: missing metadata or index")
            
            self._metadata = data["metadata"]
            self._index = data["index"]
            
            return True
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted index file at {index_path}: {e}")
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """使用 BM25 评分查询索引。
        
        Args:
            query_terms: List of terms to search for
            top_k: Maximum number of results to return
            trace: Optional TraceContext for observability
        
        Returns:
            List of results sorted by BM25 score (descending).
            Each result: {"chunk_id": str, "score": float}
        
        Raises:
            ValueError: If index not loaded or query_terms empty
        
        Example:
            >>> indexer.load("my_docs")
            >>> results = indexer.query(["machine", "learning"], top_k=5)
            >>> results[0]["score"] > 0  # True if matches found
        """
        if not self._index:
            raise ValueError("Index not loaded. Call load() or build() first.")
        
        if not query_terms:
            raise ValueError("query_terms cannot be empty")
        
        # 计算所有文档的 BM25 分数
        scores: Dict[str, float] = {}
        
        for term in query_terms:
            if term not in self._index:
                continue  # Term not in corpus, skip
            
            term_data = self._index[term]
            idf = term_data["idf"]
            
            for posting in term_data["postings"]:
                chunk_id = posting["chunk_id"]
                tf = posting["tf"]
                doc_length = posting["doc_length"]
                
                # 当前词项贡献的 BM25 分数
                term_score = self._calculate_bm25_score(
                    tf=tf,
                    doc_length=doc_length,
                    avg_doc_length=self._metadata["avg_doc_length"],
                    idf=idf
                )
                
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score
        
        # 按分数降序排序并返回 top_k
        sorted_results = sorted(
            [{"chunk_id": cid, "score": score} for cid, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def rebuild(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> None:
        """从头重建索引（语义更明确的 build 别名）。
        
        This is a convenience method that makes the intent clear when
        replacing an existing index.
        
        Args:
            term_stats: List of statistics from SparseEncoder
            collection: Collection name
            trace: Optional TraceContext for observability
        """
        self.build(term_stats, collection, trace)

    def remove_document(
        self,
        doc_id: str,
        collection: str = "default",
    ) -> bool:
        """从 BM25 索引中删除指定文档的全部倒排记录。

        Loads the index (if not already loaded), removes any postings
        whose ``chunk_id`` starts with *doc_id*, recalculates statistics,
        and re-saves the index.

        Args:
            doc_id: Document identifier (or prefix).  All postings whose
                ``chunk_id`` starts with this value are removed.
            collection: Collection name.

        Returns:
            ``True`` if any postings were removed, ``False`` otherwise.
        """
        if not self._index:
            if not self.load(collection):
                return False

        removed_any = False
        terms_to_delete: list[str] = []

        for term, term_data in self._index.items():
            original_len = len(term_data["postings"])
            term_data["postings"] = [
                p for p in term_data["postings"]
                if not p["chunk_id"].startswith(doc_id)
            ]
            if len(term_data["postings"]) < original_len:
                removed_any = True
            # 标记需要清理的空词项
            if not term_data["postings"]:
                terms_to_delete.append(term)
            else:
                term_data["df"] = len(term_data["postings"])

        # 删除空词项
        for term in terms_to_delete:
            del self._index[term]

        if removed_any:
            # 重新计算全局元数据
            all_chunk_ids: set[str] = set()
            total_length = 0
            for td in self._index.values():
                for p in td["postings"]:
                    all_chunk_ids.add(p["chunk_id"])
                    total_length += p["doc_length"]

            num_docs = len(all_chunk_ids)
            avg_doc_length = total_length / num_docs if num_docs else 0.0

            # 重新计算 IDF 值
            for td in self._index.values():
                td["idf"] = self._calculate_idf(num_docs, td["df"])

            self._metadata = {
                "num_docs": num_docs,
                "avg_doc_length": avg_doc_length,
                "total_terms": len(self._index),
                "collection": collection,
            }
            self._save(collection)

        return removed_any
    
    # ===== 私有辅助方法 =====
    
    def _calculate_idf(self, num_docs: int, df: int) -> float:
        """使用 BM25 公式计算 IDF。
        
        Formula: IDF(term) = log((N - df + 0.5) / (df + 0.5))
        
        Args:
            num_docs: Total number of documents in corpus
            df: Document frequency (number of docs containing term)
        
        Returns:
            IDF score (can be negative for very common terms)
        """
        return math.log((num_docs - df + 0.5) / (df + 0.5))
    
    def _calculate_bm25_score(
        self,
        tf: int,
        doc_length: int,
        avg_doc_length: float,
        idf: float
    ) -> float:
        """计算单个词项在文档中的 BM25 分数。
        
        Formula: score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
        
        Args:
            tf: Term frequency in document
            doc_length: Length of document (number of terms)
            avg_doc_length: Average document length in corpus
            idf: IDF score for this term
        
        Returns:
            BM25 score contribution
        """
        # 避免除零
        if avg_doc_length == 0:
            avg_doc_length = 1.0
        
        # BM25 公式
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
        
        return idf * (numerator / denominator)
    
    def _validate_term_stats(self, term_stats: List[Dict[str, Any]]) -> None:
        """校验 term_stats 数据结构。
        
        Raises:
            ValueError: If structure is invalid
        """
        for i, stat in enumerate(term_stats):
            if not isinstance(stat, dict):
                raise ValueError(f"term_stats[{i}] must be a dict, got {type(stat)}")
            
            required_fields = ["chunk_id", "term_frequencies", "doc_length"]
            for field in required_fields:
                if field not in stat:
                    raise ValueError(f"term_stats[{i}] missing required field: {field}")
            
            if not isinstance(stat["term_frequencies"], dict):
                raise ValueError(
                    f"term_stats[{i}]['term_frequencies'] must be dict, "
                    f"got {type(stat['term_frequencies'])}"
                )
            
            if not isinstance(stat["doc_length"], int) or stat["doc_length"] < 0:
                raise ValueError(
                    f"term_stats[{i}]['doc_length'] must be non-negative int, "
                    f"got {stat['doc_length']}"
                )
    
    def _get_index_path(self, collection: str) -> Path:
        """获取索引文件路径。
        
        Args:
            collection: Collection name
        
        Returns:
            Path to index file
        """
        return self.index_dir / f"{collection}_bm25.json"
    
    def _save(self, collection: str) -> None:
        """将索引保存到磁盘。
        
        Args:
            collection: Collection name
        """
        # 确保目录存在
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = self._get_index_path(collection)
        
        # 准备数据
        data = {
            "metadata": self._metadata,
            "index": self._index
        }
        
        # 原子写入（先写临时文件，再重命名）
        temp_path = index_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 原子重命名
            temp_path.replace(index_path)
            
        except Exception as e:
            # 写入失败时清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            raise
