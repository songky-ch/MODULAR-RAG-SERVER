"""用于从文本分块生成 BM25 词项统计的稀疏编码器。

This module implements the Sparse Encoder component of the Ingestion Pipeline,
responsible for extracting term statistics needed for BM25 indexing.

Design Principles:
- Stateless Processing: No internal state between encode() calls
- Observable: Accepts TraceContext for future observability integration
- Deterministic: Same inputs produce same term statistics
- Clear Contracts: Well-defined output structure for downstream BM25Indexer
"""

from typing import List, Dict, Optional, Any
from collections import Counter
import re
from src.core.types import Chunk


class SparseEncoder:
    """将文本分块编码为 BM25 词项统计。
    
    This encoder prepares term-level statistics needed for BM25 indexing.
    The actual index construction is handled by BM25Indexer (C12).
    
    Output Structure:
        For each chunk, produces:
        {
            "chunk_id": str,
            "term_frequencies": Dict[str, int],  # term -> count in this chunk
            "doc_length": int,                    # number of terms in chunk
            "unique_terms": int                   # vocabulary size in chunk
        }
    
    Design:
    - Tokenization: Simple whitespace + lowercasing (can be enhanced later)
    - Stop Words: None by default (can add in future iterations)
    - Deterministic: Same chunk text always produces same statistics
    
    Example:
        >>> from src.core.types import Chunk
        >>> encoder = SparseEncoder()
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world hello", metadata={})]
        >>> stats = encoder.encode(chunks)
        >>> stats[0]["term_frequencies"]["hello"]  # 2
        >>> stats[0]["doc_length"]  # 3
    """
    
    def __init__(
        self,
        min_term_length: int = 2,
        lowercase: bool = True,
    ):
        """初始化 SparseEncoder。
        
        Args:
            min_term_length: Minimum character length for a term (default: 2)
            lowercase: Whether to convert terms to lowercase (default: True)
        
        Raises:
            ValueError: If min_term_length < 1
        """
        if min_term_length < 1:
            raise ValueError(f"min_term_length must be >= 1, got {min_term_length}")
        
        self.min_term_length = min_term_length
        self.lowercase = lowercase
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """将分块编码为 BM25 词项统计。
        
        For each chunk, extracts:
        - Term frequencies (term -> count)
        - Document length (total terms)
        - Unique terms count
        
        Args:
            chunks: List of Chunk objects to encode
            trace: Optional TraceContext for observability (reserved for Stage F)
        
        Returns:
            List of statistics dictionaries (one per chunk, in same order).
            Each dict contains: chunk_id, term_frequencies, doc_length, unique_terms
        
        Raises:
            ValueError: If chunks list is empty
            ValueError: If any chunk has empty text
        
        Example:
            >>> chunks = [
            ...     Chunk(id="1", text="machine learning", metadata={}),
            ...     Chunk(id="2", text="deep learning networks", metadata={})
            ... ]
            >>> stats = encoder.encode(chunks)
            >>> len(stats) == len(chunks)  # True
            >>> stats[0]["term_frequencies"]["machine"]  # 1
            >>> stats[1]["doc_length"]  # 3
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        results = []
        
        for i, chunk in enumerate(chunks):
            # 校验分块文本
            if not chunk.text or not chunk.text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunk.id}) has empty or whitespace-only text"
                )
            
            # 分词并统计词项
            terms = self._tokenize(chunk.text)
            term_frequencies = Counter(terms)
            
            # 构建统计字典
            stat_dict = {
                "chunk_id": chunk.id,
                "term_frequencies": dict(term_frequencies),  # Convert Counter to dict
                "doc_length": len(terms),
                "unique_terms": len(term_frequencies),
            }
            
            results.append(stat_dict)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """将文本切分为词项。
        
        For English和数字，使用基于正则的 token（字母/数字/下划线/连字符）。
        对于连续中文片段，使用简单的 2-gram 分片，以便在缺少外部分词依赖的情况下，
        仍然对中文财报等长句有较好的 BM25 覆盖能力。
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of valid terms
        """
        # 正则：中文连续块、非中文块中的英文/数字 token
        chinese_re = re.compile(r'[\u4e00-\u9fff]+')
        alnum_re = re.compile(r'[A-Za-z0-9_-]+')

        tokens: List[str] = []

        last_end = 0
        for m in chinese_re.finditer(text):
            # 先处理前面的非中文部分
            non_zh = text[last_end:m.start()]
            if non_zh:
                tokens.extend(alnum_re.findall(non_zh))

            zh_block = m.group(0)
            # 对中文连续片段做 2-gram（例如 "中国市场" -> "中国", "国市", "市场"）
            if len(zh_block) == 1:
                tokens.append(zh_block)
            else:
                for i in range(len(zh_block) - 1):
                    tokens.append(zh_block[i : i + 2])

            last_end = m.end()

        # 处理最后一段非中文
        tail = text[last_end:]
        if tail:
            tokens.extend(alnum_re.findall(tail))

        # 按配置转换为小写（仅影响英文/数字 Token）
        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        # 按最小长度过滤
        terms = [t for t in tokens if len(t) >= self.min_term_length]

        return terms
    
    def get_corpus_stats(
        self,
        encoded_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """根据已编码分块计算语料库级统计信息。
        
        Utility method for BM25Indexer to compute:
        - Average document length
        - Document frequency (how many docs contain each term)
        - Total number of documents
        
        Args:
            encoded_chunks: List of statistics dicts from encode()
        
        Returns:
            Dictionary with corpus-level statistics:
            {
                "num_docs": int,
                "avg_doc_length": float,
                "document_frequency": Dict[str, int]  # term -> # docs containing it
            }
        """
        if not encoded_chunks:
            return {
                "num_docs": 0,
                "avg_doc_length": 0.0,
                "document_frequency": {}
            }
        
        num_docs = len(encoded_chunks)
        total_length = sum(chunk["doc_length"] for chunk in encoded_chunks)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        
        # 计算每个词项的文档频率（DF）
        doc_freq: Dict[str, int] = {}
        for chunk_stats in encoded_chunks:
            # 当前分块中的每个唯一词项为 DF 贡献 1
            for term in chunk_stats["term_frequencies"].keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        return {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "document_frequency": doc_freq,
        }
