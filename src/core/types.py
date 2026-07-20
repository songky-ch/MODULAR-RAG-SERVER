"""整个流水线使用的核心数据类型与契约。

本模块定义各流水线阶段共用的基础数据结构:
- 文档入库（加载、转换、向量化、存储）
- 检索（查询引擎、搜索、重排）
- MCP 服务（工具、响应格式化）

设计原则:
- 集中式契约: 所有阶段共用这些类型，避免相互耦合
- 可序列化: 所有类型均支持字典/JSON 转换
- 可扩展元数据: 只规定最少必填字段，同时支持灵活扩展
- 类型安全: 提供完整类型提示，便于静态分析
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class Document:
    """表示从数据源加载的原始文档。

    这是加载器（例如 PDF 加载器）在文档切分前的输出。

    属性:
        id: 文档唯一标识（例如文件哈希或基于路径的 ID）
        text: 标准 Markdown 格式的文档内容。
              图片使用占位符表示: [IMAGE: {image_id}]
        metadata: 文档级元数据，包括:
            - source_path（必填）: 原始文件路径
            - doc_type: 文档类型（例如 'pdf'、'markdown'）
            - title: 提取或推断出的文档标题
            - page_count: 总页数（如适用）
            - images: 图片引用列表（参见下方图片字段规范）
            - 其他自定义元数据

    图片字段规范（metadata.images）:
        结构: List[{"id": str, "path": str, "page": int, "text_offset": int,
                        "text_length": int, "position": dict}]
        字段:
            - id: 图片唯一标识（格式: {doc_hash}_{page}_{seq}）
            - path: 图片文件存储路径（约定: data/images/{collection}/{image_id}.png）
            - page: 图片在原文档中的页码（可选，适用于 PDF 等分页文档）
            - text_offset: 占位符在 Document.text 中的起始字符位置（从 0 开始）
            - text_length: 占位符字符串长度（通常为 len("[IMAGE: {image_id}]"））
            - position: 图片在原文档中的物理位置信息（可选，例如 PDF 坐标、像素位置）
        说明: text_offset 和 text_length 用于精确定位占位符，支持同一图片多次出现的场景

    示例:
        >>> doc = Document(
        ...     id="doc_abc123",
        ...     text="# Title\\n\\nContent...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "doc_type": "pdf",
        ...         "title": "Annual Report 2025"
        ...     }
        ... )
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """校验必填元数据字段。"""
        if "source_path" not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """从字典创建 Document。"""
        return cls(**data)


@dataclass
class Chunk:
    """表示 Document 切分后得到的文本块。

    这是切分器的输出，也是转换流水线的输入。每个分块都保留对源文档的追踪信息。

    Attributes:
        id: Unique chunk identifier (e.g., hash-based or sequential)
        text: Chunk content (subset of original document text).
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Chunk-level metadata inherited and extended from Document:
            - source_path (required): Original file path
            - chunk_index: Sequential position in document (0-based)
            - start_offset: Character offset in original document (optional)
            - end_offset: Character offset in original document (optional)
            - source_ref: Reference to parent document ID (optional)
            - images: Subset of Document.images that fall within this chunk (optional)
            - Any document-level metadata propagated from Document
        start_offset: Starting character position in original document (optional)
        end_offset: Ending character position in original document (optional)
        source_ref: Reference to parent Document.id (optional)

    说明: 如果分块包含图片占位符，metadata.images 应只包含与当前文本范围相关的图片引用。

    Example:
        >>> chunk = Chunk(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "page": 1
        ...     },
        ...     start_offset=0,
        ...     end_offset=150
        ... )
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    source_ref: Optional[str] = None

    def __post_init__(self):
        """校验必填元数据字段。"""
        if "source_path" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建 Chunk。"""
        return cls(**data)


@dataclass
class ChunkRecord:
    """表示完成全部处理、可用于存储和检索的文本块。

    这是向量化流水线的输出，也是写入向量数据库的数据结构。
    它在 Chunk 基础上增加了向量表示。

    Attributes:
        id: Unique chunk identifier (must be stable for idempotent upsert)
        text: Chunk content (same as Chunk.text).
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Extended metadata including:
            - source_path (required): Original file path
            - chunk_index: Sequential position
            - All metadata from Chunk
            - images: Image references from Chunk (see Document.images specification)
            - Any enrichment from Transform pipeline (title, summary, tags)
            - image_captions: Dict[image_id, caption_text] if multimodal enrichment applied
        dense_vector: Dense embedding vector (e.g., from OpenAI, BGE)
        sparse_vector: Sparse vector for BM25/keyword matching (optional)

    说明: ImageCaptioner 生成的图片描述保存在 metadata.image_captions 中，
          其结构为 image_id 到描述文本的映射字典。

    Example:
        >>> record = ChunkRecord(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "title": "Introduction",
        ...         "summary": "Overview of project goals"
        ...     },
        ...     dense_vector=[0.1, 0.2, ..., 0.3],
        ...     sparse_vector={"word1": 0.5, "word2": 0.3}
        ... )
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """校验必填元数据字段。"""
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """从字典创建 ChunkRecord。"""
        return cls(**data)

    @classmethod
    def from_chunk(cls, chunk: Chunk, dense_vector: Optional[List[float]] = None,
                   sparse_vector: Optional[Dict[str, float]] = None) -> "ChunkRecord":
        """使用 Chunk 及其向量创建 ChunkRecord。

        参数:
            chunk: 源 Chunk 对象
            dense_vector: 稠密向量
            sparse_vector: 稀疏向量表示

        返回:
            已从 chunk 填充全部字段的 ChunkRecord
        """
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )


# 为便于使用而定义的类型别名
Metadata = Dict[str, Any]
Vector = List[float]
SparseVector = Dict[str, float]


@dataclass
class ProcessedQuery:
    """表示完成处理、可用于检索的查询。

    这是 QueryProcessor 的输出，包含为下游稠密/稀疏检索器提取的关键词和解析后的过滤条件。

    Attributes:
        original_query: The raw user query string
        keywords: List of extracted keywords after stopword removal
        filters: Dictionary of filter conditions (e.g., {"collection": "api-docs"})
        expanded_terms: Optional list of synonyms/expanded terms (for future use)

    Example:
        >>> pq = ProcessedQuery(
        ...     original_query="如何配置 Azure OpenAI？",
        ...     keywords=["配置", "Azure", "OpenAI"],
        ...     filters={"collection": "docs"}
        ... )
    """

    original_query: str
    keywords: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    expanded_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedQuery":
        """从字典创建 ProcessedQuery。"""
        return cls(**data)


@dataclass
class RetrievalResult:
    """表示稠密/稀疏检索器返回的一条检索结果。

    这是 DenseRetriever、SparseRetriever 和 HybridSearch 的输出，
    为不同搜索方式提供统一的检索结果契约。

    Attributes:
        chunk_id: Unique identifier for the retrieved chunk
        score: Relevance score (higher = more relevant, normalized to [0, 1])
        text: The actual text content of the retrieved chunk
        metadata: Associated metadata (source_path, chunk_index, title, etc.)

    Example:
        >>> result = RetrievalResult(
        ...     chunk_id="doc1_chunk_003",
        ...     score=0.85,
        ...     text="Azure OpenAI 配置步骤如下...",
        ...     metadata={
        ...         "source_path": "docs/azure-guide.pdf",
        ...         "chunk_index": 3,
        ...         "title": "Azure Configuration"
        ...     }
        ... )
    """

    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后校验字段。"""
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"score must be numeric, got {type(self.score).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """从字典创建 RetrievalResult。"""
        return cls(**data)
