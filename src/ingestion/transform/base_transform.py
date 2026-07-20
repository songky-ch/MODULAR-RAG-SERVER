"""分块转换操作的基类。"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """分块转换操作的抽象基类。
    
    Transform operations process chunks to enhance their quality, add metadata,
    or prepare them for downstream processing (embedding, indexing).
    
    Design Principles:
        - Single Responsibility: Each transform does ONE type of enhancement
        - Atomic Operations: Failure in one chunk doesn't affect others
        - Observable: Records processing info in TraceContext
        - Graceful Degradation: Returns original chunk on unrecoverable errors
    """
    
@abstractmethod
def transform(
    self,
    chunks: List[Chunk],
    trace: Optional[TraceContext] = None
) -> List[Chunk]:
    """转换分块列表。
    
    Args:
        chunks: List of chunks to transform
        trace: Optional trace context for observability
        
    Returns:
        List of transformed chunks (same length as input)
        
    Raises:
        ValueError: If input validation fails
    """
    pass

def _clone_chunk_with_metadata_updates(
    self,
    chunk: Chunk,
    metadata_updates: dict | None = None,
) -> Chunk:
    """创建已合并元数据的新 Chunk 的辅助方法。
    
    This avoids accidental in-place mutation of existing Chunk objects.
    """
    base_metadata = dict(chunk.metadata or {})
    if metadata_updates:
        base_metadata.update(metadata_updates)
    return Chunk(
        id=chunk.id,
        text=chunk.text,
        metadata=base_metadata,
        source_ref=chunk.source_ref,
    )
