"""
Vector Store Module.

This package provides vector store abstractions backed by LangChain:
- BaseVectorStore: abstract interface
- VectorStoreFactory: config-driven creation
- LangChainChromaStore: LangChain Chroma implementation
"""

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

try:
    from src.libs.vector_store.langchain_chroma_store import LangChainChromaStore
    VectorStoreFactory.register_provider('chroma', LangChainChromaStore)
except ImportError:
    pass

__all__ = [
    'BaseVectorStore',
    'VectorStoreFactory',
    'LangChainChromaStore',
]
