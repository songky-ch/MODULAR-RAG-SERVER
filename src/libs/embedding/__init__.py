"""
Embedding Module.

This package provides embedding abstractions backed by LangChain:
- BaseEmbedding: abstract interface
- EmbeddingFactory: config-driven creation via LangChain
- LangChainEmbeddingAdapter: wraps LangChain Embeddings
"""

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.langchain_embedding import LangChainEmbeddingAdapter

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "LangChainEmbeddingAdapter",
]
