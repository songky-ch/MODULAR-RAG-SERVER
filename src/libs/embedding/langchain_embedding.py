"""LangChain Embeddings adapter for BaseEmbedding interface.

Wraps a LangChain ``Embeddings`` instance so that it satisfies the
project's :class:`BaseEmbedding` contract.  Upper-layer code (pipeline,
dense_encoder, dense_retriever) continues to call ``embed(texts)``
without knowing LangChain is involved.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings

from src.libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class LangChainEmbeddingAdapter(BaseEmbedding):
    """Adapts a LangChain ``Embeddings`` object to the ``BaseEmbedding`` interface.

    Args:
        lc_embeddings: A ready-to-use LangChain ``Embeddings`` instance
            (e.g. ``OpenAIEmbeddings``, ``OllamaEmbeddings``).
        dimensions: Optional embedding dimension hint for callers that
            need it (e.g. vector-store initialization).
    """

    def __init__(
        self,
        lc_embeddings: Embeddings,
        dimensions: Optional[int] = None,
    ) -> None:
        self._lc = lc_embeddings
        self._dimensions = dimensions

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings by delegating to the LangChain instance.

        Args:
            texts: Batch of text strings.
            trace: Optional TraceContext (unused here; tracing is done
                via LangChain callbacks at the caller level).
            **kwargs: Ignored — kept for interface compatibility.

        Returns:
            List of embedding vectors, one per input text.
        """
        self.validate_texts(texts)
        return self._lc.embed_documents(texts)

    def get_dimension(self) -> Optional[int]:
        return self._dimensions

    @property
    def langchain_embeddings(self) -> Embeddings:
        """Expose the underlying LangChain Embeddings for direct use."""
        return self._lc
