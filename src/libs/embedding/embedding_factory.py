"""Factory for creating Embedding provider instances.

Uses :class:`LCLLMFactory` to build a LangChain ``Embeddings`` object
and wraps it in :class:`LangChainEmbeddingAdapter` so that the rest of
the codebase can keep using the :class:`BaseEmbedding` interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Create a :class:`BaseEmbedding` backed by LangChain Embeddings.

    The factory delegates to ``LCLLMFactory.create_embedding(settings)``
    for the actual LangChain object construction, then wraps the result
    in a :class:`LangChainEmbeddingAdapter`.

    Example::

        settings = load_settings()
        embedding = EmbeddingFactory.create(settings)
        vectors = embedding.embed(["hello world"])
    """

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEmbedding:
        """Create an Embedding instance based on configuration.

        Args:
            settings: Application settings with ``embedding`` section.
            **override_kwargs: Forwarded to ``LCLLMFactory.create_embedding``.

        Returns:
            A :class:`BaseEmbedding` wrapping a LangChain ``Embeddings``.
        """
        from src.libs.lc_llm import LCLLMFactory
        from src.libs.embedding.langchain_embedding import LangChainEmbeddingAdapter

        lc_embeddings = LCLLMFactory.create_embedding(settings, **override_kwargs)

        dimensions = getattr(settings.embedding, "dimensions", None)

        logger.info(
            "EmbeddingFactory: created LangChain-backed embedding "
            "(provider=%s, model=%s, dimensions=%s)",
            settings.embedding.provider,
            settings.embedding.model,
            dimensions,
        )

        return LangChainEmbeddingAdapter(
            lc_embeddings=lc_embeddings,
            dimensions=dimensions,
        )

    @classmethod
    def list_providers(cls) -> list[str]:
        """List providers registered in the LangChain factory."""
        from src.libs.lc_llm import LCLLMFactory
        return LCLLMFactory.list_providers()
