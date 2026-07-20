"""Unit tests for Embedding Factory and Base Embedding.

Test Coverage:
- Factory pattern: provider registration, creation, and routing
- Configuration-driven instantiation
- Error handling for unknown/missing providers
- Validation logic in BaseEmbedding
"""

from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """Fake embedding provider for testing.
    
    Returns deterministic fake vectors for reproducible testing.
    """
    
    def __init__(self, settings: Any = None, dimension: int = 384, **kwargs: Any):
        """Initialize fake embedding provider.
        
        Args:
            settings: Optional settings (unused in fake).
            dimension: Vector dimension to return.
            **kwargs: Additional parameters (unused).
        """
        self.settings = settings
        self.dimension = dimension
        self.call_count = 0
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate fake embeddings."""
        self.validate_texts(texts)
        self.call_count += 1
        
        # Return deterministic fake vectors
        return [[float(i + j) for j in range(self.dimension)] for i in range(len(texts))]
    
    def get_dimension(self) -> int:
        """Return configured dimension."""
        return self.dimension


class TestBaseEmbedding:
    """Tests for BaseEmbedding abstract class."""
    
    def test_validate_texts_success(self):
        """Valid text list should pass validation."""
        embedding = FakeEmbedding()
        # Should not raise
        embedding.validate_texts(["hello", "world"])
    
    def test_validate_texts_empty_list(self):
        """Empty list should raise ValueError."""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError, match="cannot be empty"):
            embedding.validate_texts([])
    
    def test_validate_texts_non_string(self):
        """Non-string entries should raise ValueError."""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError, match="not a string"):
            embedding.validate_texts(["valid", 123, "text"])  # type: ignore
    
    def test_validate_texts_empty_string(self):
        """Empty or whitespace-only strings should raise ValueError."""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            embedding.validate_texts(["valid", "   ", "text"])
    
    def test_get_dimension_implemented(self):
        """FakeEmbedding should return configured dimension."""
        embedding = FakeEmbedding(dimension=512)
        assert embedding.get_dimension() == 512
    
    def test_get_dimension_not_implemented(self):
        """BaseEmbedding without override should raise NotImplementedError."""
        
        class IncompleteEmbedding(BaseEmbedding):
            def embed(self, texts: List[str], trace: Optional[Any] = None, **kwargs: Any) -> List[List[float]]:
                return [[0.0]]
        
        incomplete = IncompleteEmbedding()
        with pytest.raises(NotImplementedError, match="must implement get_dimension"):
            incomplete.get_dimension()


class TestFakeEmbedding:
    """Tests for FakeEmbedding provider implementation."""
    
    def test_embed_single_text(self):
        """Embedding single text should return one vector."""
        embedding = FakeEmbedding(dimension=3)
        result = embedding.embed(["hello"])
        
        assert len(result) == 1
        assert len(result[0]) == 3
        assert result[0] == [0.0, 1.0, 2.0]
    
    def test_embed_multiple_texts(self):
        """Embedding multiple texts should return matching number of vectors."""
        embedding = FakeEmbedding(dimension=2)
        result = embedding.embed(["hello", "world", "test"])
        
        assert len(result) == 3
        assert result[0] == [0.0, 1.0]
        assert result[1] == [1.0, 2.0]
        assert result[2] == [2.0, 3.0]
    
    def test_embed_increments_call_count(self):
        """Each embed call should increment the counter."""
        embedding = FakeEmbedding()
        assert embedding.call_count == 0
        
        embedding.embed(["test1"])
        assert embedding.call_count == 1
        
        embedding.embed(["test2", "test3"])
        assert embedding.call_count == 2
    
    def test_embed_validates_input(self):
        """embed() should call validate_texts and raise on invalid input."""
        embedding = FakeEmbedding()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embedding.embed([])
        
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            embedding.embed(["  "])


class TestEmbeddingFactory:
    """Tests for EmbeddingFactory (LangChain-backed)."""

    def test_create_delegates_to_lc_factory(self):
        """create() should delegate to LCLLMFactory.create_embedding."""
        from unittest.mock import patch

        mock_lc_emb = MagicMock()
        with patch(
            "src.libs.lc_llm.LCLLMFactory.create_embedding",
            return_value=mock_lc_emb,
        ):
            settings = MagicMock()
            settings.embedding.provider = "openai"
            settings.embedding.model = "text-embedding-3-small"
            settings.embedding.dimensions = 1536

            embedding = EmbeddingFactory.create(settings)

            from src.libs.embedding.langchain_embedding import LangChainEmbeddingAdapter
            assert isinstance(embedding, LangChainEmbeddingAdapter)
            assert embedding.get_dimension() == 1536

    def test_create_returns_base_embedding(self):
        """Returned object should satisfy BaseEmbedding interface."""
        from unittest.mock import patch

        mock_lc_emb = MagicMock()
        mock_lc_emb.embed_documents.return_value = [[0.1, 0.2]]
        with patch(
            "src.libs.lc_llm.LCLLMFactory.create_embedding",
            return_value=mock_lc_emb,
        ):
            settings = MagicMock()
            settings.embedding.provider = "openai"
            settings.embedding.model = "m"
            settings.embedding.dimensions = None

            embedding = EmbeddingFactory.create(settings)
            assert isinstance(embedding, BaseEmbedding)

            result = embedding.embed(["hello"])
            assert result == [[0.1, 0.2]]
            mock_lc_emb.embed_documents.assert_called_once_with(["hello"])

    def test_list_providers_delegates(self):
        """list_providers should delegate to LCLLMFactory."""
        from unittest.mock import patch

        with patch(
            "src.libs.lc_llm.LCLLMFactory.list_providers",
            return_value=["ollama", "openai"],
        ):
            providers = EmbeddingFactory.list_providers()
            assert providers == ["ollama", "openai"]
