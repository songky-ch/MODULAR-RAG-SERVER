"""Unit tests for LangChain LLM Factory (LCLLMFactory).

Test Coverage:
- Factory registry: register, list, case-insensitive lookup
- LLM creation: success, overrides, unknown provider, provider override
- Vision LLM creation: success, missing config
- Embedding creation: success, overrides
- Provider resolution and dependency checking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.libs.lc_llm.base_provider import BaseProviderConfig
from src.libs.lc_llm.factory import LCLLMFactory


# -----------------------------------------------------------------------------
# Mock Settings
# -----------------------------------------------------------------------------


@dataclass
class MockLLMSettings:
    provider: str = "openai"
    model: str = "test-model"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: Optional[str] = "sk-test"
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None


@dataclass
class MockEmbeddingSettings:
    provider: str = "openai"
    model: str = "text-embedding-test"
    dimensions: int = 768
    api_key: Optional[str] = "sk-test"
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None


@dataclass
class MockVisionLLMSettings:
    enabled: bool = True
    provider: str = "openai"
    model: str = "test-vision-model"
    max_image_size: int = 2048
    api_key: Optional[str] = "sk-test"
    base_url: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None


@dataclass
class MockSettings:
    llm: MockLLMSettings = field(default_factory=MockLLMSettings)
    embedding: MockEmbeddingSettings = field(default_factory=MockEmbeddingSettings)
    vision_llm: Optional[MockVisionLLMSettings] = field(default_factory=MockVisionLLMSettings)


# -----------------------------------------------------------------------------
# Fake Provider for Testing
# -----------------------------------------------------------------------------


class FakeProviderConfig(BaseProviderConfig):
    """Minimal concrete provider config for testing the factory."""

    def get_model_provider(self) -> str:
        return "fake"

    def get_required_package(self) -> str:
        return "langchain-openai"  # already installed, so check_dependency passes

    def build_llm_kwargs(self, settings: Any, **overrides: Any) -> dict[str, Any]:
        return {
            "model": overrides.get("model", settings.llm.model),
            "temperature": overrides.get("temperature", settings.llm.temperature),
        }

    def build_vision_kwargs(self, settings: Any, **overrides: Any) -> dict[str, Any]:
        return {
            "model": overrides.get("model", settings.vision_llm.model),
        }

    def build_embedding_kwargs(self, settings: Any, **overrides: Any) -> dict[str, Any]:
        return {
            "model": overrides.get("model", settings.embedding.model),
        }


class LLMOnlyProviderConfig(BaseProviderConfig):
    """Provider that only supports text LLM (no vision / embedding)."""

    def get_model_provider(self) -> str:
        return "llm_only"

    def get_required_package(self) -> str:
        return "langchain-openai"

    def build_llm_kwargs(self, settings: Any, **overrides: Any) -> dict[str, Any]:
        return {"model": settings.llm.model}


# -----------------------------------------------------------------------------
# Factory Registration Tests
# -----------------------------------------------------------------------------


class TestLCLLMFactoryRegistration:
    """Tests for provider registry management."""

    def setup_method(self):
        """Clear provider registry before each test."""
        LCLLMFactory._PROVIDERS.clear()

    def test_register_success(self):
        """Registering a valid provider config should succeed."""
        LCLLMFactory.register("fake", FakeProviderConfig)
        assert "fake" in LCLLMFactory._PROVIDERS
        assert LCLLMFactory._PROVIDERS["fake"] is FakeProviderConfig

    def test_register_case_insensitive(self):
        """Provider names should be normalized to lowercase."""
        LCLLMFactory.register("OpenAI", FakeProviderConfig)
        assert "openai" in LCLLMFactory._PROVIDERS
        assert "OpenAI" not in LCLLMFactory._PROVIDERS

    def test_register_invalid_class(self):
        """Registering a non-BaseProviderConfig class should raise TypeError."""
        class NotAProvider:
            pass

        with pytest.raises(TypeError, match="not a BaseProviderConfig subclass"):
            LCLLMFactory.register("bad", NotAProvider)  # type: ignore

    def test_list_providers_empty(self):
        """list_providers should return empty list when none registered."""
        assert LCLLMFactory.list_providers() == []

    def test_list_providers_sorted(self):
        """list_providers should return sorted provider names."""
        LCLLMFactory.register("zebra", FakeProviderConfig)
        LCLLMFactory.register("alpha", FakeProviderConfig)
        LCLLMFactory.register("beta", FakeProviderConfig)
        assert LCLLMFactory.list_providers() == ["alpha", "beta", "zebra"]

    def test_register_overwrites_existing(self):
        """Re-registering the same name should overwrite silently."""
        LCLLMFactory.register("dup", FakeProviderConfig)
        LCLLMFactory.register("dup", LLMOnlyProviderConfig)
        assert LCLLMFactory._PROVIDERS["dup"] is LLMOnlyProviderConfig


# -----------------------------------------------------------------------------
# LLM Creation Tests
# -----------------------------------------------------------------------------


class TestLCLLMFactoryCreateLLM:
    """Tests for LCLLMFactory.create() — text LLM."""

    def setup_method(self):
        LCLLMFactory._PROVIDERS.clear()
        LCLLMFactory.register("fake", FakeProviderConfig)

    @patch("src.libs.lc_llm.factory.init_chat_model")
    def test_create_success(self, mock_init):
        """Should call init_chat_model with correct args and return its result."""
        mock_init.return_value = MagicMock()
        settings = MockSettings(llm=MockLLMSettings(provider="fake"))

        result = LCLLMFactory.create(settings)

        mock_init.assert_called_once_with(
            "test-model",
            model_provider="fake",
            temperature=0.0,
        )
        assert result is mock_init.return_value

    @patch("src.libs.lc_llm.factory.init_chat_model")
    def test_create_with_overrides(self, mock_init):
        """Overrides should take precedence over settings."""
        mock_init.return_value = MagicMock()
        settings = MockSettings(llm=MockLLMSettings(provider="fake"))

        LCLLMFactory.create(settings, model="override-model", temperature=0.9)

        mock_init.assert_called_once_with(
            "override-model",
            model_provider="fake",
            temperature=0.9,
        )

    @patch("src.libs.lc_llm.factory.init_chat_model")
    def test_create_provider_from_settings(self, mock_init):
        """Should read provider from settings.llm.provider."""
        mock_init.return_value = MagicMock()
        settings = MockSettings(llm=MockLLMSettings(provider="FAKE"))

        LCLLMFactory.create(settings)

        mock_init.assert_called_once()

    @patch("src.libs.lc_llm.factory.init_chat_model")
    def test_create_provider_override(self, mock_init):
        """Provider can be overridden via the 'provider' kwarg."""
        mock_init.return_value = MagicMock()
        LCLLMFactory.register("alt", FakeProviderConfig)
        settings = MockSettings(llm=MockLLMSettings(provider="fake"))

        LCLLMFactory.create(settings, provider="alt")

        mock_init.assert_called_once()

    def test_create_unknown_provider(self):
        """Should raise ValueError for an unregistered provider."""
        settings = MockSettings(llm=MockLLMSettings(provider="nonexistent"))

        with pytest.raises(ValueError) as exc_info:
            LCLLMFactory.create(settings)

        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg
        assert "Registered providers" in error_msg


# -----------------------------------------------------------------------------
# Vision LLM Creation Tests
# -----------------------------------------------------------------------------


class TestLCLLMFactoryCreateVisionLLM:
    """Tests for LCLLMFactory.create_vision_llm()."""

    def setup_method(self):
        LCLLMFactory._PROVIDERS.clear()
        LCLLMFactory.register("fake", FakeProviderConfig)

    @patch("src.libs.lc_llm.factory.init_chat_model")
    def test_create_vision_success(self, mock_init):
        """Should create vision LLM from settings.vision_llm."""
        mock_init.return_value = MagicMock()
        settings = MockSettings(
            vision_llm=MockVisionLLMSettings(provider="fake"),
        )

        result = LCLLMFactory.create_vision_llm(settings)

        mock_init.assert_called_once_with(
            "test-vision-model",
            model_provider="fake",
        )
        assert result is mock_init.return_value

    @patch("src.libs.lc_llm.factory.init_chat_model")
    def test_create_vision_with_override(self, mock_init):
        """Model override should take precedence."""
        mock_init.return_value = MagicMock()
        settings = MockSettings(
            vision_llm=MockVisionLLMSettings(provider="fake"),
        )

        LCLLMFactory.create_vision_llm(settings, model="gpt-4o")

        mock_init.assert_called_once_with(
            "gpt-4o",
            model_provider="fake",
        )

    def test_create_vision_missing_config(self):
        """Should raise ValueError when vision_llm is not configured."""
        settings = MockSettings(vision_llm=None)

        with pytest.raises(ValueError, match="Vision LLM is not configured"):
            LCLLMFactory.create_vision_llm(settings)

    def test_create_vision_unsupported_provider(self):
        """Provider that doesn't implement build_vision_kwargs should raise."""
        LCLLMFactory.register("llm_only", LLMOnlyProviderConfig)
        settings = MockSettings(
            vision_llm=MockVisionLLMSettings(provider="llm_only"),
        )

        with pytest.raises(NotImplementedError, match="does not support Vision LLM"):
            LCLLMFactory.create_vision_llm(settings)


# -----------------------------------------------------------------------------
# Embedding Creation Tests
# -----------------------------------------------------------------------------


class TestLCLLMFactoryCreateEmbedding:
    """Tests for LCLLMFactory.create_embedding()."""

    def setup_method(self):
        LCLLMFactory._PROVIDERS.clear()
        LCLLMFactory.register("fake", FakeProviderConfig)

    @patch("src.libs.lc_llm.factory.init_embeddings")
    def test_create_embedding_success(self, mock_init):
        """Should create embedding from settings.embedding."""
        mock_init.return_value = MagicMock()
        settings = MockSettings(
            embedding=MockEmbeddingSettings(provider="fake"),
        )

        result = LCLLMFactory.create_embedding(settings)

        mock_init.assert_called_once_with(
            "text-embedding-test",
            provider="fake",
        )
        assert result is mock_init.return_value

    @patch("src.libs.lc_llm.factory.init_embeddings")
    def test_create_embedding_with_override(self, mock_init):
        """Model override should take precedence."""
        mock_init.return_value = MagicMock()
        settings = MockSettings(
            embedding=MockEmbeddingSettings(provider="fake"),
        )

        LCLLMFactory.create_embedding(settings, model="new-emb")

        mock_init.assert_called_once_with(
            "new-emb",
            provider="fake",
        )

    @patch("src.libs.lc_llm.factory.init_embeddings")
    def test_create_embedding_provider_override(self, mock_init):
        """Provider can be overridden via the 'provider' kwarg."""
        mock_init.return_value = MagicMock()
        LCLLMFactory.register("alt", FakeProviderConfig)
        settings = MockSettings(embedding=MockEmbeddingSettings(provider="fake"))

        LCLLMFactory.create_embedding(settings, provider="alt")

        mock_init.assert_called_once()

    def test_create_embedding_unsupported_provider(self):
        """Provider that doesn't implement build_embedding_kwargs should raise."""
        LCLLMFactory.register("llm_only", LLMOnlyProviderConfig)
        settings = MockSettings(
            embedding=MockEmbeddingSettings(provider="llm_only"),
        )

        with pytest.raises(NotImplementedError, match="does not support Embedding"):
            LCLLMFactory.create_embedding(settings)

    def test_create_embedding_unknown_provider(self):
        """Should raise ValueError for an unregistered provider."""
        settings = MockSettings(embedding=MockEmbeddingSettings(provider="nope"))

        with pytest.raises(ValueError, match="nope"):
            LCLLMFactory.create_embedding(settings)
