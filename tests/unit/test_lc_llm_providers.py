"""Unit tests for LangChain provider configuration mappers.

Test Coverage:
- BaseProviderConfig: dependency check, abstract enforcement
- OpenAIProviderConfig: LLM / Vision / Embedding kwargs building
- OllamaProviderConfig: LLM / Vision / Embedding kwargs building
- Settings field mapping, override precedence, env-var fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch

import pytest

from src.libs.lc_llm.base_provider import BaseProviderConfig
from src.libs.lc_llm.ollama_provider import OllamaProviderConfig
from src.libs.lc_llm.openai_provider import OpenAIProviderConfig


# -----------------------------------------------------------------------------
# Mock Settings (lightweight dataclasses)
# -----------------------------------------------------------------------------


@dataclass
class MockLLMCfg:
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.5
    max_tokens: int = 2048
    api_key: Optional[str] = "sk-settings"
    base_url: Optional[str] = "https://api.example.com/v1"
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None


@dataclass
class MockVisionCfg:
    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4o-vision"
    max_image_size: int = 2048
    api_key: Optional[str] = "sk-vision"
    base_url: Optional[str] = None
    endpoint: Optional[str] = "https://vision.example.com/v1"
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None


@dataclass
class MockEmbeddingCfg:
    provider: str = "openai"
    model: str = "text-embedding-3"
    dimensions: int = 1024
    api_key: Optional[str] = "sk-emb"
    base_url: Optional[str] = "https://emb.example.com/v1"
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None


@dataclass
class MockSettings:
    llm: MockLLMCfg
    embedding: MockEmbeddingCfg
    vision_llm: Optional[MockVisionCfg] = None


def make_settings(**overrides: Any) -> MockSettings:
    """Create MockSettings with optional overrides."""
    return MockSettings(
        llm=MockLLMCfg(**{k: v for k, v in overrides.items() if k in MockLLMCfg.__dataclass_fields__}),
        embedding=MockEmbeddingCfg(),
        vision_llm=MockVisionCfg(),
    )


# -----------------------------------------------------------------------------
# BaseProviderConfig Tests
# -----------------------------------------------------------------------------


class TestBaseProviderConfig:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_directly(self):
        """BaseProviderConfig is abstract and should not be instantiable."""
        with pytest.raises(TypeError):
            BaseProviderConfig()  # type: ignore

    def test_check_dependency_installed(self):
        """check_dependency should pass for an installed package."""
        config = OpenAIProviderConfig()
        config.check_dependency()  # langchain-openai is installed; should not raise

    def test_check_dependency_missing(self):
        """check_dependency should raise ImportError with pip hint."""

        class MissingPkgConfig(BaseProviderConfig):
            def get_model_provider(self) -> str:
                return "missing"

            def get_required_package(self) -> str:
                return "langchain-nonexistent-provider"

            def build_llm_kwargs(self, settings: Any, **overrides: Any) -> dict[str, Any]:
                return {"model": "x"}

        config = MissingPkgConfig()

        with pytest.raises(ImportError) as exc_info:
            config.check_dependency()

        msg = str(exc_info.value)
        assert "pip install langchain-nonexistent-provider" in msg
        assert "missing" in msg

    def test_build_vision_kwargs_default_raises(self):
        """Default build_vision_kwargs should raise NotImplementedError."""

        class MinimalConfig(BaseProviderConfig):
            def get_model_provider(self) -> str:
                return "minimal"

            def get_required_package(self) -> str:
                return "langchain-openai"

            def build_llm_kwargs(self, settings: Any, **overrides: Any) -> dict[str, Any]:
                return {"model": "x"}

        config = MinimalConfig()

        with pytest.raises(NotImplementedError, match="does not support Vision LLM"):
            config.build_vision_kwargs(None)  # type: ignore

    def test_build_embedding_kwargs_default_raises(self):
        """Default build_embedding_kwargs should raise NotImplementedError."""

        class MinimalConfig(BaseProviderConfig):
            def get_model_provider(self) -> str:
                return "minimal"

            def get_required_package(self) -> str:
                return "langchain-openai"

            def build_llm_kwargs(self, settings: Any, **overrides: Any) -> dict[str, Any]:
                return {"model": "x"}

        config = MinimalConfig()

        with pytest.raises(NotImplementedError, match="does not support Embedding"):
            config.build_embedding_kwargs(None)  # type: ignore


# -----------------------------------------------------------------------------
# OpenAIProviderConfig Tests
# -----------------------------------------------------------------------------


class TestOpenAIProviderConfig:
    """Tests for OpenAI/compatible endpoint config mapper."""

    def setup_method(self):
        self.config = OpenAIProviderConfig()

    def test_get_model_provider(self):
        assert self.config.get_model_provider() == "openai"

    def test_get_required_package(self):
        assert self.config.get_required_package() == "langchain-openai"

    # -- LLM kwargs --------------------------------------------------------

    def test_build_llm_kwargs_from_settings(self):
        """Should read model, temperature, max_tokens, api_key, base_url."""
        settings = MockSettings(
            llm=MockLLMCfg(),
            embedding=MockEmbeddingCfg(),
        )
        kwargs = self.config.build_llm_kwargs(settings)

        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 2048
        assert kwargs["api_key"] == "sk-settings"
        assert kwargs["base_url"] == "https://api.example.com/v1"

    def test_build_llm_kwargs_overrides(self):
        """Overrides should take precedence over settings."""
        settings = MockSettings(llm=MockLLMCfg(), embedding=MockEmbeddingCfg())
        kwargs = self.config.build_llm_kwargs(
            settings,
            model="override-model",
            temperature=0.9,
            max_tokens=512,
            api_key="sk-override",
            base_url="https://override.com/v1",
        )

        assert kwargs["model"] == "override-model"
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 512
        assert kwargs["api_key"] == "sk-override"
        assert kwargs["base_url"] == "https://override.com/v1"

    def test_build_llm_kwargs_env_fallback(self):
        """api_key should fall back to OPENAI_API_KEY env var."""
        settings = MockSettings(
            llm=MockLLMCfg(api_key=None),
            embedding=MockEmbeddingCfg(),
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}):
            kwargs = self.config.build_llm_kwargs(settings)
        assert kwargs["api_key"] == "sk-env"

    def test_build_llm_kwargs_no_api_key(self):
        """When no api_key anywhere, 'api_key' should be absent."""
        settings = MockSettings(
            llm=MockLLMCfg(api_key=None, base_url=None),
            embedding=MockEmbeddingCfg(),
        )
        with patch.dict("os.environ", {}, clear=True):
            kwargs = self.config.build_llm_kwargs(settings)
        assert "api_key" not in kwargs

    def test_build_llm_kwargs_no_base_url(self):
        """When no base_url, 'base_url' should be absent (let LangChain default)."""
        settings = MockSettings(
            llm=MockLLMCfg(base_url=None),
            embedding=MockEmbeddingCfg(),
        )
        kwargs = self.config.build_llm_kwargs(settings)
        assert "base_url" not in kwargs

    # -- Vision kwargs -----------------------------------------------------

    def test_build_vision_kwargs_from_settings(self):
        """Should read model and api_key from settings.vision_llm."""
        settings = MockSettings(
            llm=MockLLMCfg(),
            embedding=MockEmbeddingCfg(),
            vision_llm=MockVisionCfg(),
        )
        kwargs = self.config.build_vision_kwargs(settings)

        assert kwargs["model"] == "gpt-4o-vision"
        assert kwargs["api_key"] == "sk-vision"
        assert kwargs["base_url"] == "https://vision.example.com/v1"

    def test_build_vision_kwargs_uses_endpoint_field(self):
        """Should fall back to 'endpoint' when 'base_url' is None."""
        settings = MockSettings(
            llm=MockLLMCfg(),
            embedding=MockEmbeddingCfg(),
            vision_llm=MockVisionCfg(base_url=None, endpoint="https://ep.com/v1"),
        )
        kwargs = self.config.build_vision_kwargs(settings)
        assert kwargs["base_url"] == "https://ep.com/v1"

    def test_build_vision_kwargs_override(self):
        """Overrides should win."""
        settings = MockSettings(
            llm=MockLLMCfg(),
            embedding=MockEmbeddingCfg(),
            vision_llm=MockVisionCfg(),
        )
        kwargs = self.config.build_vision_kwargs(settings, model="my-vl")
        assert kwargs["model"] == "my-vl"

    # -- Embedding kwargs --------------------------------------------------

    def test_build_embedding_kwargs_from_settings(self):
        """Should read model, api_key, base_url, dimensions."""
        settings = MockSettings(llm=MockLLMCfg(), embedding=MockEmbeddingCfg())
        kwargs = self.config.build_embedding_kwargs(settings)

        assert kwargs["model"] == "text-embedding-3"
        assert kwargs["api_key"] == "sk-emb"
        assert kwargs["base_url"] == "https://emb.example.com/v1"
        assert kwargs["dimensions"] == 1024
        assert kwargs["check_embedding_ctx_length"] is False

    def test_build_embedding_kwargs_no_base_url_no_ctx_flag(self):
        """Without custom base_url, check_embedding_ctx_length should be absent."""
        settings = MockSettings(
            llm=MockLLMCfg(),
            embedding=MockEmbeddingCfg(base_url=None),
        )
        kwargs = self.config.build_embedding_kwargs(settings)
        assert "check_embedding_ctx_length" not in kwargs

    def test_build_embedding_kwargs_override(self):
        """Overrides should win for embedding too."""
        settings = MockSettings(llm=MockLLMCfg(), embedding=MockEmbeddingCfg())
        kwargs = self.config.build_embedding_kwargs(settings, model="new-emb", dimensions=512)

        assert kwargs["model"] == "new-emb"
        assert kwargs["dimensions"] == 512


# -----------------------------------------------------------------------------
# OllamaProviderConfig Tests
# -----------------------------------------------------------------------------


class TestOllamaProviderConfig:
    """Tests for Ollama config mapper."""

    def setup_method(self):
        self.config = OllamaProviderConfig()

    def test_get_model_provider(self):
        assert self.config.get_model_provider() == "ollama"

    def test_get_required_package(self):
        assert self.config.get_required_package() == "langchain-ollama"

    # -- LLM kwargs --------------------------------------------------------

    def test_build_llm_kwargs_from_settings(self):
        """Should map max_tokens to num_predict and set base_url."""
        settings = MockSettings(
            llm=MockLLMCfg(model="llama3", temperature=0.7, max_tokens=4096, base_url=None),
            embedding=MockEmbeddingCfg(),
        )
        kwargs = self.config.build_llm_kwargs(settings)

        assert kwargs["model"] == "llama3"
        assert kwargs["temperature"] == 0.7
        assert kwargs["num_predict"] == 4096
        assert kwargs["base_url"] == "http://localhost:11434"

    def test_build_llm_kwargs_custom_base_url(self):
        """Custom base_url from settings should override default."""
        settings = MockSettings(
            llm=MockLLMCfg(base_url="http://192.168.1.100:11434"),
            embedding=MockEmbeddingCfg(),
        )
        kwargs = self.config.build_llm_kwargs(settings)
        assert kwargs["base_url"] == "http://192.168.1.100:11434"

    def test_build_llm_kwargs_override_base_url(self):
        """Override base_url should take highest precedence."""
        settings = MockSettings(
            llm=MockLLMCfg(base_url="http://from-settings:11434"),
            embedding=MockEmbeddingCfg(),
        )
        kwargs = self.config.build_llm_kwargs(settings, base_url="http://override:11434")
        assert kwargs["base_url"] == "http://override:11434"

    # -- Vision kwargs -----------------------------------------------------

    def test_build_vision_kwargs_from_settings(self):
        """Should read model from settings.vision_llm."""
        settings = MockSettings(
            llm=MockLLMCfg(),
            embedding=MockEmbeddingCfg(),
            vision_llm=MockVisionCfg(model="llava", provider="ollama"),
        )
        kwargs = self.config.build_vision_kwargs(settings)

        assert kwargs["model"] == "llava"
        assert kwargs["base_url"] == "http://localhost:11434"

    def test_build_vision_kwargs_custom_base_url(self):
        """Custom base_url from vision_llm settings."""
        settings = MockSettings(
            llm=MockLLMCfg(),
            embedding=MockEmbeddingCfg(),
            vision_llm=MockVisionCfg(base_url="http://gpu-server:11434"),
        )
        kwargs = self.config.build_vision_kwargs(settings)
        assert kwargs["base_url"] == "http://gpu-server:11434"

    # -- Embedding kwargs --------------------------------------------------

    def test_build_embedding_kwargs_from_settings(self):
        """Should read model and set default base_url."""
        settings = MockSettings(
            llm=MockLLMCfg(),
            embedding=MockEmbeddingCfg(model="nomic-embed-text", base_url=None),
        )
        kwargs = self.config.build_embedding_kwargs(settings)

        assert kwargs["model"] == "nomic-embed-text"
        assert kwargs["base_url"] == "http://localhost:11434"

    def test_build_embedding_kwargs_custom_base_url(self):
        """Custom base_url should override default."""
        settings = MockSettings(
            llm=MockLLMCfg(),
            embedding=MockEmbeddingCfg(base_url="http://remote:11434"),
        )
        kwargs = self.config.build_embedding_kwargs(settings)
        assert kwargs["base_url"] == "http://remote:11434"
