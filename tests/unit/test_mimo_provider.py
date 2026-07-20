"""Tests for Xiaomi MiMo provider configuration."""

from types import SimpleNamespace

import pytest

from src.libs.lc_llm.mimo_provider import MiMoProviderConfig


def _settings():
    return SimpleNamespace(
        llm=SimpleNamespace(
            api_key=None,
            base_url=None,
            model="mimo-v2.5-pro",
            temperature=1.0,
            max_tokens=8192,
        ),
        vision_llm=SimpleNamespace(
            api_key=None,
            base_url=None,
            model="mimo-v2.5",
            timeout=120,
        ),
    )


def test_mimo_llm_uses_dedicated_environment(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "shared")
    monkeypatch.setenv("MIMO_LLM_API_KEY", "llm-key")
    monkeypatch.delenv("MIMO_BASE_URL", raising=False)

    kwargs = MiMoProviderConfig().build_llm_kwargs(_settings())

    assert kwargs["api_key"] == "llm-key"
    assert kwargs["base_url"] == "https://api.xiaomimimo.com/v1"
    assert kwargs["max_completion_tokens"] == 8192
    assert "max_tokens" not in kwargs


def test_mimo_vision_uses_shared_key(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "shared")

    kwargs = MiMoProviderConfig().build_vision_kwargs(_settings())

    assert kwargs["api_key"] == "shared"
    assert kwargs["model"] == "mimo-v2.5"


def test_mimo_requires_key(monkeypatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.delenv("MIMO_LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MIMO_API_KEY"):
        MiMoProviderConfig().build_llm_kwargs(_settings())


def test_mimo_embedding_is_not_supported():
    with pytest.raises(NotImplementedError, match="provider='ollama'"):
        MiMoProviderConfig().build_embedding_kwargs(_settings())
