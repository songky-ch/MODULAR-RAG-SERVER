"""Ragas-based evaluator for RAG quality assessment.

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?

Design Principles:
- Pluggable: Implements BaseEvaluator interface, swappable via factory.
- Config-Driven: LLM/Embedding backend read from settings.yaml.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not."""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    Ragas does NOT require ground-truth labels.  It uses an LLM to judge
    the quality of the generated answer against the retrieved context.

    Supported metrics:
        - faithfulness: Measures factual consistency with context.
        - answer_relevancy: Measures how relevant the answer is to the query.
        - context_precision: Measures relevance/ordering of retrieved chunks.

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (used to configure LLM backend).
            metrics: Metric names to compute. Defaults to all supported.
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── public API ────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings).
            generated_answer: The generated answer text. Required for Ragas.
            ground_truth: Ignored by Ragas (not needed for LLM-as-Judge).
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Dictionary mapping metric names to float scores (0.0 – 1.0).

        Raises:
            ValueError: If query/chunks are invalid or generated_answer is missing.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            result = self._run_ragas(query, contexts, generated_answer)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── private helpers ───────────────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """Execute Ragas collections metrics and return normalised scores.

        Ragas 0.4+ collections metrics use per-metric ``score()`` instead of
        the legacy ``evaluate()`` pipeline.  Each metric has its own signature:
        - Faithfulness / ContextPrecision: (user_input, response, retrieved_contexts)
        - AnswerRelevancy: (user_input, response)
        """
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
        )

        # Build LLM / Embedding wrappers from settings
        llm, embeddings = self._build_wrappers()

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == FAITHFULNESS:
                m = Faithfulness(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            elif metric_name == ANSWER_RELEVANCY:
                m = AnswerRelevancy(llm=llm, embeddings=embeddings)
                result = m.score(user_input=query, response=answer)
            elif metric_name == CONTEXT_PRECISION:
                m = ContextPrecisionWithoutReference(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            else:
                continue

            scores[metric_name] = float(result.value) if result.value is not None else 0.0

        return scores

    OPENAI_COMPATIBLE_DEFAULTS: Dict[str, str] = {
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }

    def _build_wrappers(self) -> tuple:
        """Build Ragas LLM and Embedding wrappers from project settings.

        Uses Ragas 0.4+ native API (InstructorLLM + OpenAIEmbeddings)
        instead of deprecated LangchainLLMWrapper.

        Supports azure, openai, and OpenAI-compatible providers
        (dashscope, deepseek, qwen, etc.) via base_url.

        Returns:
            Tuple of (llm_wrapper, embeddings_wrapper).
        """
        from openai import AsyncAzureOpenAI, AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        llm_client = self._create_async_client(self.settings.llm, "LLM")
        llm = llm_factory(self.settings.llm.model, client=llm_client, max_tokens=8192)

        emb_client = self._create_async_client(self.settings.embedding, "embedding")
        embeddings = OpenAIEmbeddings(model=self.settings.embedding.model, client=emb_client)

        return llm, embeddings

    def _create_async_client(self, cfg: Any, label: str) -> Any:
        """Create an AsyncOpenAI or AsyncAzureOpenAI client from a config section.

        Args:
            cfg: LLMSettings or EmbeddingSettings with provider, api_key, etc.
            label: Human-readable label for error messages.

        Returns:
            An async OpenAI-compatible client.
        """
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        provider = cfg.provider.lower()

        if provider == "azure":
            return AsyncAzureOpenAI(
                api_key=cfg.api_key,
                azure_endpoint=cfg.azure_endpoint,
                api_version=getattr(cfg, "api_version", None) or "2024-02-15-preview",
            )

        if provider == "openai":
            return AsyncOpenAI(api_key=cfg.api_key)

        base_url = getattr(cfg, "base_url", None)
        if not base_url:
            base_url = self.OPENAI_COMPATIBLE_DEFAULTS.get(provider)

        if base_url:
            return AsyncOpenAI(api_key=cfg.api_key, base_url=base_url)

        raise ValueError(
            f"Unsupported {label} provider for Ragas: '{provider}'. "
            f"Supported: azure, openai, {', '.join(sorted(self.OPENAI_COMPATIBLE_DEFAULTS))}. "
            "Or set base_url in settings for any OpenAI-compatible endpoint."
        )

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations.

        Args:
            chunks: List of chunk dicts, strings, or objects with .text.

        Returns:
            List of text strings.
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
