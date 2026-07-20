"""DashScope SDK-based Reranker implementation.

Uses the native ``dashscope.TextReRank`` SDK to call Alibaba Cloud's
reranking service, avoiding manual HTTP endpoint construction.

Supported models include ``gte-rerank``, ``qwen3-rerank``, etc.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class DashScopeRerankError(RuntimeError):
    """Raised when a DashScope rerank SDK call fails."""


class DashScopeReranker(BaseReranker):
    """Reranker backed by the DashScope ``TextReRank`` SDK.

    Args:
        settings: Application settings with ``rerank`` section.
        api_key: Explicit API key (overrides settings / env var).
        **kwargs: Extra parameters forwarded to ``TextReRank.call()``.
    """

    ENV_VAR = "DASHSCOPE_API_KEY"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self._extra = kwargs

        rerank_cfg = getattr(settings, "rerank", None)

        self.model: str = (
            kwargs.get("model")
            or (getattr(rerank_cfg, "model", None) if rerank_cfg else None)
            or "gte-rerank"
        )
        self.api_key: str = (
            api_key
            or (getattr(rerank_cfg, "api_key", None) if rerank_cfg else None)
            or os.environ.get(self.ENV_VAR, "")
        )

        if not self.api_key:
            raise ValueError(
                f"DashScope API key not provided. Set {self.ENV_VAR} env var, "
                "pass api_key, or configure rerank.api_key in settings.yaml"
            )

        logger.info(
            "DashScopeReranker initialised: model=%s",
            self.model,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Call DashScope TextReRank SDK and return candidates sorted by relevance."""
        self.validate_query(query)
        self.validate_candidates(candidates)

        if len(candidates) == 1:
            return candidates

        documents = [
            c.get("text") or c.get("content", "") for c in candidates
        ]

        top_n = kwargs.get("top_n") or kwargs.get("top_k") or len(candidates)

        try:
            from dashscope import TextReRank

            response = TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
                api_key=self.api_key,
            )
        except ImportError as exc:
            raise DashScopeRerankError(
                "dashscope package is not installed. "
                "Install it with: pip install dashscope"
            ) from exc
        except Exception as exc:
            raise DashScopeRerankError(
                f"DashScope TextReRank SDK call failed: {exc}"
            ) from exc

        status_code = response.get("status_code", 0)
        if status_code != 200:
            message = response.get("message", "")
            code = response.get("code", "")
            raise DashScopeRerankError(
                f"DashScope rerank error (status={status_code}, code={code}): {message}"
            )

        results = response.get("output", {}).get("results", [])
        if not results:
            logger.warning("DashScope rerank returned empty results, returning original order")
            return list(candidates)

        reranked: List[Dict[str, Any]] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", 0.0)
            if idx is None or idx < 0 or idx >= len(candidates):
                logger.warning("DashScope rerank returned invalid index %s, skipping", idx)
                continue
            candidate = candidates[idx].copy()
            candidate["rerank_score"] = float(score)
            reranked.append(candidate)

        reranked.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)

        logger.debug(
            "DashScopeReranker: %d -> %d candidates after rerank",
            len(candidates),
            len(reranked),
        )
        return reranked
