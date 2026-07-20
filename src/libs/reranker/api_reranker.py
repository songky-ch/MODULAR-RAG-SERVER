"""Cloud API-based Reranker implementation.

Supports any rerank API that follows the Jina / Cohere / SiliconFlow
convention:

    POST /v1/rerank   (or custom path)
    {
        "model": "...",
        "query": "...",
        "documents": ["text1", "text2", ...],
        "top_n": 5
    }

    Response:
    {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 2, "relevance_score": 0.88},
            ...
        ]
    }

Tested providers:
- Jina Reranker      (https://api.jina.ai/v1/rerank)
- Cohere Rerank      (https://api.cohere.com/v2/rerank)
- SiliconFlow        (https://api.siliconflow.cn/v1/rerank)
- DashScope Rerank   (compatible endpoint)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class APIRerankError(RuntimeError):
    """Raised when a cloud rerank API call fails."""


class APIReranker(BaseReranker):
    """Cloud API-based reranker.

    Sends ``(query, documents)`` to a remote rerank endpoint and maps
    the returned ``(index, relevance_score)`` pairs back to the original
    candidate list.

    Args:
        settings: Application settings with ``rerank`` section.
        api_key: Explicit API key (overrides settings / env var).
        base_url: Explicit base URL (overrides settings).
        timeout: HTTP request timeout in seconds.
        **kwargs: Extra provider-specific params forwarded to the API body.
    """

    DEFAULT_BASE_URL = "https://api.jina.ai"
    RERANK_PATH = "/v1/rerank"
    ENV_VAR = "RERANK_API_KEY"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.timeout = timeout
        self._extra = kwargs

        rerank_cfg = getattr(settings, "rerank", None)

        self.model: str = (
            kwargs.get("model")
            or (getattr(rerank_cfg, "model", None) if rerank_cfg else None)
            or ""
        )
        self.api_key: str = (
            api_key
            or (getattr(rerank_cfg, "api_key", None) if rerank_cfg else None)
            or os.environ.get(self.ENV_VAR, "")
        )
        self.base_url: str = (
            base_url
            or (getattr(rerank_cfg, "base_url", None) if rerank_cfg else None)
            or self.DEFAULT_BASE_URL
        ).rstrip("/")

        if not self.api_key:
            raise ValueError(
                f"Rerank API key not provided. Set {self.ENV_VAR} env var, "
                "pass api_key, or configure rerank.api_key in settings.yaml"
            )

        logger.info(
            "APIReranker initialised: base_url=%s, model=%s",
            self.base_url,
            self.model,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Call remote rerank API and return candidates sorted by relevance.

        Each returned candidate gains a ``rerank_score`` field.
        """
        self.validate_query(query)
        self.validate_candidates(candidates)

        if len(candidates) == 1:
            return candidates

        documents = [
            c.get("text") or c.get("content", "") for c in candidates
        ]

        top_n = kwargs.get("top_n") or kwargs.get("top_k") or len(candidates)

        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        url = f"{self.base_url}{self.RERANK_PATH}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                detail = self._parse_error(response)
                raise APIRerankError(
                    f"Rerank API error (HTTP {response.status_code}): {detail}"
                )

            body = response.json()
        except httpx.TimeoutException as exc:
            raise APIRerankError(
                f"Rerank API timed out after {self.timeout}s"
            ) from exc
        except httpx.RequestError as exc:
            raise APIRerankError(
                f"Rerank API connection error: {exc}"
            ) from exc

        results = body.get("results", [])
        if not results:
            logger.warning("Rerank API returned empty results, returning original order")
            return list(candidates)

        reranked: List[Dict[str, Any]] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score", 0.0))
            if idx is None or idx < 0 or idx >= len(candidates):
                logger.warning("Rerank API returned invalid index %s, skipping", idx)
                continue
            candidate = candidates[idx].copy()
            candidate["rerank_score"] = float(score)
            reranked.append(candidate)

        reranked.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)

        logger.debug(
            "APIReranker: %d -> %d candidates after rerank",
            len(candidates),
            len(reranked),
        )
        return reranked

    @staticmethod
    def _parse_error(response: httpx.Response) -> str:
        try:
            data = response.json()
            if "error" in data:
                err = data["error"]
                return err.get("message", str(err)) if isinstance(err, dict) else str(err)
            if "detail" in data:
                return str(data["detail"])
            return response.text
        except Exception:
            return response.text or "Unknown error"
