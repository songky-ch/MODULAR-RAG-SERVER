"""LangChain Chroma VectorStore adapter.

Wraps ``langchain_chroma.Chroma`` behind the project's
:class:`BaseVectorStore` interface so that upper-layer code
(pipeline, retrievers, MCP tools) remains unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.core.settings import resolve_path
from src.libs.vector_store.base_vector_store import BaseVectorStore

CHROMA_CLIENT_SETTINGS = ChromaSettings(anonymized_telemetry=False)

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class LangChainChromaStore(BaseVectorStore):
    """ChromaDB vector store backed by ``langchain_chroma.Chroma``.

    Implements every :class:`BaseVectorStore` method by delegating to
    the LangChain Chroma wrapper or its underlying collection.

    Args:
        settings: Application settings with ``vector_store`` and
            ``embedding`` sections.
        **kwargs: Overrides for ``collection_name`` or
            ``persist_directory``.
    """

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        vector_store_config = getattr(settings, "vector_store", None)
        if vector_store_config is None:
            raise ValueError(
                "Missing required configuration: settings.vector_store. "
                "Ensure 'vector_store' section exists in settings.yaml"
            )

        self.collection_name: str = kwargs.get(
            "collection_name",
            getattr(vector_store_config, "collection_name", "knowledge_hub"),
        )

        persist_dir_str: str = kwargs.get(
            "persist_directory",
            getattr(vector_store_config, "persist_directory", "./data/db/chroma"),
        )
        self.persist_directory = resolve_path(persist_dir_str)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing LangChainChromaStore: collection='%s', "
            "persist_directory='%s'",
            self.collection_name,
            self.persist_directory,
        )

        self._chroma = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
            client_settings=CHROMA_CLIENT_SETTINGS,
            collection_metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "LangChainChromaStore initialised. Collection count: %d",
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Internal accessors
    # ------------------------------------------------------------------

    @property
    def _collection(self):
        """Underlying chromadb Collection for low-level ops."""
        return self._chroma._collection

    @property
    def collection(self):
        """Public access to underlying Chroma collection.

        Used by DataService, DocumentManager, and other code that expects
        Chroma-style store.collection.get() for low-level queries.
        """
        return self._collection

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.validate_records(records)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for record in records:
            ids.append(str(record["id"]))
            embeddings.append(record["vector"])

            metadata = record.get("metadata", {})
            sanitized = self._sanitize_metadata(metadata)
            if not sanitized:
                sanitized = {"_placeholder": "true"}
            metadatas.append(sanitized)

            documents.append(str(metadata.get("text", record["id"])))

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            logger.debug("Upserted %d records via LangChain Chroma", len(records))
        except Exception as e:
            raise RuntimeError(
                f"Failed to upsert {len(records)} records: {e}"
            ) from e

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.validate_query_vector(vector, top_k)

        lc_filter = self._build_lc_filter(filters)

        try:
            results: list[tuple[Document, float]] = (
                self._chroma.similarity_search_by_vector_with_relevance_scores(
                    embedding=vector,
                    k=top_k,
                    filter=lc_filter,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query LangChain Chroma (top_k={top_k}): {e}"
            ) from e

        output: list[dict[str, Any]] = []
        for doc, score in results:
            output.append({
                "id": doc.metadata.get("_chroma_id", doc.id or ""),
                "score": float(score),
                "text": doc.page_content,
                "metadata": {
                    k: v for k, v in doc.metadata.items()
                    if k != "_chroma_id"
                },
            })

        logger.debug("Query returned %d results", len(output))
        return output

    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not ids:
            raise ValueError("IDs list cannot be empty")
        try:
            self._chroma.delete(ids=[str(i) for i in ids])
            logger.debug("Deleted %d records", len(ids))
        except Exception as e:
            raise RuntimeError(f"Failed to delete {len(ids)} records: {e}") from e

    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        target = collection_name or self.collection_name
        try:
            client = self._chroma._client
            client.delete_collection(name=target)
            self._chroma = Chroma(
                collection_name=target,
                persist_directory=str(self.persist_directory),
                client_settings=CHROMA_CLIENT_SETTINGS,
                collection_metadata={"hnsw:space": "cosine"},
            )
            logger.info("Cleared collection '%s'", target)
        except Exception as e:
            raise RuntimeError(f"Failed to clear collection '{target}': {e}") from e

    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not ids:
            raise ValueError("IDs list cannot be empty")

        str_ids = [str(i) for i in ids]

        try:
            results = self._collection.get(
                ids=str_ids,
                include=["metadatas", "documents"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get records by IDs: {e}") from e

        id_to_result: dict[str, dict[str, Any]] = {}
        if results and results.get("ids"):
            result_ids = results["ids"]
            documents = results.get("documents", [None] * len(result_ids))
            metadatas = results.get("metadatas", [{}] * len(result_ids))

            for i, record_id in enumerate(result_ids):
                id_to_result[record_id] = {
                    "id": record_id,
                    "text": documents[i] if documents and documents[i] else "",
                    "metadata": metadatas[i] if metadatas and metadatas[i] else {},
                }

        output = []
        for id_ in str_ids:
            output.append(id_to_result.get(id_, {}))

        logger.debug(
            "Retrieved %d of %d records by IDs",
            len([r for r in output if r]),
            len(ids),
        )
        return output

    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        trace: Optional[Any] = None,
    ) -> int:
        if not filter_dict:
            raise ValueError("filter_dict cannot be empty")

        try:
            where = self._build_lc_filter(filter_dict)
            results = self._collection.get(where=where, include=[])
            matching_ids = results.get("ids", [])

            if not matching_ids:
                logger.debug("delete_by_metadata: no records matched %s", filter_dict)
                return 0

            self._collection.delete(ids=matching_ids)
            logger.info(
                "delete_by_metadata: deleted %d records matching %s",
                len(matching_ids),
                filter_dict,
            )
            return len(matching_ids)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete by metadata {filter_dict}: {e}"
            ) from e

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "count": self._collection.count(),
            "name": self.collection_name,
            "metadata": self._collection.metadata,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                continue
            elif isinstance(value, (list, tuple)):
                sanitized[key] = ",".join(str(v) for v in value)
            else:
                sanitized[key] = str(value)
        return sanitized

    @staticmethod
    def _build_lc_filter(
        filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        where: dict[str, Any] = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                where[key] = value
            else:
                where[key] = value
        return where
