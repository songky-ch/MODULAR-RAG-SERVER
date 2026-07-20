"""Transform that enriches chunks with section / table metadata hints.

This transform is designed to work with the FinancialReportSplitter, which
produces chunks that roughly align with sections and tables in Markdown.
Because the splitter itself only returns plain strings, we heuristically
inspect chunk text here to infer:

- Whether the chunk is table-centric (Markdown table).
- Table title (short text immediately preceding a table header).
- Section titles from in-chunk heading patterns.
"""

from __future__ import annotations

from typing import List, Optional
import re

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.observability.logger import get_logger


logger = get_logger(__name__)


class SectionTableMetadataTransform(BaseTransform):
    """Heuristic metadata enrichment for section / table information."""

    # Reuse patterns similar to FinancialReportSplitter
    SECTION_TITLE_PATTERNS = [
        re.compile(r"^#{1,6}\s+.+"),
        re.compile(r"^[一二三四五六七八九十]+、.+"),
        re.compile(r"^[（(][一二三四五六七八九十]+[）)].+"),
        re.compile(r"^\d+\.\s+.+"),
    ]

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        if not chunks:
            return []

        enriched: List[Chunk] = []

        for idx, chunk in enumerate(chunks):
            try:
                metadata_updates: dict = {}
                text = chunk.text or ""
                lines = text.split("\n")

                # Table-related hints
                if self._looks_like_table(lines):
                    metadata_updates["is_table_chunk"] = True
                    title = self._infer_table_title(lines)
                    if title:
                        metadata_updates["table_title"] = title

                # Section-related hints
                section_title = self._infer_section_title(lines)
                if section_title:
                    metadata_updates.setdefault("section_title", section_title)

                if metadata_updates:
                    new_chunk = Chunk(
                        id=chunk.id,
                        text=chunk.text,
                        metadata={**(chunk.metadata or {}), **metadata_updates},
                        source_ref=chunk.source_ref,
                    )
                    enriched.append(new_chunk)
                else:
                    enriched.append(chunk)
            except Exception as exc:
                logger.warning("Failed to enrich chunk %s: %s", chunk.id, exc)
                enriched.append(chunk)

        if trace:
            table_count = sum(1 for c in enriched if c.metadata.get("is_table_chunk"))
            section_count = sum(1 for c in enriched if c.metadata.get("section_title"))
            trace.record_stage(
                "section_table_metadata",
                {
                    "total_chunks": len(chunks),
                    "table_chunks": table_count,
                    "section_chunks": section_count,
                },
            )

        return enriched

    def _looks_like_table(self, lines: List[str]) -> bool:
        """Return True if the chunk mostly consists of Markdown table lines."""
        non_empty = [ln for ln in lines if ln.strip()]
        if not non_empty:
            return False
        table_like = [ln for ln in non_empty if ln.strip().startswith("|") and "|" in ln]
        return len(table_like) >= max(2, len(non_empty) // 2)

    def _infer_table_title(self, lines: List[str]) -> str | None:
        """Try to extract a short title line from the top of the chunk."""
        # Look at the first few non-empty lines that are not table rows
        candidates: List[str] = []
        for ln in lines[:5]:
            stripped = ln.strip()
            if not stripped:
                continue
            if stripped.startswith("|") and "|" in stripped:
                break
            candidates.append(stripped)
        for cand in candidates:
            if 2 <= len(cand) <= 50:
                return cand
        return None

    def _infer_section_title(self, lines: List[str]) -> str | None:
        """Look for an obvious section title in the first few lines."""
        for ln in lines[:5]:
            stripped = ln.strip()
            if not stripped:
                continue
            for pattern in self.SECTION_TITLE_PATTERNS:
                if pattern.match(stripped):
                    return stripped
        return None

