"""Financial-report-aware text splitter for Chinese PDFs (Markdown input).

This splitter (v2) uses a single-pass linear scan to process Markdown text from
Chinese financial reports, maintaining a section context stack so that every chunk
carries its enclosing heading hierarchy.

Key design choices:
- Single-pass linear scan keeps chunks in strict document order.
- Section context stack propagates headings (Markdown ``#`` and Chinese
  numbering like 一、 / (一) / 1.) into every chunk as a prefix.
- Tables are kept whole (or split by rows with repeated headers) and always
  include their section context for retrieval relevance.
- Short text immediately preceding a table is treated as a *preamble* and
  merged into the table chunk instead of forming a micro-chunk.
- Text splitting is delegated to LangChain ``RecursiveCharacterTextSplitter``
  with Chinese-optimised separators.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import re

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[misc, assignment]

from src.libs.splitter.base_splitter import BaseSplitter


class FinancialReportSplitter(BaseSplitter):
    """Structure-aware splitter for Chinese financial-report Markdown."""

    DEFAULT_CHUNK_SIZE: int = 1500
    DEFAULT_CHUNK_OVERLAP: int = 200
    DEFAULT_CONTEXT_MAX_LENGTH: int = 150
    MIN_STANDALONE_TEXT_LENGTH: int = 50

    SECTION_TITLE_PATTERNS = [
        re.compile(r"^#{1,6}\s+.+"),
        re.compile(r"^[一二三四五六七八九十]+、.+"),
        re.compile(r"^[（(][一二三四五六七八九十]+[）)].+"),
        re.compile(r"^\d+\.\s+.+"),
    ]

    CHINESE_SEPARATORS = [
        "\n\n",
        "。\n", "。",
        "！\n", "！",
        "？\n", "？",
        "；\n", "；",
        "\n",
        ". ", "! ", "? ",
        "，", ", ",
        " ",
        "",
    ]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **_: Any,
    ) -> None:
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is required for FinancialReportSplitter. "
                "Install with: pip install langchain-text-splitters"
            )

        self.settings = settings

        ingestion = getattr(settings, "ingestion", None)
        base_chunk_size = getattr(ingestion, "chunk_size", self.DEFAULT_CHUNK_SIZE)
        base_overlap = getattr(ingestion, "chunk_overlap", self.DEFAULT_CHUNK_OVERLAP)

        self.chunk_size = int(chunk_size) if chunk_size is not None else int(base_chunk_size)
        self.chunk_overlap = int(chunk_overlap) if chunk_overlap is not None else int(base_overlap)

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got: {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got: {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )

        fs_cfg = self._read_financial_splitter_cfg(ingestion)
        self.context_max_length: int = int(
            fs_cfg.get("context_max_length", self.DEFAULT_CONTEXT_MAX_LENGTH)
        )
        self.table_context: bool = bool(fs_cfg.get("table_context", True))

        reserved = min(self.context_max_length + 4, self.chunk_size // 3)
        effective_size = self.chunk_size - reserved
        rcts_overlap = min(self.chunk_overlap, max(effective_size - 1, 0))
        self._rcts = RecursiveCharacterTextSplitter(
            chunk_size=effective_size,
            chunk_overlap=rcts_overlap,
            separators=self.CHINESE_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    @staticmethod
    def _read_financial_splitter_cfg(ingestion: Any) -> dict:
        if ingestion is None:
            return {}
        raw = getattr(ingestion, "financial_splitter", None)
        return raw if isinstance(raw, dict) else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Split Markdown text into ordered, context-enriched chunks."""
        _ = trace
        self.validate_text(text)

        lines = text.split("\n")
        if not lines:
            raise ValueError("Input text cannot be empty after split")

        chunks: List[str] = []
        section_context: List[Tuple[int, str]] = []
        text_buffer: List[str] = []

        i = 0
        n = len(lines)
        while i < n:
            stripped = lines[i].strip()

            if self._is_section_title(stripped):
                if text_buffer:
                    chunks.extend(self._flush_text_buffer(text_buffer, section_context))
                    text_buffer = []
                self._update_section_context(section_context, stripped)
                i += 1

            elif self._is_table_header_start(lines, i):
                preamble = ""
                buf_text = "\n".join(text_buffer).strip()
                if buf_text and len(buf_text) <= self.MIN_STANDALONE_TEXT_LENGTH:
                    preamble = buf_text
                elif text_buffer:
                    chunks.extend(self._flush_text_buffer(text_buffer, section_context))
                text_buffer = []

                table_lines, i = self._collect_table(lines, i)
                chunks.extend(
                    self._split_table_with_context(table_lines, section_context, preamble)
                )

            else:
                text_buffer.append(lines[i])
                i += 1

        if text_buffer:
            chunks.extend(self._flush_text_buffer(text_buffer, section_context))

        if not chunks:
            chunks = [text.strip()]

        self.validate_chunks(chunks)
        return chunks

    # ------------------------------------------------------------------
    # Section context stack
    # ------------------------------------------------------------------
    def _is_section_title(self, stripped_line: str) -> bool:
        if not stripped_line:
            return False
        # Real section titles never contain sentence-ending punctuation mid-line
        content_core = stripped_line.rstrip("。！？")
        if any(p in content_core for p in ("。", "！", "？")):
            return False
        return any(p.match(stripped_line) for p in self.SECTION_TITLE_PATTERNS)

    def _detect_title_level(self, title: str) -> int:
        """Map a title string to a numeric depth level.

        Markdown ``#`` headings use their native level.  Chinese numbering
        patterns are mapped to conventional depths:
        ``一、`` -> 2, ``（一）`` -> 3, ``1.`` -> 4.
        """
        md_match = re.match(r"^(#{1,6})\s+", title)
        if md_match:
            return len(md_match.group(1))
        if re.match(r"^[一二三四五六七八九十]+、", title):
            return 2
        if re.match(r"^[（(][一二三四五六七八九十]+[）)]", title):
            return 3
        if re.match(r"^\d+\.\s+", title):
            return 4
        return 5

    def _update_section_context(
        self, stack: List[Tuple[int, str]], title_line: str
    ) -> None:
        level = self._detect_title_level(title_line)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title_line))

    def _build_context_prefix(self, stack: List[Tuple[int, str]]) -> str:
        if not stack:
            return ""
        prefix = " > ".join(item[1] for item in stack)
        if len(prefix) > self.context_max_length:
            parts = [item[1] for item in stack]
            while len(" > ".join(parts)) > self.context_max_length and len(parts) > 1:
                parts.pop(0)
            prefix = " > ".join(parts)
        return prefix

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------
    def _is_table_line(self, line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and "|" in stripped

    def _is_table_header_start(self, lines: List[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False
        header = lines[index].strip()
        sep = lines[index + 1].strip()
        if not header.startswith("|") or "|" not in header:
            return False
        if not sep.startswith("|"):
            return False
        return sep.count("-") >= 3 and sep.count("|") >= 2

    def _collect_table(self, lines: List[str], start: int) -> Tuple[List[str], int]:
        """Collect all contiguous table lines from *start*.

        Returns ``(table_lines, next_index_after_table)``.
        """
        collected = [lines[start]]
        i = start + 1
        while i < len(lines) and self._is_table_line(lines[i]):
            collected.append(lines[i])
            i += 1
        return collected, i

    def _split_table_with_context(
        self,
        table_lines: List[str],
        section_context: List[Tuple[int, str]],
        preamble: str = "",
    ) -> List[str]:
        ctx = self._build_context_prefix(section_context) if self.table_context else ""
        header_parts: List[str] = []
        if ctx:
            header_parts.append(ctx)
        if preamble:
            header_parts.append(preamble)
        context_block = "\n\n".join(header_parts)
        sep = "\n\n" if context_block else ""

        full_table = "\n".join(table_lines)
        combined = f"{context_block}{sep}{full_table}".strip()
        if len(combined) <= self.chunk_size:
            return [combined]

        if len(table_lines) <= 2:
            return [combined]

        header = table_lines[:2]
        data = table_lines[2:]

        chunks: List[str] = []
        current = list(header)

        for row in data:
            candidate_table = "\n".join(current + [row])
            candidate = f"{context_block}{sep}{candidate_table}".strip()
            if len(candidate) > self.chunk_size and len(current) > len(header):
                chunk_table = "\n".join(current)
                chunks.append(f"{context_block}{sep}{chunk_table}".strip())
                current = list(header) + [row]
            else:
                current.append(row)

        if current:
            chunk_table = "\n".join(current)
            chunks.append(f"{context_block}{sep}{chunk_table}".strip())

        return [c for c in chunks if c]

    # ------------------------------------------------------------------
    # Text buffer flushing
    # ------------------------------------------------------------------
    def _flush_text_buffer(
        self,
        buffer_lines: List[str],
        section_context: List[Tuple[int, str]],
    ) -> List[str]:
        text = "\n".join(buffer_lines).strip()
        if not text:
            return []

        context_prefix = self._build_context_prefix(section_context)
        sep = "\n\n" if context_prefix else ""

        combined = f"{context_prefix}{sep}{text}".strip()
        if len(combined) <= self.chunk_size:
            return [combined]

        sub_chunks = self._rcts.split_text(text)
        if not sub_chunks:
            return [combined]

        results: List[str] = []
        for sub in sub_chunks:
            chunk = f"{context_prefix}{sep}{sub}".strip()
            if chunk:
                results.append(chunk)

        return results if results else [combined]
