"""Unit tests for FinancialReportSplitter v2 (context-aware splitting)."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.libs.splitter.financial_report_splitter import FinancialReportSplitter
from src.libs.splitter.base_splitter import BaseSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_settings(
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    financial_splitter: Any = None,
) -> Any:
    """Create a minimal mock Settings with IngestionSettings."""
    settings = MagicMock()
    ingestion = MagicMock()
    ingestion.chunk_size = chunk_size
    ingestion.chunk_overlap = chunk_overlap
    # financial_splitter must return a dict or None; MagicMock would return
    # another Mock which is not ``isinstance(raw, dict)``, so the splitter
    # correctly falls back to defaults unless we explicitly set a dict.
    ingestion.financial_splitter = financial_splitter
    settings.ingestion = ingestion
    return settings


def _make_splitter(
    chunk_size: int = 80,
    overlap: int = 10,
    context_max_length: int = 30,
    table_context: bool = True,
) -> FinancialReportSplitter:
    fs_cfg = {"context_max_length": context_max_length, "table_context": table_context}
    settings = _mock_settings(chunk_size=chunk_size, chunk_overlap=overlap, financial_splitter=fs_cfg)
    return FinancialReportSplitter(settings=settings)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestFinancialReportSplitterConfig:

    def test_initialization_from_settings(self) -> None:
        settings = _mock_settings(chunk_size=1600, chunk_overlap=200)
        splitter = FinancialReportSplitter(settings=settings)
        assert isinstance(splitter, BaseSplitter)
        assert splitter.chunk_size == 1600
        assert splitter.chunk_overlap == 200

    def test_initialization_with_overrides(self) -> None:
        settings = _mock_settings(chunk_size=1000, chunk_overlap=100)
        splitter = FinancialReportSplitter(settings=settings, chunk_size=1200, chunk_overlap=150)
        assert splitter.chunk_size == 1200
        assert splitter.chunk_overlap == 150

    @pytest.mark.parametrize("size", [0, -1])
    def test_invalid_chunk_size(self, size: int) -> None:
        settings = _mock_settings(chunk_size=size)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FinancialReportSplitter(settings=settings)

    def test_invalid_chunk_overlap_negative(self) -> None:
        settings = _mock_settings(chunk_overlap=-10)
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            FinancialReportSplitter(settings=settings)

    def test_invalid_chunk_overlap_too_large(self) -> None:
        settings = _mock_settings(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="must be less than chunk_size"):
            FinancialReportSplitter(settings=settings)

    def test_financial_splitter_config_defaults(self) -> None:
        settings = _mock_settings()
        splitter = FinancialReportSplitter(settings=settings)
        assert splitter.context_max_length == 150
        assert splitter.table_context is True

    def test_financial_splitter_config_custom(self) -> None:
        fs_cfg = {"context_max_length": 80, "table_context": False}
        settings = _mock_settings(financial_splitter=fs_cfg)
        splitter = FinancialReportSplitter(settings=settings)
        assert splitter.context_max_length == 80
        assert splitter.table_context is False


# ---------------------------------------------------------------------------
# Core behaviour tests
# ---------------------------------------------------------------------------

class TestDocumentOrderPreservation:
    """Verify that chunks appear in strict document order."""

    def test_text_table_text_order(self) -> None:
        splitter = _make_splitter(chunk_size=200, overlap=0)
        text = (
            "## 一、公司概况\n"
            "这是公司概况的正文内容。\n"
            "| 项目 | 数据 |\n"
            "| --- | --- |\n"
            "| 收入 | 100 |\n"
            "## 二、财务数据\n"
            "这是财务数据的正文内容。\n"
        )
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

        # When text + table fit, they may be merged into one chunk (desired).
        # Key invariant: section-1 content appears before section-2 content.
        overview_idx = next(i for i, c in enumerate(chunks) if "公司概况" in c)
        table_idx = next(i for i, c in enumerate(chunks) if "| 收入 |" in c)
        finance_idx = next(i for i, c in enumerate(chunks) if "财务数据" in c and "正文" in c)

        assert overview_idx <= table_idx < finance_idx

    def test_multiple_tables_preserve_order(self) -> None:
        splitter = _make_splitter(chunk_size=200, overlap=0)
        text = (
            "## 表格一\n"
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
            "## 表格二\n"
            "| C | D |\n| --- | --- |\n| 3 | 4 |\n"
        )
        chunks = splitter.split_text(text)
        t1_idx = next(i for i, c in enumerate(chunks) if "| 1 | 2 |" in c)
        t2_idx = next(i for i, c in enumerate(chunks) if "| 3 | 4 |" in c)
        assert t1_idx < t2_idx


class TestSectionContextPropagation:
    """Verify that section headings are propagated as chunk prefixes."""

    def test_heading_context_in_text_chunk(self) -> None:
        splitter = _make_splitter(chunk_size=200, overlap=0)
        text = "## 三、主要财务指标\n这里是财务指标的详细说明。\n"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert "三、主要财务指标" in chunks[0]
        assert "详细说明" in chunks[0]

    def test_heading_context_in_table_chunk(self) -> None:
        splitter = _make_splitter(chunk_size=300, overlap=0)
        text = (
            "## 三、主要财务指标\n"
            "| 项目 | 2024年 | 2023年 |\n"
            "| --- | --- | --- |\n"
            "| 营业收入 | 100亿 | 80亿 |\n"
        )
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        combined = chunks[0]
        assert "三、主要财务指标" in combined
        assert "| 营业收入 |" in combined

    def test_nested_section_context(self) -> None:
        splitter = _make_splitter(chunk_size=300, overlap=0)
        text = (
            "## 一、公司概况\n"
            "### （一）主营业务\n"
            "公司主要从事软件开发业务。\n"
        )
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert "一、公司概况" in chunks[0]
        assert "（一）主营业务" in chunks[0]
        assert "软件开发" in chunks[0]

    def test_section_context_resets_on_same_level(self) -> None:
        splitter = _make_splitter(chunk_size=300, overlap=0)
        text = (
            "## 一、公司概况\n"
            "### （一）主营业务\n"
            "主营业务内容。\n"
            "## 二、财务数据\n"
            "财务数据内容。\n"
        )
        chunks = splitter.split_text(text)
        finance_chunk = next(c for c in chunks if "财务数据内容" in c)
        assert "二、财务数据" in finance_chunk
        # Should NOT carry over the old section context
        assert "（一）主营业务" not in finance_chunk


class TestTableSplitting:
    """Verify table integrity and header repetition."""

    def test_table_preserves_header_rows(self) -> None:
        splitter = _make_splitter(chunk_size=120, overlap=0)
        text = (
            "## 数据表\n"
            "| 年度 | 收入 |\n"
            "| ---- | ---- |\n"
            "| 2021 | 10 |\n"
            "| 2022 | 20 |\n"
            "| 2023 | 30 |\n"
        )
        chunks = splitter.split_text(text)
        for c in chunks:
            if "| 2022" in c:
                assert "| 年度 | 收入 |" in c

    def test_table_preamble_merged(self) -> None:
        """Short text before a table should be merged as preamble, not a standalone chunk."""
        splitter = _make_splitter(chunk_size=300, overlap=0)
        text = (
            "表1 主要财务指标\n"
            "| 项目 | 金额 |\n"
            "| --- | --- |\n"
            "| 收入 | 100 |\n"
        )
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert "表1 主要财务指标" in chunks[0]
        assert "| 收入 | 100 |" in chunks[0]

    def test_table_context_disabled(self) -> None:
        splitter = _make_splitter(chunk_size=300, overlap=0, table_context=False)
        text = (
            "## 三、指标\n"
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        )
        chunks = splitter.split_text(text)
        table_chunk = next(c for c in chunks if "| 1 | 2 |" in c)
        assert "三、指标" not in table_chunk


class TestNoMicroChunks:
    """Section titles should not produce tiny standalone chunks."""

    def test_title_not_standalone_chunk(self) -> None:
        splitter = _make_splitter(chunk_size=300, overlap=0)
        text = (
            "## 一、概况\n"
            "这是概况内容。\n"
            "## 二、数据\n"
            "这是数据内容。\n"
        )
        chunks = splitter.split_text(text)
        for c in chunks:
            assert len(c.strip()) > 10, f"Micro-chunk detected: {c!r}"

    def test_title_between_tables_not_standalone(self) -> None:
        splitter = _make_splitter(chunk_size=300, overlap=0)
        text = (
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
            "## 中间标题\n"
            "| C | D |\n| --- | --- |\n| 3 | 4 |\n"
        )
        chunks = splitter.split_text(text)
        for c in chunks:
            assert c.strip() != "## 中间标题", "Section title should not be a standalone chunk"


class TestChinesePunctuation:
    """Verify splitting respects Chinese sentence boundaries."""

    def test_split_at_chinese_periods(self) -> None:
        splitter = _make_splitter(chunk_size=18, overlap=0, context_max_length=0)
        text = "这里有第一句话。这里有第二句话。这里有第三句话。"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        full = "".join(chunks)
        assert "第一句话" in full
        assert "第三句话" in full

    def test_mixed_title_and_content_on_same_line(self) -> None:
        """A line like ``三、其他事项。内容`` should NOT be treated as a title."""
        splitter = _make_splitter(chunk_size=60, overlap=0, context_max_length=0)
        text = "三、其他事项。这里有中文句号。以及更多内容。"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        full = "".join(chunks)
        assert "三、其他事项" in full
        assert "中文句号" in full


class TestLongTextSplitting:
    """Verify behaviour on long text segments."""

    def test_long_section_produces_multiple_chunks(self) -> None:
        splitter = _make_splitter(chunk_size=60, overlap=5, context_max_length=15)
        text = "## 财务摘要\n" + "本节内容。" * 30
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        assert all("本节内容" in c for c in chunks[1:] if len(c) > 15)

    def test_overlap_produces_shared_content(self) -> None:
        splitter = _make_splitter(chunk_size=60, overlap=10, context_max_length=10)
        text = "一句话内容。" * 30
        chunks = splitter.split_text(text)
        if len(chunks) >= 2:
            # With overlap the end of one chunk should appear near the start of the next
            tail = chunks[0][-10:]
            assert any(tail[:5] in c for c in chunks[1:3])


class TestEdgeCases:

    def test_empty_text_raises(self) -> None:
        splitter = _make_splitter()
        with pytest.raises(ValueError):
            splitter.split_text("")

    def test_whitespace_only_raises(self) -> None:
        splitter = _make_splitter()
        with pytest.raises(ValueError):
            splitter.split_text("   \n\n   ")

    def test_text_without_any_structure(self) -> None:
        splitter = _make_splitter(chunk_size=200, overlap=0)
        text = "这是一段没有任何标题和表格的纯文本内容。"
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert "纯文本内容" in chunks[0]

    def test_table_only_document(self) -> None:
        splitter = _make_splitter(chunk_size=300, overlap=0)
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert "| 1 | 2 |" in chunks[0]
