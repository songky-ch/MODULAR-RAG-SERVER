import pytest
from unittest import mock
from pathlib import Path
import hashlib

from src.core.settings import load_settings
from src.core.types import Document
from src.ingestion.pipeline import IngestionPipeline

# The path provided by the user
TARGET_MARKDOWN_FILE = r"d:\LLM Project\MODULAR-RAG-MCP-SERVER\data\markdown\3f59624725f0d79a1894d36dc727c9fedc7cfeb7cbcc11d257b1c08f6b42a00f\tmpbiafcspt.md"


class MarkdownDirectLoader:
    """
    A custom loader that ignores normal PDF/OCR extraction logic
    and directly loads a Markdown file into a Document object.
    
    This effectively mocks the 'load' step in the pipeline.
    """
    def load(self, file_path: str) -> Document:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        
        # We compute a deterministic ID based on content to simulate the real loader behavior
        doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()
        
        return Document(
            id=doc_id,
            text=content,
            metadata={
                "source_path": str(path),
                "doc_type": "markdown",
                "title": path.stem,
                "images": []  # Providing an empty array since images aren't extracted here
            }
        )


class TestSkipLoadIngestion:
    
    @pytest.fixture
    def settings(self):
        """Loads application settings."""
        # This will default to the standard priority: os env -> config/settings.yaml
        return load_settings()

    def test_ingest_markdown_directly(self, settings):
        """
        Test that we can ingest a markdown file, successfully overriding
        the loader (GLM OCR / PdfLoader) but preserving all downstream stages
        (chunking, transforms, embeddings, stores).
        """
        # Given: We ensure our target markdown file exists
        if not Path(TARGET_MARKDOWN_FILE).exists():
            pytest.skip(f"Test file not found: {TARGET_MARKDOWN_FILE}")
            
        from src.ingestion.embedding.batch_processor import BatchResult
        
        # We mock GlmOcrPdfLoader to avoid httpx proxy initialization errors during pipeline __init__
        with mock.patch("src.ingestion.pipeline.GlmOcrPdfLoader"):
            pipeline = IngestionPipeline(settings, collection="test_markdown_skip_load_v2", force=True)
            
            # => Here is the trick: Override the loader to our custom parser
            pipeline.loader = MarkdownDirectLoader()
            
            # We use the real batch process and dense encoder, allowing API calls
            # to generate real dense embeddings for these chunks.
            
            try:
                # When: We run the pipeline
                result = pipeline.run(TARGET_MARKDOWN_FILE)
                
                # Then: The pipeline should complete successfully
                assert result.success is True, f"Pipeline failed: {result.error}"
                assert result.chunk_count > 0, "No chunks were generated."
                
                # Verify that the downstream stages executed
                assert "chunking" in result.stages
                assert "transform" in result.stages
                assert "encoding" in result.stages
                assert "storage" in result.stages
                
                print(f"Integration Test Success! Ingested markdown directly. Generated {result.chunk_count} chunks.")
                
            finally:
                pipeline.close()
