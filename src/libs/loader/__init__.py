"""
Loader Module.

This package contains document loader components:
- Base loader class
- PDF loader (local MarkItDown)
- GLM-OCR PDF loader (Zhipu cloud API)
- File integrity checker
"""

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.glm_ocr_pdf_loader import GlmOcrPdfLoader
from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker

__all__ = [
    "BaseLoader",
    "PdfLoader",
    "GlmOcrPdfLoader",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
]
