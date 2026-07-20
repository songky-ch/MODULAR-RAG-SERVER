"""Image Captioner transform for enriching chunks with image descriptions.

Performance Optimizations:
1. Only processes images that are actually referenced in chunk text (via [IMAGE: id] placeholder)
2. Uses caption cache to avoid redundant Vision API calls for the same image
3. Skips chunks without image references entirely
4. Parallel processing of unique images with thread-safe caching
"""

import base64
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.core.trace.langchain_callback import build_langchain_callbacks
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.lc_llm import LCLLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Regex to find image placeholders: [IMAGE: some_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')

# Default max parallel workers for Vision API calls
DEFAULT_MAX_WORKERS = 3  # Lower than text LLM due to higher cost/latency


class ImageCaptioner(BaseTransform):
    """Generates captions for images referenced in chunks using Vision LLM.
    
    This transform identifies chunks containing image references, uses a Vision LLM
    to generate descriptive captions, and enriches the chunk text/metadata with
    these captions to improve retrieval for visual content.
    
    Key Features:
    - Only processes images actually referenced in chunk text (not all images in metadata)
    - Caches captions to avoid redundant Vision API calls
    - Thread-safe caption cache for potential future parallelization
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseChatModel] = None,
    ):
        self.settings = settings
        self.llm: Optional[BaseChatModel] = None
        self._caption_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()

        if self.settings.vision_llm and self.settings.vision_llm.enabled:
            try:
                self.llm = llm or LCLLMFactory.create_vision_llm(settings)
            except Exception as e:
                logger.error(f"Failed to initialize Vision LLM: {e}")
        else:
            logger.warning(
                "Vision LLM is disabled or not configured. "
                "ImageCaptioner will skip processing."
            )
        
        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """Load the image captioning prompt from configuration."""
        # Assuming standard relative path. In production, logic might be robust.
        from src.core.settings import resolve_path
        prompt_path = resolve_path("config/prompts/image_captioning.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Describe this image in detail for indexing purposes."

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        """Extract image IDs actually referenced in the chunk text.
        
        Args:
            text: Chunk text content
            
        Returns:
            List of image IDs found in [IMAGE: id] placeholders
        """
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _get_caption(
        self,
        img_id: str,
        img_path: str,
        trace: Optional[TraceContext] = None,
    ) -> Optional[str]:
        """Get caption for an image, using cache if available. Thread-safe."""
        with self._cache_lock:
            if img_id in self._caption_cache:
                logger.debug(f"Caption cache hit for image {img_id}")
                return self._caption_cache[img_id]

        if not img_path or not Path(img_path).exists():
            logger.warning(f"Image path not found: {img_path}")
            return None

        try:
            image_data = Path(img_path).read_bytes()
            b64 = base64.b64encode(image_data).decode("utf-8")

            suffix = Path(img_path).suffix.lower()
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime = mime_map.get(suffix, "image/png")
            data_uri = f"data:{mime};base64,{b64}"

            # DashScope (qwen/dashscope) expects type "image" with "image" field;
            # OpenAI-compatible providers expect type "image_url" with "image_url.url"
            provider = ""
            if self.settings.vision_llm:
                provider = self.settings.vision_llm.provider.lower()

            if provider in ("qwen", "dashscope"):
                image_part = {"type": "image", "image": data_uri}
            else:
                image_part = {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }

            message = HumanMessage(content=[
                {"type": "text", "text": self.prompt},
                image_part,
            ])

            config: dict = {"callbacks": build_langchain_callbacks(trace)}

            response = self.llm.invoke([message], config=config)
            caption = response.content if hasattr(response, "content") else str(response)

            with self._cache_lock:
                self._caption_cache[img_id] = caption
            logger.debug(f"Generated and cached caption for image {img_id}")
            return caption

        except Exception as e:
            logger.error(f"Failed to caption image {img_path}: {e}")
            return None

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks and add captions for referenced images.
        
        Only processes images that are actually referenced in chunk text
        via [IMAGE: id] placeholders. Uses caching to avoid redundant API calls.
        Parallel processing for unique images.
        """
        if not self.llm:
            return chunks
        
        # Build image lookup from all chunks' metadata
        image_lookup: Dict[str, dict] = {}
        for chunk in chunks:
            if chunk.metadata and "images" in chunk.metadata:
                for img_meta in chunk.metadata.get("images", []):
                    img_id = img_meta.get("id")
                    if img_id and img_id not in image_lookup:
                        image_lookup[img_id] = img_meta
        
        logger.info(f"Found {len(image_lookup)} unique images in document")
        
        # Clear cache for new document processing
        with self._cache_lock:
            self._caption_cache.clear()
        
        # First pass: collect all unique image IDs that need captioning
        images_to_caption: Dict[str, str] = {}  # img_id -> img_path
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            for img_id in referenced_ids:
                if img_id not in images_to_caption:
                    img_meta = image_lookup.get(img_id)
                    if img_meta and img_meta.get("path"):
                        images_to_caption[img_id] = img_meta.get("path")
        
        # Parallel caption generation for all unique images
        if images_to_caption:
            self._generate_captions_parallel(images_to_caption, trace)
        
        # Second pass: apply captions to chunks
        processed_chunks = []
        total_captions_added = 0
        
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            
            if not referenced_ids:
                processed_chunks.append(chunk)
                continue
            
            new_text = chunk.text
            captions = []
            
            for img_id in referenced_ids:
                img_id_stripped = img_id.strip()
                
                # Get caption from cache (already populated by parallel processing)
                with self._cache_lock:
                    caption = self._caption_cache.get(img_id_stripped)
                
                if caption:
                    captions.append({"id": img_id_stripped, "caption": caption})
                    
                    placeholder = f"[IMAGE: {img_id}]"
                    replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
                    new_text = new_text.replace(placeholder, replacement)
                    total_captions_added += 1
                    
            chunk.text = new_text
            
            if captions:
                if "image_captions" not in chunk.metadata:
                    chunk.metadata["image_captions"] = []
                chunk.metadata["image_captions"].extend(captions)
            
            processed_chunks.append(chunk)
        
        with self._cache_lock:
            api_calls = len(self._caption_cache)
        logger.info(f"Added {total_captions_added} captions, API calls: {api_calls}")
            
        return processed_chunks
    
    def _generate_captions_parallel(
        self, 
        images_to_caption: Dict[str, str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """Generate captions for multiple images in parallel.
        
        Args:
            images_to_caption: Dict of img_id -> img_path
            trace: Optional trace context
        """
        if not images_to_caption:
            return
        
        max_workers = min(DEFAULT_MAX_WORKERS, len(images_to_caption))
        logger.debug(f"Generating captions for {len(images_to_caption)} images (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_caption, img_id, img_path, trace): img_id
                for img_id, img_path in images_to_caption.items()
            }
            
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    caption = future.result()
                    if caption:
                        logger.debug(f"Caption generated for {img_id}")
                except Exception as e:
                    logger.error(f"Failed to generate caption for {img_id}: {e}")
