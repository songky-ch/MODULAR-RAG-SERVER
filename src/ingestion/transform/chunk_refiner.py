"""分块优化转换: 基于规则的清洗与可选的 LLM 增强。"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.core.settings import Settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.core.trace.langchain_callback import build_langchain_callbacks
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.lc_llm import LCLLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

# LLM 调用的默认最大并行线程数
DEFAULT_MAX_WORKERS = 5


class ChunkRefiner(BaseTransform):
    """通过规则清洗和可选的 LLM 增强来优化分块。
    
    Processing Pipeline:
        1. Rule-based refine: Remove noise (whitespace, headers/footers, HTML)
        2. (Optional) LLM refine: Intelligent content improvement
        3. On LLM failure: Gracefully fallback to rule-based result
    
    Configuration (via settings.yaml):
        - ingestion.chunk_refiner.use_llm: bool - Enable LLM enhancement
        - ingestion.chunk_refiner.prompt_path: str - Custom prompt file path
    
    Design Principles:
        - Graceful Degradation: LLM errors don't block ingestion
        - Atomic Processing: Each chunk processed independently
        - Observable: Records refined_by in metadata
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseChatModel] = None,
        prompt_path: Optional[str] = None
    ):
        """初始化 ChunkRefiner。
        
        Args:
            settings: Application settings
            llm: Optional LLM instance (for testing; auto-created if None)
            prompt_path: Optional custom prompt file path
        """
        self.settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = prompt_path or str(resolve_path("config/prompts/chunk_refinement.txt"))
        
        # 判断是否使用 LLM
        self.use_llm = getattr(
            getattr(settings, 'ingestion', None), 
            'chunk_refiner', 
            {}
        ).get('use_llm', False) if hasattr(settings, 'ingestion') else False
        
    @property
    def llm(self) -> Optional[BaseChatModel]:
        """延迟加载 LangChain LLM 实例。"""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LCLLMFactory.create(self.settings)
                logger.info("LangChain LLM initialized for chunk refinement")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """通过优化流水线转换分块。
        
        Args:
            chunks: List of chunks to refine
            trace: Optional trace context
            
        Returns:
            List of refined chunks (same length as input)
        """
        if not chunks:
            return []
        
        # 启用 LLM 时并行处理分块
        if self.use_llm and self.llm:
            return self._transform_parallel(chunks, trace)
        else:
            return self._transform_sequential(chunks, trace)
    
    def _refine_single_chunk(
        self, 
        chunk: Chunk, 
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """优化单个分块，线程安全。
        
        Args:
            chunk: Chunk to refine
            trace: Optional trace context
            
        Returns:
            Tuple of (refined_chunk, refined_by, error_message)
        """
        try:
            # 步骤 1: 基于规则的优化
            rule_refined_text = self._rule_based_refine(chunk.text)
            
            # 步骤 2: LLM 增强
            if self.use_llm and self.llm:
                llm_refined_text = self._llm_refine(rule_refined_text, trace)
                
                if llm_refined_text:
                    refined_text = llm_refined_text
                    refined_by = "llm"
                else:
                    refined_text = rule_refined_text
                    refined_by = "rule"
            else:
                refined_text = rule_refined_text
                refined_by = "rule"
            
            refined_chunk = Chunk(
                id=chunk.id,
                text=refined_text,
                metadata={
                    **(chunk.metadata or {}),
                    'refined_by': refined_by
                },
                source_ref=chunk.source_ref
            )
            return (refined_chunk, refined_by, None)
            
        except Exception as e:
            logger.error(f"Failed to refine chunk {chunk.id}: {e}")
            return (chunk, "error", str(e))
    
    def _transform_parallel(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """使用 ThreadPoolExecutor 并行处理分块。"""
        max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))
        refined_chunks = [None] * len(chunks)
        llm_enhanced_count = 0
        fallback_count = 0
        
        logger.debug(f"Processing {len(chunks)} chunks in parallel (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交全部任务
            future_to_idx = {
                executor.submit(self._refine_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }
            
            # 收集结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    refined_chunk, refined_by, error = future.result()
                    refined_chunks[idx] = refined_chunk
                    
                    if refined_by == "llm":
                        llm_enhanced_count += 1
                    elif refined_by == "rule" and error is None:
                        fallback_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error in parallel refinement: {e}")
                    refined_chunks[idx] = chunks[idx]
        
        success_count = sum(1 for c in refined_chunks if c is not None)
        
        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": True,
                "max_workers": max_workers
            })
        
        logger.info(
            f"Refined {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )
        
        return refined_chunks
    
    def _transform_sequential(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """顺序处理分块（LLM 关闭时的处理方式）。"""
        refined_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0
        
        for chunk in chunks:
            try:
                # 步骤 1: 基于规则的优化（始终执行）
                rule_refined_text = self._rule_based_refine(chunk.text)
                
                # 步骤 2: 可选的 LLM 增强
                if self.use_llm and self.llm:
                    llm_refined_text = self._llm_refine(rule_refined_text, trace)
                    
                    if llm_refined_text:
                        # LLM 处理成功
                        refined_text = llm_refined_text
                        refined_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM 处理失败，回退到规则优化结果
                        refined_text = rule_refined_text
                        refined_by = "rule"
                        fallback_count += 1
                        if chunk.metadata:
                            chunk.metadata['refine_fallback_reason'] = "llm_failed"
                else:
                    # LLM 已关闭，使用规则优化结果
                    refined_text = rule_refined_text
                    refined_by = "rule"
                
                # 创建优化后的分块
                refined_chunk = Chunk(
                    id=chunk.id,
                    text=refined_text,
                    metadata={
                        **(chunk.metadata or {}),
                        'refined_by': refined_by
                    },
                    source_ref=chunk.source_ref
                )
                refined_chunks.append(refined_chunk)
                success_count += 1
                
            except Exception as e:
                # 单个分块处理失败时记录日志并保留原内容
                logger.error(f"Failed to refine chunk {chunk.id}: {e}")
                refined_chunks.append(chunk)
        
        # 记录追踪信息
        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": False
            })
        
        logger.info(
            f"Refined {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )
        
        return refined_chunks
    
    def _rule_based_refine(self, text: str) -> str:
        """应用基于规则的文本清洗。
        
        Cleaning operations:
            1. Remove page headers/footers (separator lines + metadata)
            2. Remove HTML comments
            3. Remove HTML tags (preserve content)
            4. Normalize excessive whitespace
            5. Preserve code blocks and Markdown formatting
        
        Args:
            text: Raw chunk text
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # 文本仅包含空白字符时直接返回
        if not text.strip():
            return ""
        
        # 保留代码块（先提取，处理完成后恢复）
        code_blocks = []
        code_block_pattern = r'```[\s\S]*?```'
        
        def extract_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        text = re.sub(code_block_pattern, extract_code_block, text)
        
        # 1. 删除包含页码或页脚的分隔线
        # 格式: ────────────────
        # 后面通常跟随页码、页脚文本等
        text = re.sub(
            r'─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(r'─{10,}', '', text)  # Remove remaining separator lines
        
        # 2. 删除 HTML 注释
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # 3. 删除 HTML 标签，但保留标签内文本
        text = re.sub(r'<[^>]+>', '', text)
        
        # 4. 规范化空白字符
        # - 将多个连续空格合并为一个
        text = re.sub(r' {2,}', ' ', text)
        
        # - 将三个及以上连续换行合并为两个，以保留段落边界
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 5. 删除每行首尾空白
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # 6. 恢复代码块
        for i, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", code_block)
        
        # 最终清理
        text = text.strip()
        
        return text
    
    def _llm_refine(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """应用基于 LLM 的智能优化。
        
        Args:
            text: Rule-refined text
            trace: Optional trace context
            
        Returns:
            LLM-refined text, or None if refinement failed
        """
        if not text or not text.strip():
            return text
        
        try:
            # 加载 Prompt 模板
            prompt_template = self._load_prompt()
            if not prompt_template:
                logger.warning("Prompt template not found, skipping LLM refinement")
                return None
            
            # 填充 Prompt
            if '{text}' not in prompt_template:
                logger.error("Prompt template missing {text} placeholder")
                return None
            
            prompt = prompt_template.replace('{text}', text)
            
            messages = [HumanMessage(content=prompt)]
            config = {"callbacks": build_langchain_callbacks(trace)}
            response = self.llm.invoke(messages, config=config)

            refined_text = response.content if hasattr(response, "content") else str(response)
            
            if refined_text and refined_text.strip():
                return refined_text.strip()
            else:
                logger.warning("LLM returned empty result")
                return None
                
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
            return None
    
    def _load_prompt(self) -> Optional[str]:
        """从文件加载 Prompt 模板。
        
        Returns:
            Prompt template string, or None if file not found
        """
        if self._prompt_template is not None:
            return self._prompt_template
        
        try:
            prompt_path = Path(self._prompt_path)
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {self._prompt_path}")
                return None
            
            self._prompt_template = prompt_path.read_text(encoding='utf-8')
            logger.debug(f"Loaded prompt template from {self._prompt_path}")
            return self._prompt_template
            
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            return None
