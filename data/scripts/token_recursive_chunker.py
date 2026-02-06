"""
token_recursive_chunker.py - Token-based Recursive Chunker for LLM Training

Adapted from the RecursiveChunker in the RAG project, but:
- Token count instead of character count
- No overlap (default) for instruction datasets
- Optimized for longer, coherent chunks
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TokenRecursiveChunker:
    """Token-based chunker with semantic boundary awareness.
    
    Respects natural language boundaries (paragraphs, sentences, words)
    and uses token count instead of character count.
    
    Designed for LLM instruction dataset preparation, where:
    - Chunks should be 512-1024 tokens long
    - No overlap (each chunk becomes an independent QA pair)
    - Semantic coherence is critical
    
    Args:
        tokenizer: HuggingFace tokenizer (e.g. Mistral, Llama)
        chunk_size: Target chunk size in TOKENS (default: 512)
        chunk_overlap: Overlap in TOKENS (default: 0)
        separators: Separator hierarchy (default: paragraphs > lines > sentences > words)
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> chunker = TokenRecursiveChunker(tokenizer, chunk_size=512, chunk_overlap=0)
        >>> text = "Long document..."
        >>> chunks = chunker.chunk(text, metadata={'service': 'S3', 'hierarchy': ['S3', 'Buckets']})
    """
    
    # Default separators (same hierarchy as RecursiveChunker)
    DEFAULT_SEPARATORS = [
        "\n\n",    # Paragraphs (try first)
        "\n",      # Lines
        ". ",      # Sentences
        " ",       # Words
        ""         # Characters (fallback)
    ]
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        separators: Optional[List[str]] = None
    ):
        """Initialize token-based recursive chunker."""
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        
        logger.info("="*70)
        logger.info("TOKEN RECURSIVE CHUNKER INITIALIZATION")
        logger.info("="*70)
        logger.info(f"Tokenizer: {tokenizer.name_or_path}")
        logger.info(f"Chunk size: {chunk_size} tokens")
        logger.info(f"Chunk overlap: {chunk_overlap} tokens")
        logger.info(f"Separators: {self.separators}")
        logger.info("="*70)
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Split text into token-based chunks with semantic boundaries.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk (e.g. hierarchy, service)
        
        Returns:
            List of chunk dictionaries with:
                - content: Chunk text
                - token_count: Number of tokens
                - metadata: Source metadata + chunk position
        """
        if not text:
            logger.warning("Empty text provided for chunking")
            return []
        
        if metadata is None:
            metadata = {}
        
        # Tokenize entire text ONCE
        full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(full_tokens)
        
        logger.info(f"Chunking text: {len(text)} chars, {total_tokens} tokens")
        
        # If text fits in one chunk, return it
        if total_tokens <= self.chunk_size:
            chunk = {
                'content': text,
                'token_count': total_tokens,
                'metadata': {
                    **metadata,
                    'chunk_id': 0,
                    'chunk_start': 0,
                    'chunk_end': total_tokens,
                }
            }
            return [chunk]
        
        # Recursive split with token-aware boundary detection
        text_chunks = self._recursive_split_tokens(text, self.separators)
        
        # Add overlap if needed (boundary-aware)
        if self.chunk_overlap > 0:
            text_chunks = self._add_overlap_semantic(text_chunks)
        
        # Create final chunk objects with metadata
        result_chunks = []
        position = 0
        
        for chunk_id, chunk_text in enumerate(text_chunks):
            chunk_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=False)
            token_count = len(chunk_tokens)
            
            chunk = {
                'content': chunk_text,
                'token_count': token_count,
                'metadata': {
                    **metadata,
                    'chunk_id': chunk_id,
                    'chunk_start': position,
                    'chunk_end': position + token_count,
                }
            }
            result_chunks.append(chunk)
            
            position += token_count - self.chunk_overlap
        
        logger.info(
            f"Created {len(result_chunks)} chunks "
            f"(avg {sum(c['token_count'] for c in result_chunks) / len(result_chunks):.0f} tokens/chunk)"
        )
        
        return result_chunks
    
    def _recursive_split_tokens(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text at semantic boundaries with token-awareness.
        
        Key difference from character-based chunker:
        - Checks token count, not character count
        - Ensures chunks are within token budget
        """
        # Check token count
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Base case: fits within chunk_size
        if len(tokens) <= self.chunk_size:
            return [text] if text else []
        
        # Try each separator
        for separator in separators:
            if separator == '':
                # Last resort: token-level split
                return self._token_split(text)
            
            if separator in text:
                # Split at separator
                chunks = self._split_at_separator_tokens(text, separator)
                
                # Check if chunks fit
                final_chunks = []
                for chunk in chunks:
                    chunk_tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
                    if len(chunk_tokens) <= self.chunk_size:
                        final_chunks.append(chunk)
                    else:
                        # Recursively split large chunks with next separators
                        current_idx = separators.index(separator)
                        remaining_seps = separators[current_idx + 1:]
                        final_chunks.extend(self._recursive_split_tokens(chunk, remaining_seps))
                
                return final_chunks
        
        # Fallback
        return [text]
    
    def _split_at_separator_tokens(self, text: str, separator: str) -> List[str]:
        """Split text at separator, keeping chunks near chunk_size TOKENS."""
        if separator == '':
            return self._token_split(text)
        
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for i, split in enumerate(splits):
            # Add separator back (except last split)
            split_with_sep = split + separator if i < len(splits) - 1 else split
            
            # Check token count of combined chunk
            combined = current_chunk + split_with_sep
            combined_tokens = self.tokenizer.encode(combined, add_special_tokens=False)
            
            if current_chunk and len(combined_tokens) > self.chunk_size:
                # Current chunk is done
                chunks.append(current_chunk)
                current_chunk = split_with_sep
            else:
                # Add to current chunk
                current_chunk = combined
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _token_split(self, text: str) -> List[str]:
        """Fallback: Split text at token boundaries (for very long words/sequences)."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def _add_overlap_semantic(self, chunks: List[str]) -> List[str]:
        """Add token-based overlap while preserving semantic boundaries.
        
        Similar to RecursiveChunker, but uses token count instead of characters.
        """
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                prev_chunk = chunks[i - 1]
                
                # Tokenize previous chunk
                prev_tokens = self.tokenizer.encode(prev_chunk, add_special_tokens=False)
                
                # Find overlap region (last N tokens)
                overlap_start_token = max(0, len(prev_tokens) - self.chunk_overlap)
                overlap_tokens = prev_tokens[overlap_start_token:]
                overlap_text = self.tokenizer.decode(overlap_tokens)
                
                # Find semantic boundary in overlap region
                best_split = len(prev_chunk)  # Fallback
                
                for separator in self.separators[:-1]:  # Exclude empty string
                    if separator in overlap_text:
                        idx = overlap_text.rfind(separator)
                        # Calculate position in prev_chunk
                        overlap_decoded_prefix = self.tokenizer.decode(
                            prev_tokens[overlap_start_token:overlap_start_token + idx]
                        )
                        best_split = len(prev_chunk) - len(overlap_text) + idx + len(separator)
                        break
                
                # Extract overlap starting at semantic boundary
                overlap_final = prev_chunk[best_split:]
                
                # Prepend to current chunk
                overlapped_chunk = overlap_final + chunk
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks