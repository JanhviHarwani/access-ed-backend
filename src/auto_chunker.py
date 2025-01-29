from typing import List, Dict
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    content: str
    metadata: Dict
    size: int

class AutoChunker:
    def __init__(
        self,
        max_chunk_size: int = 500,
        min_chunk_size: int = 100,
        overlap_size: int = 50
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

    def chunk_document(self, content: str, metadata: Dict) -> List[TextChunk]:
        """
        Intelligently chunk document content while preserving semantic meaning.
        
        Args:
            content: The document content to chunk
            metadata: Document metadata to preserve with each chunk
        
        Returns:
            List of TextChunk objects
        """
        # First clean and prepare the text
        cleaned_content = self._clean_text(content)
        
        # Split into semantic sections
        sections = self._split_into_sections(cleaned_content)
        
        # Process each section into chunks
        chunks = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                # Section is small enough to be a chunk
                chunks.append(TextChunk(
                    content=section,
                    metadata=metadata.copy(),
                    size=len(section)
                ))
            else:
                # Section needs to be broken down
                section_chunks = self._chunk_section(section)
                for chunk in section_chunks:
                    chunks.append(TextChunk(
                        content=chunk,
                        metadata=metadata.copy(),
                        size=len(chunk)
                    ))
        
        # Add chunk position metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks)
            })
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', text)
        return text.strip()

    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into semantic sections based on patterns"""
        # Split on common section markers
        markers = [
            r'\n\n+',              # Multiple newlines
            r'(?<=[.!?])\s+(?=[A-Z])',  # Sentence boundaries
            r'(?<=\w)\.\s+(?=[A-Z])',   # Period followed by capital letter
            r'(?:\r?\n|\r)(?=[A-Z])',   # Newline followed by capital letter
        ]
        
        pattern = '|'.join(markers)
        sections = [s.strip() for s in re.split(pattern, text) if s.strip()]
        
        # Merge very small sections with neighbors
        merged_sections = []
        current_section = []
        current_length = 0
        
        for section in sections:
            section_length = len(section)
            if current_length + section_length <= self.max_chunk_size:
                current_section.append(section)
                current_length += section_length
            else:
                if current_section:
                    merged_sections.append(' '.join(current_section))
                current_section = [section]
                current_length = section_length
        
        if current_section:
            merged_sections.append(' '.join(current_section))
        
        return merged_sections

    def _chunk_section(self, section: str) -> List[str]:
        """Break down a large section into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(section)

        while start < text_length:
            # Find the end of the current chunk
            end = start + self.max_chunk_size
            
            if end >= text_length:
                # This is the last chunk
                chunks.append(section[start:])
                break
            
            # Try to end at a sentence boundary
            sentence_end = self._find_sentence_boundary(
                section[start:end + 50]  # Look a bit ahead
            )
            
            if sentence_end > 0:
                chunk_end = start + sentence_end
            else:
                # If no good boundary, use word boundary
                chunk_end = self._find_word_boundary(
                    section[start:end]
                )
                if chunk_end <= 0:
                    chunk_end = end
                else:
                    chunk_end += start
            
            chunks.append(section[start:chunk_end])
            
            # Move start position for next chunk, including overlap
            start = max(chunk_end - self.overlap_size, 0)
        
        return chunks

    def _find_sentence_boundary(self, text: str) -> int:
        """Find the last sentence boundary in text"""
        matches = list(re.finditer(r'[.!?]\s+', text))
        if matches:
            return matches[-1].end()
        return -1

    def _find_word_boundary(self, text: str) -> int:
        """Find the last word boundary in text"""
        matches = list(re.finditer(r'\s+', text))
        if matches:
            return matches[-1].start()
        return -1
        
    def get_chunk_info(self, chunks: List[TextChunk]) -> Dict:
        """Get statistics about the chunks"""
        sizes = [chunk.size for chunk in chunks]
        return {
            'total_chunks': len(chunks),
            'average_size': sum(sizes) / len(chunks) if chunks else 0,
            'min_size': min(sizes) if chunks else 0,
            'max_size': max(sizes) if chunks else 0
        }