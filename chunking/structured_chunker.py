"""Module for extracting structured chunks from PDF documents."""

import os
import re
import uuid
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, CHUNK_OVERLAP
from utils import normalize_text, count_tokens
from chunking.pdf_extractor import PDFExtractor

class StructuredChunker:
    """Extract structured chunks from PDF documents."""
    
    def __init__(self, 
                max_chunk_size: int = MAX_CHUNK_SIZE,
                min_chunk_size: int = MIN_CHUNK_SIZE,
                chunk_overlap: int = CHUNK_OVERLAP):
        """Initialize the chunker with configuration parameters."""
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_extractor = PDFExtractor()
        
    def extract_chunks_from_pdf(self, 
                               file_path: str, 
                               document_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract structured chunks from a PDF file."""
        if document_name is None:
            document_name = os.path.basename(file_path)
            
        # Extract text using our multi-method extractor
        full_text = self.pdf_extractor.extract_text(file_path)
        
        if not full_text or full_text.isspace():
            print(f"No text extracted from {document_name}")
            return []
            
        # Get page count for metadata
        try:
            with fitz.open(file_path) as doc:
                page_count = len(doc)
        except:
            page_count = 1  # Default if we can't determine
            
        # If we can't determine page breaks, treat as one page
        # and chunk by sections and size
        chunks = self._chunk_text(full_text, document_name, page_count)
        
        return chunks
    
    def _chunk_text(self, 
                   text: str, 
                   document_name: str,
                   page_count: int) -> List[Dict[str, Any]]:
        """Chunk text by sections and size."""
        # Identify potential section headers
        section_pattern = r'(?m)^(?:(?:\d+\.)+\s+|\b(?:[A-Z][a-z]*\s*){1,5}:|\b[A-Z][A-Z\s]{3,20}\b)'
        sections = re.split(f'({section_pattern})', text)
        
        chunks = []
        current_section = ""
        current_text = ""
        current_tokens = 0
        
        # Estimate page number based on position in the document
        # This is a rough approximation since we don't have actual page breaks
        def estimate_page(position, total_length, page_count):
            if total_length == 0:
                return 1
            return min(page_count, max(1, int(position / total_length * page_count) + 1))
        
        total_length = len(text)
        current_position = 0
        
        # Process each section
        for i, section_part in enumerate(sections):
            # Skip empty sections
            if not section_part or section_part.isspace():
                continue
                
            # Check if this part is a section header
            if i % 2 == 1:  # Odd indices are section headers from the split
                current_section = section_part.strip()
                current_position += len(section_part)
                continue
                
            text_part = section_part.strip()
            if not text_part:
                current_position += len(section_part)
                continue
                
            # If adding this part would exceed max chunk size, create a new chunk
            part_tokens = count_tokens(text_part)
            
            if current_tokens + part_tokens > self.max_chunk_size and current_tokens >= self.min_chunk_size:
                # Create a chunk with the current accumulated text
                normalized_text = normalize_text(current_text)
                if normalized_text:  # Only add non-empty chunks
                    estimated_page = estimate_page(current_position, total_length, page_count)
                    chunk_id = str(uuid.uuid4())
                    chunks.append({
                        'id': chunk_id,
                        'text': normalized_text,
                        'document': document_name,
                        'page': estimated_page,
                        'section': current_section,
                        'token_count': current_tokens
                    })
                
                # Start a new chunk with overlap
                overlap_text = " ".join(current_text.split()[-self.chunk_overlap:]) if current_text else ""
                current_text = overlap_text + " " + text_part
                current_tokens = count_tokens(current_text)
            else:
                # Add to the current chunk
                current_text += " " + text_part
                current_tokens += part_tokens
                
            current_position += len(section_part)
        
        # Add the final chunk if it's not empty
        if current_text and current_tokens >= self.min_chunk_size:
            normalized_text = normalize_text(current_text)
            if normalized_text:  # Only add non-empty chunks
                estimated_page = estimate_page(current_position, total_length, page_count)
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    'id': chunk_id,
                    'text': normalized_text,
                    'document': document_name,
                    'page': estimated_page,
                    'section': current_section,
                    'token_count': current_tokens
                })
            
        return chunks
