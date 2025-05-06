"""Module for building and managing indices for document retrieval."""

from typing import List, Dict, Any, Optional, Union
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrievers.base_retriever import BaseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_retriever import HybridRetriever
from chunking.structured_chunker import StructuredChunker
from utils import compute_file_hash

class IndexBuilder:
    """Build and manage indices for document retrieval."""
    
    def __init__(self):
        """Initialize the index builder."""
        self.chunker = StructuredChunker()
        self.retrievers = {
            "bm25": BM25Retriever(),
            "dense": DenseRetriever(),
            "hybrid": HybridRetriever()
        }
        self.current_file_hash = None
        
    def process_documents(self, 
                         file_paths: List[str], 
                         document_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process documents and build indices."""
        all_chunks = []
        
        # Process each document
        for i, file_path in enumerate(file_paths):
            doc_name = document_names[i] if document_names and i < len(document_names) else None
            chunks = self.chunker.extract_chunks_from_pdf(file_path, doc_name)
            all_chunks.extend(chunks)
            
        # Build indices for all retrievers
        for retriever in self.retrievers.values():
            retriever.add_documents(all_chunks)
            
        return all_chunks
    
    def process_file_bytes(self, 
                          file_bytes: bytes, 
                          file_name: str) -> List[Dict[str, Any]]:
        """Process a file from its bytes and build indices."""
        # Compute file hash for caching
        file_hash = compute_file_hash(file_bytes)
        self.current_file_hash = file_hash
        
        # Check if indices are already cached
        if self._load_indices(file_hash):
            # Return the chunks from one of the retrievers
            if isinstance(self.retrievers["bm25"], BM25Retriever) and self.retrievers["bm25"].chunks:
                return self.retrievers["bm25"].chunks
            return []
            
        # Save the file temporarily
        temp_path = f"temp_{file_hash}.pdf"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
            
        try:
            # Process the document
            chunks = self.chunker.extract_chunks_from_pdf(temp_path, file_name)
            
            if not chunks:
                print(f"Warning: No chunks extracted from {file_name}")
                return []
                
            # Build indices for all retrievers
            for retriever_type, retriever in self.retrievers.items():
                try:
                    retriever.add_documents(chunks)
                except Exception as e:
                    print(f"Error building {retriever_type} index: {e}")
                    
            # Save indices for future use
            self._save_indices(file_hash)
            
            return chunks
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            return []
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def get_retriever(self, retriever_type: str) -> BaseRetriever:
        """Get a retriever by type."""
        return self.retrievers.get(retriever_type)
    
    def _save_indices(self, file_hash: str) -> None:
        """Save all indices to disk."""
        for retriever in self.retrievers.values():
            retriever.save(file_hash)
            
    def _load_indices(self, file_hash: str) -> bool:
        """Load all indices from disk if available."""
        all_loaded = True
        for retriever in self.retrievers.values():
            if not retriever.load(file_hash):
                all_loaded = False
        return all_loaded
    
    def clear(self) -> None:
        """Clear all retrievers and reset the state."""
        for retriever in self.retrievers.values():
            retriever.clear()
        self.current_file_hash = None
