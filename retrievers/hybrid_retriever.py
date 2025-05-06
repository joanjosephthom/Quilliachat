"""Hybrid retriever combining BM25 and dense retrieval."""

from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrievers.base_retriever import BaseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from config import BM25_WEIGHT, DENSE_WEIGHT
from utils import merge_results

class HybridRetriever(BaseRetriever):
    """Retriever combining BM25 and dense retrieval methods."""
    
    def __init__(self, 
                bm25_weight: float = BM25_WEIGHT, 
                dense_weight: float = DENSE_WEIGHT):
        """Initialize the hybrid retriever with BM25 and dense retrievers."""
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to both BM25 and dense indices."""
        self.bm25_retriever.add_documents(chunks)
        self.dense_retriever.add_documents(chunks)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant chunks using both BM25 and dense retrieval."""
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k*2)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k*2)
        
        # Merge and rank the results
        merged_results = merge_results(
            bm25_results, 
            dense_results, 
            self.bm25_weight, 
            self.dense_weight, 
            top_k
        )
        
        return merged_results
    
    def clear(self) -> None:
        """Clear both BM25 and dense indices."""
        self.bm25_retriever.clear()
        self.dense_retriever.clear()
        
    def save(self, file_hash: str) -> None:
        """Save both BM25 and dense indices to disk."""
        self.bm25_retriever.save(file_hash)
        self.dense_retriever.save(file_hash)
        
    def load(self, file_hash: str) -> bool:
        """Load both BM25 and dense indices from disk if available."""
        bm25_loaded = self.bm25_retriever.load(file_hash)
        dense_loaded = self.dense_retriever.load(file_hash)
        return bm25_loaded and dense_loaded
