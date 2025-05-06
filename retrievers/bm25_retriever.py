"""BM25 retriever implementation."""

import os
import pickle
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrievers.base_retriever import BaseRetriever
from utils import create_cache_dir, compute_file_hash

class BM25Retriever(BaseRetriever):
    """Retriever using BM25 algorithm for sparse retrieval."""
    
    def __init__(self):
        """Initialize the BM25 retriever."""
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []
        self.cache_dir = create_cache_dir()
        
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the BM25 index."""
        if not chunks:
            print("Warning: No chunks provided to BM25Retriever")
            return
            
        self.chunks = chunks
        
        # Tokenize the chunks
        self.tokenized_chunks = [chunk['text'].lower().split() for chunk in chunks]
        
        # Filter out empty chunks to avoid division by zero
        non_empty_indices = []
        filtered_tokenized_chunks = []
        filtered_chunks = []
        
        for i, tokens in enumerate(self.tokenized_chunks):
            if tokens:  # Check if the tokenized chunk is not empty
                filtered_tokenized_chunks.append(tokens)
                filtered_chunks.append(self.chunks[i])
                non_empty_indices.append(i)
        
        if not filtered_tokenized_chunks:
            print("Warning: All chunks are empty after tokenization")
            return
            
        # Update our chunks and tokenized_chunks
        self.chunks = filtered_chunks
        self.tokenized_chunks = filtered_tokenized_chunks
        
        # Create the BM25 index
        try:
            self.bm25 = BM25Okapi(self.tokenized_chunks)
        except Exception as e:
            print(f"Error creating BM25 index: {e}")
            # If BM25 initialization fails, set to None
            self.bm25 = None
            
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a query using BM25."""
        if not self.bm25 or not self.chunks:
            print("Warning: BM25 index not initialized or no chunks available")
            return []
            
        # Tokenize the query
        tokenized_query = query.lower().split()
        
        if not tokenized_query:
            print("Warning: Empty query after tokenization")
            return []
            
        # Get BM25 scores
        try:
            scores = self.bm25.get_scores(tokenized_query)
            
            # Create (chunk, score) pairs and sort by score
            chunk_score_pairs = [(self.chunks[i], scores[i]) for i in range(len(scores))]
            chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return the top-k chunks with their scores
            return chunk_score_pairs[:top_k]
        except Exception as e:
            print(f"Error retrieving from BM25: {e}")
            return []
    
    def clear(self) -> None:
        """Clear the BM25 index."""
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []
        
    def save(self, file_hash: str) -> None:
        """Save the BM25 index to disk."""
        if not self.bm25:
            return
            
        cache_path = os.path.join(self.cache_dir, f"{file_hash}_bm25.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks': self.chunks,
                'tokenized_chunks': self.tokenized_chunks
            }, f)
            
    def load(self, file_hash: str) -> bool:
        """Load the BM25 index from disk if available."""
        cache_path = os.path.join(self.cache_dir, f"{file_hash}_bm25.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                    self.chunks = data['chunks']
                    self.tokenized_chunks = data['tokenized_chunks']
                return True
            except Exception:
                return False
        return False
