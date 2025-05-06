"""Dense retriever implementation using sentence transformers and FAISS."""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import faiss
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrievers.base_retriever import BaseRetriever
from config import EMBEDDING_MODEL
from utils import create_cache_dir, compute_file_hash

class DenseRetriever(BaseRetriever):
    """Retriever using dense embeddings and FAISS for vector search."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the dense retriever with a sentence transformer model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.cache_dir = create_cache_dir()
        
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the dense index."""
        self.chunks = chunks
        
        # Create embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create FAISS index
        vector_dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)  # Inner product for cosine similarity
        self.index.add(self.embeddings)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve the top-k most relevant chunks for a query using dense embeddings."""
        if self.index is None or len(self.chunks) == 0:
            return []
            
        # Encode the query
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return the top-k chunks with their scores
        results = [(self.chunks[idx], float(scores[0][i])) 
                  for i, idx in enumerate(indices[0])]
        
        return results
    
    def clear(self) -> None:
        """Clear the dense index."""
        self.index = None
        self.chunks = []
        self.embeddings = None
        
    def save(self, file_hash: str) -> None:
        if self.index is None:
            return
        model_tag = self.model_name.replace('/', '_')
        cache_path = os.path.join(self.cache_dir, f"{file_hash}_{model_tag}_dense.pkl")
        faiss_path = os.path.join(self.cache_dir, f"{file_hash}_{model_tag}_faiss.index")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)
        faiss.write_index(self.index, faiss_path)
            
    def load(self, file_hash: str) -> bool:
        """Load the dense index from disk if available."""
        model_tag = self.model_name.replace('/', '_')
        cache_path = os.path.join(self.cache_dir, f"{file_hash}_{model_tag}_dense.pkl")
        faiss_path = os.path.join(self.cache_dir, f"{file_hash}_{model_tag}_faiss.index")
        
        if os.path.exists(cache_path) and os.path.exists(faiss_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data['chunks']
                    self.embeddings = data['embeddings']
                
                self.index = faiss.read_index(faiss_path)
                return True
            except Exception:
                return False
        return False
