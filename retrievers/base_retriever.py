"""Base retriever interface for all retrieval methods."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    
    @abstractmethod
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the retriever's index."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a query."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the retriever's index."""
        pass
