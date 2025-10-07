from typing import Optional
from dataclasses import dataclass

from eaa.api.base import BaseConfig


@dataclass
class MemoryManagerConfig(BaseConfig):
    enabled: bool = False
    """Whether to enable long-term memory through RAG."""
    
    write_enabled: bool = True
    """Whether to enable writing to the vector store."""
    
    retrieval_enabled: bool = True
    """Whether to enable retrieval from the vector store."""
    
    top_k: int = 5
    """The number of results to return from the vector store."""
    
    score_threshold: float = 0.25
    """The minimum score threshold for a result to be considered relevant."""
    
    min_content_length: int = 12
    """The minimum length of a content to be considered relevant."""
    
    embedding_model: Optional[str] = None
    """The model to use for embedding."""
    
    vector_store_path: Optional[str] = None
    """The path to the vector store."""
    
    injection_role: str = "system"
    """The role to use for injecting the memory into the conversation."""
