from dataclasses import dataclass
from typing import Optional

from eaa.api.base import BaseConfig


@dataclass
class MemoryManagerConfig(BaseConfig):
    """Configuration for optional long-term memory integration."""

    enabled: bool = False
    write_enabled: bool = True
    retrieval_enabled: bool = True
    top_k: int = 5
    score_threshold: float = 0.25
    min_content_length: int = 12
    embedding_model: Optional[str] = None
    vector_store_path: Optional[str] = None
    injection_role: str = "system"
