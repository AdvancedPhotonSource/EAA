from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class MemoryRecord:
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


@dataclass
class MemoryQueryResult:
    record: MemoryRecord
    score: float
    highlights: Optional[str] = None


__all__ = ["MemoryRecord", "MemoryQueryResult"]
