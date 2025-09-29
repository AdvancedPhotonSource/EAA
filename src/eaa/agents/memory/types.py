from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar
import time
import uuid

T = TypeVar("T", bound="MemorySchema")


class MemorySchema:
    """Shared helper with a `from_dict` classmethod for memory dataclasses."""

    @classmethod
    def from_dict(cls: Type[T], payload: Optional[Dict[str, Any]]) -> T:
        if payload is None:
            return cls()  # type: ignore[misc]
        if not isinstance(payload, dict):
            raise TypeError("memory configuration must be a dictionary")
        allowed = {field.name for field in fields(cls)}
        kwargs = {key: payload[key] for key in allowed if key in payload}
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class MemoryRecord(MemorySchema):
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


@dataclass
class MemoryQueryResult(MemorySchema):
    record: MemoryRecord
    score: float
    highlights: Optional[str] = None


__all__ = ["MemorySchema", "MemoryRecord", "MemoryQueryResult"]
