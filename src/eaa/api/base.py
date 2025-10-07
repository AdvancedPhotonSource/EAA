from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional


@dataclass
class BaseConfig:
    """A base class for configurations."""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def fields(self) -> list[str]:
        return [f.name for f in fields(self)]
    
    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "BaseConfig":
        if payload is None:
            return cls()  # type: ignore[misc]
        if not isinstance(payload, dict):
            raise TypeError("Input must be a dictionary.")
        allowed = {field.name for field in fields(cls)}
        kwargs = {key: payload[key] for key in allowed if key in payload}
        return cls(**kwargs)  # type: ignore[arg-type]
