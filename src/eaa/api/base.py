from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class BaseConfig:
    """Base class for simple configuration objects."""

    def to_dict(self) -> dict:
        """Return the configuration as a dictionary."""
        return asdict(self)

    def fields(self) -> list[str]:
        """Return the dataclass field names."""
        return [field.name for field in fields(self)]

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "BaseConfig":
        """Build the configuration from a mapping."""
        if payload is None:
            return cls()  # type: ignore[misc]
        if not isinstance(payload, dict):
            raise TypeError("Input must be a dictionary.")
        allowed = {field.name for field in fields(cls)}
        kwargs = {key: payload[key] for key in allowed if key in payload}
        return cls(**kwargs)  # type: ignore[arg-type]
