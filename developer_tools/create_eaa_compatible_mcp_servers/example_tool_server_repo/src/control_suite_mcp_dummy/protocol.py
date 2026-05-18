"""Shared command protocol helpers for the dummy control suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import uuid


Status = Literal["ok", "error"]


@dataclass(frozen=True)
class Command:
    """A worker command sent over ZMQ.

    Parameters
    ----------
    method
        Worker method name to execute.
    params
        JSON-serializable method parameters.
    command_id
        Unique command identifier.
    """

    method: str
    params: dict[str, Any]
    command_id: str

    @classmethod
    def create(cls, method: str, params: dict[str, Any] | None = None) -> "Command":
        """Create a command with a generated UUID."""
        return cls(
            method=method,
            params={} if params is None else params,
            command_id=str(uuid.uuid4()),
        )

    def to_message(self) -> dict[str, Any]:
        """Return the JSON-serializable command envelope."""
        return {
            "id": self.command_id,
            "method": self.method,
            "params": self.params,
        }


def ok_response(command_id: str | None, result: Any) -> dict[str, Any]:
    """Build a successful worker response."""
    return {
        "id": command_id,
        "status": "ok",
        "result": result,
    }


def error_response(command_id: str | None, error: str) -> dict[str, Any]:
    """Build an error worker response."""
    return {
        "id": command_id,
        "status": "error",
        "error": error,
    }

