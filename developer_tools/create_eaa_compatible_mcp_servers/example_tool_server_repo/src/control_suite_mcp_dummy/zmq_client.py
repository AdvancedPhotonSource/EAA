"""ZMQ client used by the MCP server to reach the instrument worker."""

from __future__ import annotations

from typing import Any
import json

import zmq

from control_suite_mcp_dummy.protocol import Command


class WorkerClient:
    """Request/reply client for the instrument worker.

    Parameters
    ----------
    endpoint
        ZMQ endpoint of the worker, for example ``tcp://127.0.0.1:5555``.
    timeout_ms
        Send and receive timeout in milliseconds.
    """

    def __init__(self, endpoint: str, timeout_ms: int = 30_000) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Call a worker method and return the result payload."""
        command = Command.create(method=method, params=params)
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        try:
            socket.connect(self.endpoint)
            socket.send_string(json.dumps(command.to_message(), allow_nan=True))
            response = json.loads(socket.recv_string())
        except zmq.Again as exc:
            raise TimeoutError(
                f"Timed out waiting for worker response from {self.endpoint}"
            ) from exc
        finally:
            socket.close(linger=0)

        if not isinstance(response, dict):
            raise RuntimeError("Worker returned a non-object response.")
        if response.get("status") == "error":
            raise RuntimeError(str(response.get("error", "Unknown worker error")))
        if response.get("status") != "ok":
            raise RuntimeError(f"Worker returned invalid status: {response.get('status')}")
        return response.get("result")

