import inspect
import json
from typing import get_args

from eaa_core.tool.http_client import HTTPTransportedTool


class DummyHTTPResponse:
    """Minimal HTTP response stub for urllib tests."""

    def __init__(self, body: str, content_type: str = "application/json") -> None:
        """Initialize the response stub.

        Parameters
        ----------
        body : str
            Response body payload.
        content_type : str, optional
            HTTP content type.
        """
        self.body = body.encode("utf-8")
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        """Return the response body bytes."""
        return self.body

    def __enter__(self) -> "DummyHTTPResponse":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        """Exit the context manager."""
        return False


def test_http_transported_tool_exposes_each_endpoint_as_a_tool() -> None:
    http_tool = HTTPTransportedTool(
        base_url="http://example.com/api",
        tool_definitions=[
            {
                "tool_name": "acquire",
                "endpoint_name": "acquire",
                "description": "Acquire an image.",
                "input_signautures": {
                    "x_position": {
                        "type": "integer",
                        "description": "X coordinate.",
                        "required": True,
                    },
                    "y_position": {
                        "type": "integer",
                        "description": "Y coordinate.",
                    },
                },
                "output_signatures": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                    },
                },
            },
            {
                "tool_name": "focus",
                "endpoint_name": "/focus",
                "description": "Adjust focus.",
                "input_signautures": {
                    "z_position": {
                        "type": "number",
                        "description": "Focus position.",
                        "required": True,
                    }
                },
                "output_signatures": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                    },
                },
            },
        ],
    )

    assert http_tool.get_all_tool_names() == ["acquire", "focus"]

    acquire_spec = next(spec for spec in http_tool.exposed_tools if spec.name == "acquire")
    focus_spec = next(spec for spec in http_tool.exposed_tools if spec.name == "focus")

    acquire_signature = inspect.signature(acquire_spec.function)
    focus_signature = inspect.signature(focus_spec.function)

    assert list(acquire_signature.parameters) == ["x_position", "y_position"]
    assert acquire_signature.parameters["x_position"].default is inspect.Parameter.empty
    assert acquire_signature.parameters["y_position"].default is None
    assert get_args(acquire_signature.parameters["x_position"].annotation) == (
        int,
        "X coordinate.",
    )

    assert list(focus_signature.parameters) == ["z_position"]
    assert get_args(focus_signature.parameters["z_position"].annotation) == (
        float,
        "Focus position.",
    )

    schemas = {schema["function"]["name"]: schema for schema in http_tool.get_all_schema()}
    assert schemas["acquire"]["function"]["parameters"]["required"] == ["x_position"]
    assert schemas["focus"]["function"]["parameters"]["required"] == ["z_position"]
    assert '"status"' in schemas["acquire"]["function"]["description"]
    assert '"success"' in schemas["focus"]["function"]["description"]


def test_http_transported_tool_routes_payload_to_matching_endpoint(monkeypatch) -> None:
    captured_request = {}

    def fake_urlopen(http_request, timeout):
        captured_request["url"] = http_request.full_url
        captured_request["body"] = http_request.data.decode("utf-8")
        captured_request["headers"] = dict(http_request.headers)
        captured_request["timeout"] = timeout
        return DummyHTTPResponse(body=json.dumps({"result": {"status": "ok"}}))

    monkeypatch.setattr("eaa_core.tool.http_client.request.urlopen", fake_urlopen)

    http_tool = HTTPTransportedTool(
        base_url="http://example.com/api",
        tool_definitions=[
            {
                "tool_name": "acquire",
                "endpoint_name": "acquire",
                "description": "Acquire an image.",
                "input_signautures": {
                    "x_position": {"type": "integer", "required": True},
                    "y_position": {"type": "integer", "required": True},
                },
                "output_signatures": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                    },
                },
            },
            {
                "tool_name": "focus",
                "endpoint_name": "focus",
                "description": "Adjust focus.",
                "input_signautures": {},
                "output_signatures": {},
            },
        ],
    )

    acquire_callable = next(spec.function for spec in http_tool.exposed_tools if spec.name == "acquire")
    result = acquire_callable(x_position=100, y_position=100)

    assert result == {"status": "ok"}
    assert captured_request["url"] == "http://example.com/api/acquire"
    assert json.loads(captured_request["body"]) == {
        "x_position": 100,
        "y_position": 100,
    }
    assert captured_request["headers"]["Content-type"] == "application/json"
    assert captured_request["headers"]["Accept"] == "application/json"
    assert captured_request["timeout"] is None
