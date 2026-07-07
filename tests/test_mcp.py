import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Annotated, get_args, get_origin

import pytest
from fastmcp import Context, FastMCP

from eaa_core.task_manager.tool_executor import SerialToolExecutor
from eaa_core.tool.mcp_client import MCPTool


def load_module(module_name: str, path: Path):
    """Load a module directly from a filesystem path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mcp_tool_registers_remote_tools_and_normalizes_json_results(tmp_path):
    script_path = tmp_path / "mcp_server.py"
    repo_src = Path(__file__).resolve().parents[1] / "src"
    script_path.write_text(
        "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {str(repo_src)!r})",
                "from eaa_core.tool.base import BaseTool, tool",
                "from eaa_core.tool.mcp_server import run_mcp_server_from_tools",
                "",
                "class RemoteImageTool(BaseTool):",
                "    @tool(name='remote_image')",
                "    def remote_image(self) -> str:",
                "        return 'remote.png'",
                "",
                "if __name__ == '__main__':",
                "    run_mcp_server_from_tools(RemoteImageTool(), server_name='RemoteImageServer')",
            ]
        )
    )

    mcp_tool = MCPTool(
        {
            "mcpServers": {
                "remote_image": {
                    "command": sys.executable,
                    "args": [str(script_path)],
                }
            }
        }
    )

    try:
        schemas = mcp_tool.get_all_schema()
        assert schemas[0]["function"]["name"] == "remote_image"
        assert mcp_tool.remote_image() == {"img_path": "remote.png"}

        executor = SerialToolExecutor()
        executor.register_tools(mcp_tool)

        result = executor.execute_tool_call(
            {
                "id": "call-1",
                "function": {
                    "name": "remote_image",
                    "arguments": "{}",
                },
            }
        )

        assert json.loads(result.message["content"]) == {"img_path": "remote.png"}
    finally:
        mcp_tool._run_coroutine(mcp_tool.disconnect())


def test_mcp_tool_preserves_remote_argument_signatures_and_schemas(tmp_path):
    script_path = tmp_path / "mcp_server.py"
    repo_src = Path(__file__).resolve().parents[1] / "src"
    script_path.write_text(
        "\n".join(
            [
                "import sys",
                "from typing import Annotated",
                f"sys.path.insert(0, {str(repo_src)!r})",
                "from eaa_core.tool.base import BaseTool, tool",
                "from eaa_core.tool.mcp_server import run_mcp_server_from_tools",
                "",
                "class RemoteMathTool(BaseTool):",
                "    @tool(name='add')",
                "    def add(",
                "        self,",
                "        a: Annotated[float, 'The first addend.'],",
                "        b: Annotated[float, 'The second addend.'],",
                "    ) -> float:",
                "        return a + b",
                "",
                "if __name__ == '__main__':",
                "    run_mcp_server_from_tools(RemoteMathTool(), server_name='RemoteMathServer')",
            ]
        )
    )

    mcp_tool = MCPTool(
        {
            "mcpServers": {
                "remote_math": {
                    "command": sys.executable,
                    "args": [str(script_path)],
                }
            }
        }
    )

    try:
        add_spec = next(
            spec
            for spec in mcp_tool.exposed_tools
            if spec.name == "add"
        )
        signature = inspect.signature(add_spec.function)

        assert list(signature.parameters) == ["a", "b"]
        assert signature.parameters["a"].kind is inspect.Parameter.KEYWORD_ONLY
        assert get_origin(signature.parameters["a"].annotation) is Annotated
        assert get_args(signature.parameters["a"].annotation) == (
            float,
            "The first addend.",
        )
        assert get_args(signature.parameters["b"].annotation) == (
            float,
            "The second addend.",
        )

        executor = SerialToolExecutor()
        executor.register_tools(mcp_tool)
        schemas = {
            schema["function"]["name"]: schema
            for schema in executor.list_tool_schemas()
        }

        assert schemas["add"]["function"]["parameters"]["properties"] == {
            "a": {"type": "number", "description": "The first addend."},
            "b": {"type": "number", "description": "The second addend."},
        }
        assert schemas["add"]["function"]["parameters"]["required"] == ["a", "b"]
    finally:
        mcp_tool._run_coroutine(mcp_tool.disconnect())


def test_mcp_tool_hides_external_attribute_payload_from_model_schemas(tmp_path):
    script_path = tmp_path / "mcp_server.py"
    repo_src = Path(__file__).resolve().parents[1] / "src"
    script_path.write_text(
        "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {str(repo_src)!r})",
                "from eaa_core.tool.base import BaseTool, tool",
                "from eaa_core.tool.mcp_server import run_mcp_server_from_tools",
                "",
                "class ExternalAcquisitionTool(BaseTool):",
                "    def build(self):",
                "        self.image_k = {",
                "            'encoded_data': {",
                "                'type': 'array',",
                "                'dtype': 'float32',",
                "                'shape': [1],",
                "                'data': 'AAAAAA==',",
                "            }",
                "        }",
                "",
                "    @tool(name='acquire_image')",
                "    def acquire_image(self) -> dict:",
                "        return {'psize': 1.0}",
                "",
                "if __name__ == '__main__':",
                "    run_mcp_server_from_tools(ExternalAcquisitionTool(), server_name='ExternalAcquisitionServer')",
            ]
        )
    )

    mcp_tool = MCPTool(
        {
            "mcpServers": {
                "external_acquisition": {
                    "command": sys.executable,
                    "args": [str(script_path)],
                }
            }
        }
    )

    try:
        exposed = {spec.name: spec for spec in mcp_tool.exposed_tools}
        assert exposed["external_acquisition_tool.get_attribute_payload"].model_visible is False
        assert getattr(mcp_tool, "external_acquisition_tool.get_attribute_payload")(
            name="image_k"
        ) == {
            "encoded_data": {
                "type": "array",
                "dtype": "float32",
                "shape": [1],
                "data": "AAAAAA==",
            },
        }
        assert [
            schema["function"]["name"]
            for schema in mcp_tool.get_all_schema()
        ] == ["acquire_image"]
    finally:
        mcp_tool._run_coroutine(mcp_tool.disconnect())


def test_mcp_tool_forwards_remote_log_and_progress_notifications():
    mcp = FastMCP("LoggingServer")

    @mcp.tool(name="scan")
    async def scan(ctx: Context) -> dict[str, str]:
        await ctx.info("scan point 1")
        await ctx.report_progress(progress=1, message="scan point 1")
        await ctx.report_progress(progress=2, message="scan point 2")
        return {"status": "done"}

    class RuntimeLogSink:
        def __init__(self):
            self.logs = []

        def publish_log(self, message, **kwargs):
            self.logs.append({"message": message, **kwargs})

    runtime_log_sink = RuntimeLogSink()
    mcp_tool = MCPTool(mcp)
    mcp_tool.set_runtime_log_handler(runtime_log_sink.publish_log)

    try:
        assert mcp_tool.scan() == {"status": "done"}
    finally:
        mcp_tool._run_coroutine(mcp_tool.disconnect())

    assert [
        log["message"]
        for log in runtime_log_sink.logs
    ] == ["scan point 1", "scan point 2"]
    assert runtime_log_sink.logs[0]["level"] == "info"
    assert runtime_log_sink.logs[0]["tool_name"] == "scan"
    assert runtime_log_sink.logs[1]["level"] == "progress"
    assert runtime_log_sink.logs[1]["progress"] == 2.0


def test_model_hidden_tools_are_mcp_callable_but_not_model_visible():
    from eaa_core.tool.base import BaseTool, tool
    from eaa_core.tool.mcp_server import MCPToolServer

    class HiddenPayloadTool(BaseTool):
        @tool(name="visible")
        def visible(self) -> str:
            return "ok"

        @tool(name="hidden_payload", model_visible=False)
        def hidden_payload(self) -> dict:
            return {
                "encoded_data": {
                    "type": "array",
                    "dtype": "float32",
                    "shape": [1],
                    "data": "AAAAAA==",
                }
            }

    tool = HiddenPayloadTool()
    server = MCPToolServer(tools=tool)
    executor = SerialToolExecutor()
    executor.register_tools(tool)

    assert {spec.name for spec in tool.exposed_tools} == {
        "visible",
        "hidden_payload",
        "hidden_payload_tool.get_attribute_payload",
    }
    assert server.list_tools() == [
        "visible",
        "hidden_payload",
        "hidden_payload_tool.get_attribute_payload",
    ]
    assert tool.hidden_payload() == {
        "encoded_data": {
            "type": "array",
            "dtype": "float32",
            "shape": [1],
            "data": "AAAAAA==",
        }
    }
    assert [
        schema["function"]["name"]
        for schema in server.get_tool_schemas()
    ] == ["visible"]
    assert [
        schema["function"]["name"]
        for schema in executor.list_tool_schemas()
    ] == ["visible"]
    assert list(executor.tool_specs) == ["visible"]

    with pytest.raises(ValueError, match="Unknown tool requested"):
        executor.execute_tool_call(
            {
                "id": "call-hidden",
                "function": {
                    "name": "hidden_payload",
                    "arguments": "{}",
                },
            }
        )


def test_calculator_example_main_builds_server_with_refactored_tool_specs(monkeypatch):
    example_path = Path(__file__).resolve().parents[1] / "examples" / "mcp_calculator_server.py"
    example_module = load_module("mcp_calculator_server_example", example_path)
    captured = {}

    def fake_run_mcp_server_from_tools(*, tools, server_name):
        captured["tools"] = tools
        captured["server_name"] = server_name

    monkeypatch.setattr(
        example_module,
        "run_mcp_server_from_tools",
        fake_run_mcp_server_from_tools,
    )

    example_module.main()

    assert captured["server_name"] == "Calculator MCP Server"
    assert [spec.name for spec in captured["tools"].exposed_tools] == [
        "calculator_tool.add",
        "calculator_tool.subtract",
        "calculator_tool.multiply",
        "calculator_tool.divide",
        "calculator_tool.get_history",
        "calculator_tool.clear_history",
        "calculator_tool.get_attribute_payload",
    ]


def test_calculator_example_banner_writes_to_stderr(capsys):
    example_path = Path(__file__).resolve().parents[1] / "examples" / "mcp_calculator_server.py"
    example_module = load_module("mcp_calculator_server_banner", example_path)

    example_module.print_banner()

    captured = capsys.readouterr()
    assert "Calculator MCP Server Example" in captured.err
    assert captured.out == ""
