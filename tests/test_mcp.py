import sys
from pathlib import Path

from eaa.core.task_manager.tool_executor import SerialToolExecutor
from eaa.core.tooling.base import ToolReturnType
from eaa.tool.mcp import MCPTool


def test_mcp_tool_registers_remote_tools_and_preserves_return_type(tmp_path):
    script_path = tmp_path / "mcp_server.py"
    repo_src = Path(__file__).resolve().parents[1] / "src"
    script_path.write_text(
        "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {str(repo_src)!r})",
                "from eaa.core.tooling.base import BaseTool, ToolReturnType, tool",
                "from eaa.core.mcp.server import run_mcp_server_from_tools",
                "",
                "class RemoteImageTool(BaseTool):",
                "    @tool(name='remote_image', return_type=ToolReturnType.IMAGE_PATH)",
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

        executor = SerialToolExecutor()
        executor.register_tools(mcp_tool)

        result = executor.execute_tool_call(
            {
                "id": "call-1",
                "function": {"name": "remote_image", "arguments": "{}"},
            }
        )

        assert result.return_type == ToolReturnType.IMAGE_PATH
        assert result.message["content"] == "remote.png"
    finally:
        mcp_tool._run_coroutine(mcp_tool.disconnect())
