import logging
import sys
from unittest.mock import Mock

import pytest

# Ensure fastmcp is available before importing MCPTool
sys.modules.setdefault("fastmcp", Mock())

from eaa.agents.base import ToolManager
from eaa.tools.base import BaseTool, ToolReturnType, check
from eaa.tools.mcp import MCPTool

import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DummyFunctionTool(BaseTool):
    @check
    def __init__(self):
        super().__init__(require_approval=True)
        self.exposed_tools = [
            {
                "name": "dummy_tool",
                "function": self.dummy_tool,
                "return_type": ToolReturnType.TEXT,
            }
        ]

    def dummy_tool(self) -> str:
        return "ok"


class DummyMCPTool(MCPTool):
    name = "dummy_mcp"

    def __init__(self, *, require_approval: bool = True):
        BaseTool.__init__(self, build=False, require_approval=require_approval)
        self._connected = False
        self._client = None
        self.calls = []
        self._schemas = [
            {
                "type": "function",
                "function": {
                    "name": "dummy_tool",
                    "description": "Dummy MCP tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
        self._tool_names = [schema["function"]["name"] for schema in self._schemas]

    async def call_tool(self, tool_name: str, arguments: dict):
        self.calls.append((tool_name, arguments))
        return {"echo": arguments}

    def get_all_schema(self):
        return self._schemas

    def get_all_tool_names(self):
        return self._tool_names


class TestToolManager(tutils.BaseTester):
    def test_function_tool_registration_and_execution_with_approval(self):
        manager = ToolManager()
        tool = DummyFunctionTool()
        manager.add_tool(tool)

        with pytest.raises(PermissionError):
            manager.execute_tool("dummy_tool", {})

        decision_log = []

        def approval(name, kwargs):
            decision_log.append((name, kwargs))
            return True

        manager.set_approval_handler(approval)
        assert manager.execute_tool("dummy_tool", {}) == "ok"
        assert decision_log == [("dummy_tool", {})]

    def test_mcp_tool_registration_and_execution_with_approval(self):
        manager = ToolManager()
        tool = DummyMCPTool(require_approval=True)
        manager.add_tool(tool)

        with pytest.raises(PermissionError):
            manager.execute_tool("dummy_tool", {"value": 1})

        decisions = []

        def approval_handler(name, kwargs):
            decisions.append((name, kwargs))
            return True

        manager.set_approval_handler(approval_handler)
        result = manager.execute_tool("dummy_tool", {"value": 2})

        assert result == {"echo": {"value": 2}}
        assert tool.calls == [("dummy_tool", {"value": 2})]
        assert decisions == [("dummy_tool", {"value": 2})]


if __name__ == "__main__":
    tester = TestToolManager()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_function_tool_registration_and_execution_with_approval()
    tester.test_mcp_tool_registration_and_execution_with_approval()
