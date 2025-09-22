import logging
import sys
from unittest.mock import Mock

import pytest

# Ensure fastmcp is available before importing MCPTool
sys.modules.setdefault("fastmcp", Mock())

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool, ToolReturnType, ExposedToolSpec, check
from eaa.tools.mcp import MCPTool

import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DummyFunctionTool(BaseTool):
    @check
    def __init__(self):
        super().__init__(require_approval=True)
        self.exposed_tools = [
            ExposedToolSpec(
                name="function_tool",
                function=self.run,
                return_type=ToolReturnType.TEXT,
            )
        ]

    def run(self) -> str:
        return "function"


class DummyMCPTool(MCPTool):
    name = "dummy_mcp"

    def __init__(self, *, require_approval: bool = True):
        BaseTool.__init__(self, build=False, require_approval=require_approval)
        self._connected = False
        self._client = None
        self.calls = []
        self.exposed_tools = [
            ExposedToolSpec(
                name="mcp_placeholder",
                function=lambda: None,
                return_type=ToolReturnType.TEXT,
            )
        ]
        self._schemas = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_tool",
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


class DummyAgent:
    def __init__(self):
        self.tool_manager_calls = []

    def register_tools(self, tools):
        self.tool_manager_calls.append(list(tools))


class DummyTaskManager(BaseTaskManager):
    assistant_system_message = "dummy"

    def build_agent(self, *args, **kwargs):
        self.agent = DummyAgent()

    def build_tools(self, *args, **kwargs):
        # Skip automatic registration in base class build
        pass


class TestTaskManagerToolRegistration(tutils.BaseTester):
    def test_register_function_tool_via_task_manager(self):
        tool = DummyFunctionTool()
        task_manager = DummyTaskManager(tools=[tool], build=False)
        task_manager.build_agent()
        task_manager.register_tools(task_manager.tools)

        assert len(task_manager.agent.tool_manager_calls) == 1
        assert task_manager.agent.tool_manager_calls[0][0] is tool

    def test_register_mcp_tool_via_task_manager(self):
        tool = DummyMCPTool()
        task_manager = DummyTaskManager(tools=[tool], build=False)
        task_manager.build_agent()
        task_manager.register_tools(task_manager.tools)

        assert len(task_manager.agent.tool_manager_calls) == 1
        assert task_manager.agent.tool_manager_calls[0][0] is tool


if __name__ == "__main__":
    tester = TestTaskManagerToolRegistration()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_register_function_tool_via_task_manager()
    tester.test_register_mcp_tool_via_task_manager()
