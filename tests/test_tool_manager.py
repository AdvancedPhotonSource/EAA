import pytest

from eaa.agents.base import ToolManager
from eaa.tools.base import ToolReturnType


def test_tool_manager_enforces_approval():
    manager = ToolManager()

    def sample() -> str:
        return "ok"

    manager.add_function_tool(
        name="sample",
        tool_function=sample,
        return_type=ToolReturnType.TEXT,
        require_approval=True,
    )

    with pytest.raises(PermissionError):
        manager.execute_tool("sample", {})

    manager.set_approval_handler(lambda name, kwargs: False)
    with pytest.raises(PermissionError):
        manager.execute_tool("sample", {})

    manager.set_approval_handler(lambda name, kwargs: True)
    assert manager.execute_tool("sample", {}) == "ok"
