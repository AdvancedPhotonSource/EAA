from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import json
import logging

from eaa.core.message_proc import generate_openai_message
from eaa.core.tooling.base import (
    BaseTool,
    ExposedToolSpec,
    ToolReturnType,
    generate_openai_tool_schema,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Normalized tool execution result."""

    message: dict[str, Any]
    return_type: ToolReturnType


class SerialToolExecutor:
    """Serial, thread-free tool execution for task managers."""

    def __init__(
        self,
        approval_handler: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        allow_parallel_tool_execution: bool = False,
    ):
        """Initialize the executor."""
        self.approval_handler = approval_handler
        self.allow_parallel_tool_execution = allow_parallel_tool_execution
        self.tools: list[BaseTool] = []
        self.tool_specs: dict[str, ExposedToolSpec] = {}
        self.tool_execution_history: list[dict[str, Any]] = []

    def register_tools(self, tools: BaseTool | list[BaseTool]) -> None:
        """Register one or more tool objects."""
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError("Input should be a BaseTool or a list of BaseTool objects.")
            self.tools.append(tool)
            for exposed in tool.exposed_tools:
                spec = ExposedToolSpec(
                    name=exposed.name,
                    function=exposed.function,
                    return_type=exposed.return_type,
                    require_approval=(
                        tool.require_approval if exposed.require_approval is None else exposed.require_approval
                    ),
                )
                self.tool_specs[spec.name] = spec

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        """Return model-facing OpenAI tool schemas."""
        return [
            spec.schema or generate_openai_tool_schema(tool_name=name, func=spec.function)
            for name, spec in self.tool_specs.items()
        ]

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolExecutionResult]:
        """Execute assistant-requested tool calls serially."""
        return [self.execute_tool_call(tool_call) for tool_call in tool_calls]

    def execute_tool_call(self, tool_call: dict[str, Any]) -> ToolExecutionResult:
        """Execute one tool call and normalize its response."""
        function = tool_call.get("function", {})
        tool_name = function.get("name")
        if tool_name not in self.tool_specs:
            raise ValueError(f"Unknown tool requested: {tool_name}")
        spec = self.tool_specs[tool_name]
        arguments = self.parse_arguments(function.get("arguments"))
        if spec.require_approval and self.approval_handler is not None:
            approved = self.approval_handler(tool_name, arguments)
            if not approved:
                message = generate_openai_message(
                    content="Tool execution was denied by the user.",
                    role="tool",
                    tool_call_id=tool_call.get("id"),
                )
                return ToolExecutionResult(message=message, return_type=ToolReturnType.EXCEPTION)
        try:
            result = spec.function(**arguments)
            content = self.serialize_result(result, spec.return_type)
            return_type = spec.return_type
        except Exception as exc:
            logger.exception("Tool execution failed for %s", tool_name)
            content = str(exc)
            return_type = ToolReturnType.EXCEPTION
        message = generate_openai_message(
            content=content,
            role="tool",
            tool_call_id=tool_call.get("id"),
        )
        self.tool_execution_history.append({"tool_name": tool_name, "arguments": arguments})
        return ToolExecutionResult(message=message, return_type=return_type)

    @staticmethod
    def parse_arguments(arguments: Any) -> dict[str, Any]:
        """Parse assistant-provided tool arguments."""
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            raise ValueError(f"Unsupported tool argument payload: {type(arguments)}")
        stripped = arguments.strip()
        if len(stripped) == 0:
            return {}
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode into a dictionary.")
        return parsed

    @staticmethod
    def serialize_result(result: Any, return_type: ToolReturnType) -> str:
        """Serialize a tool result into message content."""
        if return_type == ToolReturnType.IMAGE_PATH:
            if not isinstance(result, str):
                raise ValueError("IMAGE_PATH tools must return a string path.")
            return result
        if return_type in {ToolReturnType.DICT, ToolReturnType.LIST}:
            return json.dumps(result)
        if return_type == ToolReturnType.BOOL:
            return json.dumps(bool(result))
        if return_type == ToolReturnType.NUMBER:
            return json.dumps(result)
        return str(result)
