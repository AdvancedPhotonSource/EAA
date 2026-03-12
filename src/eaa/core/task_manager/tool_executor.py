from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import json
import logging

from eaa.core.message_proc import generate_openai_message
from eaa.core.skill import SkillMetadata, split_markdown_into_message_sections
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
    ) -> None:
        """Initialize the executor."""
        self.approval_handler = approval_handler
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

    def execute_tool_calls_from_message(
        self,
        message: dict[str, Any],
        *,
        return_tool_return_types: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[ToolReturnType]]:
        """Execute tool calls found in an assistant message.

        Parameters
        ----------
        message : dict[str, Any]
            Assistant message that may contain tool calls.
        return_tool_return_types : bool, default=False
            Whether to return the normalized tool return types together with
            the tool messages.

        Returns
        -------
        list[dict[str, Any]] or tuple[list[dict[str, Any]], list[ToolReturnType]]
            Tool messages alone, or tool messages paired with their return
            types when ``return_tool_return_types`` is ``True``.
        """
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) == 0:
            empty = ([], []) if return_tool_return_types else []
            return empty
        results = self.execute_tool_calls(tool_calls)
        tool_messages = [result.message for result in results]
        tool_return_types = [result.return_type for result in results]
        if return_tool_return_types:
            return tool_messages, tool_return_types
        return tool_messages

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
    def parse_tool_response_payload(content: Any) -> Optional[Dict[str, Any]]:
        """Parse dict-like tool payloads from tool message content.

        Parameters
        ----------
        content : Any
            Tool message content payload.

        Returns
        -------
        dict[str, Any] or None
            Parsed dictionary payload when available.
        """
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @classmethod
    def extract_image_paths_from_tool_response(cls, content: Any) -> list[str]:
        """Extract one or more image paths from a tool response payload.

        Parameters
        ----------
        content : Any
            Tool message content payload.

        Returns
        -------
        list[str]
            Extracted image paths.
        """
        payload = cls.parse_tool_response_payload(content)
        if payload is not None:
            image_paths = payload.get("image_paths")
            if isinstance(image_paths, list):
                return [value for value in image_paths if isinstance(value, str)]
            image_path = payload.get("image_path")
            if isinstance(image_path, str):
                return [image_path]
            return []
        if isinstance(content, str):
            return [content]
        return []

    @classmethod
    def build_skill_doc_messages(
        cls,
        tool_response: Dict[str, Any],
        tool_call_info: Optional[Dict[str, Any]],
        skill_catalog: list[SkillMetadata],
    ) -> list[Dict[str, Any]]:
        """Expand skill-documentation tool payloads into OpenAI messages.

        Parameters
        ----------
        tool_response : dict[str, Any]
            Tool response message containing the documentation payload.
        tool_call_info : dict[str, Any], optional
            Tool call metadata from the originating assistant message.
        skill_catalog : list[SkillMetadata]
            Skills available to the task manager.

        Returns
        -------
        list[dict[str, Any]]
            Message sequence extracted from the skill documentation payload.
        """
        if tool_call_info is None:
            return []
        tool_name = tool_call_info.get("function", {}).get("name")
        skill_tool_names = {skill.tool_name for skill in skill_catalog}
        if tool_name not in skill_tool_names:
            return []
        payload = cls.parse_tool_response_payload(tool_response.get("content"))
        if payload is None or not isinstance(payload.get("files"), dict):
            return []
        skill_root = payload.get("path")
        skill_root_path = Path(skill_root) if isinstance(skill_root, str) else None
        messages = []
        for relative_path, file_content in payload["files"].items():
            if not isinstance(relative_path, str) or not isinstance(file_content, str):
                continue
            markdown_path = skill_root_path / relative_path if skill_root_path is not None else None
            for section in split_markdown_into_message_sections(
                file_content,
                markdown_path=markdown_path,
            ):
                if len(section["image_paths"]) == 0:
                    messages.append(generate_openai_message(content=section["text"], role="user"))
                    continue
                try:
                    messages.append(
                        generate_openai_message(
                            content=section["text"],
                            role="user",
                            image_path=section["image_paths"][0],
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load skill image '%s': %s",
                        section["image_paths"][0],
                        exc,
                    )
                    messages.append(generate_openai_message(content=section["text"], role="user"))
                for image_path in section["image_paths"][1:]:
                    try:
                        messages.append(generate_openai_message(content="", role="user", image_path=image_path))
                    except Exception as exc:
                        logger.warning("Failed to load skill image '%s': %s", image_path, exc)
        return messages

    @classmethod
    def build_tool_followup_messages(
        cls,
        tool_response: Dict[str, Any],
        tool_response_type: ToolReturnType,
        *,
        skill_catalog: list[SkillMetadata],
        message_with_yielded_image: str,
        allow_non_image_tool_responses: bool,
        hook_functions: Optional[dict[str, Callable]] = None,
        tool_call_info: Optional[Dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Generate follow-up messages after a tool finishes.

        Parameters
        ----------
        tool_response : dict[str, Any]
            Normalized tool response message.
        tool_response_type : ToolReturnType
            Declared return type for the executed tool.
        skill_catalog : list[SkillMetadata]
            Skills available to the task manager.
        message_with_yielded_image : str
            User-facing text used when an image path is returned.
        allow_non_image_tool_responses : bool
            Whether non-image tool results are acceptable in the current flow.
        hook_functions : dict[str, Callable], optional
            Optional post-tool hook mapping.
        tool_call_info : dict[str, Any], optional
            Tool call metadata from the originating assistant message.

        Returns
        -------
        list[dict[str, Any]]
            Follow-up messages to append after tool execution.
        """
        hook_functions = hook_functions or {}
        followup_messages = cls.build_skill_doc_messages(
            tool_response,
            tool_call_info,
            skill_catalog,
        )
        if tool_response_type in (ToolReturnType.IMAGE_PATH, ToolReturnType.DICT):
            image_paths = cls.extract_image_paths_from_tool_response(tool_response.get("content"))
            if len(image_paths) > 0:
                hook = hook_functions.get("image_path_tool_response")
                if hook is not None:
                    for image_path in image_paths:
                        hook_messages = hook(image_path) or []
                        followup_messages.extend(list(hook_messages))
                else:
                    followup_messages.append(
                        generate_openai_message(
                            content=message_with_yielded_image,
                            image_path=image_paths,
                            role="user",
                        )
                    )
            elif tool_response_type == ToolReturnType.IMAGE_PATH:
                logger.warning(
                    "Tool returned IMAGE_PATH but no valid image path was found in %s",
                    tool_response.get("content"),
                )
            elif not allow_non_image_tool_responses:
                followup_messages.append(cls.build_non_image_tool_warning(tool_response_type))
        elif not allow_non_image_tool_responses:
            followup_messages.append(cls.build_non_image_tool_warning(tool_response_type))
        return followup_messages

    @staticmethod
    def build_non_image_tool_warning(tool_response_type: ToolReturnType) -> dict[str, Any]:
        """Build a warning message for unexpected non-image tool results.

        Parameters
        ----------
        tool_response_type : ToolReturnType
            Declared return type for the executed tool.

        Returns
        -------
        dict[str, Any]
            User message warning about the unexpected tool result type.
        """
        return generate_openai_message(
            content=(
                f"The tool should return an image path, but got {str(tool_response_type)}. "
                "Make sure you call the right tool correctly."
            ),
            role="user",
        )

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
