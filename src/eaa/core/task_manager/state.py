from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from eaa.core.tooling.base import ToolReturnType


class TaskManagerState(BaseModel):
    """Minimal base state shared by graph-backed task managers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[dict[str, Any]] = Field(default_factory=list)
    full_history: list[dict[str, Any]] = Field(default_factory=list)
    await_user_input: bool = False
    round_index: int = 0
    store_all_images_in_context: bool = True
    latest_tool_return_types: list[ToolReturnType] = Field(default_factory=list)

    def get_latest_message(self, role: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Return the most recent message, optionally filtered by role."""
        for message in reversed(self.messages):
            if role is None or message.get("role") == role:
                return message
        return None

    def get_latest_message_index(self, role: Optional[str] = None) -> Optional[int]:
        """Return the index of the most recent message, optionally filtered by role."""
        for index in range(len(self.messages) - 1, -1, -1):
            if role is None or self.messages[index].get("role") == role:
                return index
        return None

    def last_message_is_from_user(self) -> bool:
        """Return whether the latest message is a user message."""
        if not self.messages:
            return False
        return self.messages[-1].get("role") == "user"

    @property
    def latest_response(self) -> Optional[dict[str, Any]]:
        """Return the most recent assistant response in the active messages."""
        return self.get_latest_message(role="assistant")

    @property
    def latest_outgoing_message(self) -> Optional[dict[str, Any]]:
        """Return the most recent user/system message preceding the latest response."""
        latest_response_index = self.get_latest_message_index(role="assistant")
        if latest_response_index is None:
            return None
        for index in range(latest_response_index - 1, -1, -1):
            role = self.messages[index].get("role")
            if role in {"user", "system"}:
                return self.messages[index]
        return None

    @property
    def latest_tool_messages(self) -> list[dict[str, Any]]:
        """Return contiguous tool messages immediately following the latest response."""
        latest_response_index = self.get_latest_message_index(role="assistant")
        if latest_response_index is None:
            return []
        tool_messages: list[dict[str, Any]] = []
        for message in self.messages[latest_response_index + 1 :]:
            if message.get("role") != "tool":
                break
            tool_messages.append(message)
        return tool_messages

    @property
    def latest_followup_messages(self) -> list[dict[str, Any]]:
        """Return messages after the latest tool results until the next assistant turn."""
        latest_response_index = self.get_latest_message_index(role="assistant")
        if latest_response_index is None:
            return []
        start_index = latest_response_index + 1
        while start_index < len(self.messages) and self.messages[start_index].get("role") == "tool":
            start_index += 1
        followup_messages: list[dict[str, Any]] = []
        for message in self.messages[start_index:]:
            if message.get("role") == "assistant":
                break
            followup_messages.append(message)
        return followup_messages


class ChatGraphState(TaskManagerState):
    """State used by the base conversation graph."""

    termination_behavior: Literal["return", "user"] = "user"
    bootstrap_message: Optional[Any] = None
    monitor_requested: bool = False
    monitor_task_description: str = ""
    subtask_requested: bool = False
    subtask_task_description: str = ""
    exit_requested: bool = False
    return_requested: bool = False


@dataclass
class ChatRuntimeContext:
    """Runtime context for the base chat graph."""

    memory_namespace: str
    memory_store: Any = None


class FeedbackLoopState(TaskManagerState):
    """State used by the base feedback-loop graph."""

    initial_prompt: str = ""
    initial_image_path: Optional[str | list[str]] = None
    initial_prompt_pending: bool = True
    message_with_yielded_image: str = "Here is the image the tool returned."
    max_rounds: int = 99
    n_first_images_to_keep_in_context: Optional[int] = None
    n_last_images_to_keep_in_context: Optional[int] = None
    allow_non_image_tool_responses: bool = True
    allow_multiple_tool_calls: bool = False
    hook_functions: Dict[str, Callable] = Field(default_factory=dict)
    expected_tool_call_sequence: Optional[list[str]] = None
    expected_tool_call_sequence_tolerance: int = 0
    termination_behavior: Literal["ask", "return"] = "ask"
    max_arounds_reached_behavior: Literal["return", "raise"] = "return"
    chat_requested: bool = False
    exit_requested: bool = False
    return_requested: bool = False
