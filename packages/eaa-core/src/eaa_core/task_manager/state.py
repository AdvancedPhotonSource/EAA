from dataclasses import dataclass
import sqlite3
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from eaa_core.task_manager.persistence import PrunableSqliteSaver


class TaskManagerState(BaseModel):
    """Minimal base state shared by graph-backed task managers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[dict[str, Any]] = Field(default_factory=list)
    full_history: list[dict[str, Any]] = Field(default_factory=list)
    await_user_input: bool = False
    round_index: int = 0
    initial_prompt: str = ""
    initial_image_path: Optional[str | list[str]] = None
    initial_prompt_pending: bool = True
    message_with_yielded_image: str = "Here is the image the tool returned."
    max_rounds: int = 99
    n_first_images_to_keep_in_context: Optional[int] = None
    n_last_images_to_keep_in_context: Optional[int] = None
    allow_multiple_tool_calls: bool = False
    termination_behavior: Literal["ask", "return", "user"] = "ask"
    max_arounds_reached_behavior: Literal["return", "raise"] = "return"
    chat_requested: bool = False
    exit_requested: bool = False
    return_requested: bool = False

    def copy_messages_and_history_from_state(
        self,
        state: "TaskManagerState",
    ) -> None:
        """Copy transcript fields from another task-manager state.

        Parameters
        ----------
        state : TaskManagerState
            Source state whose active model context and full transcript should
            be copied into this state.
        """
        self.messages = list(state.messages)
        self.full_history = list(state.full_history)

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
    max_agent_iterations: Optional[int] = None


CheckpointStateName = Literal["ChatGraphState", "TaskManagerState"]


def get_state_checkpoint_config(checkpoint_thread_id: str) -> dict[str, Any]:
    """Return the LangGraph config used for a checkpoint file."""
    return {
        "configurable": {
            "thread_id": checkpoint_thread_id,
        }
    }


def load_latest_checkpoint_state_from_connection(
    connection: sqlite3.Connection,
    prune_checkpoints: bool,
) -> Optional[dict[str, Any]]:
    """Load the newest transcript-bearing checkpoint state from a connection."""
    saver = PrunableSqliteSaver(
        connection,
        prune_checkpoints=prune_checkpoints,
    )
    saver.setup()
    latest_state, _ = saver.load_latest_checkpoint_state()
    return latest_state


def build_compatible_checkpoint_state(
    target_state_name: CheckpointStateName,
    incoming_state: TaskManagerState | dict[str, Any],
) -> Optional[TaskManagerState]:
    """Translate a checkpoint state into the target compatible state.

    Parameters
    ----------
    target_state_name : {"ChatGraphState", "TaskManagerState"}
        Name of the state model to produce.
    incoming_state : TaskManagerState or dict[str, Any]
        Source checkpoint state from any graph.

    Returns
    -------
    Optional[TaskManagerState]
        Compatible target state, or ``None`` when no transcript data is
        available to seed the new state.
    """
    state_data = (
        incoming_state.model_dump()
        if isinstance(incoming_state, TaskManagerState)
        else dict(incoming_state)
    )
    messages = state_data.get("messages")
    if not isinstance(messages, list):
        messages = []
    full_history = state_data.get("full_history")
    if not isinstance(full_history, list):
        full_history = list(messages)
    source_state = TaskManagerState.model_validate(
        {
            "messages": messages,
            "full_history": full_history,
        }
    )
    if len(source_state.messages) == 0 and len(source_state.full_history) == 0:
        return None
    shared_fields = {
        "await_user_input": True,
        "round_index": int(state_data.get("round_index", 0) or 0),
    }
    if target_state_name == "ChatGraphState":
        state = ChatGraphState(
            **shared_fields,
            bootstrap_message=None,
            termination_behavior="user",
            exit_requested=False,
            return_requested=False,
        )
        state.copy_messages_and_history_from_state(source_state)
        return state
    if target_state_name == "TaskManagerState":
        state = TaskManagerState(**shared_fields)
        state.copy_messages_and_history_from_state(source_state)
        return state
    raise ValueError(
        f"Unsupported state name for checkpoint loading: {target_state_name}."
    )


@dataclass
class ChatRuntimeContext:
    """Runtime context for the base chat graph."""

    memory_namespace: str
    memory_store: Any = None
