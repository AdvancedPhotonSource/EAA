import httpx
import pytest
import sqlite3
from langgraph.graph import START, StateGraph
from openai import UnprocessableEntityError
from types import SimpleNamespace
from typing import Any

from eaa_core.api.llm_config import OpenAIConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.message_proc import generate_openai_message
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.task_manager.state import ChatGraphState, ChatRuntimeContext, FeedbackLoopState, TaskManagerState
from eaa_core.tool.base import BaseTool, tool
from eaa_core.tool.subagent import SubagentTool


class EchoTool(BaseTool):
    """Test helper tool."""

    @tool(name="echo_test")
    def echo(self, message: str) -> str:
        """Return the provided test message."""
        return message


class CheckpointableTaskManager(BaseTaskManager):
    """Test helper with a minimal checkpointable task graph."""

    def build_task_graph(self, checkpointer: Any = None) -> Any:
        """Build a task graph that immediately marks the state as waiting."""
        graph_builder = StateGraph(TaskManagerState)

        def mark_complete(state: TaskManagerState) -> dict[str, bool]:
            """Mark the task graph state as awaiting user input."""
            return {"await_user_input": True}

        graph_builder.add_node("mark_complete", mark_complete)
        graph_builder.add_edge(START, "mark_complete")
        return graph_builder.compile(checkpointer=checkpointer)


class InterruptibleCheckpointTaskManager(BaseTaskManager):
    """Test helper whose task graph interrupts twice before completing."""

    def __init__(self, *args, **kwargs):
        self.invoke_count = 0
        super().__init__(*args, **kwargs)

    def build_task_graph(self, checkpointer: Any = None) -> Any:
        """Build a task graph with interruptible node execution."""
        graph_builder = StateGraph(TaskManagerState)

        def interrupt_twice(state: TaskManagerState) -> dict[str, Any]:
            """Raise keyboard interrupts on the first two invocations."""
            self.invoke_count += 1
            if self.invoke_count <= 2:
                raise KeyboardInterrupt()
            return {
                "messages": list(state.messages),
                "full_history": list(state.full_history),
                "await_user_input": True,
            }

        graph_builder.add_node("interrupt_twice", interrupt_twice)
        graph_builder.add_edge(START, "interrupt_twice")
        return graph_builder.compile(checkpointer=checkpointer)


class FeedbackStateTaskManager(BaseTaskManager):
    """Test helper with a task graph backed by feedback-loop state."""

    def build_task_graph(self, checkpointer: Any = None) -> Any:
        """Build a task graph that preserves feedback-specific fields."""
        graph_builder = StateGraph(FeedbackLoopState)

        def mark_chat_requested(state: FeedbackLoopState) -> dict[str, bool]:
            """Mark the feedback-style task graph as ready for chat handoff."""
            return {"chat_requested": True}

        graph_builder.add_node("mark_chat_requested", mark_chat_requested)
        graph_builder.add_edge(START, "mark_chat_requested")
        return graph_builder.compile(checkpointer=checkpointer)


class TaskChatRequestState(TaskManagerState):
    """Task graph state that can request a chat handoff."""

    chat_requested: bool = False


class TaskChatRequestTaskManager(BaseTaskManager):
    """Test helper with a task-specific chat handoff state."""

    def build_task_graph(self, checkpointer: Any = None) -> Any:
        """Build a task graph that requests chat handoff."""
        graph_builder = StateGraph(TaskChatRequestState)

        def mark_chat_requested(state: TaskChatRequestState) -> dict[str, bool]:
            """Mark the task graph state as ready for chat handoff."""
            return {"chat_requested": True}

        graph_builder.add_node("mark_chat_requested", mark_chat_requested)
        graph_builder.add_edge(START, "mark_chat_requested")
        return graph_builder.compile(checkpointer=checkpointer)


def test_merge_tools_ignores_hidden_tool_name_clashes():
    class FirstTool(BaseTool):
        @tool(name="first")
        def first(self):
            return "first"

    class SecondTool(BaseTool):
        @tool(name="second")
        def second(self):
            return "second"

    class DuplicateSecondTool(BaseTool):
        @tool(name="second")
        def second(self):
            return "duplicate"

    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    tools = [FirstTool()]

    task_manager._merge_tools(tools, [SecondTool(), DuplicateSecondTool()])

    assert [type(tool) for tool in tools] == [FirstTool, SecondTool]
    assert task_manager._collect_tool_names(tools) == {"first", "second"}


def test_active_state_owns_context_and_history():
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    message = {"role": "user", "content": "hello"}

    task_manager.update_message_history(message)

    assert task_manager.context == [message]
    assert task_manager.full_history == [message]
    assert task_manager.active_state.messages == [message]

    chat_state = ChatGraphState(messages=[{"role": "assistant", "content": "hi"}])
    task_manager.set_active_state(chat_state, "chat_graph")

    assert task_manager.active_state is task_manager.chat_state
    assert task_manager.context == [{"role": "assistant", "content": "hi"}]
    assert not hasattr(task_manager, "common_state")


def test_default_tools_include_subagent_tool():
    task_manager = BaseTaskManager(
        checkpoint_db_path=None,
        use_coding_tools=True,
    )

    assert "launch_subagent" in task_manager.tool_executor.tool_specs


def test_default_tools_include_literal_eval_tool_without_approval():
    task_manager = BaseTaskManager(
        checkpoint_db_path=None,
        use_coding_tools=True,
    )

    spec = task_manager.tool_executor.tool_specs["evaluate_python_expression"]
    assert spec.require_approval is False


def test_subagent_manager_omits_subagent_tool_and_appends_prompt():
    task_manager = BaseTaskManager(
        checkpoint_db_path=None,
        use_coding_tools=True,
        is_subagent=True,
    )

    assert "launch_subagent" not in task_manager.tool_executor.tool_specs
    assert "You are running as a sub-task manager" in task_manager.assistant_system_message


def test_launch_subagent_inherits_tools_except_subagent(monkeypatch, tmp_path):
    parent = BaseTaskManager(
        build=False,
        tools=[EchoTool()],
        use_coding_tools=False,
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
        use_webui=True,
    )
    parent.model = object()
    subagent_tool = SubagentTool(parent)
    parent.register_tools([*parent.tools, subagent_tool])
    captured = {}
    original_build = BaseTaskManager.build

    def capture_subagent_build(task_manager, *args, **kwargs):
        if task_manager.is_subagent:
            captured["checkpoint_db_path"] = task_manager.checkpoint_db_path
            captured["use_webui"] = task_manager.use_webui
        return original_build(task_manager, *args, **kwargs)

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        captured["messages"] = messages
        captured["tool_names"] = [
            schema["function"]["name"] for schema in tool_schemas
        ]
        return {"role": "assistant", "content": "subagent complete"}

    monkeypatch.setattr(BaseTaskManager, "build", capture_subagent_build)
    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    result = subagent_tool.launch_subagent("inspect this")

    assert result == {"result": "subagent complete"}
    assert captured["checkpoint_db_path"] == parent.checkpoint_db_path
    assert captured["use_webui"] is True
    assert "echo_test" in captured["tool_names"]
    assert "launch_subagent" not in captured["tool_names"]
    assert "You are running as a sub-task manager" in captured["messages"][0]["content"]


def test_task_graph_can_own_feedback_loop_state(monkeypatch):
    task_manager = FeedbackStateTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
    )
    task_manager.task_graph = task_manager.build_task_graph()
    task_manager.set_active_state(
        FeedbackLoopState(initial_prompt="task prompt", termination_behavior="return"),
        "task_graph",
    )

    captured = {"count": 0}

    def fake_run_conversation(*args, **kwargs):
        captured["count"] += 1
        captured["kwargs"] = kwargs
        captured["active_state"] = task_manager.active_state

    monkeypatch.setattr(task_manager, "run_conversation", fake_run_conversation)

    task_manager.run()

    assert isinstance(task_manager.task_state, FeedbackLoopState)
    assert task_manager.task_state.chat_requested is True
    assert task_manager.active_state is task_manager.task_state
    assert captured["count"] == 1
    assert captured["active_state"] is task_manager.task_state
    assert captured["kwargs"] == {
        "store_all_images_in_context": True,
        "termination_behavior": "user",
    }


def test_task_graph_can_request_chat_handoff_from_task_state(monkeypatch):
    task_manager = TaskChatRequestTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
    )
    task_manager.task_graph = task_manager.build_task_graph()
    task_manager.set_active_state(
        TaskChatRequestState(
            messages=[{"role": "user", "content": "task context"}],
            full_history=[{"role": "user", "content": "task context"}],
            round_index=3,
        ),
        "task_graph",
    )

    captured: dict[str, Any] = {}

    def fake_run_conversation(*args, **kwargs):
        captured["kwargs"] = kwargs
        captured["active_state"] = task_manager.active_state

    monkeypatch.setattr(task_manager, "run_conversation", fake_run_conversation)

    task_manager.run()

    assert isinstance(captured["active_state"], TaskChatRequestState)
    assert captured["active_state"].messages == [{"role": "user", "content": "task context"}]
    assert captured["active_state"].round_index == 3
    assert captured["kwargs"] == {
        "store_all_images_in_context": True,
        "termination_behavior": "user",
    }


def test_session_db_path_raises_with_new_database_guidance(tmp_path):
    with pytest.raises(ValueError, match="transcript_db_path.*checkpoint_db_path"):
        BaseTaskManager(
            build=False,
            use_coding_tools=False,
            session_db_path=str(tmp_path / "session.sqlite"),
        )


def test_start_webui_runtime_requires_webui_enabled():
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=False,
    )

    with pytest.raises(RuntimeError, match="use_webui.*True"):
        task_manager.start_webui_runtime()


def test_chat_graph_requests_user_input_after_plain_assistant_reply(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.model = object()
    task_manager.chat_graph = task_manager.build_chat_graph()

    model_calls = {"count": 0}
    input_calls = {"count": 0}

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        model_calls["count"] += 1
        return {"role": "assistant", "content": "Hello! How can I help you today?"}

    def fake_get_user_input(prompt, display_prompt_in_webui=False, *args, **kwargs):
        input_calls["count"] += 1
        return "/exit"

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(task_manager, "get_user_input", fake_get_user_input)

    task_manager.run_conversation(message="hello", termination_behavior="user")

    assert model_calls["count"] == 1
    assert input_calls["count"] == 1
    assert [message["role"] for message in task_manager.full_history] == ["user", "assistant"]


def test_feedback_initial_response_sets_await_user_input(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.model = object()

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "NEED HUMAN"}

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    state = FeedbackLoopState(initial_prompt="test prompt")
    result = task_manager.node_factory.call_model(state)

    assert result["await_user_input"] is True
    assert state.await_user_input is True
    assert result["initial_prompt_pending"] is False
    assert state.initial_prompt_pending is False


def test_base_feedback_loop_public_api_removed():
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)

    assert not hasattr(task_manager, "build_feedback_loop_graph")
    assert not hasattr(task_manager, "run_feedback_loop")
    assert not hasattr(task_manager, "run_feedback_loop_from_checkpoint")


def test_run_conversation_keyboard_interrupt_resumes_same_graph(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.model = object()

    invoke_calls = {"count": 0}

    class DummyGraph:
        def invoke(self, state, **kwargs):
            invoke_calls["count"] += 1
            if invoke_calls["count"] == 1:
                raise KeyboardInterrupt()
            return state.model_dump()

    task_manager.chat_graph = DummyGraph()

    printed_roles = []

    def fake_print_message(message, response_requested=None, return_string=False):
        printed_roles.append(message["role"])
        return None

    monkeypatch.setattr("eaa_core.task_manager.base.print_message", fake_print_message)
    monkeypatch.setattr(task_manager, "get_user_input", lambda *args, **kwargs: "resume chat")

    task_manager.run_conversation(message="hello", termination_behavior="user")

    assert invoke_calls["count"] == 2
    assert [message["role"] for message in task_manager.full_history[-2:]] == [
        "system",
        "user",
    ]
    assert "Keyboard interrupt detected" in task_manager.full_history[-2]["content"]
    assert "Warning: checkpoint recovery was unavailable" in task_manager.full_history[-2]["content"]
    assert task_manager.full_history[-1]["content"] == "resume chat"
    assert printed_roles == ["system", "user"]


def test_interruption_resume_adds_fake_tool_response_for_unmatched_tool_call(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.set_active_state(
        TaskManagerState(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "acquire_image",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            ],
            full_history=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "acquire_image",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            ],
        ),
        "task_graph",
    )

    monkeypatch.setattr(task_manager, "get_user_input", lambda *args, **kwargs: "resume")
    monkeypatch.setattr("eaa_core.task_manager.base.print_message", lambda *args, **kwargs: None)

    command = task_manager.append_interruption_resume_input(
        "Keyboard interrupt detected.",
        checkpoint_recovered=False,
    )

    assert command == "completed"
    assert [message["role"] for message in task_manager.context] == [
        "assistant",
        "tool",
        "system",
        "user",
    ]
    assert task_manager.context[1]["tool_call_id"] == "call_1"
    assert "Please call the tool again" in task_manager.context[1]["content"]


def test_checkpointed_tool_interruption_skips_execute_tools_on_resume(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "acquire_image",
                    "arguments": "{}",
                },
            }
        ],
    }
    checkpoint_state = TaskManagerState(
        messages=[assistant_message],
        full_history=[assistant_message],
    )
    checkpoint_config = {"configurable": {"thread_id": "task_graph"}}

    class DummyGraph:
        def __init__(self):
            self.invoke_count = 0
            self.updated_values = None
            self.updated_as_node = None

        def get_state(self, config):
            return SimpleNamespace(
                created_at="2026-06-04T00:00:00Z",
                values=checkpoint_state.model_dump(),
                next=("execute_tools",),
            )

        def update_state(self, config, values, **kwargs):
            self.updated_values = values
            self.updated_as_node = kwargs.get("as_node")

        def invoke(self, graph_input, **kwargs):
            self.invoke_count += 1
            if self.invoke_count == 1:
                raise KeyboardInterrupt()
            return self.updated_values

    graph = DummyGraph()

    monkeypatch.setattr(task_manager, "get_user_input", lambda *args, **kwargs: "resume")
    monkeypatch.setattr("eaa_core.task_manager.base.print_message", lambda *args, **kwargs: None)

    result = task_manager.invoke_graph_with_interruption_recovery(
        graph=graph,
        graph_input=checkpoint_state,
        graph_kwargs={"config": checkpoint_config},
        graph_name="task_graph",
        state_model=TaskManagerState,
        interruption_message="Keyboard interrupt detected.",
    )

    assert result.command == "completed"
    assert graph.invoke_count == 2
    assert graph.updated_as_node == "execute_tools"
    assert [message["role"] for message in graph.updated_values["messages"]] == [
        "assistant",
        "tool",
        "system",
        "user",
    ]
    assert graph.updated_values["messages"][1]["tool_call_id"] == "call_1"
    assert "interrupted before it completed" in graph.updated_values["messages"][1]["content"]


def test_non_tool_checkpointed_interruption_does_not_force_resume_node():
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    checkpoint_state = TaskManagerState(
        messages=[{"role": "user", "content": "seed"}],
        full_history=[{"role": "user", "content": "seed"}],
    )
    checkpoint_config = {"configurable": {"thread_id": "task_graph"}}

    class DummyGraph:
        def get_state(self, config):
            return SimpleNamespace(
                created_at="2026-06-04T00:00:00Z",
                values=checkpoint_state.model_dump(),
                next=("call_model",),
            )

    resume_node = task_manager.get_interrupted_checkpoint_resume_node(
        graph=DummyGraph(),
        checkpoint_config=checkpoint_config,
    )

    assert resume_node is None


def test_interruption_message_preview_uses_checkpoint_state_and_image_placeholder(tmp_path):
    checkpoint_base = tmp_path / "interrupt_preview.sqlite"
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=str(checkpoint_base),
    )
    graph, checkpoint_config, _ = task_manager.get_checkpointed_graph(
        "chat_graph",
        load_state=False,
    )
    recovered_text = "x" * 120
    recovered_state = ChatGraphState(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": recovered_text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
        full_history=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": recovered_text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
    )
    graph.update_state(
        checkpoint_config,
        recovered_state.model_dump(),
        as_node="await_or_ingest_user_input",
    )

    recovered = task_manager.recover_active_state_from_checkpoint(
        graph=graph,
        checkpoint_config=checkpoint_config,
        graph_name="chat_graph",
        state_model=ChatGraphState,
    )
    content = task_manager.format_interruption_message_content(
        "Keyboard interrupt detected.",
        checkpoint_recovered=recovered,
    )

    assert recovered is True
    assert "Warning: checkpoint recovery was unavailable" not in content
    assert f"Last recovered message: {'x' * 100}" in content
    assert "<image>" in content
    assert "data:image/png" not in content


def test_checkpointed_interruption_recovery_survives_multiple_interruptions(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "multi_interrupt.sqlite"
    task_manager = InterruptibleCheckpointTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=str(checkpoint_base),
    )
    task_manager.task_graph = task_manager.build_task_graph()
    task_manager.set_active_state(
        TaskManagerState(
            messages=[{"role": "user", "content": "seed"}],
            full_history=[{"role": "user", "content": "seed"}],
        ),
        "task_graph",
    )
    resume_inputs = iter(["resume one", "resume two"])

    monkeypatch.setattr(task_manager, "get_user_input", lambda *args, **kwargs: next(resume_inputs))
    monkeypatch.setattr("eaa_core.task_manager.base.print_message", lambda *args, **kwargs: None)

    task_manager.run()

    system_messages = [
        message["content"]
        for message in task_manager.full_history
        if message["role"] == "system"
    ]
    assert len(system_messages) == 2
    assert "Warning: checkpoint recovery was unavailable" not in system_messages[1]
    assert task_manager.full_history[-1]["content"] == "resume two"
    assert task_manager.invoke_count == 3


def test_run_conversation_monitor_command_hands_off_to_task_manager(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.chat_graph = task_manager.build_chat_graph()

    captured: dict[str, Any] = {}

    def fake_enter_monitoring_mode(task_description: str):
        captured["task_description"] = task_description
        captured["state"] = task_manager.active_state

    monkeypatch.setattr(task_manager, "get_user_input", lambda *args, **kwargs: "/monitor check beam drift")
    monkeypatch.setattr(task_manager, "enter_monitoring_mode", fake_enter_monitoring_mode)

    task_manager.run_conversation(termination_behavior="user")

    assert captured["task_description"] == "check beam drift"
    assert isinstance(captured["state"], ChatGraphState)
    assert captured["state"].monitor_requested is True
    assert captured["state"].messages == []
    assert captured["state"].full_history == []


def test_run_conversation_accepts_dict_and_list_messages(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.model = object()
    task_manager.chat_graph = task_manager.build_chat_graph()
    outgoing_payloads = []

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        outgoing_payloads.append(messages[-1])
        return {"role": "assistant", "content": "done"}

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    task_manager.run_conversation(
        message={"role": "user", "content": "dict"},
        termination_behavior="return",
    )
    task_manager.run_conversation(
        message=[{"role": "user", "content": "list"}],
        termination_behavior="return",
        inherit_activate_state_messages=False,
    )

    assert outgoing_payloads == [
        {"role": "user", "content": "dict"},
        {"role": "user", "content": "list"},
    ]


def test_run_conversation_converts_tagged_text_bootstrap(tmp_path, monkeypatch):
    from PIL import Image

    image_path = tmp_path / "image.png"
    Image.new("RGB", (1, 1), color=(0, 255, 0)).save(image_path)
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.model = object()
    task_manager.chat_graph = task_manager.build_chat_graph()
    captured = {}

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        captured["message"] = messages[-1]
        return {"role": "assistant", "content": "done"}

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    task_manager.run_conversation(
        message=f"inspect <img {image_path}>",
        termination_behavior="return",
    )

    assert captured["message"]["content"][0] == {"type": "text", "text": "inspect "}
    assert captured["message"]["content"][1]["type"] == "image_url"


def test_chat_tool_loop_continues_until_non_tool_response(monkeypatch):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
        tools=[EchoTool()],
    )
    task_manager.model = object()
    task_manager.build_tools()
    task_manager.chat_graph = task_manager.build_chat_graph()
    responses = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo_test",
                        "arguments": "{\"message\": \"hi\"}",
                    },
                }
            ],
        },
        {"role": "assistant", "content": "done"},
    ]

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return responses.pop(0)

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    task_manager.run_conversation(
        message="start",
        max_agent_iterations=2,
        termination_behavior="return",
    )

    assert [message["role"] for message in task_manager.context] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert task_manager.chat_state.round_index == 0


def test_chat_non_tool_response_resets_round_index(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.model = object()

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "done"}

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    state = ChatGraphState(
        messages=[{"role": "user", "content": "finish"}],
        round_index=3,
    )
    result = task_manager.node_factory.call_model(state)

    assert result["round_index"] == 0
    assert state.round_index == 0


def test_chat_max_agent_iterations_completes_unanswered_tool_call(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.model = object()
    task_manager.chat_graph = task_manager.build_chat_graph()

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "missing_tool", "arguments": "{}"},
                }
            ],
        }

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    task_manager.run_conversation(
        message="start",
        max_agent_iterations=0,
        termination_behavior="return",
    )

    assert task_manager.context[-1] == {
        "role": "tool",
        "content": "<Incomplete tool response>",
        "tool_call_id": "call_1",
    }


def test_chat_round_prunes_context_images_without_pruning_full_history(tmp_path):
    from PIL import Image

    image_paths = []
    for index in range(3):
        image_path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (1, 1), color=(index, 0, 0)).save(image_path)
        image_paths.append(image_path)
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    state = ChatGraphState(
        messages=[
            generate_openai_message(content=f"image {index}", image_path=str(image_path))
            for index, image_path in enumerate(image_paths)
        ],
        full_history=[
            generate_openai_message(content=f"image {index}", image_path=str(image_path))
            for index, image_path in enumerate(image_paths)
        ],
        n_first_images_to_keep_in_context=1,
        n_last_images_to_keep_in_context=1,
    )

    task_manager.node_factory.finalize_chat_round(state)

    context_images = [
        message for message in state.messages if isinstance(message["content"], list)
    ]
    history_images = [
        message for message in state.full_history if isinstance(message["content"], list)
    ]
    assert len(context_images) == 2
    assert len(history_images) == 3


def test_terminate_is_plain_chat_response(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    task_manager.model = object()
    task_manager.chat_graph = task_manager.build_chat_graph()

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "TERMINATE"}

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    task_manager.run_conversation(message="start", termination_behavior="return")

    assert task_manager.context[-1] == {"role": "assistant", "content": "TERMINATE"}
    assert task_manager.chat_state.await_user_input is True


def test_run_conversation_can_resume_from_checkpoint(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "checkpoint.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "Hello! How can I help you today?"}

    first_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=str(checkpoint_base),
    )
    first_manager.model = object()

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(first_manager, "get_user_input", lambda *args, **kwargs: "/exit")
    first_manager.run_conversation(message="hello", termination_behavior="user")

    resumed_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=str(checkpoint_base),
    )
    resumed_manager.model = object()

    input_calls = {"count": 0}

    def fake_get_user_input(*args, **kwargs):
        input_calls["count"] += 1
        return "/exit"

    monkeypatch.setattr(resumed_manager, "get_user_input", fake_get_user_input)

    resumed_manager.run_conversation_from_checkpoint()

    assert input_calls["count"] == 1
    assert resumed_manager.full_history == first_manager.full_history
    assert checkpoint_base.exists()


def test_run_conversation_can_resume_from_override_checkpoint_path(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "override_chat.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "Hello! How can I help you today?"}

    first_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=str(checkpoint_base),
    )
    first_manager.model = object()

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(first_manager, "get_user_input", lambda *args, **kwargs: "/exit")
    first_manager.run_conversation(message="hello", termination_behavior="user")

    resumed_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
    )
    resumed_manager.model = object()

    input_calls = {"count": 0}

    def fake_get_user_input(*args, **kwargs):
        input_calls["count"] += 1
        return "/exit"

    monkeypatch.setattr(resumed_manager, "get_user_input", fake_get_user_input)

    resumed_manager.run_conversation_from_checkpoint(
        checkpoint_db_path=str(checkpoint_base),
    )

    assert input_calls["count"] == 1
    assert resumed_manager.full_history == first_manager.full_history
    assert checkpoint_base.exists()


def test_removed_feedback_loop_checkpoint_name_is_rejected(tmp_path):
    checkpoint_base = tmp_path / "removed_feedback.sqlite"
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=str(checkpoint_base),
    )

    with pytest.raises(ValueError, match="Unsupported graph name"):
        task_manager.get_checkpointed_graph("feedback_loop_graph")  # type: ignore[arg-type]


def test_run_conversation_can_resume_from_chat_checkpoint_after_exit(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "chat_checkpoint.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "Hello! How can I help you today?"}

    first_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=str(checkpoint_base),
    )
    first_manager.model = object()
    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(first_manager, "get_user_input", lambda *args, **kwargs: "/exit")

    first_manager.run_conversation(message="hello", termination_behavior="user")

    resumed_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
    )
    resumed_manager.model = object()

    input_calls = {"count": 0}

    def fake_get_user_input(*args, **kwargs):
        input_calls["count"] += 1
        return "/exit"

    monkeypatch.setattr(resumed_manager, "get_user_input", fake_get_user_input)

    resumed_manager.run_conversation_from_checkpoint(
        checkpoint_db_path=str(checkpoint_base),
    )

    assert input_calls["count"] == 1
    assert resumed_manager.full_history == first_manager.full_history
    assert resumed_manager.context == first_manager.context


def test_run_task_graph_can_resume_from_override_checkpoint_path(tmp_path):
    checkpoint_base = tmp_path / "override_task.sqlite"

    first_manager = CheckpointableTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=str(checkpoint_base),
    )
    checkpointed_graph, checkpoint_config, _ = first_manager.get_checkpointed_graph(
        "task_graph"
    )

    initial_state = TaskManagerState(messages=[{"role": "user", "content": "start"}])
    final_state = checkpointed_graph.invoke(initial_state, config=checkpoint_config)
    first_manager.set_active_state(
        TaskManagerState.model_validate(final_state),
        "task_graph",
    )

    resumed_manager = CheckpointableTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
    )
    resumed_manager.run_from_checkpoint(checkpoint_db_path=str(checkpoint_base))

    assert resumed_manager.active_state.await_user_input is True
    assert checkpoint_base.exists()


def test_shared_checkpoint_db_can_prune_history(tmp_path, monkeypatch):
    shared_db = tmp_path / "shared.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "Hello! How can I help you today?"}

    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=str(shared_db),
        prune_checkpoints=True,
    )
    task_manager.model = object()
    checkpointed_graph, checkpoint_config, _ = task_manager.get_checkpointed_graph(
        "chat_graph"
    )

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(task_manager, "get_user_input", lambda *args, **kwargs: "/exit")

    initial_state = ChatGraphState(
        messages=[],
        full_history=[],
        round_index=0,
        termination_behavior="user",
        store_all_images_in_context=True,
        bootstrap_message="hello",
        await_user_input=False,
    )
    task_manager.set_active_state(initial_state, "chat_graph")
    final_state = checkpointed_graph.invoke(
        initial_state,
        config=checkpoint_config,
        context=task_manager.memory_manager.get_runtime_context(),
    )
    task_manager.set_active_state(
        ChatGraphState.model_validate(final_state),
        "chat_graph",
    )

    assert checkpoint_config["configurable"]["thread_id"] == "chat_graph"
    assert shared_db.exists()
    assert not (tmp_path / "shared.chat_graph.sqlite").exists()

    connection = sqlite3.connect(shared_db)
    checkpoint_rows = connection.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
        (checkpoint_config["configurable"]["thread_id"],),
    ).fetchone()
    assert checkpoint_rows is not None
    assert checkpoint_rows[0] == 1

    write_rows = connection.execute(
        "SELECT COUNT(*) FROM writes WHERE thread_id != ?",
        (checkpoint_config["configurable"]["thread_id"],),
    ).fetchone()
    connection.close()
    assert write_rows is not None
    assert write_rows[0] == 0

    _, _, resumed_state = task_manager.get_checkpointed_graph("chat_graph")
    assert resumed_state is not None
    assert resumed_state.full_history == task_manager.full_history


def test_get_user_input_reads_from_webui_runtime_queue(tmp_path):
    transcript_db = tmp_path / "transcript.sqlite"
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        transcript_db_path=str(transcript_db),
    )
    task_manager.build_db()
    task_manager.runtime_controller.submit_input("queued response")

    message = task_manager.get_user_input("Prompt: ", display_prompt_in_webui=True)

    assert message == "queued response"
    assert task_manager.runtime_controller.input_requested is False


def test_webui_runtime_commands_are_not_persisted_in_transcript_db(tmp_path):
    transcript_db = tmp_path / "transcript.sqlite"
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        transcript_db_path=str(transcript_db),
    )
    task_manager.build_db()
    task_manager.runtime_controller.submit_input("queued only in memory")

    connection = sqlite3.connect(transcript_db)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    connection.close()
    assert "webui_inputs" not in tables


def test_task_manager_state_derives_latest_messages():
    state = TaskManagerState(
        messages=[
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "call tool", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "tool result", "tool_call_id": "1"},
            {"role": "user", "content": "follow-up"},
        ]
    )

    assert state.latest_response == state.messages[1]
    assert state.latest_outgoing_message == state.messages[0]
    assert state.latest_tool_messages == [state.messages[2]]
    assert state.latest_followup_messages == [state.messages[3]]


def test_execute_tools_accepts_base_task_manager_state(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)
    captured = {}

    def fake_execute_tools_for_state(
        state,
        *,
        message_with_yielded_image,
        allow_non_image_tool_responses,
        store_all_images_in_context=True,
    ):
        captured["state"] = state
        captured["message_with_yielded_image"] = message_with_yielded_image
        captured["allow_non_image_tool_responses"] = allow_non_image_tool_responses
        captured["store_all_images_in_context"] = store_all_images_in_context
        return state.model_dump()

    monkeypatch.setattr(task_manager, "execute_tools_for_state", fake_execute_tools_for_state)

    state = TaskManagerState(
        messages=[
            {"role": "assistant", "content": "call tool", "tool_calls": [{"id": "1"}]},
        ]
    )
    result = task_manager.node_factory.execute_tools(state)

    assert result == state.model_dump()
    assert captured["state"] is state
    assert captured["message_with_yielded_image"] == "Here is the image the tool returned."
    assert captured["allow_non_image_tool_responses"] is True
    assert captured["store_all_images_in_context"] is True


def test_image_followup_converts_followup_exceptions_into_messages(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, checkpoint_db_path=None)

    def fake_build_tool_followup_messages(*args, **kwargs):
        raise FileNotFoundError("missing.png")

    monkeypatch.setattr(
        task_manager.node_factory,
        "build_tool_followup_messages",
        fake_build_tool_followup_messages,
    )

    state = TaskManagerState(
        messages=[
            {"role": "assistant", "content": "call tool", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": '{"img_path": "missing.png"}', "tool_call_id": "1"},
        ],
        full_history=[
            {"role": "assistant", "content": "call tool", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": '{"img_path": "missing.png"}', "tool_call_id": "1"},
        ],
    )

    result = task_manager.node_factory.image_followup(state)

    assert result == state.model_dump()
    assert state.messages[-1]["role"] == "user"
    assert "processing the tool follow-up output" in state.messages[-1]["content"]
    assert "missing.png" in state.messages[-1]["content"]
    assert state.full_history[-1] == state.messages[-1]


def test_memory_llm_config_overrides_embedding_client_connection(monkeypatch):
    captured = {}

    def fake_openai_embeddings(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("eaa_core.task_manager.memory_manager.OpenAIEmbeddings", fake_openai_embeddings)

    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
        llm_config=OpenAIConfig(model="chat-model", base_url="https://chat.example", api_key="chat-key"),
        memory_config=MemoryManagerConfig(
            enabled=True,
            embedding_model="memory-embedding-model",
            llm_config={"base_url": "https://memory.example", "api_key": "memory-key", "model": "ignored"},
        ),
    )

    task_manager.memory_manager.build_embeddings_client()

    assert captured == {
        "model": "memory-embedding-model",
        "api_key": "memory-key",
        "base_url": "https://memory.example",
    }


def test_memory_retrieval_falls_back_to_string_input_on_422():
    def make_error():
        request = httpx.Request("POST", "https://memory.example/v1/embeddings")
        response = httpx.Response(status_code=422, request=request)
        body = {
            "detail": [
                {
                    "type": "string_type",
                    "loc": ["body", "input", "str"],
                    "msg": "Input should be a valid string",
                    "input": [[1, 2, 3]],
                }
            ]
        }
        return UnprocessableEntityError("Error code: 422", response=response, body=body)

    class FakeStore:
        def __init__(self, manager):
            self.manager = manager

        def similarity_search_with_relevance_scores(self, query, k, filter):
            if self.manager.check_embedding_ctx_length:
                raise make_error()
            return [(SimpleNamespace(page_content="stored memory"), 0.9)]

    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
        llm_config=OpenAIConfig(model="chat-model", api_key="chat-key"),
        memory_config=MemoryManagerConfig(enabled=True),
    )

    build_calls = []

    def fake_build_store():
        build_calls.append(task_manager.memory_manager.check_embedding_ctx_length)
        task_manager.memory_manager.store = FakeStore(task_manager.memory_manager)

    task_manager.memory_manager.build_store = fake_build_store
    task_manager.memory_manager.build_store()

    memories = task_manager.memory_manager.retrieve_user_memories(
        {"role": "user", "content": "what did I tell you?"},
        namespace="test",
    )

    assert build_calls == [True, False]
    assert task_manager.memory_manager.check_embedding_ctx_length is False
    assert len(memories) == 1
    assert memories[0][0].page_content == "stored memory"


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lowered = text.lower()
        return [
            float(lowered.count("sky")),
            float(lowered.count("blue")),
            float(lowered.count("fragile")),
            float(len(lowered.split())),
        ]


def test_memory_manager_saves_image_caption_with_text(monkeypatch):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
        llm_config=OpenAIConfig(model="gpt-test", api_key="test"),
        memory_config=MemoryManagerConfig(enabled=True),
    )
    task_manager.model = object()
    saved = {}

    class FakeStore:
        def add_texts(self, texts, metadatas, ids):
            saved["texts"] = texts
            saved["metadatas"] = metadatas
            saved["ids"] = ids

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        saved["caption_messages"] = messages
        return {"role": "assistant", "content": "A plot with a fragile sample response."}

    task_manager.memory_manager.store = FakeStore()
    monkeypatch.setattr("eaa_core.task_manager.memory_manager.invoke_chat_model", fake_invoke_chat_model)

    task_manager.memory_manager.save_user_memory(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Remember this: the sample is fragile."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,example"},
                },
            ],
        },
        namespace="test",
    )

    assert saved["texts"] == [
        "the sample is fragile.\n"
        "Image descriptions:\n"
        "1. A plot with a fragile sample response."
    ]
    assert saved["metadatas"][0]["namespace"] == "test"
    assert saved["metadatas"][0]["kind"] == "user_memory"
    assert saved["caption_messages"][0]["content"][1]["type"] == "image_url"


def test_memory_manager_retrieves_with_image_caption_query(monkeypatch):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
        llm_config=OpenAIConfig(model="gpt-test", api_key="test"),
        memory_config=MemoryManagerConfig(enabled=True),
    )
    task_manager.model = object()
    captured = {}

    class FakeStore:
        def similarity_search_with_relevance_scores(self, query, k, filter):
            captured["query"] = query
            captured["k"] = k
            captured["filter"] = filter
            return [(SimpleNamespace(page_content="stored image memory"), 0.9)]

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "An image showing blue sky."}

    task_manager.memory_manager.store = FakeStore()
    monkeypatch.setattr("eaa_core.task_manager.memory_manager.invoke_chat_model", fake_invoke_chat_model)

    memories = task_manager.memory_manager.retrieve_user_memories(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What matches this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,example"},
                },
            ],
        },
        namespace="test",
    )

    assert "What matches this image?" in captured["query"]
    assert "Image descriptions:\n1. An image showing blue sky." in captured["query"]
    assert captured["k"] == task_manager.memory_manager.config.top_k
    assert captured["filter"] == {
        "$and": [
            {"namespace": "test"},
            {"kind": "user_memory"},
        ]
    }
    assert len(memories) == 1


def test_feedback_followup_triggered_memory_reuses_memory_manager(monkeypatch):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
        llm_config=OpenAIConfig(model="gpt-test", api_key="test"),
        memory_config=MemoryManagerConfig(enabled=True),
    )
    task_manager.model = object()
    saved = {}

    class FakeStore:
        def add_texts(self, texts, metadatas, ids):
            saved["texts"] = texts
            saved["metadatas"] = metadatas
            saved["ids"] = ids

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        saved["caption_messages"] = messages
        return {"role": "assistant", "content": "A microscope image with a target feature."}

    monkeypatch.setattr("eaa_core.task_manager.memory_manager.invoke_chat_model", fake_invoke_chat_model)

    state = FeedbackLoopState()
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "<embed-this> Here is the image the tool returned.",
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,example"},
            },
        ],
    }
    runtime_context = ChatRuntimeContext(
        memory_namespace="feedback-test",
        memory_store=FakeStore(),
    )
    task_manager.memory_manager.store = runtime_context.memory_store

    task_manager.node_factory.apply_followup_messages_for_state(
        state,
        [message],
        runtime_context=runtime_context,
    )

    assert len(state.full_history) == 1
    assert saved["texts"] == [
        "Here is the image the tool returned.\n"
        "Image descriptions:\n"
        "1. A microscope image with a target feature."
    ]
    assert saved["metadatas"][0]["namespace"] == "feedback-test"
    assert saved["caption_messages"][0]["content"][1]["type"] == "image_url"


def test_chat_graph_saves_keyword_triggered_long_term_memory(monkeypatch, tmp_path):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
        llm_config=OpenAIConfig(model="gpt-test", api_key="test"),
        memory_config=MemoryManagerConfig(
            enabled=True,
            persist_directory=str(tmp_path),
            collection_name="test_memory_save",
        ),
    )
    task_manager.model = object()
    monkeypatch.setattr(task_manager.memory_manager, "build_embeddings_client", lambda: FakeEmbeddings())
    task_manager.build_memory_store()
    task_manager.chat_graph = task_manager.build_chat_graph()

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "I'll remember that."}

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    task_manager.run_conversation(
        message="Remember this: the sky is blue.",
        termination_behavior="return",
    )

    results = task_manager.memory_manager.store.similarity_search_with_relevance_scores(
        "what color is the sky?",
        k=1,
        filter={
            "$and": [
                {"namespace": task_manager.memory_manager.get_namespace()},
                {"kind": "user_memory"},
            ]
        },
    )

    assert len(results) == 1
    assert results[0][0].page_content == "the sky is blue."


def test_chat_graph_retrieves_long_term_memory_into_model_context(monkeypatch, tmp_path):
    stored_memory = "MEMORY_TEST_TOKEN: the sky is blue"

    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        checkpoint_db_path=None,
        llm_config=OpenAIConfig(model="gpt-test", api_key="test"),
        memory_config=MemoryManagerConfig(
            enabled=True,
            persist_directory=str(tmp_path),
            collection_name="test_memory_recall",
        ),
    )
    task_manager.model = object()
    monkeypatch.setattr(task_manager.memory_manager, "build_embeddings_client", lambda: FakeEmbeddings())
    task_manager.build_memory_store()
    task_manager.chat_graph = task_manager.build_chat_graph()

    task_manager.memory_manager.store.add_texts(
        [stored_memory],
        metadatas=[{"namespace": task_manager.memory_manager.get_namespace(), "kind": "user_memory"}],
        ids=["memory-1"],
    )

    captured = {"messages": None}

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        captured["messages"] = messages
        return {"role": "assistant", "content": "The sky is blue."}

    monkeypatch.setattr("eaa_core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    task_manager.run_conversation(
        message="What color is the sky?",
        termination_behavior="return",
    )

    memory_messages = [
        message
        for message in captured["messages"]
        if message.get("role") == "system" and "Relevant long-term memory:" in str(message.get("content"))
    ]

    assert len(memory_messages) == 1
    assert stored_memory in str(memory_messages[0]["content"])
