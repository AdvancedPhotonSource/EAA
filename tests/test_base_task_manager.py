import httpx
import sqlite3
from langgraph.graph import START, StateGraph
from openai import UnprocessableEntityError
from types import SimpleNamespace
from typing import Any

from eaa.api.llm_config import OpenAIConfig
from eaa.api.memory import MemoryManagerConfig
from eaa.core.task_manager.base import BaseTaskManager
from eaa.core.task_manager.state import ChatGraphState, FeedbackLoopState, TaskManagerState


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


def test_chat_graph_requests_user_input_after_plain_assistant_reply(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, session_db_path=None)
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

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(task_manager, "get_user_input", fake_get_user_input)

    task_manager.run_conversation(message="hello", termination_behavior="user")

    assert model_calls["count"] == 1
    assert input_calls["count"] == 1
    assert [message["role"] for message in task_manager.full_history] == ["user", "assistant"]


def test_feedback_initial_response_sets_await_user_input(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, session_db_path=None)
    task_manager.model = object()

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "NEED HUMAN"}

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    state = FeedbackLoopState(initial_prompt="test prompt")
    result = task_manager.node_factory.call_model(state)

    assert result["await_user_input"] is True
    assert state.await_user_input is True
    assert result["initial_prompt_pending"] is False
    assert state.initial_prompt_pending is False


def test_feedback_graph_preserves_feedback_loop_state_in_call_model(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, session_db_path=None)
    task_manager.model = object()
    task_manager.feedback_loop_graph = task_manager.build_feedback_loop_graph()

    seen = {"type_name": None}
    original_call_model = task_manager.node_factory.call_model

    def wrapped_call_model(state):
        seen["type_name"] = type(state).__name__
        return original_call_model(state)

    task_manager.node_factory.call_model = wrapped_call_model
    task_manager.feedback_loop_graph = task_manager.build_feedback_loop_graph()

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "NEED HUMAN"}

    def fake_get_user_input(prompt, display_prompt_in_webui=False, *args, **kwargs):
        return "/exit"

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(task_manager, "get_user_input", fake_get_user_input)

    task_manager.run_feedback_loop(initial_prompt="test prompt", max_rounds=1)

    assert seen["type_name"] == "FeedbackLoopState"


def test_run_conversation_keyboard_interrupt_reenters_chat(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, session_db_path=None)
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

    monkeypatch.setattr("eaa.core.task_manager.base.print_message", fake_print_message)

    task_manager.run_conversation(message="hello", termination_behavior="user")

    assert invoke_calls["count"] == 2
    assert task_manager.full_history[-1]["role"] == "system"
    assert "Keyboard interrupt detected" in task_manager.full_history[-1]["content"]
    assert printed_roles == ["system"]


def test_run_feedback_loop_keyboard_interrupt_enters_chat_mode(monkeypatch):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, session_db_path=None)
    task_manager.model = object()

    class DummyGraph:
        def invoke(self, state, **kwargs):
            raise KeyboardInterrupt()

    task_manager.feedback_loop_graph = DummyGraph()

    calls = {"count": 0, "kwargs": None}
    printed_roles = []

    def fake_run_conversation(*args, **kwargs):
        calls["count"] += 1
        calls["kwargs"] = kwargs

    def fake_print_message(message, response_requested=None, return_string=False):
        printed_roles.append(message["role"])
        return None

    monkeypatch.setattr(task_manager, "run_conversation", fake_run_conversation)
    monkeypatch.setattr("eaa.core.task_manager.base.print_message", fake_print_message)

    task_manager.run_feedback_loop(initial_prompt="test prompt", termination_behavior="ask")

    assert calls["count"] == 1
    assert calls["kwargs"] == {
        "store_all_images_in_context": True,
        "termination_behavior": "user",
    }
    assert task_manager.full_history[-1]["role"] == "system"
    assert "Keyboard interrupt detected" in task_manager.full_history[-1]["content"]
    assert printed_roles == ["system"]


def test_run_conversation_can_resume_from_checkpoint(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "session.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "Hello! How can I help you today?"}

    first_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(checkpoint_base),
    )
    first_manager.model = object()

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(first_manager, "get_user_input", lambda *args, **kwargs: "/exit")
    first_manager.run_conversation(message="hello", termination_behavior="user")

    resumed_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(checkpoint_base),
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
        session_db_path=str(checkpoint_base),
    )
    first_manager.model = object()

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(first_manager, "get_user_input", lambda *args, **kwargs: "/exit")
    first_manager.run_conversation(message="hello", termination_behavior="user")

    resumed_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=None,
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


def test_run_conversation_can_seed_from_feedback_checkpoint(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "cross_graph_chat.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "NEED HUMAN"}

    first_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(checkpoint_base),
    )
    first_manager.model = object()

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(first_manager, "get_user_input", lambda *args, **kwargs: "/exit")
    first_manager.run_feedback_loop(initial_prompt="test prompt", termination_behavior="ask")

    resumed_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=None,
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


def test_feedback_loop_checkpoint_supports_runtime_hook_functions(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "feedback.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "TERMINATE"}

    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(checkpoint_base),
    )
    task_manager.model = object()
    checkpointed_graph, checkpoint_config, _ = task_manager.get_checkpointed_graph(
        "feedback_loop_graph"
    )

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

    initial_state = FeedbackLoopState(
        initial_prompt="test prompt",
        termination_behavior="return",
    )
    task_manager.state = initial_state
    final_state = checkpointed_graph.invoke(initial_state, config=checkpoint_config)
    task_manager.state = FeedbackLoopState.model_validate(final_state)

    hook_functions = {
        "image_path_tool_response": test_feedback_loop_checkpoint_supports_runtime_hook_functions
    }
    task_manager.run_feedback_loop_from_checkpoint(hook_functions=hook_functions)

    assert task_manager.active_feedback_hook_functions == hook_functions

    assert checkpoint_base.exists()


def test_run_feedback_loop_from_checkpoint_reopens_human_gate(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "feedback_resume.sqlite"
    model_calls = {"count": 0}

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        model_calls["count"] += 1
        return {"role": "assistant", "content": "NEED HUMAN"}

    first_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(checkpoint_base),
    )
    first_manager.model = object()
    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(first_manager, "get_user_input", lambda *args, **kwargs: "/exit")

    first_manager.run_feedback_loop(initial_prompt="test prompt", termination_behavior="ask")

    _, _, saved_state = first_manager.get_checkpointed_graph("feedback_loop_graph")
    assert saved_state is not None
    assert saved_state.exit_requested is True

    resumed_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(checkpoint_base),
    )
    resumed_manager.model = object()

    input_calls = {"count": 0}

    def fake_get_user_input(*args, **kwargs):
        input_calls["count"] += 1
        return "/exit"

    monkeypatch.setattr(resumed_manager, "get_user_input", fake_get_user_input)

    resumed_manager.run_feedback_loop_from_checkpoint()

    assert input_calls["count"] == 1
    assert model_calls["count"] == 1


def test_run_feedback_loop_can_resume_from_override_checkpoint_path(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "override_feedback.sqlite"
    model_calls = {"count": 0}

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        model_calls["count"] += 1
        return {"role": "assistant", "content": "NEED HUMAN"}

    first_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(checkpoint_base),
    )
    first_manager.model = object()
    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(first_manager, "get_user_input", lambda *args, **kwargs: "/exit")

    first_manager.run_feedback_loop(initial_prompt="test prompt", termination_behavior="ask")

    resumed_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=None,
    )
    resumed_manager.model = object()

    input_calls = {"count": 0}

    def fake_get_user_input(*args, **kwargs):
        input_calls["count"] += 1
        return "/exit"

    monkeypatch.setattr(resumed_manager, "get_user_input", fake_get_user_input)

    resumed_manager.run_feedback_loop_from_checkpoint(
        checkpoint_db_path=str(checkpoint_base),
    )

    assert input_calls["count"] == 1
    assert model_calls["count"] == 1


def test_run_feedback_loop_can_seed_from_chat_checkpoint(tmp_path, monkeypatch):
    checkpoint_base = tmp_path / "cross_graph_feedback.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "Hello! How can I help you today?"}

    first_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(checkpoint_base),
    )
    first_manager.model = object()
    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
    monkeypatch.setattr(first_manager, "get_user_input", lambda *args, **kwargs: "/exit")

    first_manager.run_conversation(message="hello", termination_behavior="user")

    resumed_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=None,
    )
    resumed_manager.model = object()

    input_calls = {"count": 0}

    def fake_get_user_input(*args, **kwargs):
        input_calls["count"] += 1
        return "/exit"

    monkeypatch.setattr(resumed_manager, "get_user_input", fake_get_user_input)

    resumed_manager.run_feedback_loop_from_checkpoint(
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
        session_db_path=str(checkpoint_base),
    )
    checkpointed_graph, checkpoint_config, _ = first_manager.get_checkpointed_graph(
        "task_graph"
    )

    initial_state = TaskManagerState(messages=[{"role": "user", "content": "start"}])
    final_state = checkpointed_graph.invoke(initial_state, config=checkpoint_config)
    first_manager.state = TaskManagerState.model_validate(final_state)

    resumed_manager = CheckpointableTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=None,
    )
    resumed_manager.run_from_checkpoint(checkpoint_db_path=str(checkpoint_base))

    assert resumed_manager.state.await_user_input is True
    assert checkpoint_base.exists()


def test_shared_checkpoint_db_can_prune_history(tmp_path, monkeypatch):
    shared_db = tmp_path / "shared.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "Hello! How can I help you today?"}

    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(shared_db),
        prune_checkpoints=True,
    )
    task_manager.model = object()
    checkpointed_graph, checkpoint_config, _ = task_manager.get_checkpointed_graph(
        "chat_graph"
    )

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)
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
    task_manager.state = initial_state
    final_state = checkpointed_graph.invoke(
        initial_state,
        config=checkpoint_config,
        context=task_manager.memory_manager.get_runtime_context(),
    )
    task_manager.state = ChatGraphState.model_validate(final_state)

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


def test_get_user_input_reads_from_webui_inputs_table(tmp_path):
    shared_db = tmp_path / "webui.sqlite"
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        session_db_path=str(shared_db),
    )
    task_manager.build_db()
    task_manager.persistence.enqueue_webui_input("queued response")

    message = task_manager.get_user_input("Prompt: ", display_prompt_in_webui=True)

    assert message == "queued response"
    status_row = task_manager.persistence.connection.execute(
        "SELECT user_input_requested FROM status WHERE id = 1"
    ).fetchone()
    assert status_row == (0,)


def test_get_user_input_consumes_pending_webui_input_from_before_build(tmp_path):
    shared_db = tmp_path / "webui.sqlite"
    bootstrap_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        session_db_path=str(shared_db),
    )
    bootstrap_manager.build_db()
    bootstrap_manager.persistence.enqueue_webui_input("queued before wait")
    bootstrap_manager.persistence.connection.close()

    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        session_db_path=str(shared_db),
    )
    task_manager.build_db()

    message = task_manager.get_user_input("Prompt: ", display_prompt_in_webui=True)

    assert message == "queued before wait"
    remaining_rows = task_manager.persistence.connection.execute(
        "SELECT COUNT(*) FROM webui_inputs"
    ).fetchone()
    assert remaining_rows == (0,)


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
    task_manager = BaseTaskManager(build=False, use_coding_tools=False, session_db_path=None)
    captured = {}

    def fake_execute_tools_for_state(
        state,
        *,
        message_with_yielded_image,
        allow_non_image_tool_responses,
        hook_functions=None,
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


def test_memory_llm_config_overrides_embedding_client_connection(monkeypatch):
    captured = {}

    def fake_openai_embeddings(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("eaa.core.task_manager.memory_manager.OpenAIEmbeddings", fake_openai_embeddings)

    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=None,
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
        session_db_path=None,
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


def test_chat_graph_saves_keyword_triggered_long_term_memory(monkeypatch, tmp_path):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=None,
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

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

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
        session_db_path=None,
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

    monkeypatch.setattr("eaa.core.task_manager.base.invoke_chat_model", fake_invoke_chat_model)

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
