import sqlite3

from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.task_manager.state import ChatGraphState
from eaa_core.gui.chat import _parse_images_field
from eaa_core.gui.chat import _insert_user_message, _query_messages, set_message_db_path


def test_parse_images_field_single_data_url():
    image = "data:image/png;base64,AAA"
    assert _parse_images_field(image) == [image]


def test_parse_images_field_json_list():
    image = '["data:image/png;base64,AAA","data:image/png;base64,BBB"]'
    assert _parse_images_field(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_parse_images_field_python_literal_list():
    image = "['data:image/png;base64,AAA','data:image/png;base64,BBB']"
    assert _parse_images_field(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_parse_images_field_double_encoded_json_list():
    image = '"[\\"data:image/png;base64,AAA\\", \\"data:image/png;base64,BBB\\"]"'
    assert _parse_images_field(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_query_messages_reads_checkpoint_history(tmp_path, monkeypatch):
    shared_db = tmp_path / "shared.sqlite"

    def fake_invoke_chat_model(llm, messages, tool_schemas=None):
        return {"role": "assistant", "content": "Checkpoint response"}

    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        session_db_path=str(shared_db),
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
    final_state = checkpointed_graph.invoke(
        initial_state,
        config=checkpoint_config,
        context=task_manager.memory_manager.get_runtime_context(),
    )
    task_manager.set_active_state(
        ChatGraphState.model_validate(final_state),
        "chat_graph",
    )

    set_message_db_path(str(shared_db))
    messages = _query_messages()

    assert len(messages) == len(task_manager.full_history)
    assert messages[-1]["content"] == "Checkpoint response"


def test_insert_user_message_writes_to_webui_inputs_table(tmp_path):
    shared_db = tmp_path / "shared.sqlite"
    set_message_db_path(str(shared_db))

    _insert_user_message("submitted from browser")

    connection = sqlite3.connect(shared_db)
    input_rows = connection.execute(
        "SELECT content FROM webui_inputs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    connection.close()

    assert input_rows == ("submitted from browser",)


def test_query_messages_reads_explicit_webui_messages(tmp_path):
    shared_db = tmp_path / "shared.sqlite"
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        session_db_path=str(shared_db),
    )
    task_manager.build_db()
    task_manager.add_webui_message_to_db({"role": "system", "content": "display now"})

    set_message_db_path(str(shared_db))
    messages = _query_messages()

    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "display now"
