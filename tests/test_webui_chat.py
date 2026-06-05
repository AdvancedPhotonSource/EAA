import sqlite3

from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.task_manager.state import ChatGraphState
from eaa_core.gui.html import (
    HTMLWebUIBase,
    build_parser,
    launch_html_webui_subprocess,
)
from eaa_core.gui.relay import SQLiteWebUIRelay


def test_parse_images_field_single_data_url():
    image = "data:image/png;base64,AAA"
    assert SQLiteWebUIRelay.parse_images(image) == [image]


def test_parse_images_field_json_list():
    image = '["data:image/png;base64,AAA","data:image/png;base64,BBB"]'
    assert SQLiteWebUIRelay.parse_images(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_parse_images_field_python_literal_list():
    image = "['data:image/png;base64,AAA','data:image/png;base64,BBB']"
    assert SQLiteWebUIRelay.parse_images(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_parse_images_field_double_encoded_json_list():
    image = '"[\\"data:image/png;base64,AAA\\", \\"data:image/png;base64,BBB\\"]"'
    assert SQLiteWebUIRelay.parse_images(image) == [
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

    messages = SQLiteWebUIRelay(str(shared_db)).load_messages()

    assert len(messages) == len(task_manager.full_history)
    assert messages[-1]["content"] == "Checkpoint response"


def test_insert_user_message_writes_to_webui_inputs_table(tmp_path):
    shared_db = tmp_path / "shared.sqlite"
    SQLiteWebUIRelay(str(shared_db)).enqueue_user_input("submitted from browser")

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

    messages = SQLiteWebUIRelay(str(shared_db)).load_messages()

    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "display now"


def test_sqlite_webui_relay_queues_user_input(tmp_path):
    shared_db = tmp_path / "shared.sqlite"
    relay = SQLiteWebUIRelay(str(shared_db))

    relay.enqueue_user_input("queued through relay")

    connection = sqlite3.connect(shared_db)
    input_rows = connection.execute(
        "SELECT content FROM webui_inputs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    connection.close()

    assert input_rows == ("queued through relay",)


def test_html_webui_base_uses_sqlite_relay(tmp_path):
    shared_db = tmp_path / "shared.sqlite"
    webui = HTMLWebUIBase(str(shared_db), title="Custom UI")

    assert webui.title == "Custom UI"
    assert isinstance(webui.relay, SQLiteWebUIRelay)
    assert webui.relay.db_path == str(shared_db)


def test_html_webui_base_consumes_matching_pending_message(tmp_path):
    shared_db = tmp_path / "shared.sqlite"
    webui = HTMLWebUIBase(str(shared_db))
    webui.pending_messages["pending-0"] = "hello"

    consumed = webui.consume_pending_message({"role": "user", "content": "hello"})

    assert consumed is True
    assert webui.pending_messages == {}


def test_html_webui_preserves_mathjax_delimiters_for_markdown(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    content = (
        r"\[x = \frac{-b \pm \sqrt{b^{2} - 4ac}}{2a}\]" "\n"
        r"Inline: \(e^{i\pi} + 1 = 0\)"
    )

    preserved = webui.preserve_math_delimiters(content)

    assert preserved == (
        r"\\[x = \frac{-b \pm \sqrt{b^{2} - 4ac}}{2a}\\]" "\n"
        r"Inline: \\(e^{i\pi} + 1 = 0\\)"
    )


def test_html_webui_keeps_multiline_display_math_in_one_markdown_text_run(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    content = (
        r"\[" "\n"
        r"x = \frac{-b \pm \sqrt{b^{2} - 4ac}}{2a}" "\n"
        r"\]"
    )

    preserved = webui.preserve_math_delimiters(content)

    assert preserved == r"\\[x = \frac{-b \pm \sqrt{b^{2} - 4ac}}{2a}\\]"


def test_html_webui_keeps_multiline_dollar_display_math_in_one_text_run(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    content = "$$\n" r"\sum_{n=0}^{\infty} x^{n}" "\n$$"

    preserved = webui.preserve_math_delimiters(content)

    assert preserved == r"$$\sum_{n=0}^{\infty} x^{n}$$"


def test_html_webui_does_not_preserve_mathjax_delimiters_in_code(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    content = (
        r"Text \[x\]" "\n"
        r"`code \[x\]`" "\n"
        "```" "\n"
        r"block \[x\]" "\n"
        "```"
    )

    preserved = webui.preserve_math_delimiters(content)

    assert preserved == (
        r"Text \\[x\\]" "\n"
        r"`code \[x\]`" "\n"
        "```" "\n"
        r"block \[x\]" "\n"
        "```"
    )


def test_html_webui_styles_include_input_and_code_block_rules(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    styles = webui.styles()

    assert ".eaa-input-panel" in styles
    assert ".eaa-markdown pre" in styles
    assert ".eaa-markdown code:not(pre code)" in styles
    assert ".eaa-browser-image-preview" in styles
    assert ".eaa-markdown-folded" in styles
    assert ".eaa-message-expand" in styles
    assert ".eaa-tool-call-details" in styles
    assert ".eaa-approval-arguments" in styles
    assert ".eaa-approval-extracted-field" in styles


def test_html_webui_page_includes_long_message_folding_script(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    page = webui.page_html()

    assert "eaa-markdown-folded" in page
    assert "Show more" in page
    assert "lines.length > 10" in page


def test_html_webui_formats_tool_calls_from_payload(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))

    tool_calls = webui.format_message_tool_calls(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": "call_1: acquire_image\nArguments: {\"x\": 1}",
        }
    )

    assert tool_calls == 'call_1: acquire_image\nArguments: {"x": 1}'


def test_html_webui_ignores_missing_tool_calls(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))

    tool_calls = webui.format_message_tool_calls(
        {
            "role": "assistant",
            "content": "no tools",
        }
    )

    assert tool_calls == ""


def test_html_webui_formats_approval_arguments_with_extracted_code(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    prompt = (
        "Tool 'python' requires approval before execution.\n"
        'Arguments: {"code": "print(1)\\nprint(2)", "timeout": 5}\n'
        "Approve? [y/N]: "
    )

    formatted = webui.format_approval_message(prompt)

    assert formatted is not None
    assert formatted["summary"] == "Tool `python` requires approval before execution."
    assert '"code": "<code rendered below>"' in formatted["arguments_json"]
    assert '"timeout": 5' in formatted["arguments_json"]
    assert formatted["extracted_fields"] == [
        {"label": "code", "value": "print(1)\nprint(2)"}
    ]


def test_html_webui_formats_nested_approval_content_fields(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    prompt = (
        "Tool 'write_file' requires approval before execution.\n"
        'Arguments: {"file_path": "a.py", "payload": {"content": "a\\nb"}}\n'
        "Approve? [y/N]: "
    )

    formatted = webui.format_approval_message(prompt)

    assert formatted is not None
    assert '"content": "<payload.content rendered below>"' in formatted["arguments_json"]
    assert formatted["extracted_fields"] == [
        {"label": "payload.content", "value": "a\nb"}
    ]


def test_html_webui_returns_none_for_unparseable_approval_arguments(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    prompt = (
        "Tool 'python' requires approval before execution.\n"
        "Arguments: not-json\n"
        "Approve? [y/N]: "
    )

    assert webui.format_approval_message(prompt) is None


def test_html_webui_image_html_uses_lazy_loading(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    html = webui.image_html("/tmp/image.png?a=1&b=2", "example-image")

    assert 'loading="lazy"' in html
    assert 'decoding="async"' in html
    assert 'data-eaa-full-src=' in html
    assert "&amp;" in html


def test_html_webui_script_binds_keyboard_shortcuts_to_textarea(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    script = webui.script()

    assert "keydown" in script
    assert "event.shiftKey" in script
    assert "Enter" in script


def test_sqlite_webui_relay_image_response_sets_cache_headers(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png-data")
    relay = SQLiteWebUIRelay(str(tmp_path / "shared.sqlite"))

    response = relay.image_response(str(image_path))

    assert response.headers["cache-control"] == "public, max-age=3600"
    assert response.headers["etag"]
    assert response.headers["last-modified"]


def test_html_webui_parser_reads_launch_arguments():
    parser = build_parser()

    args = parser.parse_args(
        [
            "session.sqlite",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--title",
            "Beamline UI",
            "--upload-dir",
            "uploads",
            "--poll-interval",
            "0.5",
        ]
    )

    assert args.session_db_path == "session.sqlite"
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.title == "Beamline UI"
    assert args.upload_dir == "uploads"
    assert args.poll_interval == 0.5


def test_launch_html_webui_subprocess_returns_popen(tmp_path, monkeypatch):
    shared_db = tmp_path / "shared.sqlite"
    popen_calls = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            popen_calls.append((command, kwargs))

    monkeypatch.setattr("eaa_core.gui.html.subprocess.Popen", FakePopen)

    process = launch_html_webui_subprocess(
        str(shared_db),
        host="0.0.0.0",
        port=9000,
        title="Beamline UI",
        upload_dir="uploads",
        poll_interval=0.5,
        python_executable="/usr/bin/python",
        cwd="/tmp",
        env={"EAA_TEST": "1"},
    )

    assert isinstance(process, FakePopen)
    assert popen_calls == [
        (
            [
                "/usr/bin/python",
                "-m",
                "eaa_core.gui.html",
                str(shared_db),
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--title",
                "Beamline UI",
                "--upload-dir",
                "uploads",
                "--poll-interval",
                "0.5",
            ],
            {
                "cwd": "/tmp",
                "env": {"EAA_TEST": "1"},
                "stdout": None,
                "stderr": None,
            },
        )
    ]
