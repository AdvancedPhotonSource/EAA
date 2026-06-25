import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from fastapi.responses import Response

from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.task_manager.state import ChatGraphState
from eaa_core.task_manager.persistence import SQLiteTranscriptStore, parse_persisted_images
from eaa_core.gui.html import (
    HTMLWebUIBase,
    build_parser,
    launch_html_webui_subprocess,
)
from eaa_core.gui.runtime import WebUIRuntimeController
from eaa_core.gui.runtime import WebUIRuntimeServer


def test_parse_images_field_single_data_url():
    image = "data:image/png;base64,AAA"
    assert parse_persisted_images(image) == [image]


def test_parse_images_field_json_list():
    image = '["data:image/png;base64,AAA","data:image/png;base64,BBB"]'
    assert parse_persisted_images(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_parse_images_field_python_literal_list():
    image = "['data:image/png;base64,AAA','data:image/png;base64,BBB']"
    assert parse_persisted_images(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_parse_images_field_double_encoded_json_list():
    image = '"[\\"data:image/png;base64,AAA\\", \\"data:image/png;base64,BBB\\"]"'
    assert parse_persisted_images(image) == [
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
        checkpoint_db_path=str(shared_db),
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

    messages = SQLiteTranscriptStore(str(shared_db)).load_messages()

    assert messages == []


def test_runtime_controller_queues_user_input(tmp_path):
    task_manager = BaseTaskManager(build=False, use_coding_tools=False)
    controller = WebUIRuntimeController(task_manager)
    controller.submit_input("submitted from browser")

    assert controller.input_queue.get_nowait() == "submitted from browser"


def test_transcript_messages_are_agent_owned_without_webui(tmp_path):
    transcript_db = tmp_path / "shared.sqlite"
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=False,
        transcript_db_path=str(transcript_db),
    )
    task_manager.build_db()

    task_manager.update_message_history({"role": "system", "content": "agent side"})

    messages = SQLiteTranscriptStore(str(transcript_db)).load_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "agent side"


def test_query_messages_reads_explicit_transcript_messages(tmp_path):
    shared_db = tmp_path / "shared.sqlite"
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        transcript_db_path=str(shared_db),
    )
    task_manager.build_db()
    task_manager.record_transcript_message({"role": "system", "content": "display now"})

    messages = SQLiteTranscriptStore(str(shared_db)).load_messages()

    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "display now"


def test_webui_publish_does_not_persist_transcript(tmp_path):
    transcript_db = tmp_path / "shared.sqlite"
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        transcript_db_path=str(transcript_db),
    )
    task_manager.build_db()

    task_manager.update_message_history(
        {"role": "system", "content": "display only"},
        update_context=False,
        update_full_history=False,
        write_to_webui=True,
    )

    assert SQLiteTranscriptStore(str(transcript_db)).load_messages() == []


def test_webui_publish_does_not_use_transcript_payload(tmp_path, monkeypatch):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        transcript_db_path=str(tmp_path / "shared.sqlite"),
    )
    task_manager.build_db()
    published = []

    monkeypatch.setattr(
        task_manager.runtime_controller,
        "publish_message",
        lambda message: published.append(message),
    )

    message = {"role": "system", "content": "display and persist"}
    task_manager.update_message_history(message)

    assert published == [message]
    assert "id" not in published[0]


def test_webui_publish_does_not_wait_for_transcript_write(tmp_path, monkeypatch):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        transcript_db_path=str(tmp_path / "shared.sqlite"),
    )
    task_manager.build_db()
    published = []

    monkeypatch.setattr(
        task_manager.runtime_controller,
        "publish_message",
        lambda message: published.append(message),
    )
    monkeypatch.setattr(
        task_manager,
        "record_transcript_message",
        lambda message: (_ for _ in ()).throw(RuntimeError("transcript failed")),
    )

    message = {"role": "system", "content": "live first"}
    try:
        task_manager.update_message_history(message)
    except RuntimeError:
        pass

    assert published == [message]


def test_runtime_transcript_store_uses_transcript_table(tmp_path):
    transcript_db = tmp_path / "transcript.sqlite"
    store = SQLiteTranscriptStore(str(transcript_db))
    store.append_message({"role": "system", "content": "stored"})

    assert store.load_messages()[0]["content"] == "stored"


def test_transcript_store_uses_configured_safe_table(tmp_path):
    transcript_db = tmp_path / "transcript.sqlite"
    primary_store = SQLiteTranscriptStore(str(transcript_db))
    subagent_store = SQLiteTranscriptStore(
        str(transcript_db),
        table_name="transcript_messages_subagent_1",
    )

    primary_store.append_message({"role": "system", "content": "primary"})
    subagent_store.append_message({"role": "system", "content": "subagent"})

    assert primary_store.load_messages()[0]["content"] == "primary"
    assert subagent_store.load_messages()[0]["content"] == "subagent"
    connection = sqlite3.connect(transcript_db)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"transcript_messages", "transcript_messages_subagent_1"} <= tables


def test_runtime_state_and_approval_are_conversation_scoped(tmp_path):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        transcript_db_path=str(tmp_path / "transcript.sqlite"),
    )
    controller = WebUIRuntimeController(task_manager, upload_dir=str(tmp_path))
    controller.build()
    conversation = controller.create_conversation(label="Subagent 1")

    controller.publish_message(
        {"role": "assistant", "content": "child"},
        conversation_id=conversation["id"],
    )
    snapshot = controller.snapshot()

    assert [item["id"] for item in snapshot["conversations"]] == [
        "primary",
        conversation["id"],
    ]
    assert snapshot["conversations"][1]["messages"][0]["content"] == "child"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(controller.request_approval, "primary_tool", {}, "primary")
        for _ in range(50):
            if controller.has_pending_approval_for_conversation("primary"):
                break
            time.sleep(0.01)
        controller.submit_approval(True, conversation["id"])
        time.sleep(0.05)
        assert not future.done()
        controller.submit_approval(False, "primary")
        assert future.result(timeout=1) is False


def test_runtime_fastapi_routes_handle_core_commands(tmp_path):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        transcript_db_path=str(tmp_path / "transcript.sqlite"),
    )
    task_manager.build_db()
    controller = WebUIRuntimeController(
        task_manager,
        upload_dir=str(tmp_path),
    )
    controller.build()
    server = WebUIRuntimeServer(controller)
    client = TestClient(server.build_app())

    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.json()["messages"] == []
    assert state.json()["conversations"][0]["id"] == "primary"
    assert state.json()["status"] == "idle"

    input_response = client.post("/api/input", json={"content": "hello"})
    assert input_response.status_code == 201
    assert controller.input_queue.get_nowait() == "hello"

    interrupt_response = client.post("/api/interrupt")
    assert interrupt_response.status_code == 200
    assert controller.interrupt_event.is_set()

    approval_response = client.post("/api/approval", json={"approved": True})
    assert approval_response.status_code == 200
    assert controller.approval_queue.get_nowait() is True

    catalog_response = client.get("/api/skill-catalog")
    assert catalog_response.status_code == 200
    skill_names = {skill["name"] for skill in catalog_response.json()["skills"]}
    assert skill_names == {"bayesian-optimization", "monitor-status"}

    upload_response = client.post(
        "/api/upload-image",
        json={"image_data": "data:image/png;base64,cG5nLWRhdGE="},
    )
    assert upload_response.status_code == 201
    image_response = client.get(
        "/api/image",
        params={"path": upload_response.json()["file_path"]},
    )
    assert image_response.status_code == 200


def test_runtime_state_uses_live_messages_not_transcript_db(tmp_path):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        transcript_db_path=str(tmp_path / "transcript.sqlite"),
    )
    task_manager.build_db()
    task_manager.record_transcript_message(
        {"role": "system", "content": "persisted only"}
    )
    controller = WebUIRuntimeController(task_manager)
    client = TestClient(WebUIRuntimeServer(controller).build_app())

    assert client.get("/api/state").json()["messages"] == []

    controller.publish_message({"role": "system", "content": "live only"})

    assert client.get("/api/state").json()["messages"] == [
        {"role": "system", "content": "live only", "id": "runtime-1"}
    ]


def test_runtime_publish_normalizes_structured_content_for_display(tmp_path):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        use_webui=True,
        transcript_db_path=str(tmp_path / "transcript.sqlite"),
    )
    task_manager.build_db()

    task_manager.publish_webui_message(
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Image result"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        }
    )

    assert task_manager.runtime_controller is not None
    messages = task_manager.runtime_controller.snapshot()["messages"]
    assert messages == [
        {
            "role": "system",
            "content": "Image result\n<image>",
            "tool_calls": None,
            "image": "data:image/png;base64,abc",
            "images": ["data:image/png;base64,abc"],
            "id": "runtime-1",
        }
    ]


def test_record_transcript_message_does_not_read_back_row(tmp_path, monkeypatch):
    task_manager = BaseTaskManager(
        build=False,
        use_coding_tools=False,
        transcript_db_path=str(tmp_path / "transcript.sqlite"),
    )
    task_manager.build_db()
    load_calls = []

    monkeypatch.setattr(
        task_manager.transcript_store,
        "load_messages",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    result = task_manager.record_transcript_message(
        {"role": "system", "content": "stored"}
    )

    assert result is None
    assert load_calls == []


def test_runtime_message_ids_are_unique_under_concurrent_publish():
    task_manager = BaseTaskManager(build=False, use_coding_tools=False)
    controller = WebUIRuntimeController(task_manager)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: controller.publish_message(
                    {"role": "system", "content": f"message {index}"}
                ),
                range(50),
            )
        )

    ids = [message["id"] for message in controller.snapshot()["messages"]]
    assert len(ids) == 50
    assert len(set(ids)) == 50


def test_runtime_interrupt_publishes_explicit_event():
    task_manager = BaseTaskManager(build=False, use_coding_tools=False)
    controller = WebUIRuntimeController(task_manager)
    subscriber = controller.subscribe()

    controller.request_interrupt()

    interrupt_event = subscriber.get_nowait()
    status_event = subscriber.get_nowait()
    assert interrupt_event.type == "interrupt.requested"
    assert interrupt_event.payload["interrupt_requested"] is True
    assert status_event.type == "status.changed"


def test_runtime_input_and_approval_restore_previous_status():
    task_manager = BaseTaskManager(build=False, use_coding_tools=False)
    controller = WebUIRuntimeController(task_manager)

    controller.submit_input("queued")
    assert controller.request_input("Prompt") == "queued"
    assert controller.snapshot()["status"] == "idle"

    controller.set_status("running", input_requested=False)
    controller.submit_approval(True)
    assert controller.request_approval("tool", {}) is True
    assert controller.snapshot()["status"] == "running"


def test_html_webui_base_uses_runtime_url():
    webui = HTMLWebUIBase("http://127.0.0.1:9999", title="Custom UI")

    assert webui.title == "Custom UI"
    assert webui.runtime_url == "http://127.0.0.1:9999"


def test_html_webui_page_uses_same_origin_runtime_routes():
    webui = HTMLWebUIBase("http://127.0.0.1:9999")
    page = webui.page_html()

    assert '"state": "/api/state"' in page
    assert '"events": "/api/events"' in page
    assert '"send": "/api/input"' in page
    assert '"http://127.0.0.1:9999/api/state"' not in page


def test_html_webui_proxies_runtime_api(monkeypatch):
    webui = HTMLWebUIBase("http://127.0.0.1:9999")
    calls = []

    def fake_proxy_request(self, *, method, url, body, headers):
        calls.append((method, url, body, headers))
        return Response(content=b'{"ok": true}', media_type="application/json")

    monkeypatch.setattr(HTMLWebUIBase, "blocking_proxy_request", fake_proxy_request)
    client = TestClient(webui.build_app())

    response = client.post("/api/input?x=1", json={"content": "hello"})

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == [
        (
            "POST",
            "http://127.0.0.1:9999/api/input?x=1",
            b'{"content":"hello"}',
            {"Content-Type": "application/json"},
        )
    ]


def test_html_webui_proxies_event_stream(monkeypatch):
    webui = HTMLWebUIBase("http://127.0.0.1:9999")

    monkeypatch.setattr(
        webui,
        "proxy_event_stream",
        lambda: iter([b"event: status.changed\n", b"data: {}\n\n"]),
    )
    client = TestClient(webui.build_app())

    with client.stream("GET", "/api/events") as response:
        assert response.status_code == 200
        assert "event: status.changed" in response.read().decode()


def test_html_webui_page_serves_built_react_assets(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    page = webui.page_html()

    assert "/static/webui/assets/" in page
    assert "window.EAA_WEBUI_CONFIG" in page
    assert "/static/mathjax/es5/tex-svg-full.js" in page


def test_html_webui_static_assets_are_mounted(tmp_path):
    webui = HTMLWebUIBase(str(tmp_path / "shared.sqlite"))
    client = TestClient(webui.build_app())

    response = client.get("/static/webui/index.html")

    assert response.status_code == 200
    assert "root" in response.text


def test_runtime_image_response_sets_cache_headers(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png-data")
    task_manager = BaseTaskManager(build=False, use_coding_tools=False)
    controller = WebUIRuntimeController(task_manager)

    response = controller.image_response(str(image_path))

    assert response.headers["cache-control"] == "public, max-age=3600"
    assert response.headers["etag"]
    assert response.headers["last-modified"]


def test_html_webui_parser_reads_launch_arguments():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--runtime-url",
            "http://127.0.0.1:9999",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--title",
            "Beamline UI",
            "--poll-interval",
            "0.5",
        ]
    )

    assert args.runtime_url == "http://127.0.0.1:9999"
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.title == "Beamline UI"
    assert args.poll_interval == 0.5


def test_launch_html_webui_subprocess_returns_popen(tmp_path, monkeypatch):
    popen_calls = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            popen_calls.append((command, kwargs))

    monkeypatch.setattr("eaa_core.gui.html.subprocess.Popen", FakePopen)

    process = launch_html_webui_subprocess(
        "http://127.0.0.1:9999",
        host="0.0.0.0",
        port=9000,
        title="Beamline UI",
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
                "--runtime-url",
                "http://127.0.0.1:9999",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--title",
                "Beamline UI",
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
