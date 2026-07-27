import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eaa_core.gui.runtime import WebUIRuntimeController
from eaa_core.signals import ControlSignal
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.task_manager.tool_executor import RELEASE_DESCRIPTION, SerialToolExecutor
from eaa_core.tool.base import BaseTool, ExposedToolSpec, tool


def build_tool_call(name, arguments=None, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}),
        },
    }


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


@pytest.mark.parametrize("value", [True, False, "1", object()])
def test_release_timeout_rejects_non_numeric_values(value):
    with pytest.raises(TypeError, match="release_timeout"):
        BaseTool(build=False, release_timeout=value)
    with pytest.raises(TypeError, match="release_timeout"):
        ExposedToolSpec(name="invalid", function=lambda: None, release_timeout=value)


@pytest.mark.parametrize("value", [-1, math.inf, -math.inf, math.nan])
def test_release_timeout_rejects_invalid_numeric_values(value):
    with pytest.raises(ValueError, match="release_timeout"):
        BaseTool(build=False, release_timeout=value)
    with pytest.raises(ValueError, match="release_timeout"):
        tool(name="invalid", release_timeout=value)


def test_release_timeout_decorator_precedence_and_constructor_inheritance():
    class TimeoutTool(BaseTool):
        @tool(name="inherited", release_timeout=None)
        def inherited(self):
            return "inherited"

        @tool(name="overridden", release_timeout=0)
        def overridden(self):
            return "overridden"

    timeout_tool = TimeoutTool(release_timeout=1.25)
    specs = {spec.name: spec for spec in timeout_tool.exposed_tools}

    assert specs["inherited"].release_timeout == 1.25
    assert specs["overridden"].release_timeout == 0.0


def test_fast_releasable_call_retains_ordinary_tool_response():
    class FastTool(BaseTool):
        @tool(name="fast", release_timeout=0.5)
        def fast(self):
            return {"value": 3}

    released = []
    executor = SerialToolExecutor()
    executor.set_queue_handlers(release_handler=released.append)
    executor.register_tools(FastTool())
    run_token = executor.begin_graph_run()

    result = executor.execute_tool_call(
        build_tool_call("fast"),
        graph_run_token=run_token,
    )

    assert json.loads(result.message["content"]) == {"value": 3}
    assert result.released_job_id is None
    assert released == []
    executor.finish_graph_run(run_token)


def test_released_call_returns_exact_payload_and_queues_success():
    started = threading.Event()
    finish = threading.Event()

    class SlowTool(BaseTool):
        @tool(name="slow", release_timeout=0)
        def slow(self):
            started.set()
            assert finish.wait(1)
            return {"value": 7}

    released = []
    completed = []
    dequeued = []
    executor = SerialToolExecutor()
    executor.set_queue_handlers(
        release_handler=released.append,
        completion_handler=completed.append,
        dequeue_handler=dequeued.extend,
    )
    executor.register_tools(SlowTool())
    run_token = executor.begin_graph_run()

    result = executor.execute_tool_call(
        build_tool_call("slow"),
        graph_run_token=run_token,
        conversation_id="subagent-1",
        conversation_label="Sample Search",
    )

    payload = json.loads(result.message["content"])
    assert payload == {
        "job_id": result.released_job_id,
        "status": "executing",
        "description": RELEASE_DESCRIPTION,
    }
    assert len(payload["job_id"]) == 8
    assert released == [
        {
            "job_id": payload["job_id"],
            "tool_name": "slow",
            "conversation_id": "subagent-1",
            "conversation_label": "Sample Search",
            "status": "executing",
            "timestamp": released[0]["timestamp"],
        }
    ]

    assert started.wait(1)
    finish.set()
    assert executor.completion_event.wait(1)
    completions = executor.drain_ready_completions(run_token)

    assert len(completions) == 1
    assert json.loads(completions[0].system_message["content"]) == {
        "job_id": payload["job_id"],
        "tool_name": "slow",
        "status": "completed",
        "result": {"value": 7},
    }
    assert completed[0]["status"] == "completed"
    assert json.loads(completed[0]["content"])["result"] == {"value": 7}
    assert dequeued == [payload["job_id"]]
    assert not executor.completion_event.is_set()
    executor.finish_graph_run(run_token)


def test_released_call_queues_failure_and_serializes_supported_values():
    finish = threading.Event()

    class ResultTool(BaseTool):
        @tool(name="failure", release_timeout=0)
        def failure(self):
            assert finish.wait(1)
            raise RuntimeError("delayed failure")

        @tool(name="array", release_timeout=0.5)
        def array(self):
            return {"array": np.array([1, 2]), "path": Path("/tmp/value")}

    executor = SerialToolExecutor()
    executor.register_tools(ResultTool())
    run_token = executor.begin_graph_run()

    failed = executor.execute_tool_call(
        build_tool_call("failure"),
        graph_run_token=run_token,
    )
    finish.set()
    assert executor.completion_event.wait(1)
    completion = executor.drain_ready_completions(run_token)[0]
    assert json.loads(completion.system_message["content"]) == {
        "job_id": failed.released_job_id,
        "tool_name": "failure",
        "status": "failed",
        "error": "delayed failure",
    }

    array_result = executor.execute_tool_call(
        build_tool_call("array", call_id="call_2"),
        graph_run_token=run_token,
    )
    assert json.loads(array_result.message["content"]) == {
        "array": [1, 2],
        "path": "/tmp/value",
    }
    executor.finish_graph_run(run_token)


def test_approval_denial_and_forced_blocking_never_release():
    finish = threading.Event()

    class ApprovedTool(BaseTool):
        @tool(name="approved", release_timeout=0)
        def approved(self):
            assert finish.wait(1)
            return "done"

    released = []
    denied_executor = SerialToolExecutor(
        approval_handler=lambda _name, _arguments: False
    )
    denied_tool = ApprovedTool(require_approval=True)
    denied_executor.register_tools(denied_tool)
    denied_token = denied_executor.begin_graph_run()
    denied = denied_executor.execute_tool_call(
        build_tool_call("approved"),
        graph_run_token=denied_token,
    )
    assert json.loads(denied.message["content"]) == {
        "error": "Tool execution was denied by the user."
    }
    denied_executor.finish_graph_run(denied_token)

    executor = SerialToolExecutor()
    executor.set_queue_handlers(release_handler=released.append)
    executor.register_tools(ApprovedTool())
    run_token = executor.begin_graph_run()
    timer = threading.Timer(0.03, finish.set)
    timer.start()
    started_at = time.monotonic()
    result = executor.execute_tool_call(
        build_tool_call("approved"),
        allow_release=False,
        graph_run_token=run_token,
    )
    timer.join()

    assert time.monotonic() - started_at >= 0.02
    assert json.loads(result.message["content"]) == {"result": "done"}
    assert released == []
    executor.finish_graph_run(run_token)


def test_new_tool_call_runs_while_released_call_is_still_executing():
    slow_started = threading.Event()
    finish_slow = threading.Event()
    slow_finished = threading.Event()

    class OverlappingTool(BaseTool):
        @tool(name="slow", release_timeout=0)
        def slow(self):
            slow_started.set()
            assert finish_slow.wait(1)
            slow_finished.set()
            return "slow"

        @tool(name="fast")
        def fast(self):
            assert not slow_finished.is_set()
            return "fast"

    executor = SerialToolExecutor()
    executor.register_tools(OverlappingTool())
    run_token = executor.begin_graph_run()

    slow_result = executor.execute_tool_call(
        build_tool_call("slow", call_id="call_slow"),
        graph_run_token=run_token,
    )
    assert slow_result.released_job_id is not None
    assert slow_started.wait(1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor.execute_tool_call,
            build_tool_call("fast", call_id="call_fast"),
            graph_run_token=run_token,
        )
        try:
            fast_result = future.result(timeout=0.2)
            assert json.loads(fast_result.message["content"]) == {"result": "fast"}
            assert not slow_finished.is_set()
        finally:
            finish_slow.set()

    assert executor.completion_event.wait(1)
    executor.finish_graph_run(run_token)


def test_graph_termination_discards_released_and_future_results():
    finish = threading.Event()
    completed = []
    cleaned = []

    class SlowTool(BaseTool):
        @tool(name="slow", release_timeout=0)
        def slow(self):
            assert finish.wait(1)
            return "late"

    executor = SerialToolExecutor()
    executor.set_queue_handlers(
        completion_handler=completed.append,
        cleanup_handler=cleaned.extend,
    )
    executor.register_tools(SlowTool())
    run_token = executor.begin_graph_run()
    result = executor.execute_tool_call(
        build_tool_call("slow"),
        graph_run_token=run_token,
    )

    executor.finish_graph_run(run_token)
    finish.set()
    time.sleep(0.05)

    assert cleaned == [result.released_job_id]
    assert completed == []
    assert executor.drain_ready_completions(run_token) == []
    assert not executor.completion_event.is_set()


def test_chat_wakeup_batches_completions_and_preserves_followups(
    tmp_path,
    monkeypatch,
):
    gates = [threading.Event(), threading.Event()]
    started = [threading.Event(), threading.Event()]
    image_path = tmp_path / "result.png"
    Image.new("RGB", (2, 2), color=(0, 255, 0)).save(image_path)

    class BatchTool(BaseTool):
        @tool(name="batch", release_timeout=0)
        def batch(self, index: int):
            started[index].set()
            assert gates[index].wait(1)
            if index == 0:
                return {
                    "messages": [
                        {"role": "user", "content": "tool follow-up"}
                    ]
                }
            return {"img_path": str(image_path)}

    task_manager = BaseTaskManager(
        build=False,
        checkpoint_db_path=None,
        transcript_db_path=str(tmp_path / "transcript.sqlite"),
        use_webui=True,
        tools=[BatchTool()],
    )
    task_manager.model = object()
    task_manager.build_tools()
    task_manager.chat_graph = task_manager.build_chat_graph()
    model_contexts = []

    def fake_invoke_chat_model(_llm, messages, tool_schemas=None):
        model_contexts.append(messages)
        call_index = len(model_contexts)
        if call_index == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    build_tool_call("batch", {"index": 0}, "call_1"),
                    build_tool_call("batch", {"index": 1}, "call_2"),
                ],
            }
        if call_index == 2:
            assert started[0].wait(1)
            gates[0].set()
            assert started[1].wait(1)
            gates[1].set()
            wait_for(
                lambda: len(
                    task_manager.runtime_controller.snapshot()["message_queue"]
                )
                == 2
            )
            return {"role": "assistant", "content": "Waiting for the tools."}
        task_manager.runtime_controller.submit_input("/exit")
        return {"role": "assistant", "content": "Both tools finished."}

    monkeypatch.setattr(
        "eaa_core.task_manager.base.invoke_chat_model",
        fake_invoke_chat_model,
    )

    task_manager.run_conversation(
        message="start",
        termination_behavior="user",
    )

    assert len(model_contexts) == 3
    completion_payloads = []
    for message in model_contexts[2]:
        if message.get("role") != "system":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except json.JSONDecodeError:
            continue
        if "job_id" in payload:
            completion_payloads.append(payload)
    assert [payload["status"] for payload in completion_payloads] == [
        "completed",
        "completed",
    ]
    assert any(
        message.get("role") == "user"
        and message.get("content") == "tool follow-up"
        for message in model_contexts[2]
    )
    assert any(
        message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and any(
            part.get("type") == "image_url"
            for part in message["content"]
            if isinstance(part, dict)
        )
        for message in model_contexts[2]
    )
    snapshot = task_manager.runtime_controller.snapshot()
    assert snapshot["tool_execution_queue"] == []
    assert snapshot["message_queue"] == []


def test_chat_graph_termination_cleans_queue_and_discards_late_result(
    tmp_path,
    monkeypatch,
):
    finish = threading.Event()
    started = threading.Event()

    class LateTool(BaseTool):
        @tool(name="late", release_timeout=0)
        def late(self):
            started.set()
            assert finish.wait(1)
            return "too late"

    task_manager = BaseTaskManager(
        build=False,
        checkpoint_db_path=None,
        transcript_db_path=str(tmp_path / "transcript.sqlite"),
        use_webui=True,
        tools=[LateTool()],
    )
    task_manager.model = object()
    task_manager.build_tools()
    task_manager.chat_graph = task_manager.build_chat_graph()
    model_calls = 0

    def fake_invoke_chat_model(_llm, messages, tool_schemas=None):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [build_tool_call("late")],
            }
        return {"role": "assistant", "content": "Waiting."}

    monkeypatch.setattr(
        "eaa_core.task_manager.base.invoke_chat_model",
        fake_invoke_chat_model,
    )

    task_manager.run_conversation(
        message="start",
        termination_behavior="return",
    )

    assert started.wait(1)
    assert task_manager.runtime_controller.snapshot()["tool_execution_queue"] == []
    finish.set()
    time.sleep(0.05)
    snapshot = task_manager.runtime_controller.snapshot()
    assert snapshot["tool_execution_queue"] == []
    assert snapshot["message_queue"] == []
    assert model_calls == 2


def test_terminal_input_wait_wakes_for_tool_completion(monkeypatch):
    task_manager = BaseTaskManager(
        build=False,
        checkpoint_db_path=None,
        transcript_db_path=None,
    )
    read_fd, write_fd = os.pipe()
    read_stream = os.fdopen(read_fd)
    monkeypatch.setattr(sys, "stdin", read_stream)
    run_token = task_manager.tool_executor.begin_graph_run()
    task_manager._active_graph_run_token = run_token
    result = []
    waiter = threading.Thread(
        target=lambda: result.append(
            task_manager._get_terminal_input_or_tool_completion(
                "Prompt: ",
                task_manager.tool_executor.completion_event,
            )
        ),
    )
    waiter.start()

    task_manager.tool_executor.completion_event.set()
    waiter.join(timeout=1)

    os.close(write_fd)
    read_stream.close()
    task_manager.tool_executor.finish_graph_run(run_token)
    task_manager._active_graph_run_token = None
    assert not waiter.is_alive()
    assert result[0] is ControlSignal.BACKGROUND_TOOL_COMPLETION_WAKEUP


def test_runtime_queue_snapshots_events_and_transitions():
    task_manager = BaseTaskManager(build=False)
    controller = WebUIRuntimeController(task_manager)
    controller.create_conversation(
        conversation_id="subagent-1",
        label="Long Sample Search",
    )
    subscriber = controller.subscribe()
    execution = {
        "job_id": "12345678",
        "tool_name": "scan",
        "conversation_id": "subagent-1",
        "conversation_label": "Long Sample Search",
        "status": "executing",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    message = {
        "job_id": "12345678",
        "tool_name": "scan",
        "conversation_id": "subagent-1",
        "conversation_label": "Long Sample Search",
        "status": "completed",
        "content": '{"job_id":"12345678","status":"completed","result":{}}',
        "queued_at": "2026-01-01T00:00:01Z",
    }

    controller.add_tool_execution(execution)
    released_event = subscriber.get_nowait()
    controller.complete_tool_execution(message)
    completed_event = subscriber.get_nowait()

    assert released_event.type == "queue.changed"
    assert released_event.payload == {
        "tool_execution_queue": [execution],
        "message_queue": [],
    }
    assert completed_event.payload == {
        "tool_execution_queue": [],
        "message_queue": [message],
    }
    assert controller.snapshot()["message_queue"] == [message]

    controller.dequeue_tool_messages(["12345678"])
    dequeued_event = subscriber.get_nowait()
    assert dequeued_event.payload == {
        "tool_execution_queue": [],
        "message_queue": [],
    }


def test_runtime_queue_updates_are_thread_safe():
    task_manager = BaseTaskManager(build=False)
    controller = WebUIRuntimeController(task_manager)

    def transition(index):
        job_id = f"{index:08d}"
        controller.add_tool_execution(
            {
                "job_id": job_id,
                "tool_name": "scan",
                "conversation_id": "primary",
                "conversation_label": "Primary",
                "status": "executing",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
        controller.complete_tool_execution(
            {
                "job_id": job_id,
                "tool_name": "scan",
                "conversation_id": "primary",
                "conversation_label": "Primary",
                "status": "completed",
                "content": "{}",
                "queued_at": "2026-01-01T00:00:01Z",
            }
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(transition, range(40)))

    snapshot = controller.snapshot()
    assert snapshot["tool_execution_queue"] == []
    assert len(snapshot["message_queue"]) == 40
    assert len({entry["job_id"] for entry in snapshot["message_queue"]}) == 40

    controller.cleanup_tool_jobs(
        [entry["job_id"] for entry in snapshot["message_queue"]]
    )
    assert controller.snapshot()["message_queue"] == []
