import copy
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence
import json
import logging
import threading
import uuid

import numpy as np

from eaa_core.message_proc import generate_openai_message
from eaa_core.tool.base import (
    BaseTool,
    ExposedToolSpec,
    TOOL_IMAGE_PATH_FIELD,
    normalize_tool_result,
    generate_openai_tool_schema,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Normalized tool execution result."""

    message: dict[str, Any]
    released_job_id: str | None = None


@dataclass
class BackgroundToolCompletion:
    """One completed released tool call ready for graph-thread delivery."""

    job_id: str
    tool_name: str
    conversation_id: str
    conversation_label: str
    status: str
    system_message: dict[str, Any]
    tool_response: dict[str, Any]
    queued_at: str


@dataclass
class _ToolJob:
    """Internal state shared by a tool worker and its submitting graph thread."""

    job_id: str
    tool_name: str
    arguments: dict[str, Any]
    function: Callable[..., Any]
    tool_call_id: str | None
    graph_run_token: str
    conversation_id: str
    conversation_label: str
    submitted_at: str
    done_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    released: bool = False
    completion: BackgroundToolCompletion | None = None


RELEASE_DESCRIPTION = (
    "The tool call has been successfully submitted but execution is still in "
    "progress. Now wait; when tool execution finishes, the result will be given "
    "in a follow-up message."
)


class SerialToolExecutor:
    """Tool execution with opt-in background release.

    This executor is also the source of truth for WebUI tool schemas and
    tool-to-owner metadata.
    """

    def __init__(
        self,
        approval_handler: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        tools: Optional[list[BaseTool]] = None,
    ) -> None:
        """Initialize the executor."""
        self.approval_handler = approval_handler
        self.tools: list[BaseTool] = tools if tools is not None else []
        self.tool_specs: dict[str, ExposedToolSpec] = {}
        self.tool_spec_owners: dict[str, BaseTool] = {}
        self.tool_execution_history: list[dict[str, Any]] = []
        self.release_handler: Callable[[dict[str, Any]], None] | None = None
        self.completion_handler: Callable[[dict[str, Any]], None] | None = None
        self.dequeue_handler: Callable[[list[str]], None] | None = None
        self.cleanup_handler: Callable[[list[str]], None] | None = None
        self.completion_event = threading.Event()
        self._state_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._active_graph_runs: set[str] = set()
        self._jobs_by_run: dict[str, dict[str, _ToolJob]] = {}
        self._ready_by_run: dict[str, deque[BackgroundToolCompletion]] = {}

    def set_queue_handlers(
        self,
        *,
        release_handler: Callable[[dict[str, Any]], None] | None = None,
        completion_handler: Callable[[dict[str, Any]], None] | None = None,
        dequeue_handler: Callable[[list[str]], None] | None = None,
        cleanup_handler: Callable[[list[str]], None] | None = None,
    ) -> None:
        """Configure display-queue callbacks owned by the task manager."""
        self.release_handler = release_handler
        self.completion_handler = completion_handler
        self.dequeue_handler = dequeue_handler
        self.cleanup_handler = cleanup_handler

    def begin_graph_run(self) -> str:
        """Create and register a token for one graph invocation."""
        token = uuid.uuid4().hex
        with self._state_lock:
            self._active_graph_runs.add(token)
            self._jobs_by_run[token] = {}
            self._ready_by_run[token] = deque()
        return token

    def finish_graph_run(self, graph_run_token: str) -> None:
        """Invalidate a graph run and remove all of its display-queue entries."""
        with self._state_lock:
            self._active_graph_runs.discard(graph_run_token)
            jobs = self._jobs_by_run.pop(graph_run_token, {})
            self._ready_by_run.pop(graph_run_token, None)
            job_ids = list(jobs)
            self._refresh_completion_event_locked()
        if job_ids and self.cleanup_handler is not None:
            self.cleanup_handler(job_ids)

    def has_ready_completions(self, graph_run_token: str | None) -> bool:
        """Return whether a graph run has completed messages waiting."""
        if graph_run_token is None:
            return False
        with self._state_lock:
            return bool(self._ready_by_run.get(graph_run_token))

    def has_background_jobs(self, graph_run_token: str | None) -> bool:
        """Return whether a graph run owns released or queued worker jobs."""
        if graph_run_token is None:
            return False
        with self._state_lock:
            return bool(self._jobs_by_run.get(graph_run_token))

    def drain_ready_completions(
        self,
        graph_run_token: str | None,
    ) -> list[BackgroundToolCompletion]:
        """Drain ready completions for one active graph run in FIFO order."""
        if graph_run_token is None:
            return []
        with self._state_lock:
            ready = self._ready_by_run.get(graph_run_token)
            if ready is None:
                return []
            completions = list(ready)
            ready.clear()
            jobs = self._jobs_by_run.get(graph_run_token, {})
            for completion in completions:
                jobs.pop(completion.job_id, None)
            self._refresh_completion_event_locked()
        if completions and self.dequeue_handler is not None:
            self.dequeue_handler([completion.job_id for completion in completions])
        return completions

    def register_tools(self, tools: BaseTool | list[BaseTool]) -> None:
        """Register one or more tool objects."""
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError("Input should be a BaseTool or a list of BaseTool objects.")
            if tool not in self.tools:
                self.tools.append(tool)
            for exposed in tool.exposed_tools:
                if not exposed.model_visible:
                    continue
                spec = ExposedToolSpec(
                    name=exposed.name,
                    function=exposed.function,
                    require_approval=(
                        tool.require_approval if exposed.require_approval is None else exposed.require_approval
                    ),
                    schema=exposed.schema,
                    model_visible=exposed.model_visible,
                    release_timeout=exposed.release_timeout,
                )
                self.tool_specs[spec.name] = spec
                self.tool_spec_owners[spec.name] = tool

    def unregister_tool(self, tool: BaseTool) -> None:
        """Unregister one tool object and its exposed model-visible specs."""
        if not isinstance(tool, BaseTool):
            raise ValueError("Input should be a BaseTool object.")
        if tool in self.tools:
            self.tools.remove(tool)
        self.unregister_tool_specs(tool.exposed_tools)

    def unregister_tool_specs(self, exposed_tools: Sequence[ExposedToolSpec]) -> None:
        """Remove exposed tool specs matching the given exposed tool records."""
        for exposed in exposed_tools:
            if exposed.model_visible:
                self.tool_specs.pop(exposed.name, None)
                self.tool_spec_owners.pop(exposed.name, None)

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        """Return model-facing OpenAI tool schemas."""
        return [
            spec.schema or generate_openai_tool_schema(tool_name=name, func=spec.function)
            for name, spec in self.tool_specs.items()
            if spec.model_visible
        ]

    def list_tool_ui_schemas(self) -> list[dict[str, Any]]:
        """Return WebUI tool schemas with display-only metadata."""
        schemas = []
        for name, spec in self.tool_specs.items():
            if not spec.model_visible:
                continue
            schema = copy.deepcopy(
                spec.schema or generate_openai_tool_schema(tool_name=name, func=spec.function)
            )
            owner = self.tool_spec_owners.get(name)
            metadata_getter = getattr(owner, "get_mcp_tool_metadata", None)
            if callable(metadata_getter):
                schema["mcp"] = metadata_getter(name)
            schemas.append(schema)
        return schemas

    def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        allow_release: bool = True,
        graph_run_token: str | None = None,
        conversation_id: str = "primary",
        conversation_label: str = "Primary",
    ) -> list[ToolExecutionResult]:
        """Submit assistant-requested tool calls in message order."""
        return [
            self.execute_tool_call(
                tool_call,
                allow_release=allow_release,
                graph_run_token=graph_run_token,
                conversation_id=conversation_id,
                conversation_label=conversation_label,
            )
            for tool_call in tool_calls
        ]

    def execute_tool_calls_from_message(
        self,
        message: dict[str, Any],
        *,
        allow_release: bool = True,
        graph_run_token: str | None = None,
        conversation_id: str = "primary",
        conversation_label: str = "Primary",
    ) -> list[dict[str, Any]]:
        """Execute tool calls found in an assistant message.

        Parameters
        ----------
        message : dict[str, Any]
            Assistant message that may contain tool calls.
        Returns
        -------
        list[dict[str, Any]]
            Tool messages generated by the executed tool calls.
        """
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) == 0:
            return []
        results = self.execute_tool_calls(
            tool_calls,
            allow_release=allow_release,
            graph_run_token=graph_run_token,
            conversation_id=conversation_id,
            conversation_label=conversation_label,
        )
        return [result.message for result in results]

    def execute_tool_call(
        self,
        tool_call: dict[str, Any],
        *,
        allow_release: bool = True,
        graph_run_token: str | None = None,
        conversation_id: str = "primary",
        conversation_label: str = "Primary",
    ) -> ToolExecutionResult:
        """Execute one tool call and normalize its response."""
        function = tool_call.get("function", {})
        tool_name = function.get("name")
        if tool_name not in self.tool_specs:
            raise ValueError(f"Unknown tool requested: {tool_name}")
        spec = self.tool_specs[tool_name]
        if not spec.model_visible:
            raise ValueError(f"Tool {tool_name!r} is not available for model tool calls.")
        arguments = self.parse_arguments(function.get("arguments"))
        try:
            approval_required = (
                spec.require_approval(arguments)
                if callable(spec.require_approval)
                else spec.require_approval
            )
            if approval_required and self.approval_handler is not None:
                approved = self.approval_handler(tool_name, arguments)
                if not approved:
                    message = generate_openai_message(
                        content=self.serialize_result({"error": "Tool execution was denied by the user."}),
                        role="tool",
                        tool_call_id=tool_call.get("id"),
                    )
                    return ToolExecutionResult(message=message)
        except Exception as exc:
            logger.exception("Tool execution failed for %s", tool_name)
            content = self.serialize_result({"error": str(exc)})
            return ToolExecutionResult(
                message=generate_openai_message(
                    content=content,
                    role="tool",
                    tool_call_id=tool_call.get("id"),
                )
            )

        can_release = (
            allow_release
            and graph_run_token is not None
            and spec.release_timeout is not None
        )
        if not can_release:
            content, _failed = self._execute_tool_body(
                tool_name,
                arguments,
                spec.function,
            )
            return ToolExecutionResult(
                message=generate_openai_message(
                    content=content,
                    role="tool",
                    tool_call_id=tool_call.get("id"),
                )
            )

        job = self._create_job(
            tool_name=tool_name,
            arguments=arguments,
            function=spec.function,
            tool_call_id=tool_call.get("id"),
            graph_run_token=graph_run_token,
            conversation_id=conversation_id,
            conversation_label=conversation_label,
        )
        worker = threading.Thread(
            target=self._run_job,
            args=(job,),
            name=f"eaa-tool-{job.job_id}",
            daemon=True,
        )
        worker.start()
        if job.done_event.wait(spec.release_timeout):
            return ToolExecutionResult(
                message=self._immediate_message_from_job(job),
            )
        with job.lock:
            if job.completion is not None:
                completed_before_release = True
            else:
                completed_before_release = False
                job.released = True
                self._publish_released_job(job)
        if completed_before_release:
            return ToolExecutionResult(
                message=self._immediate_message_from_job(job),
            )
        release_payload = {
            "job_id": job.job_id,
            "status": "executing",
            "description": RELEASE_DESCRIPTION,
        }
        return ToolExecutionResult(
            message=generate_openai_message(
                content=json.dumps(release_payload),
                role="tool",
                tool_call_id=tool_call.get("id"),
            ),
            released_job_id=job.job_id,
        )

    def _create_job(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        function: Callable[..., Any],
        tool_call_id: str | None,
        graph_run_token: str,
        conversation_id: str,
        conversation_label: str,
    ) -> _ToolJob:
        """Create and register one releasable worker job."""
        with self._state_lock:
            if graph_run_token not in self._active_graph_runs:
                raise RuntimeError("Cannot release a tool call for an inactive graph run.")
            existing_ids = {
                job_id
                for jobs in self._jobs_by_run.values()
                for job_id in jobs
            }
            job_id = uuid.uuid4().hex[:8]
            while job_id in existing_ids:
                job_id = uuid.uuid4().hex[:8]
            job = _ToolJob(
                job_id=job_id,
                tool_name=tool_name,
                arguments=arguments,
                function=function,
                tool_call_id=tool_call_id,
                graph_run_token=graph_run_token,
                conversation_id=conversation_id,
                conversation_label=conversation_label,
                submitted_at=self._timestamp(),
            )
            self._jobs_by_run[graph_run_token][job_id] = job
        return job

    def _run_job(self, job: _ToolJob) -> None:
        """Execute and normalize one tool call in a daemon worker."""
        content, failed = self._execute_tool_body(
            job.tool_name,
            job.arguments,
            job.function,
        )
        parsed = json.loads(content)
        if failed:
            completion_payload = {
                "job_id": job.job_id,
                "tool_name": job.tool_name,
                "status": "failed",
                "error": str(parsed["error"]),
            }
        else:
            completion_payload = {
                "job_id": job.job_id,
                "tool_name": job.tool_name,
                "status": "completed",
                "result": parsed,
            }
        queued_at = self._timestamp()
        completion = BackgroundToolCompletion(
            job_id=job.job_id,
            tool_name=job.tool_name,
            conversation_id=job.conversation_id,
            conversation_label=job.conversation_label,
            status=completion_payload["status"],
            system_message=generate_openai_message(
                content=json.dumps(completion_payload),
                role="system",
            ),
            tool_response=generate_openai_message(
                content=content,
                role="tool",
                tool_call_id=job.tool_call_id,
            ),
            queued_at=queued_at,
        )
        with job.lock:
            job.completion = completion
            job.done_event.set()
            if job.released:
                self._queue_completed_job(job, completion)

    def _execute_tool_body(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        function: Callable[..., Any],
    ) -> tuple[str, bool]:
        """Execute one tool body and normalize its result."""
        try:
            result = function(**arguments)
            content = self.serialize_result(result)
            failed = False
        except Exception as exc:
            logger.exception("Tool execution failed for %s", tool_name)
            content = self.serialize_result({"error": str(exc)})
            failed = True
        with self._history_lock:
            self.tool_execution_history.append(
                {"tool_name": tool_name, "arguments": arguments}
            )
        return content, failed

    def _immediate_message_from_job(self, job: _ToolJob) -> dict[str, Any]:
        """Build the ordinary tool response for a worker that met its deadline."""
        with job.lock:
            completion = job.completion
        if completion is None:
            raise RuntimeError("Tool worker completed without a normalized result.")
        with self._state_lock:
            jobs = self._jobs_by_run.get(job.graph_run_token)
            if jobs is not None:
                jobs.pop(job.job_id, None)
        return completion.tool_response

    def _publish_released_job(self, job: _ToolJob) -> None:
        """Publish one execution-queue entry while holding the job lock."""
        entry = {
            "job_id": job.job_id,
            "tool_name": job.tool_name,
            "conversation_id": job.conversation_id,
            "conversation_label": job.conversation_label,
            "status": "executing",
            "timestamp": job.submitted_at,
        }
        with self._state_lock:
            if job.graph_run_token not in self._active_graph_runs:
                return
            if self.release_handler is not None:
                self.release_handler(entry)

    def _queue_completed_job(
        self,
        job: _ToolJob,
        completion: BackgroundToolCompletion,
    ) -> None:
        """Move one released execution into the graph and display message queues."""
        entry = {
            "job_id": completion.job_id,
            "tool_name": completion.tool_name,
            "conversation_id": completion.conversation_id,
            "conversation_label": completion.conversation_label,
            "status": completion.status,
            "content": completion.system_message["content"],
            "queued_at": completion.queued_at,
        }
        with self._state_lock:
            if job.graph_run_token not in self._active_graph_runs:
                return
            ready = self._ready_by_run.get(job.graph_run_token)
            if ready is None:
                return
            ready.append(completion)
            self.completion_event.set()
            if self.completion_handler is not None:
                self.completion_handler(entry)

    def _refresh_completion_event_locked(self) -> None:
        """Synchronize the wake event with all run-scoped ready queues."""
        if any(self._ready_by_run.values()):
            self.completion_event.set()
        else:
            self.completion_event.clear()

    @staticmethod
    def _timestamp() -> str:
        """Return an ISO UTC timestamp for queue records."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00",
            "Z",
        )

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
            image_path = payload.get(TOOL_IMAGE_PATH_FIELD)
            if isinstance(image_path, list):
                return [value for value in image_path if isinstance(value, str) and value.strip()]
            if isinstance(image_path, str) and image_path.strip():
                return [image_path]
            legacy_image_paths = payload.get("image_paths")
            if isinstance(legacy_image_paths, list):
                return [value for value in legacy_image_paths if isinstance(value, str) and value.strip()]
            legacy_image_path = payload.get("image_path")
            if isinstance(legacy_image_path, str) and legacy_image_path.strip():
                return [legacy_image_path]
            return []
        return []

    @classmethod
    def extract_followup_messages_from_tool_response(cls, content: Any) -> list[Dict[str, Any]]:
        """Extract follow-up messages emitted by a tool response payload."""
        payload = cls.parse_tool_response_payload(content)
        if payload is None:
            return []
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return []
        return [message for message in messages if isinstance(message, dict)]

    @classmethod
    def build_tool_followup_messages(
        cls,
        tool_response: Dict[str, Any],
        *,
        message_with_yielded_image: str,
    ) -> list[dict[str, Any]]:
        """Generate follow-up messages after a tool finishes.

        Parameters
        ----------
        tool_response : dict[str, Any]
            Normalized tool response message.
        message_with_yielded_image : str
            User-facing text used when an image path is returned.

        Returns
        -------
        list[dict[str, Any]]
            Follow-up messages to append after tool execution.
        """
        followup_messages = cls.extract_followup_messages_from_tool_response(
            tool_response.get("content")
        )
        image_paths = cls.extract_image_paths_from_tool_response(tool_response.get("content"))
        if len(image_paths) > 0:
            followup_messages.append(
                generate_openai_message(
                    content=message_with_yielded_image,
                    image_path=image_paths,
                    role="user",
                )
            )
        return followup_messages

    @staticmethod
    def serialize_result(result: Any) -> str:
        """Serialize a tool result into normalized JSON content."""
        return json.dumps(normalize_tool_result(result), default=SerialToolExecutor._json_default)

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Convert common non-JSON-native tool values into serializable forms."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
