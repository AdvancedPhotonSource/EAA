"""Agent-side WebUI runtime controller and FastAPI adapter."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import formatdate
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from eaa_core.task_manager.skills import skill_catalog_to_dicts


@dataclass
class RuntimeEvent:
    """One WebUI runtime event."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConversation:
    """Server-owned display state for one agent conversation."""

    id: str
    label: str
    kind: str
    status: str = "idle"
    terminated: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable conversation snapshot."""
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "status": self.status,
            "terminated": self.terminated,
            "messages": list(self.messages),
            "pending_approval": self.pending_approval,
        }


class WebUIRuntimeController:
    """Thread-safe runtime state and command bus for one task manager."""

    def __init__(
        self,
        task_manager: Any,
        *,
        upload_dir: str = ".tmp",
    ) -> None:
        self.task_manager = task_manager
        self.upload_dir = upload_dir
        self.input_queue: queue.Queue[str] = queue.Queue()
        self.approval_queue: queue.Queue[bool] = queue.Queue()
        self.approval_queues: dict[str, queue.Queue[bool]] = {
            "primary": self.approval_queue,
        }
        self.interrupt_event = threading.Event()
        self.lock = threading.Lock()
        self.subscribers: set[queue.Queue[RuntimeEvent]] = set()
        self.messages: list[dict[str, Any]] = []
        self.message_event_counter = 0
        self.subagent_counter = 0
        self.status = "idle"
        self.input_requested = False
        self.queued_input_count = 0
        self.pending_approval: dict[str, Any] | None = None
        self.conversations: dict[str, RuntimeConversation] = {
            "primary": RuntimeConversation(
                id="primary",
                label="Primary",
                kind="primary",
            )
        }

    def build(self) -> None:
        """Initialize runtime state."""
        with self.lock:
            self.conversations.setdefault(
                "primary",
                RuntimeConversation(id="primary", label="Primary", kind="primary"),
            )
            self.approval_queues.setdefault("primary", self.approval_queue)

    def create_conversation(
        self,
        *,
        label: str | None = None,
        kind: str = "subagent",
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Create and publish a conversation tab."""
        with self.lock:
            if conversation_id is None:
                self.subagent_counter += 1
                conversation_id = f"subagent-{self.subagent_counter}"
            if label is None:
                label = conversation_id.replace("-", " ").title()
            conversation = RuntimeConversation(
                id=conversation_id,
                label=label,
                kind=kind,
            )
            self.conversations[conversation_id] = conversation
            self.approval_queues[conversation_id] = queue.Queue()
            snapshot = conversation.snapshot()
        self.publish("conversation.created", {"conversation": snapshot})
        return snapshot

    def ensure_conversation(self, conversation_id: str) -> RuntimeConversation:
        """Return an existing conversation, creating a generic one if needed."""
        with self.lock:
            conversation = self.conversations.get(conversation_id)
            if conversation is None:
                conversation = RuntimeConversation(
                    id=conversation_id,
                    label=conversation_id.replace("-", " ").title(),
                    kind="subagent" if conversation_id != "primary" else "primary",
                )
                self.conversations[conversation_id] = conversation
                self.approval_queues.setdefault(conversation_id, queue.Queue())
            return conversation

    def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Publish one event to connected browser streams.

        Supported event types are:

        - ``message.created``: a live display message was added.
          Payload: ``{"message": <message dict>}``.
        - ``status.changed``: runtime status or input/interrupt flags changed.
          Payload: ``{"status": str, "input_requested": bool,
          "interrupt_requested": bool}``.
        - ``input.requested``: the agent is waiting for user input.
          Payload: ``{"prompt": str}``.
        - ``approval.requested``: a tool call is waiting for approval.
          Payload: ``{"tool_name": str, "arguments": dict}``.
        - ``interrupt.requested``: cooperative interruption was requested.
          Payload matches ``status.changed``.
        - ``interrupt.cleared``: the pending interrupt flag was cleared.
          Payload matches ``status.changed``.

        Planned but not yet emitted event types are ``tool.started``,
        ``tool.finished``, and ``ui.tab.opened``.
        """
        event = RuntimeEvent(event_type, payload or {})
        with self.lock:
            subscribers = list(self.subscribers)
        for subscriber in subscribers:
            subscriber.put(event)

    def subscribe(self) -> queue.Queue[RuntimeEvent]:
        """Register and return an event queue for one SSE client."""
        subscriber: queue.Queue[RuntimeEvent] = queue.Queue()
        with self.lock:
            self.subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[RuntimeEvent]) -> None:
        """Remove an SSE client event queue."""
        with self.lock:
            self.subscribers.discard(subscriber)

    def snapshot(self) -> dict[str, Any]:
        """Return current server-owned WebUI state."""
        with self.lock:
            messages = list(self.messages)
            status = self.status
            input_requested = self.input_requested
            interrupt_requested = self.interrupt_event.is_set()
            pending_approval = self.pending_approval
            conversations = [
                conversation.snapshot()
                for conversation in self.conversations.values()
            ]
        return {
            "conversations": conversations,
            "messages": messages,
            "status": status,
            "input_requested": input_requested,
            "interrupt_requested": interrupt_requested,
            "pending_approval": pending_approval,
        }

    def publish_message(self, message: dict[str, Any], conversation_id: str = "primary") -> None:
        """Publish one message to live WebUI clients."""
        payload = dict(message)
        with self.lock:
            conversation = self.conversations.setdefault(
                conversation_id,
                RuntimeConversation(
                    id=conversation_id,
                    label=conversation_id.replace("-", " ").title(),
                    kind="subagent" if conversation_id != "primary" else "primary",
                ),
            )
            if "id" not in payload:
                self.message_event_counter += 1
                payload["id"] = f"runtime-{self.message_event_counter}"
            conversation.messages.append(payload)
            if conversation_id == "primary":
                self.messages.append(payload)
        self.publish("message.created", {"conversation_id": conversation_id, "message": payload})

    def status_payload(self) -> dict[str, Any]:
        """Return the current runtime status payload."""
        with self.lock:
            return {
                "status": self.status,
                "input_requested": self.input_requested,
                "interrupt_requested": self.interrupt_event.is_set(),
            }

    def set_status(
        self,
        status: str,
        *,
        input_requested: bool | None = None,
        conversation_id: str = "primary",
    ) -> None:
        """Update session status and publish it."""
        with self.lock:
            conversation = self.conversations.setdefault(
                conversation_id,
                RuntimeConversation(
                    id=conversation_id,
                    label=conversation_id.replace("-", " ").title(),
                    kind="subagent" if conversation_id != "primary" else "primary",
                ),
            )
            conversation.status = status
            if conversation_id == "primary":
                self.status = status
            if input_requested is not None:
                self.input_requested = input_requested
            payload = {
                "status": self.status,
                "input_requested": self.input_requested,
                "interrupt_requested": self.interrupt_event.is_set(),
                "conversation_id": conversation_id,
                "conversation": conversation.snapshot(),
            }
        self.publish("status.changed", payload)

    def request_input(self, prompt: str | None = None, conversation_id: str = "primary") -> str:
        """Block until the WebUI submits ordered user input."""
        with self.lock:
            conversation = self.conversations.get(conversation_id)
            previous_status = conversation.status if conversation is not None else self.status
        self.set_status("waiting_for_input", input_requested=True, conversation_id=conversation_id)
        if prompt:
            self.publish("input.requested", {"conversation_id": conversation_id, "prompt": prompt})
        value = self.input_queue.get()
        with self.lock:
            self.queued_input_count = max(0, self.queued_input_count - 1)
        self.set_status(previous_status, input_requested=False, conversation_id=conversation_id)
        return value

    def submit_input(self, content: str) -> bool:
        """Queue user input from the WebUI when no input is already queued."""
        with self.lock:
            if self.queued_input_count:
                return False
            self.queued_input_count += 1
        self.input_queue.put(content)
        return True

    def request_interrupt(self) -> None:
        """Request cooperative interruption."""
        self.interrupt_event.set()
        payload = self.status_payload()
        self.publish("interrupt.requested", payload)
        self.publish("status.changed", payload)

    def clear_interrupt(self) -> None:
        """Clear the cooperative interrupt flag."""
        self.interrupt_event.clear()
        payload = self.status_payload()
        self.publish("interrupt.cleared", payload)
        self.publish("status.changed", payload)

    def check_interrupt(self) -> None:
        """Raise ``KeyboardInterrupt`` if a WebUI interrupt is pending."""
        if self.interrupt_event.is_set():
            self.clear_interrupt()
            raise KeyboardInterrupt

    def request_approval(
        self,
        tool_name: str,
        tool_kwargs: dict[str, Any],
        conversation_id: str = "primary",
    ) -> bool:
        """Publish an approval request and wait for an ordered response."""
        with self.lock:
            conversation = self.conversations.setdefault(
                conversation_id,
                RuntimeConversation(
                    id=conversation_id,
                    label=conversation_id.replace("-", " ").title(),
                    kind="subagent" if conversation_id != "primary" else "primary",
                ),
            )
            previous_status = conversation.status
            pending_approval = {
                "conversation_id": conversation_id,
                "tool_name": tool_name,
                "arguments": tool_kwargs,
            }
            conversation.pending_approval = dict(pending_approval)
            if conversation_id == "primary":
                self.pending_approval = dict(pending_approval)
            approval_queue = self.approval_queues.setdefault(conversation_id, queue.Queue())
        self.publish("approval.requested", pending_approval)
        self.set_status("waiting_for_approval", input_requested=True, conversation_id=conversation_id)
        approved = approval_queue.get()
        with self.lock:
            conversation = self.conversations[conversation_id]
            conversation.pending_approval = None
            if conversation_id == "primary":
                self.pending_approval = None
        self.set_status(previous_status, input_requested=False, conversation_id=conversation_id)
        return approved

    def submit_approval(self, approved: bool, conversation_id: str = "primary") -> None:
        """Queue a tool approval decision."""
        with self.lock:
            approval_queue = self.approval_queues.setdefault(conversation_id, queue.Queue())
        approval_queue.put(approved)

    def has_pending_approval(self) -> bool:
        """Return whether a tool approval request is waiting for a response."""
        with self.lock:
            return any(
                conversation.pending_approval is not None
                for conversation in self.conversations.values()
            )

    def has_pending_approval_for_conversation(self, conversation_id: str = "primary") -> bool:
        """Return whether a conversation is waiting for approval."""
        with self.lock:
            conversation = self.conversations.get(conversation_id)
            return bool(conversation and conversation.pending_approval is not None)

    def terminate_conversation(self, conversation_id: str, message: str | None = None) -> None:
        """Mark a conversation terminated and publish the terminal event."""
        with self.lock:
            conversation = self.conversations.setdefault(
                conversation_id,
                RuntimeConversation(
                    id=conversation_id,
                    label=conversation_id.replace("-", " ").title(),
                    kind="subagent" if conversation_id != "primary" else "primary",
                ),
            )
            conversation.terminated = True
            conversation.status = "terminated"
            payload = {
                "conversation_id": conversation_id,
                "message": message,
                "conversation": conversation.snapshot(),
            }
        self.publish("conversation.terminated", payload)

    def ensure_upload_dir(self) -> str:
        """Ensure and return the configured upload directory."""
        os.makedirs(self.upload_dir, exist_ok=True)
        return self.upload_dir

    def save_base64_image(self, image_data: str) -> str:
        """Save a browser-submitted base64 image."""
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(self.ensure_upload_dir(), f"pasted_image_{timestamp}.png")
        with open(path, "wb") as image_file:
            image_file.write(image_bytes)
        return path

    @staticmethod
    def guess_mime_type_from_path(path: str) -> str:
        """Return a basic image MIME type."""
        lower = path.lower()
        if lower.endswith(".png"):
            return "image/png"
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            return "image/jpeg"
        if lower.endswith(".gif"):
            return "image/gif"
        if lower.endswith(".webp"):
            return "image/webp"
        return "application/octet-stream"

    def image_response(self, path: str = Query(...)) -> FileResponse:
        """Build an image file response."""
        normalized_path = os.path.abspath(path)
        if not os.path.exists(normalized_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {normalized_path}")
        stat_result = os.stat(normalized_path)
        headers = {
            "Cache-Control": "public, max-age=3600",
            "ETag": f'"{stat_result.st_mtime_ns:x}-{stat_result.st_size:x}"',
            "Last-Modified": formatdate(stat_result.st_mtime, usegmt=True),
        }
        return FileResponse(
            normalized_path,
            media_type=self.guess_mime_type_from_path(normalized_path),
            headers=headers,
        )

    def upload_image_response(self, payload: dict[str, Any]) -> JSONResponse:
        """Build an upload response."""
        image_data = payload.get("image_data", "")
        if not image_data:
            return JSONResponse({"error": "No image data provided"}, status_code=400)
        try:
            return JSONResponse({"file_path": self.save_base64_image(str(image_data))}, status_code=201)
        except Exception as exc:
            return JSONResponse({"error": f"Invalid image data: {exc}"}, status_code=400)


class WebUIRuntimeServer:
    """FastAPI runtime server owned by the task-manager process."""

    def __init__(
        self,
        controller: WebUIRuntimeController,
        *,
        host: str = "127.0.0.1",
        port: int = 8010,
    ) -> None:
        self.controller = controller
        self.host = host
        self.port = port
        self._server: Any | None = None
        self._thread: threading.Thread | None = None

    def build_app(self) -> FastAPI:
        """Build the FastAPI runtime API."""
        app = FastAPI(title="EAA WebUI Runtime")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/api/state")
        async def state() -> dict[str, Any]:
            return self.controller.snapshot()

        @app.get("/api/events")
        async def events() -> StreamingResponse:
            subscriber = self.controller.subscribe()

            async def stream() -> Any:
                try:
                    while True:
                        event = await asyncio.to_thread(subscriber.get)
                        payload = json.dumps(event.payload)
                        yield f"event: {event.type}\ndata: {payload}\n\n"
                finally:
                    self.controller.unsubscribe(subscriber)

            return StreamingResponse(stream(), media_type="text/event-stream")

        @app.post("/api/input")
        async def submit_input(payload: dict[str, Any]) -> JSONResponse:
            content = str(payload.get("content") or "").strip()
            if not content:
                return JSONResponse({"error": "No content provided"}, status_code=400)
            if self.controller.has_pending_approval_for_conversation("primary"):
                normalized = content.lower()
                if normalized in {"y", "yes"}:
                    self.controller.submit_approval(True, "primary")
                    return JSONResponse({"ok": True, "handled_as": "approval"}, status_code=201)
                if normalized in {"n", "no"}:
                    self.controller.submit_approval(False, "primary")
                    return JSONResponse({"ok": True, "handled_as": "approval"}, status_code=201)
                return JSONResponse(
                    {
                        "error": "Approval response must be yes or no",
                        "message": "Please enter only Yes or No to respond to a tool call approval.",
                    },
                    status_code=409,
                )
            if not self.controller.submit_input(content):
                return JSONResponse({"error": "User input is already queued"}, status_code=409)
            return JSONResponse({"ok": True}, status_code=201)

        @app.post("/api/interrupt")
        async def interrupt() -> dict[str, bool]:
            self.controller.request_interrupt()
            return {"ok": True}

        @app.post("/api/approval")
        async def approval(payload: dict[str, Any]) -> dict[str, bool]:
            conversation_id = str(payload.get("conversation_id") or "primary")
            self.controller.submit_approval(bool(payload.get("approved")), conversation_id)
            return {"ok": True}

        @app.get("/api/skill-catalog")
        async def skill_catalog() -> dict[str, list[dict[str, str]]]:
            return {
                "skills": skill_catalog_to_dicts(
                    self.controller.task_manager.skill_catalog
                )
            }

        @app.get("/api/image")
        def image(path: str) -> Any:
            return self.controller.image_response(path=path)

        @app.post("/api/upload-image")
        async def upload_image(payload: dict[str, Any]) -> Any:
            return self.controller.upload_image_response(payload)

        return app

    def start(self) -> None:
        """Start the runtime API in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        import uvicorn

        config = uvicorn.Config(
            self.build_app(),
            host=self.host,
            port=self.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the runtime API if it is running."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
