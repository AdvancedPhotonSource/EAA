"""Reusable NiceGUI WebUI base for EAA task managers."""

from __future__ import annotations

import argparse
import html as html_module
import json
import re
import subprocess
import sys
from typing import Any
from urllib.parse import quote

from eaa_core.gui.relay import SQLiteWebUIRelay, set_message_db_path

IMAGE_TAG_PATTERN = re.compile(r"<img\s+([^>\s]+)>")
APPROVAL_PATTERN = re.compile(r"Approve\?\s*\[y/N\]:", re.IGNORECASE)
APPROVAL_ARGUMENTS_PATTERN = re.compile(
    r"Arguments:\s*(?P<arguments>.*?)\nApprove\?\s*\[y/N\]:",
    re.IGNORECASE | re.DOTALL,
)
APPROVAL_TOOL_PATTERN = re.compile(
    r"Tool\s+'(?P<tool_name>[^']+)'\s+requires approval",
    re.IGNORECASE,
)


class NiceGUIWebUIBase:
    """Base NiceGUI chat UI backed by the shared EAA SQLite relay.

    Subclasses can override the ``build_*`` hook methods to add task-specific
    controls while preserving the SQL communication protocol.

    Parameters
    ----------
    session_db_path : str
        SQLite session database shared with the task manager.
    title : str, default="EAA WebUI"
        Browser and header title.
    upload_dir : str, default=".tmp"
        Directory used to store pasted images.
    poll_interval : float, default=1.0
        Message and status polling interval in seconds.
    """

    image_route = "/api/image"
    upload_route = "/api/upload-image"
    skill_catalog_route = "/api/skill-catalog"
    markdown_extras = ["fenced-code-blocks", "tables", "code-friendly", "break-on-newline"]
    foldable_roles = {"user", "user_webui", "tool"}
    folded_message_line_limit = 10

    def __init__(
        self,
        session_db_path: str,
        *,
        title: str = "EAA WebUI",
        upload_dir: str = ".tmp",
        poll_interval: float = 1.0,
    ) -> None:
        self.session_db_path = session_db_path
        self.title = title
        self.poll_interval = poll_interval
        self.relay = SQLiteWebUIRelay(session_db_path, upload_dir=upload_dir)
        self.last_message_id: int | None = None
        self.rendered_message_ids: set[str] = set()
        self.pending_messages: dict[str, str] = {}
        self.pending_message_counter = 0
        self.messages_container: Any | None = None
        self.images_container: Any | None = None
        self.input_box: Any | None = None
        self.connection_status_label: Any | None = None
        self.input_status_label: Any | None = None
        self.image_dialog: Any | None = None
        self.image_dialog_image: Any | None = None
        self.skill_catalog: list[dict[str, str]] = []

    def run(self, host: str = "127.0.0.1", port: int = 8008) -> None:
        """Run the NiceGUI WebUI.

        Parameters
        ----------
        host : str, default="127.0.0.1"
            Bind host.
        port : int, default=8008
            Bind port.
        """
        from nicegui import app, ui

        self.register_api_routes(app)

        @ui.page("/")
        def index() -> None:
            self.build_page()

        print(f"EAA NiceGUI WebUI running at http://{host}:{port} (DB: {self.session_db_path})")
        ui.run(host=host, port=port, title=self.title, reload=False)

    def register_api_routes(self, app: Any) -> None:
        """Register image routes on a NiceGUI/FastAPI app.

        Parameters
        ----------
        app : Any
            NiceGUI application object.
        """

        async def upload_image(payload: dict[str, Any]) -> Any:
            return self.relay.upload_image_response(payload)

        async def skill_catalog() -> Any:
            return self.relay.skill_catalog_response()

        app.add_api_route(
            self.image_route,
            self.relay.image_response,
            methods=["GET"],
        )
        app.add_api_route(
            self.upload_route,
            upload_image,
            methods=["POST"],
        )
        app.add_api_route(
            self.skill_catalog_route,
            skill_catalog,
            methods=["GET"],
        )

    def build_page(self) -> None:
        """Build the default page layout."""
        from nicegui import ui

        ui.add_head_html(self.styles())
        self.install_keyboard_shortcuts()
        self.install_clipboard_handler()
        self.install_image_preview_handler()

        with ui.column().classes("eaa-page"):
            self.build_header()
            self.build_before_messages()
            with ui.row().classes("eaa-main"):
                with ui.column().classes("eaa-chat"):
                    self.messages_container = ui.column().classes("eaa-messages")
                with ui.column().classes("eaa-sidebar"):
                    self.build_sidebar_header()
                    self.images_container = ui.column().classes("eaa-images")
            self.build_after_messages()
            self.build_input_area()

        self.build_image_dialog()
        self.install_skill_autocomplete_handler()

        ui.timer(self.poll_interval, self.poll_once)
        ui.timer(0.1, self.poll_once, once=True)

    def build_image_dialog(self) -> None:
        """Build a shared full-size image preview dialog."""
        from nicegui import ui

        with ui.dialog() as self.image_dialog:
            self.image_dialog_image = ui.html("").classes("eaa-image-dialog-card")

    def open_image_dialog(self, src: str) -> None:
        """Open the full-size image dialog for a given source.

        Parameters
        ----------
        src : str
            Image source URL.
        """
        if self.image_dialog is None or self.image_dialog_image is None:
            return
        safe_src = html_module.escape(src, quote=True)
        self.image_dialog_image.set_content(
            f'<img src="{safe_src}" class="eaa-image-dialog-image" alt="">'
        )
        self.image_dialog.open()

    def build_header(self) -> None:
        """Build the default header."""
        from nicegui import ui

        with ui.row().classes("eaa-header"):
            ui.label(self.title).classes("eaa-title")
            self.connection_status_label = ui.label("Connecting...").classes("eaa-status")

    def build_before_messages(self) -> None:
        """Hook for subclasses to add content above the chat layout."""

    def build_after_messages(self) -> None:
        """Hook for subclasses to add content below the chat layout."""

    def build_sidebar_header(self) -> None:
        """Build the default image sidebar header."""
        from nicegui import ui

        ui.label("Images").classes("eaa-sidebar-title")

    def build_input_area(self) -> None:
        """Build the default message input area."""
        from nicegui import ui

        with ui.column().classes("eaa-input-panel"):
            self.input_status_label = ui.label("").classes("eaa-processing hidden")
            with ui.row().classes("eaa-input-row"):
                self.input_box = ui.textarea(
                    placeholder=(
                        "Type a message. Paste an image or use "
                        "<img /absolute/path/to/image.png>."
                    )
                ).props("autogrow outlined").classes("eaa-input")
                ui.html('<div class="eaa-skill-suggestions hidden"></div>')
                with ui.row().classes("eaa-actions"):
                    self.build_extra_input_controls()
                    ui.button("Send", on_click=self.send_current_message).classes("eaa-send")

    def build_extra_input_controls(self) -> None:
        """Hook for subclasses to add controls beside the Send button."""

    def poll_once(self) -> None:
        """Poll the relay database once and refresh visible UI state."""
        try:
            messages = self.relay.load_messages(since_id=self.last_message_id)
            status = self.relay.get_user_input_requested()
            if messages:
                self.render_messages(messages)
            self.update_processing_status(status)
            if self.connection_status_label is not None:
                self.connection_status_label.text = "Connected"
        except Exception:
            if self.connection_status_label is not None:
                self.connection_status_label.text = "Reconnecting..."

    def render_messages(self, messages: list[dict[str, Any]]) -> None:
        """Render relay messages.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            WebUI-formatted messages from the relay.
        """
        rendered_any = False
        rendered_images = False
        for message in messages:
            if not self.should_display_message(message):
                continue
            message_id = str(message.get("id", ""))
            if message_id in self.rendered_message_ids:
                continue
            consumed_pending = self.consume_pending_message(message)
            if consumed_pending:
                self.rendered_message_ids.add(message_id)
                raw_id = message.get("id")
                if isinstance(raw_id, int):
                    self.last_message_id = raw_id
                continue
            rendered_images = self.render_message(message) or rendered_images
            rendered_any = True
            self.rendered_message_ids.add(message_id)
            raw_id = message.get("id")
            if isinstance(raw_id, int):
                self.last_message_id = raw_id
        if rendered_any:
            self.scroll_messages_to_bottom()
        if rendered_images:
            self.scroll_images_to_bottom()
        self.on_messages_loaded(messages)

    def scroll_messages_to_bottom(self) -> None:
        """Auto-scroll the message container to the bottom while in follow mode."""
        self.scroll_container_to_bottom(".eaa-messages")

    def scroll_images_to_bottom(self) -> None:
        """Auto-scroll the sidebar image panel to the bottom while in follow mode."""
        self.scroll_container_to_bottom(".eaa-images")

    def scroll_container_to_bottom(self, selector: str) -> None:
        """Auto-scroll a container to the bottom while in follow mode.

        Maintains a per-element ``__eaaFollow`` flag so the view stays pinned
        to the newest content until the user scrolls up. Scrolling is instant
        so rapid bursts of updates don't break the follow chain.

        Parameters
        ----------
        selector : str
            CSS selector identifying the scrollable container.
        """
        from nicegui import ui

        ui.run_javascript(
            f"""
            (() => {{
              const el = document.querySelector({selector!r});
              if (!el) return;
              const NEAR_BOTTOM = 100;
              if (!el.__eaaScrollInit) {{
                el.__eaaScrollInit = true;
                el.__eaaFollow = true;
                const updateFollow = () => {{
                  const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
                  el.__eaaFollow = distance <= NEAR_BOTTOM;
                }};
                el.addEventListener('wheel', updateFollow, {{ passive: true }});
                el.addEventListener('touchmove', updateFollow, {{ passive: true }});
                el.addEventListener('keydown', (event) => {{
                  const keys = ['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End'];
                  if (keys.includes(event.key)) {{
                    requestAnimationFrame(updateFollow);
                  }}
                }});
              }}
              if (el.__eaaFollow) {{
                requestAnimationFrame(() => {{
                  el.scrollTop = el.scrollHeight;
                }});
              }}
            }})();
            """
        )

    def should_display_message(self, message: dict[str, Any]) -> bool:
        """Return whether a relay message should be displayed.

        Parameters
        ----------
        message : dict[str, Any]
            WebUI-formatted message.

        Returns
        -------
        bool
            Whether to render the message.
        """
        return True

    def render_message(self, message: dict[str, Any]) -> bool:
        """Render one message in the transcript.

        Parameters
        ----------
        message : dict[str, Any]
            WebUI-formatted message.

        Returns
        -------
        bool
            Whether the message rendered one or more images.
        """
        from nicegui import ui

        if self.messages_container is None:
            return False

        role = str(message.get("role") or "message")
        content = self.format_message_content(message)
        rendered_images = False
        with self.messages_container:
            with ui.column().classes(f"eaa-message eaa-message-{role}"):
                ui.label(self.format_role_label(role)).classes("eaa-role")
                if content:
                    if self.is_approval_message(message):
                        self.render_approval_message_content(content)
                    else:
                        self.render_message_content(role, content)
                self.render_message_tool_calls(message)
                rendered_images = self.render_message_images(message)
                self.maybe_add_approval_actions(message)
        return rendered_images

    def render_message_tool_calls(self, message: dict[str, Any]) -> None:
        """Render assistant tool calls attached to a transcript message.

        Parameters
        ----------
        message : dict[str, Any]
            WebUI-formatted message containing an optional ``tool_calls`` field.
        """
        from nicegui import ui

        tool_calls = self.format_message_tool_calls(message)
        if not tool_calls:
            return
        safe_tool_calls = html_module.escape(tool_calls)
        ui.html(
            "<details class=\"eaa-tool-call-details\" open>"
            "<summary>Tool calls</summary>"
            f"<pre>{safe_tool_calls}</pre>"
            "</details>"
        )

    def render_approval_message_content(self, content: str) -> None:
        """Render a tool-approval prompt with compact arguments and code blocks.

        Parameters
        ----------
        content : str
            Approval prompt text emitted by the task manager.
        """
        from nicegui import ui

        approval = self.format_approval_message(content)
        if approval is None:
            self.render_message_content("system", content)
            return

        ui.markdown(approval["summary"], extras=list(self.markdown_extras)).classes(
            "eaa-markdown"
        )
        safe_arguments = html_module.escape(approval["arguments_json"])
        ui.html(
            "<div class=\"eaa-approval-arguments\">"
            "<div class=\"eaa-approval-section-title\">Arguments</div>"
            f"<pre>{safe_arguments}</pre>"
            "</div>"
        )
        for field in approval["extracted_fields"]:
            safe_label = html_module.escape(field["label"])
            safe_value = html_module.escape(field["value"])
            ui.html(
                "<div class=\"eaa-approval-extracted-field\">"
                f"<div class=\"eaa-approval-section-title\">{safe_label}</div>"
                f"<pre><code>{safe_value}</code></pre>"
                "</div>"
            )

    def render_message_content(self, role: str, content: str) -> None:
        """Render message text, folding long user/tool content by default.

        Parameters
        ----------
        role : str
            Message role.
        content : str
            Markdown content to render.
        """
        from nicegui import ui

        lines = content.replace("\r\n", "\n").splitlines()
        if role not in self.foldable_roles or len(lines) <= self.folded_message_line_limit:
            ui.markdown(content, extras=list(self.markdown_extras)).classes("eaa-markdown")
            return

        markdown = ui.markdown(content, extras=list(self.markdown_extras)).classes(
            "eaa-markdown eaa-markdown-folded"
        )

        def expand_message() -> None:
            markdown.classes(remove="eaa-markdown-folded")
            button.set_visibility(False)

        button = ui.button("Show more", on_click=expand_message).props(
            "flat dense no-caps"
        ).classes("eaa-message-expand")

    def maybe_add_approval_actions(self, message: dict[str, Any]) -> None:
        """Render Yes/No buttons for tool-approval system prompts.

        Detects the ``Approve? [y/N]:`` prompt emitted by the task manager
        and routes button clicks back through the relay user-input queue.

        Parameters
        ----------
        message : dict[str, Any]
            WebUI-formatted message.
        """
        from nicegui import ui

        if str(message.get("role") or "") != "system":
            return
        if not APPROVAL_PATTERN.search(str(message.get("content") or "")):
            return

        state: dict[str, Any] = {"submitted": False, "yes": None, "no": None}

        def submit(value: str) -> None:
            if state["submitted"]:
                return
            state["submitted"] = True
            for button_key in ("yes", "no"):
                button = state[button_key]
                if button is not None:
                    button.disable()
            self.relay.enqueue_user_input(value)
            self.append_pending_message(value)

        with ui.row().classes("eaa-approval-actions"):
            state["yes"] = (
                ui.button("Yes", on_click=lambda: submit("yes"))
                .classes("eaa-approval-button eaa-approval-yes")
            )
            state["no"] = (
                ui.button("No", on_click=lambda: submit("no"))
                .classes("eaa-approval-button eaa-approval-no")
            )

    def render_message_images(self, message: dict[str, Any]) -> bool:
        """Render message images in both transcript and sidebar.

        Combines images attached to the message (``images``/``image`` fields)
        with paths embedded as ``<img /path/to/file>`` tags in the content.

        Parameters
        ----------
        message : dict[str, Any]
            WebUI-formatted message.

        Returns
        -------
        bool
            Whether one or more image elements were rendered.
        """
        from nicegui import ui

        sources: list[str] = []
        seen_sources: set[str] = set()
        attached = message.get("images") or []
        if not attached and message.get("image"):
            attached = [message["image"]]
        for image in attached:
            source = self.image_source(str(image))
            if source not in seen_sources:
                seen_sources.add(source)
                sources.append(source)

        role = str(message.get("role") or "")
        if role != "system":
            for path in self.parse_content_image_paths(message.get("content")):
                source = self.image_source(path)
                if source not in seen_sources:
                    seen_sources.add(source)
                    sources.append(source)

        if len(sources) == 0:
            return False

        with ui.row().classes("eaa-message-images"):
            for src in sources:
                ui.html(
                    self.image_html(src, "eaa-message-image eaa-clickable-image")
                ).classes("eaa-image-html")

        if self.images_container is None:
            return True
        with self.images_container:
            for src in sources:
                ui.html(
                    self.image_html(src, "eaa-sidebar-image eaa-clickable-image")
                ).classes("eaa-image-html")
        return True

    @staticmethod
    def image_html(src: str, class_name: str) -> str:
        """Return lightweight lazy image HTML for transcript and sidebar images.

        Parameters
        ----------
        src : str
            Image source URL or data URL.
        class_name : str
            CSS class names to place on the image element.

        Returns
        -------
        str
            Safe image HTML.
        """
        safe_src = html_module.escape(src, quote=True)
        safe_class = html_module.escape(class_name, quote=True)
        return (
            f'<img src="{safe_src}" class="{safe_class}" '
            f'data-eaa-full-src="{safe_src}" loading="lazy" decoding="async" alt="">'
        )

    @staticmethod
    def parse_content_image_paths(content: Any) -> list[str]:
        """Return image paths embedded as ``<img path>`` tags in text.

        Parameters
        ----------
        content : Any
            Message content text.

        Returns
        -------
        list[str]
            Extracted image paths (in document order).
        """
        if not content:
            return []
        return [match.group(1) for match in IMAGE_TAG_PATTERN.finditer(str(content))]

    def attach_image_click(self, image: Any, src: str) -> None:
        """Attach a click handler that opens the full-size dialog.

        Parameters
        ----------
        image : Any
            NiceGUI image element.
        src : str
            Image source URL to open.
        """
        image.classes("eaa-clickable-image")

    def install_image_preview_handler(self) -> None:
        """Install delegated browser-side image preview handling."""
        from nicegui import ui

        ui.add_body_html(
            """
            <div id="eaa-image-preview" class="eaa-browser-image-preview" aria-hidden="true">
              <button
                class="eaa-browser-image-preview-close"
                type="button"
                aria-label="Close image preview"
              >&times;</button>
              <img class="eaa-browser-image-preview-image" alt="">
            </div>
            <script>
            (() => {
              if (window.__eaaImagePreviewInstalled) return;
              window.__eaaImagePreviewInstalled = true;

              const getPreview = () => document.getElementById('eaa-image-preview');
              const closePreview = () => {
                const preview = getPreview();
                if (!preview) return;
                preview.classList.remove('open');
                preview.setAttribute('aria-hidden', 'true');
                const image = preview.querySelector('img');
                if (image) image.removeAttribute('src');
              };

              document.addEventListener('click', (event) => {
                const target = event.target;
                if (!(target instanceof Element)) return;
                const close = target.closest('.eaa-browser-image-preview-close');
                if (close || target.id === 'eaa-image-preview') {
                  closePreview();
                  return;
                }
                const image = target.closest('img[data-eaa-full-src]');
                if (!image) return;
                const preview = getPreview();
                if (!preview) return;
                const previewImage = preview.querySelector('img');
                if (!previewImage) return;
                previewImage.src = image.getAttribute('data-eaa-full-src') || image.src;
                preview.classList.add('open');
                preview.setAttribute('aria-hidden', 'false');
              });

              document.addEventListener('keydown', (event) => {
                if (event.key === 'Escape') closePreview();
              });
            })();
            </script>
            """
        )

    def format_message_content(self, message: dict[str, Any]) -> str:
        """Return the markdown content rendered for a message.

        Parameters
        ----------
        message : dict[str, Any]
            WebUI-formatted message.

        Returns
        -------
        str
            Markdown text.
        """
        return str(message.get("content") or "").strip()

    def is_approval_message(self, message: dict[str, Any]) -> bool:
        """Return whether a message is a tool-approval system prompt.

        Parameters
        ----------
        message : dict[str, Any]
            WebUI-formatted message.

        Returns
        -------
        bool
            Whether the message should use approval-specific rendering.
        """
        return str(message.get("role") or "") == "system" and bool(
            APPROVAL_PATTERN.search(str(message.get("content") or ""))
        )

    def format_approval_message(self, content: str) -> dict[str, Any] | None:
        """Parse and format a tool-approval prompt for display.

        Parameters
        ----------
        content : str
            Prompt emitted by ``BaseTaskManager`` for tool approval.

        Returns
        -------
        dict[str, Any] | None
            Display fields for a parsed approval prompt, or ``None`` when the
            prompt does not match the expected shape.
        """
        arguments_match = APPROVAL_ARGUMENTS_PATTERN.search(content)
        if arguments_match is None:
            return None
        try:
            arguments = json.loads(arguments_match.group("arguments"))
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None

        scrubbed_arguments, extracted_fields = self.extract_approval_text_fields(
            arguments
        )
        tool_match = APPROVAL_TOOL_PATTERN.search(content)
        tool_name = tool_match.group("tool_name") if tool_match is not None else "tool"
        return {
            "summary": f"Tool `{tool_name}` requires approval before execution.",
            "arguments_json": json.dumps(scrubbed_arguments, indent=2, default=str),
            "extracted_fields": extracted_fields,
        }

    def extract_approval_text_fields(
        self,
        value: Any,
        *,
        path: str = "",
    ) -> tuple[Any, list[dict[str, str]]]:
        """Replace large text fields in approval arguments with placeholders.

        Parameters
        ----------
        value : Any
            Parsed JSON-compatible value.
        path : str, default=""
            Dot/bracket path to ``value`` in the original arguments.

        Returns
        -------
        tuple[Any, list[dict[str, str]]]
            Scrubbed value and extracted text fields for separate rendering.
        """
        extracted_fields: list[dict[str, str]] = []
        if isinstance(value, dict):
            scrubbed: dict[str, Any] = {}
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key.lower() in {"code", "content"} and isinstance(item, str):
                    extracted_fields.append({"label": child_path, "value": item})
                    scrubbed[key] = f"<{child_path} rendered below>"
                    continue
                scrubbed_item, child_fields = self.extract_approval_text_fields(
                    item,
                    path=child_path,
                )
                scrubbed[key] = scrubbed_item
                extracted_fields.extend(child_fields)
            return scrubbed, extracted_fields
        if isinstance(value, list):
            scrubbed_list: list[Any] = []
            for index, item in enumerate(value):
                child_path = f"{path}[{index}]" if path else f"[{index}]"
                scrubbed_item, child_fields = self.extract_approval_text_fields(
                    item,
                    path=child_path,
                )
                scrubbed_list.append(scrubbed_item)
                extracted_fields.extend(child_fields)
            return scrubbed_list, extracted_fields
        return value, extracted_fields

    def format_message_tool_calls(self, message: dict[str, Any]) -> str:
        """Return display text for tool calls attached to a message.

        Parameters
        ----------
        message : dict[str, Any]
            WebUI-formatted message.

        Returns
        -------
        str
            Tool-call text, or an empty string when no calls are present.
        """
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            return ""
        if isinstance(tool_calls, str):
            return tool_calls.strip()
        return str(tool_calls).strip()

    def format_role_label(self, role: str) -> str:
        """Return the display label for a message role.

        Parameters
        ----------
        role : str
            Message role.

        Returns
        -------
        str
            Display label.
        """
        if role == "user_webui":
            return "user"
        return role

    def on_messages_loaded(self, messages: list[dict[str, Any]]) -> None:
        """Hook called after a polling batch has been rendered.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Polled messages.
        """

    def update_processing_status(self, user_input_requested: int | None) -> None:
        """Update the input status text from the relay status flag.

        Parameters
        ----------
        user_input_requested : int | None
            ``0`` while the agent is processing, otherwise input is available.
        """
        if self.input_status_label is None:
            return
        if user_input_requested == 0:
            self.input_status_label.text = "Agent is processing..."
            self.input_status_label.classes(remove="hidden")
        else:
            self.input_status_label.text = ""
            self.input_status_label.classes(add="hidden")

    def send_current_message(self) -> None:
        """Queue the current input box value for the task manager."""
        if self.input_box is None:
            return
        content = str(self.input_box.value or "").strip()
        if not content:
            return
        self.relay.enqueue_user_input(content)
        self.append_pending_message(content)
        self.input_box.value = ""

    def append_pending_message(self, content: str) -> None:
        """Render an optimistic local user message.

        Parameters
        ----------
        content : str
            User-submitted content.
        """
        pending_id = f"pending-{self.pending_message_counter}"
        self.pending_message_counter += 1
        self.pending_messages[pending_id] = content.strip()
        self.render_message(
            {
                "id": pending_id,
                "role": "user",
                "content": content,
                "images": [],
                "pending": True,
            }
        )
        self.scroll_messages_to_bottom()

    def consume_pending_message(self, message: dict[str, Any]) -> bool:
        """Forget an optimistic local message once it appears in the relay.

        Parameters
        ----------
        message : dict[str, Any]
            Relay message.

        Returns
        -------
        bool
            Whether a matching optimistic message was already rendered.
        """
        role = message.get("role")
        if role not in {"user", "user_webui"}:
            return False
        content = str(message.get("content") or "").strip()
        matched_id = None
        for pending_id, pending_content in self.pending_messages.items():
            if pending_content == content:
                matched_id = pending_id
                break
        if matched_id is not None:
            del self.pending_messages[matched_id]
            return True
        return False

    def image_source(self, image: str) -> str:
        """Return a browser-loadable image source.

        Parameters
        ----------
        image : str
            Persisted image data URL or file path.

        Returns
        -------
        str
            Image source URL.
        """
        if image.startswith("data:image"):
            return image
        return f"{self.image_route}?path={quote(image)}"

    def install_clipboard_handler(self) -> None:
        """Install browser JavaScript for clipboard image paste support."""
        from nicegui import ui

        ui.add_body_html(
            f"""
            <script>
            document.addEventListener('paste', async (event) => {{
              const items = event.clipboardData && event.clipboardData.items;
              if (!items) return;
              for (const item of items) {{
                if (!item.type || !item.type.startsWith('image/')) continue;
                const file = item.getAsFile();
                if (!file) continue;
                event.preventDefault();
                const reader = new FileReader();
                reader.onload = async () => {{
                  const response = await fetch('{self.upload_route}', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{image_data: reader.result}})
                  }});
                  if (!response.ok) return;
                  const result = await response.json();
                  const active = document.activeElement;
                  const tag = `<img ${{result.file_path}}> `;
                  if (active && typeof active.value === 'string') {{
                    const start = active.selectionStart || active.value.length;
                    const end = active.selectionEnd || active.value.length;
                    active.value = active.value.slice(0, start) + tag + active.value.slice(end);
                    active.selectionStart = active.selectionEnd = start + tag.length;
                    active.dispatchEvent(new Event('input', {{bubbles: true}}));
                  }}
                }};
                reader.readAsDataURL(file);
                break;
              }}
            }});
            </script>
            """
        )

    def install_keyboard_shortcuts(self) -> None:
        """Install Enter and Shift+Enter keyboard behavior for the input box."""
        from nicegui import ui

        ui.add_body_html(f"<script>{self.keyboard_shortcuts_script()}</script>")

    def install_skill_autocomplete_handler(self) -> None:
        """Install browser JavaScript for ``/skill`` autocomplete."""
        from nicegui import ui

        try:
            self.skill_catalog = self.relay.load_skill_catalog()
        except Exception:
            self.skill_catalog = []
        skill_json = json.dumps(self.skill_catalog)
        ui.add_body_html(f"<script>{self.skill_autocomplete_script(skill_json)}</script>")

    def skill_autocomplete_script(self, skill_json: str) -> str:
        """Return JavaScript used to suggest configured skills."""
        return f"""
            (() => {{
              let skills = {skill_json};
              let skillsLoaded = skills.length > 0;
              const refreshSkills = async () => {{
                if (skillsLoaded) return;
                try {{
                  const response = await fetch('{self.skill_catalog_route}');
                  if (!response.ok) return;
                  const payload = await response.json();
                  skills = Array.isArray(payload.skills) ? payload.skills : [];
                  skillsLoaded = skills.length > 0;
                }} catch (_error) {{
                  return;
                }}
              }};
              const escapeHtml = (value) => String(value)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
              const attachSkillAutocomplete = () => {{
                const input = document.querySelector('.eaa-input textarea');
                const panel = document.querySelector('.eaa-skill-suggestions');
                if (!input || !panel) return false;
                if (input.dataset.eaaSkillAutocompleteBound === '1') return true;
                input.dataset.eaaSkillAutocompleteBound = '1';
                const hide = () => {{
                  panel.classList.add('hidden');
                  panel.innerHTML = '';
                }};
                const render = async () => {{
                  const value = input.value || '';
                  if (!value.startsWith('/skill')) {{
                    hide();
                    return;
                  }}
                  await refreshSkills();
                  const typed = value.slice('/skill'.length).trimStart().toLowerCase();
                  const matches = skills
                    .filter((skill) => !typed || String(skill.name || '').toLowerCase().startsWith(typed))
                    .slice(0, 8);
                  if (!matches.length) {{
                    hide();
                    return;
                  }}
                  panel.innerHTML = matches.map((skill) => `
                    <button type="button" class="eaa-skill-suggestion" data-skill="${{escapeHtml(skill.name || '')}}">
                      <span class="eaa-skill-name">${{escapeHtml(skill.name || '')}}</span>
                      <span class="eaa-skill-description">${{escapeHtml(skill.description || '')}}</span>
                    </button>
                  `).join('');
                  panel.classList.remove('hidden');
                }};
                input.addEventListener('input', render);
                input.addEventListener('blur', () => setTimeout(hide, 150));
                panel.addEventListener('mousedown', (event) => {{
                  const button = event.target.closest('.eaa-skill-suggestion');
                  if (!button) return;
                  event.preventDefault();
                  input.value = `/skill ${{button.dataset.skill}} `;
                  input.focus();
                  input.dispatchEvent(new Event('input', {{bubbles: true}}));
                  hide();
                }});
                return true;
              }};
              if (attachSkillAutocomplete()) return;
              const observer = new MutationObserver(() => {{
                if (attachSkillAutocomplete()) observer.disconnect();
              }});
              observer.observe(document.body, {{childList: true, subtree: true}});
            }})();
            """.strip()

    def keyboard_shortcuts_script(self) -> str:
        """Return the JavaScript used to bind Enter-to-send behavior.

        Returns
        -------
        str
            JavaScript snippet.
        """
        return """
            (() => {
              const attachShortcuts = () => {
                const input = document.querySelector('.eaa-input textarea');
                const send = document.querySelector('.eaa-send button, .eaa-send');
                if (!input || !send) return false;
                if (input.dataset.eaaSendShortcutBound === '1') return true;
                input.dataset.eaaSendShortcutBound = '1';
                input.addEventListener('keydown', (event) => {
                  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
                  event.preventDefault();
                  send.click();
                });
                return true;
              };
              if (attachShortcuts()) return;
              const observer = new MutationObserver(() => {
                if (attachShortcuts()) observer.disconnect();
              });
              observer.observe(document.body, { childList: true, subtree: true });
            })();
            """.strip()

    def styles(self) -> str:
        """Return CSS for the default NiceGUI WebUI.

        Returns
        -------
        str
            CSS wrapped in a style tag.
        """
        return """
        <style>
        .eaa-page {
            height: 100vh;
            width: 100%;
            gap: 0;
            background: #f6f7f9;
            color: #20242a;
        }
        .eaa-header {
            width: 100%;
            min-height: 52px;
            align-items: center;
            justify-content: space-between;
            padding: 0 18px;
            border-bottom: 1px solid #d9dde3;
            background: #ffffff;
        }
        .eaa-title {
            font-size: 18px;
            font-weight: 650;
        }
        .eaa-status {
            color: #586170;
            font-size: 13px;
        }
        .eaa-main {
            width: 100%;
            flex: 1;
            min-height: 0;
            gap: 0;
        }
        .eaa-chat {
            flex: 1;
            min-width: 0;
            height: 100%;
            border-right: 1px solid #d9dde3;
        }
        .eaa-messages {
            width: 100%;
            height: 100%;
            overflow-y: auto;
            gap: 12px;
            padding: 16px;
        }
        .eaa-sidebar {
            width: 280px;
            height: 100%;
            min-width: 220px;
            background: #ffffff;
            padding: 12px;
            gap: 10px;
        }
        .eaa-sidebar-title {
            font-size: 13px;
            font-weight: 650;
            color: #586170;
        }
        .eaa-images {
            width: 100%;
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            gap: 10px;
        }
        .eaa-message {
            max-width: 860px;
            min-width: 0;
            gap: 6px;
            padding: 10px 12px;
            border: 1px solid #dfe3e8;
            border-radius: 8px;
            background: #ffffff;
        }
        .eaa-message-user,
        .eaa-message-user_webui {
            align-self: flex-end;
            background: #eef6ff;
            border-color: #c9ddf2;
        }
        .eaa-message-assistant {
            background: #f7f9fc;
            border-color: #e1e6ee;
        }
        .eaa-message-system {
            background: #fff7e6;
            border-color: #f0d9a8;
        }
        .eaa-message-tool {
            background: #f1f6f0;
            border-color: #d3e2cf;
        }
        .eaa-role {
            font-size: 12px;
            color: #667085;
            font-weight: 650;
            text-transform: uppercase;
        }
        .eaa-markdown {
            font-size: 14px;
            line-height: 1.45;
            min-width: 0;
            max-width: 100%;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .eaa-markdown pre {
            margin: 0.75rem 0;
            padding: 0.875rem 1rem;
            overflow-x: auto;
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }
        .eaa-markdown pre code {
            background: transparent;
            padding: 0;
        }
        .eaa-markdown code:not(pre code) {
            background: #f3f4f6;
            padding: 0.1rem 0.3rem;
            border-radius: 4px;
        }
        .eaa-message-images {
            gap: 8px;
            flex-wrap: wrap;
        }
        .eaa-message-image {
            display: block;
            width: 234px;
            max-height: 234px;
            object-fit: contain;
            border: 1px solid #d9dde3;
            border-radius: 6px;
            background: #ffffff;
        }
        .eaa-sidebar-image {
            display: block;
            width: 100%;
            max-height: 220px;
            object-fit: contain;
            flex-shrink: 0;
            border: 1px solid #d9dde3;
            border-radius: 6px;
            background: #ffffff;
        }
        .eaa-image-html {
            line-height: 0;
        }
        .eaa-clickable-image {
            cursor: zoom-in;
        }
        .eaa-markdown-folded {
            max-height: 16rem;
            overflow: hidden;
        }
        .eaa-message-expand {
            align-self: flex-start;
            color: #2563eb;
        }
        .eaa-tool-call-details {
            width: 100%;
            font-size: 13px;
        }
        .eaa-tool-call-details summary {
            color: #475467;
            cursor: pointer;
            font-weight: 650;
            user-select: none;
        }
        .eaa-tool-call-details pre {
            max-height: 20rem;
            overflow: auto;
            margin: 6px 0 0;
            padding: 8px 10px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            background: #eef2f7;
            border: 1px solid #d7dde8;
            border-radius: 6px;
        }
        .eaa-approval-actions {
            gap: 8px;
            margin-top: 4px;
        }
        .eaa-approval-button {
            min-width: 72px;
        }
        .eaa-approval-arguments,
        .eaa-approval-extracted-field {
            width: 100%;
            gap: 4px;
        }
        .eaa-approval-section-title {
            margin: 2px 0 4px;
            color: #475467;
            font-size: 12px;
            font-weight: 650;
            text-transform: uppercase;
        }
        .eaa-approval-arguments pre,
        .eaa-approval-extracted-field pre {
            max-height: 24rem;
            overflow: auto;
            margin: 0;
            padding: 8px 10px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            background: #f8fafc;
            border: 1px solid #d7dde8;
            border-radius: 6px;
        }
        .eaa-approval-extracted-field pre {
            background: #111827;
            color: #f9fafb;
        }
        .eaa-approval-extracted-field code {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            white-space: pre-wrap;
        }
        .eaa-image-dialog-card {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2px;
            background: #d9dde3;
            border-radius: 6px;
            max-width: 95vw;
            max-height: 95vh;
        }
        .eaa-image-dialog-image {
            display: block;
            max-width: 90vw;
            max-height: 90vh;
            width: auto;
            height: auto;
            object-fit: contain;
        }
        .eaa-browser-image-preview {
            position: fixed;
            inset: 0;
            z-index: 10000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 24px;
            background: rgba(15, 23, 42, 0.72);
        }
        .eaa-browser-image-preview.open {
            display: flex;
        }
        .eaa-browser-image-preview-image {
            display: block;
            max-width: 92vw;
            max-height: 92vh;
            object-fit: contain;
            background: #ffffff;
            border-radius: 6px;
        }
        .eaa-browser-image-preview-close {
            position: absolute;
            top: 16px;
            right: 18px;
            width: 36px;
            height: 36px;
            border: 0;
            border-radius: 18px;
            background: #ffffff;
            color: #20242a;
            font-size: 24px;
            line-height: 36px;
            cursor: pointer;
        }
        .eaa-input-panel {
            width: 100%;
            gap: 8px;
            margin: 12px 16px 16px;
            padding: 12px 14px 14px;
            border: 1px solid #d9dde3;
            border-radius: 10px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .eaa-processing {
            color: #8a4b00;
            font-size: 13px;
        }
        .hidden {
            display: none;
        }
        .eaa-input-row {
            width: 100%;
            align-items: stretch;
            flex-wrap: nowrap;
            gap: 8px;
            position: relative;
        }
        .eaa-input {
            flex: 1;
            min-width: 0;
        }
        .eaa-skill-suggestions {
            position: absolute;
            left: 0;
            right: 104px;
            bottom: 100%;
            margin-bottom: 6px;
            z-index: 20;
            max-height: 260px;
            overflow-y: auto;
            background: #ffffff;
            border: 1px solid #d8dde6;
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(21, 30, 45, 0.14);
            padding: 6px;
        }
        .eaa-skill-suggestion {
            display: block;
            width: 100%;
            border: 0;
            background: transparent;
            text-align: left;
            padding: 8px 10px;
            border-radius: 6px;
            cursor: pointer;
        }
        .eaa-skill-suggestion:hover {
            background: #eef3fb;
        }
        .eaa-skill-name {
            display: block;
            font-weight: 650;
            color: #20242a;
        }
        .eaa-skill-description {
            display: block;
            margin-top: 2px;
            color: #5f6b7a;
            font-size: 12px;
            line-height: 1.35;
        }
        .eaa-actions {
            align-items: stretch;
            gap: 8px;
        }
        .eaa-send {
            min-width: 96px;
            height: 100%;
        }
        @media (max-width: 800px) {
            .eaa-main {
                flex-direction: column;
            }
            .eaa-sidebar {
                width: 100%;
                height: 190px;
                border-top: 1px solid #d9dde3;
            }
            .eaa-chat {
                border-right: none;
            }
        }
        </style>
        """


def run_nicegui_webui(
    session_db_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8008,
    title: str = "EAA WebUI",
    upload_dir: str = ".tmp",
    poll_interval: float = 1.0,
) -> None:
    """Run the default reusable NiceGUI WebUI.

    Parameters
    ----------
    session_db_path : str
        SQLite session database shared with the task manager.
    host : str, default="127.0.0.1"
        Bind host.
    port : int, default=8008
        Bind port.
    title : str, default="EAA WebUI"
        Browser and header title.
    upload_dir : str, default=".tmp"
        Directory used to store pasted images.
    poll_interval : float, default=1.0
        Message and status polling interval in seconds.
    """
    set_message_db_path(session_db_path)
    webui = NiceGUIWebUIBase(
        session_db_path,
        title=title,
        upload_dir=upload_dir,
        poll_interval=poll_interval,
    )
    webui.run(host=host, port=port)


def launch_nicegui_webui_subprocess(
    session_db_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8008,
    title: str = "EAA WebUI",
    upload_dir: str = ".tmp",
    poll_interval: float = 1.0,
    python_executable: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    stdout: Any | None = None,
    stderr: Any | None = None,
) -> subprocess.Popen:
    """Launch the default NiceGUI WebUI in a non-blocking subprocess.

    Parameters
    ----------
    session_db_path : str
        SQLite session database shared with the task manager.
    host : str, default="127.0.0.1"
        Bind host.
    port : int, default=8008
        Bind port.
    title : str, default="EAA WebUI"
        Browser and header title.
    upload_dir : str, default=".tmp"
        Directory used to store pasted images.
    poll_interval : float, default=1.0
        Message and status polling interval in seconds.
    python_executable : str | None, optional
        Python executable used to launch the subprocess. Defaults to the
        current interpreter.
    cwd : str | None, optional
        Working directory for the subprocess.
    env : dict[str, str] | None, optional
        Environment override for the subprocess.
    stdout : Any | None, optional
        Subprocess stdout target. Defaults to inheriting the parent stdout.
    stderr : Any | None, optional
        Subprocess stderr target. Defaults to inheriting the parent stderr.

    Returns
    -------
    subprocess.Popen
        Running subprocess handle. Call ``terminate`` or ``kill`` to stop it.
    """
    executable = python_executable or sys.executable
    command = [
        executable,
        "-m",
        "eaa_core.gui.nicegui",
        session_db_path,
        "--host",
        host,
        "--port",
        str(port),
        "--title",
        title,
        "--upload-dir",
        upload_dir,
        "--poll-interval",
        str(poll_interval),
    ]
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the NiceGUI WebUI module.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(prog="python -m eaa_core.gui.nicegui")
    parser.add_argument("session_db_path", help="Shared SQLite session database.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", default=8008, type=int, help="Bind port.")
    parser.add_argument("--title", default="EAA WebUI", help="Browser and header title.")
    parser.add_argument(
        "--upload-dir",
        default=".tmp",
        help="Directory used for pasted image uploads.",
    )
    parser.add_argument(
        "--poll-interval",
        default=1.0,
        type=float,
        help="Message and status polling interval in seconds.",
    )
    return parser


def main() -> None:
    """Run the NiceGUI WebUI from command-line arguments."""
    args = build_parser().parse_args()
    run_nicegui_webui(
        args.session_db_path,
        host=args.host,
        port=args.port,
        title=args.title,
        upload_dir=args.upload_dir,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
