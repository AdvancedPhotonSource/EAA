# WebUI

## Purpose

The EAA WebUI is a lightweight standalone browser interface for watching agent
progress and sending user input. The task-manager process owns the agent runtime
API; the WebUI process serves the browser app and proxies browser API calls to
that runtime over HTTP.

## Interface

The Chat tab is the main operator view for interactive agent runs.

![EAA WebUI chat view](assets/webui-chat.png)

The visible components are:

- Sidebar navigation: switches between `Chat`, `Tools`, and `Settings`.
- Agent process status: shows whether the browser is connected to the
  task-manager runtime.
- Conversation tabs: the primary run appears as `Main Agent`; subagent runs can
  appear as additional conversations.
- Message timeline: shows user, assistant, and tool messages with timestamps.
  Assistant messages can include collapsible tool-call cards showing the tool
  name and arguments that were requested.
- Composer: sends user input to the active conversation and supports image
  attachment/upload.
- Image panel: collects images returned by tools so operators can inspect
  outputs without searching the text transcript.
- Live Log panel: shows runtime log entries emitted by tools or MCP-backed
  integrations.
- Top controls: include Help and runtime control actions such as cooperative
  interruption when available.

The Tools tab lists the registered model-visible tools and their schemas. For
MCP-backed tools, it also exposes reconnect controls so users can reconnect an
MCP server without restarting the task-manager process.

## Launch pattern

Configure the task manager with separate checkpoint and transcript databases,
then start its runtime server explicitly:

```python
task_manager = BaseTaskManager(
    ...,
    checkpoint_db_path="checkpoint.sqlite",
    transcript_db_path="transcript.sqlite",
    use_webui=True,
)
task_manager.start_webui_runtime()
try:
    task_manager.run_conversation()
finally:
    task_manager.stop_webui_runtime()
```

In a separate process, start the browser-facing WebUI and point it at the local
runtime URL. The browser only needs to reach the WebUI server; runtime API calls
are proxied by the WebUI process.

```python
from eaa_core.gui.html import run_html_webui

run_html_webui(
    "http://127.0.0.1:8010",
    host="127.0.0.1",
    port=8008,
)
```

To launch the WebUI without blocking the current Python process, use the
subprocess launcher:

```python
from eaa_core.gui.html import launch_html_webui_subprocess

process = launch_html_webui_subprocess(
    "http://127.0.0.1:8010",
    host="127.0.0.1",
    port=8008,
)

# Later, when the UI is no longer needed:
process.terminate()
```

## Runtime API

The agent-side runtime exposes:

- `GET /api/state` for initial state and reconnect recovery
- `GET /api/events` for Server-Sent Events
- `POST /api/input` for ordered user input
- `POST /api/interrupt` for cooperative interruption
- `POST /api/approval` for tool approval decisions
- `GET /api/skill-catalog` for available skill metadata
- `GET /api/tool-schemas` for model-visible tool schemas
- `POST /api/mcp/reconnect` for reconnecting MCP-backed tools
- `GET /api/image` and `POST /api/upload-image` for image display and upload

The browser calls these routes on the WebUI origin. The WebUI server forwards
them to the configured runtime URL, so the runtime can remain bound to
`127.0.0.1` while the WebUI binds to a remote-accessible interface.

## Persistence

`checkpoint_db_path` stores LangGraph checkpoints only. `transcript_db_path`
stores the agent-owned durable transcript log. The WebUI does not read this
database for message transport or initial state. Browser-visible messages,
commands, approvals, interrupts, and runtime status are held by the
task-manager-owned runtime controller, not by SQLite relay tables.

## Extending the WebUI

Task-specific packages can subclass `HTMLWebUIBase` and override `page_html()`,
`styles()`, `script()`, or `build_app()` while keeping the same runtime API
contract:

```python
from eaa_core.gui.html import HTMLWebUIBase


class FocusingWebUI(HTMLWebUIBase):
    def styles(self) -> str:
        return super().styles() + "<style>.eaa-title::after { content: ' focusing'; }</style>"


FocusingWebUI("http://127.0.0.1:8010").run(host="127.0.0.1", port=8008)
```
