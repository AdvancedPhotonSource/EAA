# WebUI

## Purpose

The EAA WebUI is a lightweight standalone browser interface for watching agent
progress and sending user input. The task-manager process owns the agent runtime
API; the WebUI process serves the browser app and proxies browser API calls to
that runtime over HTTP.

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
task_manager.run_conversation()
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
