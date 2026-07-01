# Quick Start

This page shows the common setup path for a chat-style EAA agent:

1. configure an LLM
2. choose skills
3. register domain and built-in tools
4. optionally enable long-term memory
5. optionally launch the WebUI
6. run `BaseTaskManager.run_conversation()`

Run the script from a repository checkout after `uv sync --all-extras`. Set
`OPENAI_API_KEY` in your shell first. The memory example uses the built-in
Chroma-backed store, so it requires the `memory_chroma` extra.

```python
import os
from pathlib import Path

from skimage import data

from eaa_core.api.llm_config import OpenAIConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.gui.html import launch_html_webui_subprocess
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.tool.workspace import FileSystemTool
from eaa_imaging.tool.imaging.acquisition import SimulatedAcquireImage


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_URL = "http://127.0.0.1:8010"


def main() -> None:
    llm_config = OpenAIConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    memory_config = MemoryManagerConfig(
        enabled=True,
        persist_directory=str(PROJECT_ROOT / ".eaa_memory"),
        namespace="quick-start",
    )
    skill_dirs = [
        str(PROJECT_ROOT / "packages/eaa-core/src/eaa_core/skills"),
        str(PROJECT_ROOT / "packages/eaa-imaging/src/eaa_imaging/skills"),
    ]
    acquisition_tool = SimulatedAcquireImage(
        whole_image=data.camera(),
        add_axis_ticks=True,
    )
    workspace_tool = FileSystemTool(
        workspace_path=str(PROJECT_ROOT),
        read_whitelist_paths=skill_dirs,
    )

    task_manager = BaseTaskManager(
        llm_config=llm_config,
        memory_config=memory_config,
        tools=[acquisition_tool, workspace_tool],
        skill_dirs=skill_dirs,
        checkpoint_db_path=str(PROJECT_ROOT / "checkpoint.sqlite"),
        transcript_db_path=str(PROJECT_ROOT / "transcript.sqlite"),
        use_webui=True,
        webui_runtime_host="127.0.0.1",
        webui_runtime_port=8010,
    )
    task_manager.tool_manager.set_coding_tool_request_approval(True)

    task_manager.start_webui_runtime()
    webui_process = launch_html_webui_subprocess(
        RUNTIME_URL,
        host="127.0.0.1",
        port=8008,
        title="EAA Quick Start",
    )
    print("WebUI: http://127.0.0.1:8008")
    try:
        task_manager.run_conversation()
    finally:
        webui_process.terminate()
        task_manager.stop_webui_runtime()


if __name__ == "__main__":
    main()
```

For terminal-only chat, set `use_webui=False` and remove
`start_webui_runtime()`, `launch_html_webui_subprocess()`, and
`stop_webui_runtime()`.

## What the driver configures

- `OpenAIConfig` is passed to `BaseTaskManager.build_model()`.
- `MemoryManagerConfig(enabled=True, ...)` enables retrieval and triggered
  saving in the chat graph.
- `skill_dirs` controls the skill catalog. The same paths are passed to
  `FileSystemTool(read_whitelist_paths=...)` so the agent can inspect skill
  files without an approval prompt.
- `SimulatedAcquireImage` is a domain tool registered through the task-manager
  `tools` argument.
- `FileSystemTool` replaces the default filesystem tool handle with an
  explicitly configured workspace root.
- `use_webui=True` creates the task-manager-owned WebUI runtime controller.
  `start_webui_runtime()` starts the agent-side API, and
  `launch_html_webui_subprocess()` starts the browser-facing server.

## Slash commands

The base task-manager input parser supports these slash commands:

- `/exit`: request exit from the active chat or task graph.
- `/return`: return from chat to the caller or upper-level task when the chat
  graph is running.
- `/chat`: switch from a task graph back into chat mode when a task graph
  boundary is waiting for user input.
- `/skill`: list discovered skills.
- `/skill <name>`: inject the selected skill's `SKILL.md` into context.
- `/skill <name> <message>`: inject the selected skill and send `<message>` as
  the next user instruction in the same turn.
- `/setcodingtoolapproval true|false`: toggle approval for the Python and Bash
  coding tools. The parser also accepts `yes|no`, `on|off`, and `1|0`.
- `/setcodingtoolsandboxtype none|bubblewrap|container [visible_dir ...]`:
  configure coding-tool sandboxing. Extra paths are used as visible
  directories for bubblewrap.

Unknown slash-prefixed input is treated as normal user text and sent to the
model.
