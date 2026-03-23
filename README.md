# Experiment Automation Agents (EAA)

EAA is a Python toolkit for building experiment-facing agents around a shared set
of task-manager, tool, memory, and WebUI primitives. The current codebase centers
on a LangGraph-backed `BaseTaskManager`, stateful `BaseTool` integrations,
SQLite-backed checkpointing/WebUI relay, and a small set of concrete task
managers for imaging and tuning workflows.

## Current Status

- Core agent runtime: `src/eaa/core/task_manager/base.py`
- Reusable built-in graphs: chat and feedback loop
- Concrete workflows: ROI search, feature tracking, parameter tuning, Bayesian
  optimization, and analytical task managers under `src/eaa/task_manager/`
- Tool system: `BaseTool`, serialized execution, optional approval gates, MCP
  server/client helpers
- Long-term memory: optional chat-memory layer configured through
  `MemoryManagerConfig`
- WebUI: FastAPI + static frontend, connected to the agent through a shared
  SQLite database

## Installation

### Option 1: `uv` (recommended)

```bash
uv sync
source .venv/bin/activate
which python
```

`which python` should resolve to `.venv/bin/python`.

Optional extras are defined in `pyproject.toml`. Common ones are:

- `docs` for Sphinx documentation
- `asksage` for AskSage-specific LLM access
- `aps_mic` for APS microscopy integrations
- `postgresql_vector_store` for external PostgreSQL/pgvector client dependencies

Install an extra set with:

```bash
uv sync --extra docs
```

### Option 2: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

With extras:

```bash
pip install -e .[docs]
```

## Quickstart

The smallest useful setup is a task manager, an LLM config, and one or more
tools. This example starts a free-form chat with a simulated image-acquisition
tool:

```python
from skimage import data

from eaa.api.llm_config import OpenAIConfig
from eaa.core.task_manager.base import BaseTaskManager
from eaa.tool.imaging.acquisition import SimulatedAcquireImage

llm_config = OpenAIConfig(
    model="your-model-name",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
)

acquisition_tool = SimulatedAcquireImage(
    whole_image=data.camera(),
    add_axis_ticks=True,
    show_image_in_real_time=False,
)

task_manager = BaseTaskManager(
    llm_config=llm_config,
    tools=[acquisition_tool],
    session_db_path="session.sqlite",
    use_webui=False,
)

task_manager.run_conversation()
```

For a workflow-oriented manager, see `examples/roi_search.py`.

## Architecture

- `LLMConfig` objects describe how the chat model is constructed. The shipped
  config classes are `OpenAIConfig`, `AskSageConfig`, and `ArgoConfig`.
- `BaseTaskManager` owns model invocation, tool registration, memory,
  persistence, and graph execution.
- `SerialToolExecutor` runs tool calls one at a time. This is intentional:
  many experiment tools are stateful, should not be driven concurrently, or not thread-safe.
- `MemoryManager` adds optional long-term memory retrieval/saving on chat turns.
- The WebUI is a separate FastAPI process. It communicates with the agent
  through the same SQLite database used for WebUI relay state and checkpointing.

## Built-In Graphs and Workflows

The reusable graphs shipped in the base runtime are:

- `chat_graph` for interactive conversation
- `feedback_loop_graph` for iterative tool-driven workflows

`build_task_graph()` is available as a subclass hook for custom LangGraph
workflows, but the task managers currently in this repository mostly either:

- reuse `run_feedback_loop()` / `run_conversation()`, or
- implement an analytical workflow directly in Python while still updating
  message history and WebUI state through task-manager helpers

## WebUI and Checkpointing

Set `use_webui=True` on the task manager and give it a `session_db_path`. Then
launch the standalone WebUI process against the same SQLite file:

```python
from eaa.gui.chat import run_webui, set_message_db_path

set_message_db_path("session.sqlite")
run_webui(host="127.0.0.1", port=8008)
```

Checkpointing and the WebUI relay share the same SQLite database by default.
Each resume entrypoint also accepts ``checkpoint_db_path`` if you need to load
checkpoints from a different SQLite file. The base task manager exposes:

- `run_conversation_from_checkpoint()`
- `run_feedback_loop_from_checkpoint()`
- `run_from_checkpoint()` for subclasses that implement `task_graph`

## Long-Term Memory

Long-term memory is configured with `MemoryManagerConfig`. In the current
codebase, the built-in memory manager creates a Chroma-backed vector store and
can:

- retrieve relevant memories on chat turns
- inject retrieved memories back into the model context
- save new memories when a user message contains trigger phrases such as
  `"remember this"` or `"keep in mind"`

The `postgresql_vector_store` extra is present in `pyproject.toml`, but the
built-in `MemoryManager` path in this repository currently wires up Chroma.

## MCP Integration

EAA supports both directions of MCP integration.

Expose EAA tools as an MCP server:

```python
from eaa.core.mcp.server import run_mcp_server_from_tools
from eaa.tool.example_calculator import CalculatorTool

run_mcp_server_from_tools(
    tools=CalculatorTool(),
    server_name="Calculator MCP Server",
)
```

Use an external MCP server as a normal EAA tool:

```python
from eaa.tool.mcp import MCPTool

mcp_tool = MCPTool(
    {
        "mcpServers": {
            "remote_tools": {
                "command": "python",
                "args": ["./path/to/server.py"],
            }
        }
    }
)
```

Use an MCP server over HTTP from another machine:

```python
from eaa.tool.mcp import MCPTool

mcp_tool = MCPTool(
    {
        "mcpServers": {
            "calculator": {
                "url": "http://SERVER_IP:8050/mcp",
                "transport": "http",
            }
        }
    }
)
```

For this remote HTTP setup, the server side must be started with
``run_mcp_server_from_tools(..., transport="http", host="0.0.0.0", port=8050, path="/mcp")``.
The client config must keep the server definition under ``mcpServers``; passing
only ``{"url": ..., "transport": "http"}`` is not enough.

## Skills

Skills are reusable, markdown-first task packages that EAA can discover and
load at runtime. In the current implementation, each skill is a directory with
at least a `SKILL.md` file. Additional markdown files and referenced images can
live alongside it.

Bundled skills live under `src/eaa/skills/`. A typical layout looks like:

```text
my-skill/
  SKILL.md
  references/
    api_reference.md
    figure.png
```

To use skills, point a task manager at one or more skill directories:

```python
task_manager = BaseTaskManager(
    llm_config=llm_config,
    tools=[acquisition_tool],
    skill_dirs=["./src/eaa/skills", "~/.eaa_skills"],
)
```

At build time, EAA scans those directories for `SKILL.md`, turns each skill
into a documentation-fetching tool, and makes the discovered skills available
inside the agent. In an interactive chat session you can:

- run `/skill` to list the loaded skills
- run `/subtask <task description>` to let the agent choose a skill, fetch its
  docs, and launch a skill-driven subtask flow

If you want to copy the bundled skills out of the package tree, use:

```bash
python -m eaa.cli install-skills --destination ~/.eaa_skills
```

## Documentation

Sphinx documentation lives under `docs/` and is configured for Read the Docs.
Build it locally with:

```bash
uv sync --extra docs
source .venv/bin/activate
cd docs
make html
```

The generated site will be in `docs/_build/html/`.
