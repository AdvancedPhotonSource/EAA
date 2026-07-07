---
name: create-eaa-compatible-mcp-servers
description: Create or review external MCP server repositories for EAA-compatible instrument-control tools, including image artifact payloads, get_attribute_payload, and safe execution patterns such as Bluesky Queue Server gateways, existing execution-server gateways, or MCP-gateway/ZMQ-worker separation.
---

# Create EAA-Compatible MCP Servers

Use this skill when creating, reviewing, or refactoring an external MCP server
that exposes facility, beamline, endstation, or instrument-control tools for
use by EAA or another MCP client.

## Background

EAA (Experiment Automation Agents) is a Python toolkit for building
experiment-facing agents around task-manager, tool, memory, skill, WebUI, and
MCP integration primitives. It is published at
<https://github.com/AdvancedPhotonSource/EAA>.

Treat EAA as an MCP client that can call external MCP tools. Implement
compatibility through the small payload contracts below, not by importing EAA
or modifying EAA.

## Scope

- Implement or review the external MCP server only. Do not modify EAA.
- Keep facility-specific control logic outside EAA.
- Do not make the MCP server depend on EAA packages.
- Build a normal MCP server that can work with generic MCP clients. EAA
  compatibility should come from clear tool schemas and payload contracts.
- Use the safest execution pattern that matches the facility control stack.

## First Step: Ask for the Execution Pattern

Before implementing anything, ask the user which execution pattern they want:

1. **Bluesky Queue Server gateway**: the MCP server exposes tools and sends
   requests to an existing Bluesky Queue Server.
2. **Existing execution-server gateway**: the MCP server exposes tools and
   sends requests to another already-separated server that owns instrument
   command execution.
3. **MCP gateway plus local worker backend**: the MCP server gateway talks to a
   separate local worker process, typically through ZMQ, because instrument
   control would otherwise run inside the MCP server process.

Also ask for the endpoint, transport, authentication requirements, artifact
directory, simulation or hardware-safety mode, expected tool list, and any
server-side attributes or buffers that `get_attribute_payload` should expose.

Apply these pattern-specific gates before writing code:

- If the user selects **Bluesky Queue Server gateway**, read the current
  Bluesky Queue Server documentation at
  <https://blueskyproject.io/bluesky-queueserver/>. Then generate an
  implementation plan and discuss it with the user until they confirm the
  plan. Do not implement before the plan is finalized.
- If the user selects **Existing execution-server gateway**, ask for the
  connection and API details of that server before planning or implementing.
  At minimum, collect the endpoint, protocol or client library, authentication,
  command/request schema, response schema, status or progress API, result and
  artifact handling, timeout behavior, error semantics, concurrency guarantees,
  and any safety constraints.

If the user has already specified the pattern and required connection details,
state that assumption briefly and continue through the relevant
pattern-specific gate. If not, wait for the answer before writing code.

## Execution Patterns

### Bluesky Queue Server Gateway

Use this when the beamline already runs a Bluesky Queue Server and the queue
server owns RunEngine execution and device state.

Before implementing this pattern, read the current Bluesky Queue Server
documentation at <https://blueskyproject.io/bluesky-queueserver/> and produce
a plan for user review. The plan should cover the selected communication route
such as the Queue Server ZMQ API or Bluesky HTTP Server, environment handling,
queue item submission, queue start/stop behavior, status polling, result
retrieval, artifact handling, authentication or permissions, locking, timeouts,
and safety constraints. Implement only after the user confirms the plan.

The MCP server should:

- expose FastMCP tools for the agent or workflow;
- query, enqueue, monitor, or otherwise interact with the Queue Server;
- avoid importing or running a local RunEngine in the MCP process;
- convert queue results into JSON-serializable MCP tool returns;
- handle job completion, timeouts, and queue-side failures with clear messages.

### Existing Execution-Server Gateway

Use this when the facility already has a separate server that executes
instrument commands, serializes hardware access, and owns control-library
state.

Before implementing this pattern, ask the user how to connect to the execution
server and how its API behaves. Collect endpoint details, protocol or client
library, authentication, request and response examples, command names,
status/progress/result flow, artifact path conventions, timeout and error
semantics, concurrency guarantees, and safety constraints. Generate a plan from
those details and confirm it with the user if important behavior is still
ambiguous.

The MCP server should:

- expose FastMCP tools as a gateway;
- call the existing execution server through its supported API;
- avoid executing instrument-control commands locally;
- preserve the execution server's safety, serialization, and timeout behavior;
- translate execution-server responses into clean MCP payloads.

### MCP Gateway Plus Local Worker Backend

Use this when the MCP server would otherwise need to call instrument-control
libraries directly, such as a local Bluesky RunEngine or EPICS control logic.

Separate the system into two OS processes:

```text
MCP client
  -> MCP over HTTP
  -> FastMCP gateway process
  -> ZMQ request/reply, or equivalent local RPC
  -> synchronous instrument worker process
```

This separation ensures the event loops in the MCP gateway and the instrument
control library (such as Bluesky) do not clash with each other.

The gateway should own MCP and HTTP request handling. The worker should own
the instrument-control library, device state, blocking control logic, and any
runtime required by the control stack.

For this pattern, include or adapt these modules:

- `protocol.py`: JSON-serializable command and response envelopes.
- `zmq_client.py`: request/reply client used by the MCP gateway.
- `worker.py`: blocking instrument worker process.
- `mcp_server.py`: FastMCP gateway that forwards tool calls.
- `launcher.py`: supervisor that starts the worker, waits for health, then
  starts the MCP gateway.

Recommended console scripts:

```text
<facility-suite>
<facility-suite>-worker
<facility-suite>-mcp
```

The combined launcher should start the worker first, poll a health command,
then start the MCP gateway. It should terminate both children on shutdown or
when one child exits unexpectedly.

Recommended option names:

```bash
<facility-suite> \
  --config configs/example.yaml \
  --worker-endpoint tcp://127.0.0.1:5555 \
  --worker-startup-timeout-s 10 \
  --request-timeout-ms 30000 \
  --mcp-host 0.0.0.0 \
  --mcp-port 8050 \
  --mcp-path /mcp
```

```bash
<facility-suite>-worker \
  --config configs/example.yaml \
  --bind tcp://127.0.0.1:5555
```

```bash
<facility-suite>-mcp \
  --config configs/example.yaml \
  --worker tcp://127.0.0.1:5555 \
  --timeout-ms 30000 \
  --host 0.0.0.0 \
  --port 8050 \
  --path /mcp
```

## Repository Shape

Create a Python package managed with `uv`, with `pyproject.toml`, `uv.lock`, a
`src/<package_name>/` layout, and console scripts under `[project.scripts]`.

Runtime dependencies should match the chosen pattern:

- MCP/FastMCP for all patterns.
- Queue Server or facility API client libraries for gateway-only patterns.
- ZMQ only when using the local worker pattern.
- Facility control-library dependencies only in the process that actually
  executes instrument commands.

Include a `configs/` directory with at least one example YAML configuration.
Use clear field names and include units where useful, such as
`request_timeout_ms` or `worker_startup_timeout_s`. Explicit CLI flags should
override YAML values.

## Tool Design

MCP tools should be clear, JSON-oriented, and useful to both LLM agents and
logic-driven clients.

Each tool function should:

- use descriptive facility-specific names; EAA users can map names on the
  client side when a logic-driven workflow expects a different local name;
- annotate every argument, preferably with `Annotated[type, "description"]`;
- annotate the return type;
- include a docstring that states behavior, units, coordinate order, safety
  implications, blocking behavior, and returned payload fields when applicable;
- return only JSON-serializable values: `str`, `int`, `float`, `bool`, `list`,
  `dict`, or `None`.

## Artifact and Payload Contracts

### Images

Any tool that yields an image for the agent to inspect should:

- render the display image as a PNG file;
- save it to a path readable by the agent or EAA process;
- return the absolute path in the `img_path` field of the JSON payload.

Example:

```json
{
  "img_path": "/path/to/image.png"
}
```

Do not return images as base64 in ordinary tool payloads. File paths are more
portable across MCP clients and avoid sending large opaque strings to the LLM.

### Numerical Data and Buffers

When a tool needs to expose large arrays for analysis, prefer one of these:

- save the array to a data file such as `.npy` or TIFF and return the path; or
- expose the value through `get_attribute_payload` when a logic-driven client
  needs attribute-style access.

Avoid embedding large arrays directly in model-visible tool responses.

### `get_attribute_payload`

Every EAA-compatible MCP server should expose this normal MCP tool:

```python
get_attribute_payload(name: str) -> object
```

Use it to fetch server-side attributes, buffers, or cached results by name.
The server controls which names are supported.

Return JSON-serializable scalar, list, dictionary, or `None` values directly.
For NumPy arrays or other dense numerical arrays, return this portable array
payload:

```json
{
  "encoded_data": {
    "type": "array",
    "dtype": "float32",
    "shape": [1024, 1024],
    "data": "<base64-encoded-data>"
  }
}
```

When implementing this in Python, use standard `base64` plus NumPy byte buffers;
do not import EAA to encode or decode the payload.

## Safety and Concurrency

Instrument operations should be serialized unless the facility control stack
explicitly supports concurrent calls. Put serialization, hardware interlocks,
argument bounds, timeout policy, and recovery behavior in the execution owner:
the Queue Server, existing execution server, or local worker.

The MCP gateway should treat execution timeouts and execution-side exceptions
as ordinary tool failures with clear error messages. It should not leave
requests running indefinitely in the MCP event loop.

## README Requirements

Write a `README.md` for every server repository. It should not assume EAA.

Include:

- a short description of the server and controlled instrument or facility;
- setup instructions using `uv sync`;
- the selected execution pattern and why it is used;
- launch commands for the MCP gateway and any worker or supervisor process;
- how to copy and edit the example YAML under `configs/`;
- the precedence rule for config files and CLI overrides;
- the MCP endpoint URL produced by the examples;
- artifact directory requirements and path visibility expectations;
- important safety flags, timeout settings, and instrument-specific options;
- a generic MCP client JSON configuration, for example:

```json
{
  "mcpServers": {
    "facility_tools": {
      "url": "http://127.0.0.1:8050/mcp",
      "transport": "http"
    }
  }
}
```

Briefly document the EAA compatibility contracts: image-yielding tools return
`img_path`, and the server exposes `get_attribute_payload`.

## Review Checklist

- The user selected one of the supported execution patterns before
  implementation.
- For the Bluesky Queue Server pattern, the agent read the current Queue
  Server documentation, proposed an implementation plan, discussed it with the
  user, and waited for confirmation before implementing.
- For the existing execution-server pattern, the agent collected concrete
  connection and API details before planning or implementing.
- The repository implements an MCP server, not changes to EAA.
- The server has no EAA package dependency.
- Gateway-only patterns send commands to the Queue Server or existing
  execution server instead of executing instrument commands locally.
- The local worker pattern separates MCP gateway and instrument execution into
  different OS processes.
- Blocking control-library calls happen only in the execution owner.
- Tool and launcher settings can be loaded from YAML, with CLI overrides.
- MCP tool returns are JSON-serializable.
- Image-yielding tools save PNG files and return `img_path`.
- Large arrays are returned as files or through `get_attribute_payload`, not as
  ordinary model-visible JSON.
- `get_attribute_payload` exists and handles arrays with an `encoded_data`
  object containing `type`, `dtype`, `shape`, and `data`.
- Artifact paths returned to clients are readable from the client process that
  needs them.
- Tool arguments, return types, docstrings, units, coordinate order, and safety
  implications are clear.
- Timeouts and execution-side exceptions produce clear tool failures.
- The README documents setup, launch, configuration, endpoint, artifacts,
  safety options, and the two EAA compatibility contracts.
