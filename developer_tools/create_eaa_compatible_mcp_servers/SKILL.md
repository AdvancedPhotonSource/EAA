---
name: create-eaa-compatible-mcp-servers
description: Create or review facility-specific MCP server repositories that are compatible with EAA and safe for instrument-control libraries.
---

# Create EAA-Compatible MCP Servers

Use this skill when creating, reviewing, or refactoring an external MCP server
for EAA.

## Goals

- Keep facility and endstation-specific control logic outside EAA.
- Do not add EAA as a dependency of the MCP server repository.
- Use the process pattern `MCP frontend -> ZMQ -> instrument worker`.
- Keep instrument worker functions synchronous and blocking unless the facility
  control library requires an internal runtime.
- Make chat-facing MCP tools clear and self-describing.
- Document MCP tool methods in a way compatible with FastMCP tool schema
  generation.
- Implement EAA adapter contracts only when logic-driven task managers need
  direct method calls.

## Repository Shape

Create a normal Python package managed with `uv`. The repository should include
`pyproject.toml`, `uv.lock`, a `src/<package_name>/` layout, and console scripts
declared under `[project.scripts]`.

The runtime dependencies should include MCP, ZMQ, and the facility control
library dependencies. Do not depend on EAA packages. The server should be usable
by any MCP client; EAA compatibility should come from tool schemas and payloads,
not from importing EAA.

Tool and launcher settings must be configurable through a YAML file. Include a
`configs/` directory in each server repository with at least one example YAML
configuration that users can copy and edit. The YAML should cover shared
launcher settings, worker connection settings, MCP HTTP settings, artifact
paths, simulation or hardware safety flags, and server-specific tool defaults.
Use clear field names and keep units in the field name where practical, such as
`request_timeout_ms` or `worker_startup_timeout_s`.

Use these modules or equivalents:

- `protocol.py`: JSON-serializable command and response envelopes.
- `zmq_client.py`: request/reply client used only by the MCP frontend.
- `worker.py`: blocking instrument worker process that owns control-library
  state.
- `mcp_server.py`: FastMCP frontend that forwards tools to the worker.
- `launcher.py`: supervisor that starts the worker, waits for health, then
  starts the MCP frontend.

Expose three console scripts:

- `<facility-suite>`
- `<facility-suite>-worker`
- `<facility-suite>-mcp`

## Launch Interface

Keep CLI options consistent across facility MCP servers:

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

All console scripts should accept `--config PATH` for loading the YAML file.
Explicit CLI flags should override YAML values so operators can make temporary
runtime changes without editing the file.

The combined launcher command is required. It should:

- resolve the worker and MCP frontend console scripts from `PATH`;
- start the worker subprocess first;
- poll a `health` command through the ZMQ client until the worker is ready;
- start the MCP frontend subprocess after worker readiness;
- monitor both subprocesses and return a nonzero exit code if either child
  fails;
- handle `SIGINT` and `SIGTERM` by terminating both child processes, then
  killing them after a short timeout if needed.

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

## Worker Pattern

The worker should:

- own the instrument-control library and instrument state;
- serve a blocking ZMQ `REP` loop;
- expose a `health` command returning `{"status": "ok"}`;
- dispatch `method` and `params` to synchronous handler functions;
- return `{"status": "ok", "result": ...}` or
  `{"status": "error", "error": ...}`;
- write large data such as images or arrays to files and return paths.

The MCP frontend should:

- own FastMCP and HTTP transport;
- not import or initialize the control library unless it is harmless;
- forward each MCP tool call to the worker through the ZMQ client;
- use `asyncio.to_thread(...)` if an async MCP handler calls a blocking ZMQ
  client.

Getter tools must return serialized Python objects such as `str`, `int`,
`float`, `bool`, `list`, `dict`, or `None`. Convert control-library objects,
NumPy scalars, and arrays before returning them; write large arrays to artifact
files and return paths instead of embedding array data in the tool result.

MCP tool methods must be documented so FastMCP can generate useful tool
schemas. Each tool function should:

- annotate every argument as `Annotated[type, "description"]`;
- annotate the return type;
- include a docstring that describes the behavior, units, coordinate order,
  safety implications, and returned payload fields when applicable.

## README Requirements

Write a `README.md` for every server repository. It should be useful to users
of any MCP client and should not assume EAA.

Include:

- a short description of the server and the instrument or facility it controls;
- setup instructions using `uv sync`;
- the one-command launcher example that starts both the worker and MCP frontend;
- separate worker and MCP frontend launch commands for debugging;
- how to copy and edit the example YAML configuration under `configs/`;
- the precedence rule for config files and CLI overrides;
- the MCP endpoint URL produced by the examples;
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

Also document important server-specific options such as host, port, ZMQ
endpoint, timeout, hardware safety flags, simulation mode, output directory,
artifact path requirements, and tool defaults configured in YAML.

## EAA Contracts

Do not force EAA-specific names on general chat tools. The following contracts
are only needed for EAA logic-driven task managers that use adapter proxies.

### Parameter Tuning

Expose:

```python
set_parameters(parameters: list[float]) -> str | dict
```

The order must match the task manager's `initial_parameters` keys. EAA supplies
parameter names and ranges locally and records parameter history in the proxy.

### Imaging Acquisition

Expose:

```python
acquire_image(...) -> dict
acquire_line_scan(...) -> dict
```

For `acquire_image`, return:

- `img_path`: display image path, when a plot should be shown;
- `array_path`: `.npy` array path readable by the EAA process, when analytical
  workflows or registration need numerical image data;
- one pixel-size key: `psize`, `pixel_size`, `scan_step`, or `stepsize_x`.

For `acquire_line_scan`, return:

- `fwhm`: numeric Gaussian-fit FWHM for analytical focusing;
- `img_path`: display image path, recommended;
- optional fit metadata: `a`, `mu`, `sigma`, `c`, `normalized_residual`,
  `x_min`, `x_max`.

Prefer default argument names:

- image and line coordinates: `x_center`, `y_center`;
- image size: `size_x`, `size_y`;
- scan step: `scan_step`;
- line scan length: `length`;
- line scan angle: `angle`.

Optionally expose:

```python
set_attribute(name: str, value: object) -> dict
```

or:

```python
set_config(name: str, value: object) -> dict
```

EAA uses one of these to request
`line_scan_return_gaussian_fit=True` when available.

## Review Checklist

- The MCP frontend and instrument worker are separate OS processes.
- The launcher supervises both children and terminates them on shutdown.
- The worker health check happens before the MCP server starts.
- Tool and launcher settings can be loaded from a YAML configuration file.
- The repository includes a useful example YAML file under `configs/`.
- CLI flags override YAML configuration values.
- ZMQ messages and results are JSON-serializable.
- Getter tools return serializable Python objects, not NumPy arrays, NumPy
  scalars, or control-library objects.
- Blocking control-library calls happen in the worker process.
- Large arrays are returned as artifacts, not embedded in JSON.
- Artifact paths returned to EAA are readable from the EAA process.
- Tool arguments use `Annotated[type, "description"]`, return types are
  annotated, and docstrings clearly describe units, coordinate order, and safety
  implications.
- Logic-driven EAA task managers have the required tool names and return keys.
