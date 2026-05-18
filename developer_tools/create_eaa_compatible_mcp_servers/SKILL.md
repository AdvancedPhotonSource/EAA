---
name: create-eaa-compatible-mcp-servers
description: Create or review facility-specific MCP server repositories that are compatible with EAA and safe for instrument-control libraries.
---

# Create EAA-Compatible MCP Servers

Use this skill when creating, reviewing, or refactoring an external MCP server
for EAA.

## Goals

- Keep facility and endstation-specific control logic outside EAA.
- The MCP server should not depend on EAA, and should be able to work with
  more MCP clients than EAA.
- However, it should observe a collection of contracts with EAA in order to
  exploit EAA's full capability.
- Use the process pattern `MCP frontend -> ZMQ -> instrument worker`.
- Keep instrument worker functions synchronous and blocking unless the facility
  control library requires an internal runtime.
- Make chat-facing MCP tools clear and self-describing.
- Document MCP tool methods in a way compatible with FastMCP tool schema
  generation.

## Repository Shape

Create a normal Python package managed with `uv`. The repository should include
`pyproject.toml`, `uv.lock`, a `src/<package_name>/` layout, and console scripts
declared under `[project.scripts]`.

The runtime dependencies should include MCP, ZMQ, and the facility control
library dependencies. Do not depend on EAA packages. The server should be usable
by any MCP client; EAA compatibility should come from tool schemas and payloads,
not from importing EAA.

The server should consist of an MCP frontend and a backend instrument worker.
At runtime, the frontend and worker are launched in different processes and
communicate through ZMQ. All calls of instrument control libraries, such as
Bluesky and EPICS, should only happen in the worker process. The worker process
should listen to ZMQ messages and execute instrument control functions using
blocking and synchronous routines. Avoiding async and multithreading is essential
to ensure that
- The tool server is thread- and async-safe with the instrument control libraries;
- All instrument movements are sequential and deterministic.

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

## Launch Interface

Expose three console scripts:

- `<facility-suite>`: a script that launches both the worker process and the 
  MCP frontend process
- `<facility-suite>-worker`: a script that launches the worker process
- `<facility-suite>-mcp`: a script that launches the MCP frontend process

Example, and recommended argument naming for the combined launch script:

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

The combined launcher command should:

- resolve the worker and MCP frontend console scripts from `PATH`;
- start the worker subprocess first;
- poll a `health` command through the ZMQ client until the worker is ready;
- start the MCP frontend subprocess after worker readiness;
- monitor both subprocesses and return a nonzero exit code if either child
  fails;
- handle `SIGINT` and `SIGTERM` by terminating both child processes, then
  killing them after a short timeout if needed.

For the seperate worker/frontend launching scripts, here are the examples:

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
- write display artifacts such as plots and preview images to files and return
  paths.

The MCP frontend should:

- own FastMCP and HTTP transport;
- not import or initialize the control library unless it absolutely 
  won't cause any thread- or async-safety issue;
- forward each MCP tool call to the worker through the ZMQ client;
- use `asyncio.to_thread(...)` if an async MCP handler calls a blocking ZMQ
  client.

### Tool returns

Tools and getters must return serialized Python objects such as `str`, `int`,
`float`, `bool`, `list`, `dict`, or `None`. Non-literal objects need to
be handled properly:
- If a tool is supposed to yield an image, the image should be rendered
to PNG, saved to hard drive, and the tool should return the path in the
`img_path` field of a JSON obejct.
- If a tool is supposed to return a NumPy array, use the 
`get_attribute_payload` contract below.

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
endpoint, timeout, hardware safety flags, output directory,
artifact path requirements, and instrument-specific parameters 
(such as exposure time) configured in YAML.

## EAA Contracts

EAA can use MCP tools in two workflow styles:

- **LLM-driven workflows**: an agent chooses and calls tools from the
  model-facing tool schema.
- **Logic-driven workflows**: task-manager code calls tools explicitly through
  EAA adapter proxies.

Tools should ideally support both workflow styles. LLM-driven workflows mostly
need clear schemas and file-based artifacts that the model can reason about.
Logic-driven workflows require slightly more contract surface because analytical
code often needs numerical arrays or other non-JSON-native data from the
server's internal state.

Do not force EAA-specific names on every general chat tool. Add these contracts
when a server is intended to participate in EAA acquisition, tuning, or
analysis workflows.

### LLM-Driven Workflow Contracts

LLM-driven workflows call visible MCP tools and operate on JSON responses plus
filesystem artifacts.

#### General Return Rules

- Tools must return serialized Python objects such as `str`, `int`, `float`,
  `bool`, `list`, `dict`, or `None`.
- Images must be rendered to PNG, saved to disk, and returned in the `img_path`
  field of a JSON object.
- NumPy arrays must be saved to disk as `.npy` files, and the tool must return
  the array path in JSON.
- Artifact paths returned to EAA must be readable from the EAA process.

The subsequent sub-sections present the contracts used for some specific tool types.
Not all tools should follow these API requirements; they apply only to tools with
matching functions. For example, the requirements for "Image Acquisition Tools" only
apply to tools for acquiring images.

#### Image Acquisition Tools

Expose an image acquisition tool:

```python
acquire_image(...) -> dict
```

For `acquire_image`, return:

- `img_path`: path to a PNG display image;
- one pixel-size key: `psize`, `pixel_size`, `scan_step`, or `stepsize_x`.

`acquire_image` must update server-side image buffers. The server must keep at
least:

- `image_0`: first 2D image acquired in the current run;
- `image_km1`: image immediately before the current image;
- `image_k`: most recent 2D image.

Expose an array dump tool in the image acquisition contract:

```python
dump_array(buffer_name: str) -> dict
```

`dump_array` must:

- accept native `buffer_name` values `image_k`, `image_km1`, and `image_0`;
- save the selected buffer as a `.npy` file;
- return the path to the saved array in JSON;
- avoid embedding large array values directly in the model-visible response.

Optionally, expose a line-scan acquisition tool when the instrument supports focusing or
line-profile analysis:

```python
acquire_line_scan(...) -> dict
```

For `acquire_line_scan`, return:

- `fwhm`: numeric Gaussian-fit FWHM for analytical focusing;
- `img_path`: path to a PNG image of the scanned line profile plot, optionally with the Gaussian fit;
- optional fit metadata: `a`, `mu`, `sigma`, `c`, `normalized_residual`,
  `x_min`, `x_max`.

Prefer default argument names:

- image and line coordinates: `x_center`, `y_center`;
- image size: `size_x`, `size_y`;
- scan step: `scan_step`;
- line scan length: `length`;
- line scan direction (0 degree is horizontal; position angles rotate counter-clockwise): `angle`.

#### Instrument Parameter Setting Tools

Expose a setter for setting instrument parameters (motor positions, exposure time, etc.).
For example:

```python
set_zp_z_position(parameters: list[float]) -> str | dict
```

The input can be a list if the parameter is multi-dimensional.

The response should be JSON-serializable and should report whether the setting
operation succeeded. Include current values or relevant status fields when they
help the agent decide the next action.

### Logic-Driven Workflow Contracts

Logic-driven workflows call tools explicitly from task-manager code. They can
use the same visible tools as LLM-driven workflows, but they often need direct
access to numerical arrays held in server-side buffers.

Expose an adapter method for logic-driven attribute transfer:

```python
get_attribute_payload(name: str) -> object
```

`get_attribute_payload` must:

- accept the native server-side attribute name to fetch;
- fetch the value with attribute semantics equivalent to `getattr(tool, name)`;
- return JSON literal values such as `str`, `int`, `float`, `bool`, `list`,
  `dict`, or `None` as-is;
- encode NumPy arrays as a JSON-serializable payload;
- preserve the array dtype and shape when encoding arrays;
- avoid requiring EAA as a dependency of the external MCP server.

For acquisition buffers, EAA callers use the native attribute names rather than
the old aliases:

- `image_k` for the current image;
- `image_km1` for the previous image;
- `image_0` for the initial image.

Array payloads must use this format:

```python
{
    "encoding": "numpy_base64",
    "dtype": "float32",
    "shape": [256, 256],
    "data": "<base64-encoded array bytes>",
}
```

Use this dependency-light encoding pattern in external MCP servers:

```python
import base64
import numpy as np


def encode_array_payload(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array)
    return {
        "encoding": "numpy_base64",
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def get_attribute_payload(name: str) -> object:
    value = getattr(tool, name)
    if isinstance(value, np.ndarray):
        return encode_array_payload(value)
    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise ValueError(f"Attribute {name!r} has unsupported payload type.")
```

External servers should expose `get_attribute_payload` as a normal FastMCP
tool. EAA filters this method out of the LLM-facing tool schema, so the model
cannot see or call it. Analytical task-manager code can still call it over MCP
through EAA's adapter layer. EAA's built-in tools inherit this method from
`BaseTool`, but external MCP servers do not use `BaseTool`; server authors must
implement this contract explicitly.

EAA's acquisition proxy can use `get_attribute_payload` to implement
`dump_array(buffer_name: str)` by retrieving `image_k`, `image_km1`, or
`image_0`, writing it to a local `.npy` file, and returning the path to the
LLM-visible workflow.

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
