# Creating Async-Safe MCP Servers

Facility and endstation tools should live in separate MCP server repositories
instead of being added directly to EAA. This keeps EAA focused on orchestration
while each facility owns its instrument-control dependencies, deployment model,
and safety policy.

The standard design is a small Python package with separate modules for the
command protocol, ZMQ client, blocking worker, FastMCP frontend, and launcher.
External MCP servers should not depend on EAA packages. They should expose a
standard MCP interface that any MCP client can consume; EAA-specific
compatibility is limited to tool names, schemas, and return payloads needed by
adapter proxies.

## Pattern

Use a three-layer process boundary:

```text
EAA or another MCP client
  -> MCP over HTTP
  -> FastMCP frontend process
  -> ZMQ request/reply
  -> instrument worker process
```

The FastMCP process owns MCP and HTTP request handling. The instrument worker
process owns the instrument-control library, device state, blocking control
logic, and any runtime required by the control stack. ZMQ carries a small
JSON-serializable command envelope between the two processes.

This split is intentional. Many instrument libraries are synchronous, stateful,
or built around their own event loop or callback runtime. Tool worker functions
should therefore be synchronous and blocking by design. The MCP frontend may be
async internally, but it should forward each request to the worker without
running the control library in the MCP event loop.

## Recommended Launch Interface

Expose three console scripts with a consistent launch convention:

```text
facility-suite
facility-suite-worker
facility-suite-mcp
```

The top-level launcher starts and supervises the two child processes. It should
start the worker first, wait for a health response over ZMQ, then start the MCP
server. It should also terminate both child processes on shutdown or when one
child exits unexpectedly.

Recommended launcher options:

```bash
facility-suite \
  --worker-endpoint tcp://127.0.0.1:5555 \
  --worker-startup-timeout-s 10 \
  --request-timeout-ms 30000 \
  --mcp-host 0.0.0.0 \
  --mcp-port 8050 \
  --mcp-path /mcp
```

Recommended worker options:

```bash
facility-suite-worker --bind tcp://127.0.0.1:5555
```

Recommended MCP frontend options:

```bash
facility-suite-mcp \
  --worker tcp://127.0.0.1:5555 \
  --timeout-ms 30000 \
  --host 0.0.0.0 \
  --port 8050 \
  --path /mcp
```

The exact package and script names should be facility-specific, but the option
names and behavior should stay consistent. The MCP endpoint is then consumed by
EAA with a normal FastMCP-compatible client configuration:

```python
from eaa_core.tool.mcp_client import MCPTool

tools = MCPTool(
    {
        "mcpServers": {
            "facility_tools": {
                "url": "http://127.0.0.1:8050/mcp",
                "transport": "http",
            }
        }
    }
)
```

## General EAA Contract

For chat interactions and agent-driven processes, EAA should not require an
external MCP server to obey EAA-specific tool names, attribute names, or special
handling rules. The server should expose clear tool names, schemas, and
descriptions that match the facility's real operations. The agent-facing EAA
loop registers those tools through `MCPTool` and lets the model choose them by
schema.

Logic-driven task managers are different. Some task managers call methods and
attributes from Python code instead of asking the model to choose tools. For
MCP-backed tools, EAA bridges that gap with adapter proxies:

- `packages/eaa-core/src/eaa_core/tool/mcp_adapter.py`
- `packages/eaa-imaging/src/eaa_imaging/tool/imaging/mcp_acquisition.py`

Only MCP servers intended for these logic-driven task managers need the
contracts below.

## Parameter-Setting Contract

`BaseParameterTuningTaskManager` wraps an `MCPTool` with
`MCPParameterSettingProxy`. The remote MCP server must expose:

`set_parameters(parameters: list[float])`

: Set the controlled parameters in the same order as the task manager's
  `initial_parameters` keys. Return a string or a JSON-serializable object. The
  proxy records local parameter history after the remote call succeeds.

The task manager supplies `parameter_names` and `parameter_ranges` locally; the
MCP server does not need to expose those attributes. The proxy provides the
local `get_current_parameters` behavior from EAA's `SetParameters` base class.

## Imaging Acquisition Contract

Imaging task managers wrap an `MCPTool` with `MCPAcquireImageProxy`. For
analytical focusing workflows, the remote MCP server should expose:

`acquire_image(...)`

: Acquire a 2D image, update the server-side acquisition buffers, and return a
  JSON object. Include pixel-size metadata with one of `psize`, `pixel_size`,
  `scan_step`, or `stepsize_x`. Include `img_path` when a displayable image
  should be shown in the chat or WebUI.

`get_image_array_payload(buffer_name: str)`

: Return a JSON-serializable payload for one buffered image. `buffer_name` must
  accept `"current"`, `"previous"`, and `"initial"`. EAA decodes this payload
  through `MCPAcquireImageProxy.get_image_array(...)` so built-in and
  MCP-backed acquisition tools have the same task-manager API. External servers
  should expose this as a normal FastMCP tool. EAA's MCP client treats this name
  as an adapter-only method: analytical task-manager code can call it over MCP,
  but it is omitted from model-facing tool schemas. The payload must use this
  shape:

```json
{
  "encoding": "numpy_base64",
  "dtype": "float32",
  "shape": [256, 256],
  "data": "base64-encoded array bytes"
}
```

External MCP servers do not need to depend on EAA to produce this payload:

```python
import base64
import numpy as np


def encode_image_array_payload(image: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(image)
    return {
        "encoding": "numpy_base64",
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }
```

`acquire_line_scan(...)`

: Acquire a line scan and return a JSON object. Analytical focusing expects a
  numeric `fwhm` value. `img_path` is recommended for line-scan validation and
  user-visible reporting. Gaussian-fit metadata such as `a`, `mu`, `sigma`,
  `c`, `normalized_residual`, `x_min`, and `x_max` is useful when available.

`set_attribute(name: str, value: object)` or `set_config(name: str, value: object)`

: Optional. If present, the proxy uses it to set
  `line_scan_return_gaussian_fit=True` when an analytical focusing manager
  starts. Servers may also always include the fit metadata needed by the
  workflow.

Default analytical focusing argument names are:

- line scan coordinates: `x_center` and `y_center`
- image acquisition coordinates: `x_center` and `y_center`
- image size: `size_x` and `size_y`; `width` and `height` are also recognized
  for local history recording
- scan step: `scan_step` or `stepsize_x`
- line scan length: `length`
- line scan angle: `angle`

Task managers can be configured with alternate coordinate argument names, but
external servers should prefer the defaults when possible.

## Image Artifacts

`img_path` should point to a displayable image file such as PNG when the result
should appear in chat or the WebUI. Numerical image arrays must be transferred
through `get_image_array_payload`.

The MCP server should keep at least three image buffers:

- `initial`: the first 2D image acquired in the current run
- `previous`: the image immediately before the current image
- `current`: the most recent 2D image

These buffers may live in the instrument worker or the MCP frontend, but the
frontend must be able to return them through `get_image_array_payload`.

LLM-visible image registration uses path-based tools. EAA acquisition tools and
`MCPAcquireImageProxy` expose `dump_array(buffer_name: str)` to save a buffered
image array and return `{"array_path": "..."}`. An agent can dump the `current`
and `previous` or `initial` buffers, then call the registration tool's
`get_offset_from_paths` method with those paths. External MCP image acquisition
servers provide the buffers through `get_image_array_payload`; the EAA proxy
provides the dumping helper on the EAA side.

## Worker Protocol

Use this ZMQ command envelope:

```json
{
  "id": "uuid",
  "method": "acquire_image",
  "params": {"x_center": 1.0}
}
```

Successful responses contain:

```json
{
  "id": "uuid",
  "status": "ok",
  "result": {}
}
```

Failed responses contain:

```json
{
  "id": "uuid",
  "status": "error",
  "error": "message"
}
```

Keep the command boundary JSON-serializable. Large numerical arrays should be
written as artifacts and returned by path rather than embedded in the response.

## Agent Skill

An agent-facing playbook for creating compatible servers is available at
`developer_tools/create_eaa_compatible_mcp_servers/SKILL.md`. Add that
directory to an agent skill path when you want an agent to scaffold or review a
new facility MCP server against this pattern.
