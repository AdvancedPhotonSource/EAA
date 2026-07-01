# Creating Async-Safe MCP Servers

Facility and endstation tools should live in separate MCP server repositories
instead of being added directly to EAA. This keeps EAA focused on orchestration
while each facility owns its instrument-control dependencies, deployment model,
and safety policy.

The recommended server design is a small Python package with separate modules
for the command protocol, ZMQ client, blocking worker, FastMCP frontend, and
launcher. External MCP servers should not depend on EAA packages. They should
expose normal MCP tools whose names, arguments, schemas, and descriptions match
the facility's real operations.

EAA compatibility is deliberately narrow:

- Operational tools should be normal MCP tools with facility-specific names and
  schemas.
- Tool argument names should match the facility API and should not be shaped
  around EAA task-manager internals.
- The only EAA-specific support tool that a stateful server should expose is
  `get_attribute_payload(name: str)`.

## Process Pattern

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
names and behavior should stay consistent.

## Tool Naming and Argument Naming

Do not name remote tools or arguments for EAA's internal classes. Name them for
the facility operation they actually perform. For example, an endstation might
choose names such as `collect_fly_scan`, `move_zone_plate`,
`set_mirror_voltages`, or `measure_probe_profile`. Another facility might
choose completely different names for equivalent operations.

The same rule applies to arguments. Use the names that are natural for the
instrument API and clear to users of that facility's MCP server. EAA should not
force names such as image coordinates, scan sizes, scan steps, parameter
vectors, or line-scan fields at the remote MCP boundary.

The server should not duplicate tools or aliases solely to match a particular
client's local terminology.

## State Sync Contract

The only standardized EAA support tool for stateful logic-driven workflows is:

```python
def get_attribute_payload(name: str) -> object:
    ...
```

This tool returns the current value of a server-side attribute identified by
`name`. The `name` values are server-defined attribute identifiers, not EAA
attribute names.

The support tool name should be exactly `get_attribute_payload` or a dotted
tool name ending with `.get_attribute_payload`. Attribute names are
facility-defined; examples include `detector.last_frame`,
`detector.last_metadata`, `stage.last_position`, or any other stable names
that fit the server's state model.

## Payload Format

`get_attribute_payload` may return any JSON-serializable scalar, list, or
dictionary directly. For NumPy arrays or other dense numerical arrays, return a
portable array payload:

```json
{
  "encoding": "numpy_base64",
  "dtype": "float32",
  "shape": [256, 256],
  "data": "base64-encoded contiguous array bytes"
}
```

EAA decodes this payload with `BaseTool.decode_array_payload`. External MCP
servers do not need to depend on EAA to produce it:

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
```

Use JSON literals for metadata, histories, counters, configuration flags, and
small structured records. Use the array payload only for dense arrays.

## Server-Side State

State that a logic-driven task manager will read later should be owned by the
instrument worker or by the MCP frontend. The owner is a facility decision. The
only requirement is that the MCP frontend can return the selected server-side
attributes through `get_attribute_payload`.

Typical state includes:

- the latest measurement array or image;
- previous or initial arrays needed for registration or comparison;
- pixel sizes, timestamps, file paths, or scan metadata;
- acquisition histories or parameter histories;
- status flags or configuration values needed by a later local decision.

Do not build EAA-specific state names into the server. Keep facility state names
stable.

## Client-Side Mapping

EAA clients that use the LLM chat loop can call facility-specific MCP tool
names directly through `MCPTool`. Logic-driven task managers are different:
they often call fixed Python methods such as `acquire_image(...)` or
`set_parameters(...)`.

For those workflows, wrap the `MCPTool` client with
`eaa_core.tool.mcp_adapter.MCPRPCWrapper`. The wrapper maps local method names
and local argument names to the facility-specific MCP tool and argument names.
It can also sync local wrapper attributes from server-side state by calling the
`get_attribute_payload(name=...)` support tool described above.

See [Tools](tools.md#calling-mcp-tools-from-logic-driven-task-managers) for the
client-side mapping example.

## Artifacts and Large Data

Tool results should remain JSON-serializable. Use file paths for artifacts that
humans or downstream tools need to inspect, such as display PNGs, logs, or raw
data files. The field names for those paths are part of the facility API or the
facility workflow; they are not standardized by this server contract.

Dense numerical arrays that EAA must keep in memory should be exposed through
`get_attribute_payload` rather than embedded directly in ordinary tool results.
Very large arrays may still be written to disk and returned as paths when the
downstream workflow is path-based.

## Worker Protocol

Use a small JSON command envelope between the FastMCP frontend and the
instrument worker:

```json
{
  "id": "uuid",
  "method": "facility_specific_operation",
  "params": {"facility_argument": 1.0}
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

Keep the worker boundary JSON-serializable. Large numerical arrays should be
kept in worker/frontend state and returned through `get_attribute_payload` or
written as artifacts and returned by path.

## Safety and Concurrency

Instrument-control operations should be serialized unless the facility control
stack explicitly supports concurrent calls. The worker process is the right
place to enforce serialization, hardware interlocks, argument bounds, timeout
policy, and recovery behavior.

The MCP frontend should treat worker timeouts and worker-side exceptions as
ordinary tool failures with clear error messages. It should not leave requests
running indefinitely in the MCP event loop.

## Minimal Example

This example shows the EAA-specific support surface only. The operational tool
names remain facility-specific.

```python
import base64
import numpy as np
from fastmcp import FastMCP

app = FastMCP("Example Facility Server")
state = {
    "detector.last_frame": None,
    "detector.last_metadata": {},
}


def encode_array_payload(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array)
    return {
        "encoding": "numpy_base64",
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


@app.tool()
def collect_detector_frame(exposure_ms: float, roi_width: int, roi_height: int) -> dict:
    frame = np.zeros((roi_height, roi_width), dtype=np.float32)
    state["detector.last_frame"] = frame
    state["detector.last_metadata"] = {
        "exposure_ms": exposure_ms,
        "roi_width": roi_width,
        "roi_height": roi_height,
    }
    return {"status": "ok"}


@app.tool()
def get_attribute_payload(name: str) -> object:
    value = state[name]
    if isinstance(value, np.ndarray):
        return encode_array_payload(value)
    return value
```

## Agent Skill

An agent-facing playbook for creating compatible servers is available at
`developer_tools/create_eaa_compatible_mcp_servers/SKILL.md`. Add that
directory to an agent skill path when you want an agent to scaffold or review a
new facility MCP server against this pattern.
