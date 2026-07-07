# Creating MCP Servers

For maintainability and cross-client transferrability, beamline-specific
experiment control tools are never embedded in EAA itself. To enable instrument
control, these tools must be served as external Model Context Protocol (MCP)
or HTTP servers. We recommend MCP as it is the widely adopted standard for
agent tools. MCP servers are not just for LLM agents; a logic-driven workflow
can also call tool functions in an MCP server like making RPC calls. 

## Server-client contracts

### For agent-driven workflows

EAA does not require any **strict** contract with MCP servers if the tools are
intended to be called only by LLM agents. As long as the server sends and
receives messages consistent with the MCP format, it works with most of the
MCP clients - which includes EAA and common agent harnesses like Codex, 
Claude Code, Hermes Agent. If the server is built with a common MCP toolkit
such as FastMCP with the correct API, this requirement should be satisfied.

A non-strict, but highly recommended contract for MCP tools that yield images - for
example, one that acquires an image with the microscope - should save the image
that the LLM agent should see as a PNG file to a path accessible by the agent
process, and return the absolute path to the PNG file in the `img_path` field
of the payload:
```json
{
  "img_path": "/pat/to/image.png"
}
```
EAA's tool executor looks for this field in the returned payload. If it is
present, it loads the image and send it to the LLM in a follow-up message.
Agent may still see the image as long as the path is somewhere in the returned
payload without following that contract as they can use their built-in image
viewing tool to "see" the image; however, with EAA, following the contract
saves this additional tool call. 

We do not recommend returning images as base64 encodings in the payload.
Encoding images in tool responses is not supported by all LLM providers'
APIs. Also, if the agent harness can't properly handle the tool return,
the base64 encoding might be sent to the LLM as text. LLM cannot understand
base64 directly. Since the encodings of images are long and unstructured, 
it may generate surprising token usage, take very long for the LLM to respond,
or result in a request error.

If the agent needs to get the raw data arrays from the tools for data analysis,
consider exposing a tool that dumps a buffer or attribute in the MCP server
as a data file such as NPY or TIFF, then return the path. Similar to the case
of images, base64-encoding the data is not recommended unless you are certain
that the agent harness knows how to properly handle the message containing
encoded data.

### For logic-driven workflows

Logic-driven workflows are a bit different: the tool functions in an MCP server
is called in Python code, not by an LLM. Python code cannot automatically
find the right tool to use semantically (for example, knowing that `acquire_image`
and `collect_image` are both for image acquisition). Rather, tool functions
must be explicitly matched by name. Fortunately, EAA provides the `MCPRPCWrapper`
class that allows the user of EAA to map known MCP-side tool names to local
tool names, ensuring that the logic-driven workflows in EAA call the right functions.
For example, if a logic-driven workflow expects `acquire_image` while the tool
in the MCP server is called `collect_image`, `MCPRPCWrapper` can wrap the direct MCP
client so that when the workflow calls `tool.acquire_image`, the wrapper calls
`collect_image` of the MCP server. For MCP server developers, this removes the
needs of using certain exact tool names. See 
[Tools](tools.md#calling-mcp-tools-from-logic-driven-task-managers) for more details on
client-side mapping.

However, if the logical workflow needs to fetch buffers, attributes, or data arrays
from the MCP server, EAA does require the server to expose a `get_attribute_payload`
tool, and the structure of the payload returned by this tool must satisfy a certain
format. This is necessary for `MCPRPCWrapper` to fetch the data from the server and
parse the payload. 

`get_attribute_payload` may return any JSON-serializable scalar, list, or
dictionary directly. For NumPy arrays or other dense numerical arrays, return a
portable array payload:

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

EAA reads `encoded_data`, decodes the base64 array bytes, and reconstructs the
array from the declared dtype and shape. On the server side, the encoding of the
data can be done with the `base64` library (if it is written in Python) without
depending on EAA. For example:
```python
import numpy as np
import base64

# Original array
arr = np.array([[1, 2, 3],
                [4, 5, 6]], dtype=np.float32)

# Encode
encoded = base64.b64encode(arr.tobytes()).decode("ascii")
```

## Async and thread safety

The gateway of the MCP server runs its own async event loop which may clash
if the execution runtime of the instrument control library (*e.g.* Bluesky)
is directly called inside a tool function. To avoid this issue, we recommend
separating the MCP gateway and the instrument worker which executes instrument
operations into different processes.

### Bluesky Queue Server

If you are using the Bluesky Queue Server at the beamline, you already have
a clean separation. Simply querying the Queue Server from your MCP server
is in principle safe because the execution is performed on the Queue Server,
avoiding the event loop conflict.

### In-server instrument control logic

If you don't use the Queue Server and need to directly execute instrument control
tools (*e.g.*, running a Bluesky RunEngine locally), the execution must be separated
from the MCP server gateway. We recommend dividing the MCP server into 2 parts:
the gateway that exposes the tools and interfaces with the agent, and the worker backend
that executes the instrument commands. The worker backend should be entirely synchronous.
The gateway and the backend can communicate through ZeroMQ:

```text
EAA or another MCP client
  -> MCP over HTTP
  -> FastMCP gate process
  -> ZMQ request/reply
  -> instrument worker process
```

The FastMCP process owns MCP and HTTP request handling. The instrument worker
process owns the instrument-control library, device state, blocking control
logic, and any runtime required by the control stack. ZMQ carries a small
JSON-serializable command envelope between the two processes.

#### Recommended Launch Interface

For the gateway-worker setup, expose three console scripts with a consistent launch convention
(assuming `facility-suite` is the name of the MCP server; choose your own name for it, for
example, `aps2idd-control-suite`):

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
        "encoded_data": {
            "type": "array",
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
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
