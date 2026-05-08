Creating Async-Safe MCP Servers
===============================

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

Pattern
-------

Use a three-layer process boundary:

.. code-block:: text

   EAA or another MCP client
     -> MCP over HTTP
     -> FastMCP frontend process
     -> ZMQ request/reply
     -> instrument worker process

The FastMCP process owns MCP and HTTP request handling. The instrument worker
process owns the instrument-control library, device state, blocking control
logic, and any runtime required by the control stack. ZMQ carries a small
JSON-serializable command envelope between the two processes.

This split is intentional. Many instrument libraries are synchronous, stateful,
or built around their own event loop or callback runtime. Tool worker functions
should therefore be synchronous and blocking by design. The MCP frontend may be
async internally, but it should forward each request to the worker without
running the control library in the MCP event loop.

Recommended Launch Interface
----------------------------

Expose three console scripts with a consistent launch convention:

.. code-block:: text

   facility-suite
   facility-suite-worker
   facility-suite-mcp

The top-level launcher starts and supervises the two child processes. It should
start the worker first, wait for a health response over ZMQ, then start the MCP
server. It should also terminate both child processes on shutdown or when one
child exits unexpectedly.

Recommended launcher options:

.. code-block:: bash

   facility-suite \
     --worker-endpoint tcp://127.0.0.1:5555 \
     --worker-startup-timeout-s 10 \
     --request-timeout-ms 30000 \
     --mcp-host 0.0.0.0 \
     --mcp-port 8050 \
     --mcp-path /mcp

Recommended worker options:

.. code-block:: bash

   facility-suite-worker --bind tcp://127.0.0.1:5555

Recommended MCP frontend options:

.. code-block:: bash

   facility-suite-mcp \
     --worker tcp://127.0.0.1:5555 \
     --timeout-ms 30000 \
     --host 0.0.0.0 \
     --port 8050 \
     --path /mcp

The exact package and script names should be facility-specific, but the option
names and behavior should stay consistent. The MCP endpoint is then consumed by
EAA with a normal FastMCP-compatible client configuration:

.. code-block:: python

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

General EAA Contract
--------------------

For chat interactions and agent-driven processes, EAA should not require an
external MCP server to obey EAA-specific tool names, attribute names, or special
handling rules. The server should expose clear tool names, schemas, and
descriptions that match the facility's real operations. The agent-facing EAA
loop registers those tools through ``MCPTool`` and lets the model choose them by
schema.

Logic-driven task managers are different. Some task managers call methods and
attributes from Python code instead of asking the model to choose tools. For
MCP-backed tools, EAA bridges that gap with adapter proxies:

- ``packages/eaa-core/src/eaa_core/tool/mcp_adapter.py``
- ``packages/eaa-imaging/src/eaa_imaging/tool/imaging/mcp_acquisition.py``

Only MCP servers intended for these logic-driven task managers need the
contracts below.

Parameter-Setting Contract
--------------------------

``BaseParameterTuningTaskManager`` wraps an ``MCPTool`` with
``MCPParameterSettingProxy``. The remote MCP server must expose:

``set_parameters(parameters: list[float])``
   Set the controlled parameters in the same order as the task manager's
   ``initial_parameters`` keys. Return a string or a JSON-serializable object.
   The proxy records local parameter history after the remote call succeeds.

The task manager supplies ``parameter_names`` and ``parameter_ranges`` locally;
the MCP server does not need to expose those attributes. The proxy provides the
local ``get_current_parameters`` behavior from EAA's ``SetParameters`` base
class.

Imaging Acquisition Contract
----------------------------

Imaging task managers wrap an ``MCPTool`` with ``MCPAcquireImageProxy``. For
analytical focusing workflows, the remote MCP server should expose:

``acquire_image(...)``
   Acquire a 2D image and return a JSON object. For workflows that need image
   registration or local image buffers, include ``array_path`` pointing to a
   ``.npy`` array readable by the EAA process. Include pixel-size metadata with
   one of ``psize``, ``pixel_size``, ``scan_step``, or ``stepsize_x``. Include
   ``img_path`` when a displayable image should be shown in the chat or WebUI.

``acquire_line_scan(...)``
   Acquire a line scan and return a JSON object. Analytical focusing expects a
   numeric ``fwhm`` value. ``img_path`` is recommended for line-scan validation
   and user-visible reporting. Gaussian-fit metadata such as ``a``, ``mu``,
   ``sigma``, ``c``, ``normalized_residual``, ``x_min``, and ``x_max`` is
   useful when available.

``set_attribute(name: str, value: object)`` or ``set_config(name: str, value: object)``
   Optional. If present, the proxy uses it to set
   ``line_scan_return_gaussian_fit=True`` when an analytical focusing manager
   starts. Servers may instead always include the fit metadata needed by the
   workflow.

Default analytical focusing argument names are:

- line scan coordinates: ``x_center`` and ``y_center``
- image acquisition coordinates: ``x_center`` and ``y_center``
- image size: ``size_x`` and ``size_y``; ``width`` and ``height`` are also
  recognized for local history recording
- scan step: ``scan_step`` or ``stepsize_x``
- line scan length: ``length``
- line scan angle: ``angle``

Task managers can be configured with alternate coordinate argument names, but
external servers should prefer the defaults when possible.

Image Artifacts
---------------

``img_path`` should point to a displayable image file such as PNG when the
result should appear in chat or the WebUI. ``array_path`` should point to a
``.npy`` array when EAA needs to load the numerical image for registration,
buffer tracking, or analytical workflows.

When the MCP server runs on a different machine, these paths must still be
readable from the EAA process, for example through a shared filesystem or a
deliberate artifact synchronization layer. A path local only to the instrument
worker host is not enough for EAA-side registration.

Worker Protocol
---------------

Use this ZMQ command envelope:

.. code-block:: json

   {
     "id": "uuid",
     "method": "acquire_image",
     "params": {"x_center": 1.0}
   }

Successful responses contain:

.. code-block:: json

   {
     "id": "uuid",
     "status": "ok",
     "result": {}
   }

Failed responses contain:

.. code-block:: json

   {
     "id": "uuid",
     "status": "error",
     "error": "message"
   }

Keep the command boundary JSON-serializable. Large numerical arrays should be
written as artifacts and returned by path rather than embedded in the response.

Agent Skill
-----------

An agent-facing playbook for creating compatible servers is available at
``developer_tools/create_eaa_compatible_mcp_servers/SKILL.md``. Add that
directory to an agent skill path when you want an agent to scaffold or review a
new facility MCP server against this pattern.
