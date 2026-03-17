Tools
=====

BaseTool
--------

EAA tools are stateful Python objects derived from ``BaseTool``. Tool methods
are exposed by decorating them with ``@tool(name=...)``.

A typical tool looks like this:

.. code-block:: python

   from eaa.core.tooling.base import BaseTool, tool


   class ExampleTool(BaseTool):
       @tool(name="add")
       def add(self, a: float, b: float) -> float:
           return a + b

Tool execution normalizes every result into a JSON object. Scalar returns are
wrapped as ``{"result": ...}``, while image-producing tools should surface the
path through ``{"img_path": "..."}``.

When a ``BaseTool`` instance is created, it discovers decorated methods and
builds ``exposed_tools`` metadata that the task manager can register with the
model-facing tool executor.

Thread safety and serial execution
----------------------------------

EAA intentionally executes tools serially through ``SerialToolExecutor``.

Why this matters:

- many experiment tools are stateful
- tool calls often mutate instrument state
- parallel tool calls would make ordering and rollback ambiguous

The executor therefore runs assistant-requested tool calls one at a time and
records a normalized tool message for each result. This is the default safety
model for the current codebase.

Approval gates
--------------

Each tool instance can require approval by setting ``require_approval=True``.
When a tool call needs approval, the task manager routes the decision through
its normal input path, including the WebUI path when ``use_webui=True``.

Serving built-in tools as MCP servers
-------------------------------------

EAA can expose any ``BaseTool`` instance as an MCP server by wrapping it with
the helpers in ``eaa.core.mcp.server``.

.. code-block:: python

   from eaa.core.mcp.server import run_mcp_server_from_tools
   from eaa.tool.example_calculator import CalculatorTool

   run_mcp_server_from_tools(
       tools=CalculatorTool(),
       server_name="Calculator MCP Server",
   )

The MCP wrapper preserves the normalized EAA JSON result contract by
publishing an object output schema.

Using external MCP servers
--------------------------

EAA can also consume remote MCP tools through ``MCPTool`` in ``eaa.tool.mcp``.
The wrapper connects to one or more MCP servers using a FastMCP-compatible
configuration and exposes the remote tools through the normal ``BaseTool``
interface.

.. code-block:: python

   from eaa.tool.mcp import MCPTool

   mcp_tool = MCPTool(
       {
           "mcpServers": {
               "image_acquisition": {
                   "command": "python",
                   "args": ["./image_acquisition_mcp_server.py"],
               }
           }
       }
   )

   task_manager.register_tools(mcp_tool)

Notes:

- EAA normalizes tool results to JSON for tools served through the EAA MCP
  server helper
- arbitrary third-party MCP servers may still return non-EAA payloads, so the
  agent loop only treats results as image-bearing when an ``img_path`` is
  present
