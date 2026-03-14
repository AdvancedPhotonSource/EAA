Tools
=====

BaseTool
--------

EAA tools are stateful Python objects derived from ``BaseTool``. Tool methods
are exposed by decorating them with ``@tool(name=..., return_type=...)``.

A typical tool looks like this:

.. code-block:: python

   from eaa.core.tooling.base import BaseTool, ToolReturnType, tool


   class ExampleTool(BaseTool):
       @tool(name="add", return_type=ToolReturnType.NUMBER)
       def add(self, a: float, b: float) -> float:
           return a + b

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

The MCP wrapper preserves EAA tool return metadata such as ``IMAGE_PATH`` by
publishing an output schema extension.

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

- EAA preserves return types for tools served through the EAA MCP server helper
- arbitrary third-party MCP servers may not provide EAA return-type metadata,
  so their results may degrade to plain text in the agent loop
