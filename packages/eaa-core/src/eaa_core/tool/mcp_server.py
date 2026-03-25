"""MCP server helpers for exposing EAA tools."""

import inspect
import logging
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Union

from fastmcp import FastMCP

from eaa_core.tool.base import (
    BaseTool,
    ExposedToolSpec,
    build_mcp_output_schema,
    generate_openai_tool_schema,
    normalize_tool_result,
)

logger = logging.getLogger(__name__)


class MCPToolServer:
    """Expose `BaseTool` methods as MCP tools."""

    def __init__(
        self,
        name: str = "BaseTool MCP Server",
        tools: Optional[List[BaseTool]] = None,
    ):
        """Initialize the server.

        Parameters
        ----------
        name : str, optional
            MCP server name.
        tools : list[BaseTool], optional
            Tools to register immediately.
        """
        self.name = name
        self.mcp_server = FastMCP(name)
        self._tool_instances: Dict[str, BaseTool] = {}
        self._registered_tools: Dict[str, ExposedToolSpec] = {}
        if tools:
            self.register_tools(tools)

    def register_tools(self, tools: Union[BaseTool, List[BaseTool]]) -> None:
        """Register one or more `BaseTool` instances.

        Parameters
        ----------
        tools : BaseTool or list[BaseTool]
            Tool instances to expose through MCP.
        """
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError(f"Tool must be a BaseTool instance, got {type(tool)}")
            if not hasattr(tool, "exposed_tools") or not tool.exposed_tools:
                raise ValueError(
                    f"BaseTool {tool.__class__.__name__} must have non-empty `exposed_tools`."
                )
            self._register_tool_instance(tool)

    def _register_tool_instance(self, tool: BaseTool) -> None:
        """Register every exposed tool method from a tool instance.

        Parameters
        ----------
        tool : BaseTool
            Tool instance whose exposed methods should be registered.
        """
        for spec in tool.exposed_tools:
            if not isinstance(spec, ExposedToolSpec):
                raise TypeError("Items in `exposed_tools` must be ExposedToolSpec instances.")
            if spec.name in self._registered_tools:
                raise ValueError(f"Tool '{spec.name}' is already registered")
            self._tool_instances[spec.name] = tool
            self._registered_tools[spec.name] = spec

            self.mcp_server.tool(
                name=spec.name,
                output_schema=build_mcp_output_schema(),
            )(self._wrap_tool_function(spec.function))

    @staticmethod
    def _wrap_tool_function(function):
        """Wrap a tool function so MCP always receives normalized JSON."""

        @wraps(function)
        def wrapped_function(*args, **kwargs):
            return normalize_tool_result(function(*args, **kwargs))

        wrapped_function.__signature__ = inspect.signature(function)
        return wrapped_function

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible schemas for registered MCP tools.

        Returns
        -------
        list[dict[str, Any]]
            Model-facing schemas for the registered tools.
        """
        return [
            spec.schema or generate_openai_tool_schema(tool_name=name, func=spec.function)
            for name, spec in self._registered_tools.items()
        ]

    def list_tools(self) -> List[str]:
        """Return the registered tool names.

        Returns
        -------
        list[str]
            Registered MCP tool names.
        """
        return list(self._registered_tools.keys())

    def get_server(self) -> FastMCP:
        """Return the underlying `FastMCP` server.

        Returns
        -------
        FastMCP
            Underlying FastMCP server instance.
        """
        return self.mcp_server

    def run(
        self,
        transport: Optional[Literal["stdio", "http", "sse", "streamable-http"]] = "stdio",
        **server_kwargs,
    ) -> None:
        """Run the MCP server.

        Parameters
        ----------
        transport : {"stdio", "http", "sse", "streamable-http"}, optional
            Transport used to serve the MCP endpoint.
        **server_kwargs
            Additional FastMCP server options.
        """
        logger.info(
            "Starting MCP server '%s' with %d tools",
            self.name,
            len(self._registered_tools),
        )
        for tool_name in self.list_tools():
            logger.info("  - %s", tool_name)
        self.mcp_server.run(transport=transport, **server_kwargs)


def create_mcp_server_from_tools(
    tools: Union[BaseTool, List[BaseTool]],
    server_name: str = "BaseTool MCP Server",
) -> MCPToolServer:
    """Create an MCP server from EAA tools.

    Parameters
    ----------
    tools : BaseTool or list[BaseTool]
        Tool instances to expose.
    server_name : str, optional
        MCP server name.

    Returns
    -------
    MCPToolServer
        Configured MCP server.
    """
    server = MCPToolServer(name=server_name)
    server.register_tools(tools)
    return server


def run_mcp_server_from_tools(
    tools: Union[BaseTool, List[BaseTool]],
    server_name: str = "BaseTool MCP Server",
    transport: Optional[Literal["stdio", "http", "sse", "streamable-http"]] = "stdio",
    **server_kwargs,
) -> None:
    """Create and run an MCP server from EAA tools.

    Parameters
    ----------
    tools : BaseTool or list[BaseTool]
        Tool instances to expose.
    server_name : str, optional
        MCP server name.
    transport : {"stdio", "http", "sse", "streamable-http"}, optional
        Transport used to serve the MCP endpoint.
    **server_kwargs
        Additional FastMCP server options.
    """
    server = create_mcp_server_from_tools(tools, server_name)
    server.run(transport=transport, **server_kwargs)
