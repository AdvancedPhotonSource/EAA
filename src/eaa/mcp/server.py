"""
MCP Server component for exposing BaseTool subclasses as MCP tools.

This module provides functionality to create and run MCP servers that expose
methods from BaseTool subclasses as standardized MCP tools.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Literal

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "The 'fastmcp' package is required to use the MCP server. "
        "Install it with: pip install fastmcp"
    )

from eaa.tools.base import BaseTool
from eaa.agents.base import generate_openai_tool_schema

logger = logging.getLogger(__name__)


class MCPToolServer:
    """
    An MCP server that exposes BaseTool methods as MCP tools.
    
    This class creates an MCP server that automatically converts BaseTool
    subclasses into MCP-compatible tools, allowing them to be used by
    MCP clients like Cursor or other AI applications.
    """
    
    def __init__(
        self, 
        name: str = "BaseTool MCP Server",
        tools: Optional[List[BaseTool]] = None
    ):
        """
        Initialize the MCP Tool Server.
        
        Parameters
        ----------
        name : str, optional
            The name of the MCP server.
        tools : List[BaseTool], optional
            List of BaseTool instances to expose via MCP.
        """
        self.name = name
        self.mcp_server = FastMCP(name)
        self._tool_instances: Dict[str, BaseTool] = {}
        self._registered_tools: Dict[str, Dict[str, Any]] = {}
        
        # Register tools if provided
        if tools:
            self.register_tools(tools)
    
    def register_tools(self, tools: Union[BaseTool, List[BaseTool]]) -> None:
        """
        Register BaseTool instances with the MCP server.
        
        Parameters
        ----------
        tools : Union[BaseTool, List[BaseTool]]
            BaseTool instance(s) to register.
        """
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError(f"Tool must be a BaseTool instance, got {type(tool)}")
            
            if not hasattr(tool, "exposed_tools") or not tool.exposed_tools:
                raise ValueError(
                    f"BaseTool {tool.__class__.__name__} must have non-empty "
                    "`exposed_tools` attribute"
                )
            
            self._register_tool_instance(tool)
    
    def _register_tool_instance(self, tool: BaseTool) -> None:
        """
        Register all exposed tool methods of a BaseTool instance.
        
        Parameters
        ----------
        tool : BaseTool
            The BaseTool instance to register.
        """
        for tool_dict in tool.exposed_tools:
            tool_name = tool_dict["name"]
            tool_function = tool_dict["function"]
            
            if tool_name in self._registered_tools:
                raise ValueError(f"Tool '{tool_name}' is already registered")
            
            # Store the tool instance for later method calls
            self._tool_instances[tool_name] = tool
            self._registered_tools[tool_name] = tool_dict
            
            # Register the tool with the FastMCP server using the @tool decorator
            # The FastMCP 2.x API uses the @server.tool() decorator
            self.mcp_server.tool(name=tool_name)(tool_function)
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool schemas for all registered tools.
        
        Note that the schemas returned are NOT what's used by the MCP server.
        The MCP SDK creates the schemas itself using annotations and docstrings
        in the tool functions. Only use this function for reference.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of tool schemas.
        """
        schemas = []
        for tool_name, tool_dict in self._registered_tools.items():
            schema = generate_openai_tool_schema(tool_name, tool_dict["function"])
            schemas.append(schema)
        return schemas
    
    def list_tools(self) -> List[str]:
        """
        List all registered tool names.
        
        Returns
        -------
        List[str]
            List of tool names.
        """
        return list(self._registered_tools.keys())
    
    def get_server(self) -> FastMCP:
        """
        Get the underlying FastMCP server instance.
        
        Returns
        -------
        FastMCP
            The FastMCP server instance.
        """
        return self.mcp_server
    
    def run(
        self, 
        transport: Optional[Literal["stdio", "http", "sse", "streamable-http"]] = "stdio", 
        **server_kwargs
    ) -> None:
        """
        Run the MCP server.
        
        Parameters
        ----------
        transport : str, optional
            Transport protocol to use ("stdio", "http", "sse", or "streamable-http").
            Defaults to "stdio" for compatibility with MCP clients like Cursor.
        **server_kwargs
            Additional arguments passed to the FastMCP server run method.
        """
        logger.info(f"Starting MCP server '{self.name}' with {len(self._registered_tools)} tools")
        for tool_name in self.list_tools():
            logger.info(f"  - {tool_name}")
        
        # Run the server with the specified transport
        self.mcp_server.run(transport=transport, **server_kwargs)


def create_mcp_server_from_tools(
    tools: Union[BaseTool, List[BaseTool]],
    server_name: str = "BaseTool MCP Server"
) -> MCPToolServer:
    """
    Convenience function to create an MCP server from BaseTool instances.
    
    Parameters
    ----------
    tools : Union[BaseTool, List[BaseTool]]
        BaseTool instance(s) to expose via MCP.
    server_name : str, optional
        Name of the MCP server.
    
    Returns
    -------
    MCPToolServer
        Configured MCP server ready to run.
    """
    server = MCPToolServer(name=server_name)
    server.register_tools(tools)
    return server


def run_mcp_server_from_tools(
    tools: Union[BaseTool, List[BaseTool]], 
    server_name: str = "BaseTool MCP Server",
    transport: Optional[Literal["stdio", "http", "sse", "streamable-http"]] = "stdio",
    **server_kwargs
) -> None:
    """
    Create and run an MCP server from BaseTool instances.
    
    Parameters
    ----------
    tools : Union[BaseTool, List[BaseTool]]
        BaseTool instance(s) to expose via MCP.
    transport : str, optional
        Transport protocol to use ("stdio", "http", "sse", or "streamable-http").
        Defaults to "stdio" for compatibility with MCP clients like Cursor.
    server_name : str, optional
        Name of the MCP server.
    **server_kwargs
        Additional arguments passed to the FastMCP.run method.
    """
    server = create_mcp_server_from_tools(tools, server_name)
    server.run(transport=transport, **server_kwargs) 
