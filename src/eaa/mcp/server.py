"""
MCP Server component for exposing BaseTool subclasses as MCP tools.

This module provides functionality to create and run MCP servers that expose
methods from BaseTool subclasses as standardized MCP tools.
"""

import inspect
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
import asyncio

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "The 'mcp' package is required to use the MCP server. "
        "Install it with: pip install mcp"
    )

from eaa.tools.base import BaseTool, ToolReturnType
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
        self.tools: List[BaseTool] = tools or []
        self.mcp_server = FastMCP(name)
        self._tool_instances: Dict[str, BaseTool] = {}
        self._registered_tools: Dict[str, Dict[str, Any]] = {}
        
        # Register tools if provided
        if self.tools:
            self.register_tools(self.tools)
    
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
        Register a single BaseTool instance.
        
        Parameters
        ----------
        tool : BaseTool
            The BaseTool instance to register.
        """
        for tool_dict in tool.exposed_tools:
            tool_name = tool_dict["name"]
            tool_function = tool_dict["function"]
            return_type = tool_dict["return_type"]
            
            if tool_name in self._registered_tools:
                raise ValueError(f"Tool '{tool_name}' is already registered")
            
            # Store the tool instance for later method calls
            self._tool_instances[tool_name] = tool
            self._registered_tools[tool_name] = tool_dict
            
            # Create the MCP tool wrapper
            self._create_mcp_tool(tool_name, tool_function, return_type)
    
    def _create_mcp_tool(
        self, 
        tool_name: str, 
        tool_function: Callable, 
        return_type: ToolReturnType
    ) -> None:
        """
        Create an MCP tool from a BaseTool method.
        
        Parameters
        ----------
        tool_name : str
            Name of the tool.
        tool_function : Callable
            The callable method from the BaseTool.
        return_type : ToolReturnType
            Expected return type of the tool.
        """
        # Get function signature for parameter validation
        sig = inspect.signature(tool_function)
        
        # Create wrapper function that handles tool execution
        def mcp_tool_wrapper(**kwargs):
            """Wrapper function for MCP tool execution."""
            try:
                # Filter kwargs to match function signature
                filtered_kwargs = {}
                for param_name, param in sig.parameters.items():
                    if param_name in kwargs:
                        filtered_kwargs[param_name] = kwargs[param_name]
                    elif param.default == inspect.Parameter.empty:
                        raise ValueError(f"Required parameter '{param_name}' is missing")
                
                # Execute the tool function
                result = tool_function(**filtered_kwargs)
                
                # Handle different return types
                if return_type == ToolReturnType.TEXT:
                    return str(result)
                elif return_type == ToolReturnType.IMAGE_PATH:
                    return str(result)
                elif return_type == ToolReturnType.NUMBER:
                    return float(result) if result is not None else None
                elif return_type == ToolReturnType.BOOL:
                    return bool(result)
                elif return_type in [ToolReturnType.LIST, ToolReturnType.DICT]:
                    return result
                else:
                    return str(result)
                    
            except Exception as e:
                logger.error(f"Error executing tool '{tool_name}': {str(e)}")
                return f"Error: {str(e)}"
        
        # Set the docstring from the original function
        mcp_tool_wrapper.__doc__ = tool_function.__doc__ or f"Execute {tool_name}"
        
        # Set the function name to match the tool name for proper registration
        mcp_tool_wrapper.__name__ = tool_name
        
        # Register the tool with the MCP server using the decorator
        decorated_wrapper = self.mcp_server.tool()(mcp_tool_wrapper)
        
        logger.info(f"Registered MCP tool: {tool_name}")
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool schemas for all registered tools.
        
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
    
    def run(self, **kwargs) -> None:
        """
        Run the MCP server.
        
        Parameters
        ----------
        **kwargs
            Additional arguments passed to the FastMCP server.
        """
        logger.info(f"Starting MCP server '{self.name}' with {len(self._registered_tools)} tools")
        for tool_name in self.list_tools():
            logger.info(f"  - {tool_name}")
        
        # Run the server
        self.mcp_server.run(**kwargs)


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
    **server_kwargs
) -> None:
    """
    Create and run an MCP server from BaseTool instances.
    
    Parameters
    ----------
    tools : Union[BaseTool, List[BaseTool]]
        BaseTool instance(s) to expose via MCP.
    server_name : str, optional
        Name of the MCP server.
    **server_kwargs
        Additional arguments passed to the server run method.
    """
    server = create_mcp_server_from_tools(tools, server_name)
    server.run(**server_kwargs) 
