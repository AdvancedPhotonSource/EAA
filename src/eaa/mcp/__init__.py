"""
MCP (Model Context Protocol) components for EAA.

This package provides functionality to expose BaseTool subclasses as MCP servers,
allowing them to be used by MCP clients like Cursor or other AI applications.
"""

from .server import MCPToolServer, create_mcp_server_from_tools, run_mcp_server_from_tools

__all__ = [
    "MCPToolServer",
    "create_mcp_server_from_tools", 
    "run_mcp_server_from_tools"
] 