import asyncio

import fastmcp

from eaa.tools.base import BaseTool


class MCPTool(BaseTool):
    
    def __init__(
        self,
        config: dict,
        *args, **kwargs
    ):
        """Initialize an MCP tool.

        Parameters
        ----------
        config : dict
            A dictionary giving the configurations of one or multiple MCP
            servers. The structure of the dictionary should follow the standard
            of FastMCP (https://gofastmcp.com/clients/client):
            ```
            config = {
                "mcpServers": {
                    "server_name": {
                        # Remote HTTP/SSE server
                        "transport": "http",  # or "sse" 
                        "url": "https://api.example.com/mcp",
                        "headers": {"Authorization": "Bearer token"},
                        "auth": "oauth"  # or bearer token string
                    },
                    "local_server": {
                        # Local stdio server
                        "transport": "stdio",
                        "command": "python",
                        "args": ["./server.py", "--verbose"],
                        "env": {"DEBUG": "true"},
                        "cwd": "/path/to/server",
                    }
                }
            }
            ```
            Below is a multi-server example from the FastMCP documentation:
            ```
            config = {
                "mcpServers": {
                    "weather": {"url": "https://weather-api.example.com/mcp"},
                    "assistant": {"command": "python", "args": ["./assistant_server.py"]}
                }
            }
            ```
        """
        super().__init__(*args, **kwargs)
        self.config = config

    async def list_tools(self):
        """List the tools available on the MCP server."""
        fastmcp_client = fastmcp.Client(self.config)
        async with fastmcp_client:
            return await fastmcp_client.list_tools()
    
    async def list_resources(self):
        """List the resources available on the MCP server."""
        fastmcp_client = fastmcp.Client(self.config)
        async with fastmcp_client:
            return await fastmcp_client.list_resources()
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the MCP server."""
        fastmcp_client = fastmcp.Client(self.config)
        async with fastmcp_client:
            result = await fastmcp_client.call_tool(tool_name, arguments)
            return result.structured_content["result"]
        
    def get_all_schema(self):
        """Get the function call-like schema for all the tools
        available on the MCP server.
        """
        tools = asyncio.run(self.list_tools())
        schemas = []
        for tool in tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.inputSchema["properties"],
                        "required": tool.inputSchema["required"]
                    }
                }
            }
            schemas.append(schema)
        return schemas
    
    def get_all_tool_names(self):
        """Get the names of all the tools available on the MCP server."""
        tools = asyncio.run(self.list_tools())
        return [tool.name for tool in tools]
