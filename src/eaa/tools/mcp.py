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
        self._client = None
        self._connected = False
        self._loop = asyncio.get_event_loop()

    async def _ensure_connected(self):
        """Ensure the MCP client is connected."""
        if not self._connected or self._client is None:
            await self.connect()

    async def connect(self):
        """Connect to the MCP server."""
        if self._client is not None:
            await self.disconnect()
        
        self._client = fastmcp.Client(self.config)
        await self._client.__aenter__()
        self._connected = True

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self._client is not None and self._connected:
            await self._client.__aexit__(None, None, None)
            self._connected = False
            self._client = None

    async def list_tools(self):
        """List the tools available on the MCP server."""
        await self._ensure_connected()
        return await self._client.list_tools()
    
    async def list_resources(self):
        """List the resources available on the MCP server."""
        await self._ensure_connected()
        return await self._client.list_resources()
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the MCP server."""
        await self._ensure_connected()
        result = await self._client.call_tool(tool_name, arguments)
        return result.structured_content["result"]
        
    def get_all_schema(self):
        """Get the function call-like schema for all the tools
        available on the MCP server.
        """
        tools = self._loop.run_until_complete(self.list_tools())
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
        tools = self._loop.run_until_complete(self.list_tools())
        return [tool.name for tool in tools]

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if self._connected and self._client is not None:
            # Try to clean up the connection, but don't fail if event loop is closed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    loop.create_task(self.disconnect())
                else:
                    # If loop is not running, run cleanup synchronously
                    loop.run_until_complete(self.disconnect())
            except RuntimeError:
                # Event loop might be closed, ignore cleanup
                pass
