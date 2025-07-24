# MCP Server for BaseTool Integration

This guide explains how to use the MCP (Model Context Protocol) server component to expose any `BaseTool` subclass as MCP tools for use by AI applications like Cursor.

## Overview

The MCP server component automatically converts `BaseTool` subclasses into MCP-compatible tools, allowing AI applications to call the tool methods directly through the standardized MCP protocol.

## Key Components

### MCPToolServer

The main class that creates and manages an MCP server:

```python
from eaa.mcp import MCPToolServer
from eaa.tools.example_calculator import CalculatorTool

# Create a calculator tool
calculator = CalculatorTool()

# Create and configure the MCP server
server = MCPToolServer(name="My Calculator Server")
server.register_tools(calculator)

# Run the server
server.run()
```

### Convenience Functions

For quick setup, use the convenience functions:

```python
from eaa.mcp import run_mcp_server_from_tools
from eaa.tools.example_calculator import CalculatorTool

# Create and run server in one step
calculator = CalculatorTool()
run_mcp_server_from_tools(
    tools=calculator,
    server_name="Calculator MCP Server"
)
```

## Creating Compatible BaseTool Subclasses

To create a `BaseTool` that works with the MCP server:

1. **Inherit from BaseTool**
2. **Define exposed_tools** - List the methods you want to expose
3. **Use proper type annotations** - Required for schema generation
4. **Add docstrings** - Used as tool descriptions

### Example BaseTool Implementation

```python
from typing import Dict, List, Any
from eaa.tools.base import BaseTool, ToolReturnType, check

class MyTool(BaseTool):
    name: str = "my_tool"
    
    @check
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.exposed_tools: List[Dict[str, Any]] = [
            {
                "name": "calculate_something",
                "function": self.calculate_something,
                "return_type": ToolReturnType.NUMBER
            },
            {
                "name": "get_status",
                "function": self.get_status,
                "return_type": ToolReturnType.TEXT
            }
        ]
    
    def calculate_something(self, x: float, y: float) -> float:
        """
        Calculate something with two numbers.
        
        Parameters
        ----------
        x : float
            First number
        y : float
            Second number
            
        Returns
        -------
        float
            The result of the calculation
        """
        return x * y + 42
    
    def get_status(self) -> str:
        """
        Get the current status of the tool.
        
        Returns
        -------
        str
            Status message
        """
        return "Tool is ready"
```

## Return Types

The MCP server handles different return types automatically:

- `ToolReturnType.TEXT` - Returns as string
- `ToolReturnType.NUMBER` - Returns as float
- `ToolReturnType.BOOL` - Returns as boolean
- `ToolReturnType.LIST` - Returns as list
- `ToolReturnType.DICT` - Returns as dictionary
- `ToolReturnType.IMAGE_PATH` - Returns path as string
- `ToolReturnType.EXCEPTION` - Handled as error

## Running the MCP Server

### Basic Usage

```python
from eaa.tools.example_calculator import CalculatorTool
from eaa.mcp import run_mcp_server_from_tools

# Create tool instance
calculator = CalculatorTool()

# Run server
run_mcp_server_from_tools(
    tools=calculator,
    server_name="Calculator Server"
)
```

### Multiple Tools

```python
from eaa.mcp import MCPToolServer

# Create multiple tools
tool1 = MyTool1()
tool2 = MyTool2()

# Create server with multiple tools
server = MCPToolServer("Multi-Tool Server")
server.register_tools([tool1, tool2])
server.run()
```

### Advanced Configuration

```python
from eaa.mcp import create_mcp_server_from_tools

# Create server without running
server = create_mcp_server_from_tools(
    tools=[tool1, tool2],
    server_name="Advanced Server"
)

# Inspect tool schemas
schemas = server.get_tool_schemas()
print(f"Registered {len(schemas)} tools:")
for schema in schemas:
    print(f"  - {schema['function']['name']}")

# List tool names
tools = server.list_tools()
print(f"Available tools: {', '.join(tools)}")

# Run with custom options
server.run(debug=True)
```

## Connecting to MCP Clients

### Cursor IDE

1. Create your MCP server script
2. Add MCP configuration to Cursor settings
3. Reference your server script in the configuration

Example Cursor MCP configuration:

```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["path/to/your/mcp_server.py"]
    }
  }
}
```

### Other MCP Clients

The server follows the standard MCP protocol and should work with any compatible client.

## Example: Calculator MCP Server

See `examples/mcp_calculator_server.py` for a complete working example:

```bash
# Run the calculator server
python examples/mcp_calculator_server.py
```

This exposes the following tools:
- `add(a, b)` - Add two numbers
- `subtract(a, b)` - Subtract two numbers
- `multiply(a, b)` - Multiply two numbers
- `divide(a, b)` - Divide two numbers
- `get_history()` - Get calculation history
- `clear_history()` - Clear calculation history

## Error Handling

The MCP server automatically handles errors:

- **Missing parameters** - Returns error message
- **Type conversion errors** - Attempts conversion, falls back to string
- **Tool execution errors** - Catches exceptions and returns error message
- **Duplicate tool names** - Raises ValueError during registration

## Schema Generation

Tool schemas are automatically generated from:

- **Function signatures** - Parameter names and types
- **Type annotations** - Converted to JSON schema types
- **Docstrings** - Used as tool descriptions
- **Default values** - Handled for optional parameters

## Best Practices

1. **Use descriptive docstrings** - They become tool descriptions
2. **Add proper type annotations** - Required for schema generation
3. **Handle errors gracefully** - Use try/catch in tool methods
4. **Avoid naming conflicts** - Ensure unique tool names across all tools
5. **Keep tools focused** - One responsibility per tool method
6. **Test tools independently** - Before exposing via MCP

## Troubleshooting

### Common Issues

**ImportError: mcp package not found**
```bash
pip install mcp
```

**Tool not appearing in client**
- Check that `exposed_tools` is properly defined
- Verify type annotations are present
- Ensure docstrings are added

**Naming conflicts**
- Make sure tool names are unique across all registered tools
- Consider namespacing for similar tools

**Parameter errors**
- Verify type annotations match expected types
- Check that required parameters are properly marked

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your MCP server code here
```

Inspect generated schemas:

```python
server = create_mcp_server_from_tools(tools)
schemas = server.get_tool_schemas()
import json
print(json.dumps(schemas, indent=2))
```

## Dependencies

Required packages:
- `mcp` - MCP protocol implementation
- Standard Python packages (inspect, json, logging, typing)

## License

This component follows the same license as the EAA project. 