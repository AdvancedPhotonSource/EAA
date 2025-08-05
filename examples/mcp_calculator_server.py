#!/usr/bin/env python3
"""
Example MCP Server using CalculatorTool.

This script demonstrates how to create and run an MCP server that exposes
BaseTool methods as MCP tools for use by AI applications like Cursor.

Usage:
    python examples/mcp_calculator_server.py

The server will expose the calculator tool methods (add, subtract, multiply, 
divide, get_history, clear_history) as MCP tools.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eaa.tools.example_calculator import CalculatorTool
from eaa.mcp import run_mcp_server_from_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run the MCP calculator server."""
    try:
        # Create the calculator tool
        calculator = CalculatorTool()
        
        logger.info("Created calculator tool with the following methods:")
        for tool_dict in calculator.exposed_tools:
            logger.info(f"  - {tool_dict['name']}: {tool_dict['function'].__doc__.split('.')[0] if tool_dict['function'].__doc__ else 'No description'}")
        
        # Create and run the MCP server
        logger.info("Starting MCP server...")
        run_mcp_server_from_tools(
            tools=calculator,
            server_name="Calculator MCP Server",
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        raise


if __name__ == "__main__":
    print("""
    Calculator MCP Server Example
    ============================
    
    This server exposes calculator operations as MCP tools:
    - add(a, b): Add two numbers
    - subtract(a, b): Subtract two numbers  
    - multiply(a, b): Multiply two numbers
    - divide(a, b): Divide two numbers
    - get_history(): Get calculation history
    - clear_history(): Clear calculation history
    
    Connect this server to your MCP client (like Cursor) to use these tools
    in AI conversations.
    
    Press Ctrl+C to stop the server.
    """)
    
    main() 