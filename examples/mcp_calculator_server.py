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

import logging
import inspect
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eaa.core.mcp.server import run_mcp_server_from_tools
from eaa.tool.example_calculator import CalculatorTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def log_available_tools(calculator: CalculatorTool) -> None:
    """Log the exposed calculator tool methods."""
    logger.info("Created calculator tool with the following methods:")
    for spec in calculator.exposed_tools:
        doc = inspect.getdoc(spec.function) or "No description"
        logger.info("  - %s: %s", spec.name, doc.split(".")[0])


def print_banner() -> None:
    """Print usage information without corrupting stdio MCP transport."""
    print(
        """
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
""",
        file=sys.stderr,
    )


def main() -> None:
    """Main function to run the MCP calculator server."""
    try:
        # Create the calculator tool
        calculator = CalculatorTool()
        log_available_tools(calculator)

        # Create and run the MCP server
        logger.info("Starting MCP server...")
        # To expose this server over HTTP for remote clients instead of stdio, use:
        # run_mcp_server_from_tools(
        #     tools=calculator,
        #     server_name="Calculator MCP Server",
        #     transport="http",
        #     host="0.0.0.0",
        #     port=8000,
        #     path="/mcp",
        # )
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
    print_banner()
    main()
