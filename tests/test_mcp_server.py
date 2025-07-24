#!/usr/bin/env python3
"""
Test script for MCP server functionality.

This script tests the MCPToolServer component with the example calculator tool
to ensure proper tool registration and schema generation.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eaa.tools.example_calculator import CalculatorTool
from eaa.mcp import MCPToolServer, create_mcp_server_from_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_tool_creation():
    """Test that the calculator tool is created correctly."""
    logger.info("Testing tool creation...")
    
    calculator = CalculatorTool()
    
    # Check that exposed_tools is properly set
    assert hasattr(calculator, "exposed_tools"), "Calculator should have exposed_tools attribute"
    assert len(calculator.exposed_tools) > 0, "Calculator should have at least one exposed tool"
    
    # Check expected tools
    tool_names = [tool["name"] for tool in calculator.exposed_tools]
    expected_tools = ["add", "subtract", "multiply", "divide", "get_history", "clear_history"]
    
    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"Expected tool '{expected_tool}' not found"
    
    logger.info(f"✓ Calculator tool created successfully with {len(tool_names)} tools")
    return calculator


def test_server_creation():
    """Test that the MCP server can be created."""
    logger.info("Testing server creation...")
    
    calculator = CalculatorTool()
    server = MCPToolServer(name="Test Calculator Server")
    
    # Register tools
    server.register_tools(calculator)
    
    # Check that tools are registered
    registered_tools = server.list_tools()
    assert len(registered_tools) == len(calculator.exposed_tools), "All tools should be registered"
    
    logger.info(f"✓ Server created successfully with {len(registered_tools)} registered tools")
    return server


def test_tool_schemas():
    """Test that tool schemas are generated correctly."""
    logger.info("Testing tool schema generation...")
    
    calculator = CalculatorTool()
    server = create_mcp_server_from_tools(calculator, "Schema Test Server")
    
    # Get schemas
    schemas = server.get_tool_schemas()
    
    # Check schema structure
    assert len(schemas) > 0, "Should have at least one schema"
    
    for schema in schemas:
        # Check basic schema structure
        assert "type" in schema, "Schema should have 'type' field"
        assert "function" in schema, "Schema should have 'function' field"
        assert schema["type"] == "function", "Schema type should be 'function'"
        
        func_def = schema["function"]
        assert "name" in func_def, "Function should have 'name' field"
        assert "description" in func_def, "Function should have 'description' field"
        assert "parameters" in func_def, "Function should have 'parameters' field"
        
        params = func_def["parameters"]
        assert "type" in params, "Parameters should have 'type' field"
        assert "properties" in params, "Parameters should have 'properties' field"
        assert params["type"] == "object", "Parameters type should be 'object'"
    
    logger.info(f"✓ Generated {len(schemas)} valid tool schemas")
    
    # Print schema details for debugging
    for schema in schemas:
        func_name = schema["function"]["name"]
        func_desc = schema["function"]["description"]
        logger.info(f"  - {func_name}: {func_desc[:50]}...")
    
    return schemas


def test_tool_execution():
    """Test that tools can be executed correctly."""
    logger.info("Testing tool execution...")
    
    calculator = CalculatorTool()
    
    # Test basic arithmetic
    result = calculator.add(5, 3)
    assert result == 8, f"Expected 8, got {result}"
    
    result = calculator.subtract(10, 4)
    assert result == 6, f"Expected 6, got {result}"
    
    result = calculator.multiply(3, 4)
    assert result == 12, f"Expected 12, got {result}"
    
    result = calculator.divide(15, 3)
    assert result == 5, f"Expected 5, got {result}"
    
    # Test history
    history = calculator.get_history()
    assert len(history) == 4, f"Expected 4 history entries, got {len(history)}"
    
    # Test clear history
    clear_msg = calculator.clear_history()
    assert "Cleared" in clear_msg, "Clear message should contain 'Cleared'"
    
    history = calculator.get_history()
    assert len(history) == 0, f"Expected empty history, got {len(history)}"
    
    # Test error handling
    try:
        calculator.divide(5, 0)
        assert False, "Division by zero should raise an error"
    except ValueError as e:
        assert "Cannot divide by zero" in str(e), "Expected division by zero error"
    
    logger.info("✓ All tool executions completed successfully")


def test_multiple_tools():
    """Test server with multiple tool instances."""
    logger.info("Testing multiple tool instances...")
    
    calc1 = CalculatorTool()
    calc2 = CalculatorTool()  # Different instance
    
    # This should fail due to naming conflicts
    server = MCPToolServer(name="Multi-Tool Test")
    try:
        server.register_tools([calc1, calc2])
        assert False, "Should have failed due to naming conflicts"
    except ValueError as e:
        assert "already registered" in str(e), "Expected naming conflict error"
        logger.info("✓ Correctly detected naming conflicts")


def run_all_tests():
    """Run all tests."""
    logger.info("Starting MCP server tests...")
    
    try:
        # Test individual components
        calculator = test_tool_creation()
        server = test_server_creation()
        schemas = test_tool_schemas()
        test_tool_execution()
        test_multiple_tools()
        
        logger.info("=" * 50)
        logger.info("✓ All tests passed successfully!")
        logger.info("=" * 50)
        
        # Print summary
        logger.info("Test Summary:")
        logger.info(f"  - Tool creation: ✓")
        logger.info(f"  - Server creation: ✓")
        logger.info(f"  - Schema generation: ✓ ({len(schemas)} schemas)")
        logger.info(f"  - Tool execution: ✓")
        logger.info(f"  - Error handling: ✓")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n🎉 All tests passed! The MCP server component is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs above.")
        sys.exit(1) 