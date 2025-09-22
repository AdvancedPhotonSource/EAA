"""
Example Calculator Tool for demonstrating MCP server functionality.

This module provides a simple calculator tool that can be exposed via MCP.
"""

from typing import Dict, List, Any
import logging

from eaa.tools.base import BaseTool, ToolReturnType, ExposedToolSpec, check

logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    """
    A simple calculator tool for basic arithmetic operations.
    
    This tool demonstrates how to create a BaseTool that can be exposed
    via MCP server for use by AI applications.
    """
    
    name: str = "calculator"
    
    @check
    def __init__(
        self,
        *args,
        require_approval: bool = False,
        **kwargs,
    ):
        """Initialize the calculator tool."""
        super().__init__(*args, require_approval=require_approval, **kwargs)
        
        self.calculation_history: List[str] = []
        
        self.exposed_tools = [
            ExposedToolSpec(
                name="add",
                function=self.add,
                return_type=ToolReturnType.NUMBER,
            ),
            ExposedToolSpec(
                name="subtract",
                function=self.subtract,
                return_type=ToolReturnType.NUMBER,
            ),
            ExposedToolSpec(
                name="multiply",
                function=self.multiply,
                return_type=ToolReturnType.NUMBER,
            ),
            ExposedToolSpec(
                name="divide",
                function=self.divide,
                return_type=ToolReturnType.NUMBER,
            ),
            ExposedToolSpec(
                name="get_history",
                function=self.get_history,
                return_type=ToolReturnType.LIST,
            ),
            ExposedToolSpec(
                name="clear_history",
                function=self.clear_history,
                return_type=ToolReturnType.TEXT,
            ),
        ]
    
    def add(self, a: float, b: float) -> float:
        """
        Add two numbers together.
        
        Parameters
        ----------
        a : float
            The first number.
        b : float
            The second number.
            
        Returns
        -------
        float
            The sum of a and b.
        """
        result = a + b
        self.calculation_history.append(f"{a} + {b} = {result}")
        logger.info(f"Added {a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """
        Subtract the second number from the first.
        
        Parameters
        ----------
        a : float
            The first number (minuend).
        b : float
            The second number (subtrahend).
            
        Returns
        -------
        float
            The difference of a and b.
        """
        result = a - b
        self.calculation_history.append(f"{a} - {b} = {result}")
        logger.info(f"Subtracted {a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """
        Multiply two numbers together.
        
        Parameters
        ----------
        a : float
            The first number.
        b : float
            The second number.
            
        Returns
        -------
        float
            The product of a and b.
        """
        result = a * b
        self.calculation_history.append(f"{a} * {b} = {result}")
        logger.info(f"Multiplied {a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """
        Divide the first number by the second.
        
        Parameters
        ----------
        a : float
            The dividend.
        b : float
            The divisor.
            
        Returns
        -------
        float
            The quotient of a and b.
            
        Raises
        ------
        ValueError
            If b is zero (division by zero).
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        
        result = a / b
        self.calculation_history.append(f"{a} / {b} = {result}")
        logger.info(f"Divided {a} / {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """
        Get the calculation history.
        
        Returns
        -------
        List[str]
            List of all calculations performed.
        """
        logger.info(f"Retrieved calculation history with {len(self.calculation_history)} entries")
        return self.calculation_history.copy()
    
    def clear_history(self) -> str:
        """
        Clear the calculation history.
        
        Returns
        -------
        str
            Confirmation message.
        """
        count = len(self.calculation_history)
        self.calculation_history.clear()
        message = f"Cleared {count} calculations from history"
        logger.info(message)
        return message 
