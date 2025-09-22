
import argparse
import logging
import os
from typing import Callable

import pytest
import numpy as np

from eaa.agents.base import print_message
from eaa.agents.openai import OpenAIAgent
from eaa.tools.base import BaseTool, ToolReturnType, check

import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_function_tool(name: str, func: Callable, return_type: ToolReturnType) -> BaseTool:
    class _FunctionTool(BaseTool):
        tool_name = name

        @check
        def __init__(self):
            super().__init__()
            self.exposed_tools = [
                {
                    "name": name,
                    "function": func,
                    "return_type": return_type,
                }
            ]

    _FunctionTool.name = f"{name}_wrapper"
    return _FunctionTool()


class TestOpenAIAPI(tutils.BaseTester):
    
    @pytest.mark.local
    def test_openai_api(self):
        
        def list_sum(numbers: list[float]) -> float:
            """
            Sum all the numbers in the list.
            
            Parameters
            ----------
            numbers : list[float]
                The list of numbers to sum.
                
            Returns
            -------
            float
                The sum of the numbers in the list.
            """
            return np.sum(numbers)
        
        
        context = []
        
        agent = OpenAIAgent(
            llm_config={
                "model": "gpt-4o-2024-11-20",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": "https://api.openai.com/v1",
            },
            system_message="You are a helpful assistant."
        )
        
        agent.register_tools([
            build_function_tool("list_sum", list_sum, ToolReturnType.NUMBER)
        ])
        
        # `store_message=True` ensures the user message is saved in the message history.
        response, outgoing = agent.receive("Can you sum these numbers: 2, 4, 6, 6, 7?", context=context, return_outgoing_message=True)
        context.append(outgoing)
        context.append(response)
        tool_responses = agent.handle_tool_call(response)
        if len(tool_responses) > 0:
            response, outgoing = agent.receive(tool_responses[0], role="tool", context=context, return_outgoing_message=True)
            context.append(outgoing)
            context.append(response)
            print(response)
            
    @pytest.mark.local
    def test_openai_api_with_image(self):
        image_path = os.path.join(self.get_ci_input_data_dir(), "simulated_images", "cameraman.png")
        
        def get_image() -> str:
            """Get the acquired image.

            Returns
            -------
            str
                The acquired image.
            """
            return image_path
        
        context = []
        
        agent = OpenAIAgent(
            llm_config={
                "model": "gpt-4o-2024-11-20",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": "https://api.openai.com/v1",
            },
            system_message="You are a helpful assistant."
        )
        
        agent.register_tools([
            build_function_tool("get_image", get_image, ToolReturnType.IMAGE_PATH)
        ])
        
        response, outgoing = agent.receive(
            "Please use your tool to get the image, and tell me what you see.",
            context=context,
            return_outgoing_message=True
        )
        context.append(outgoing)
        context.append(response)
        tool_responses = agent.handle_tool_call(response)
        if len(tool_responses) > 0:
            print_message(tool_responses[0])
            context.append(tool_responses[0])
            # Tools are not allowed to return images; it only returns the path to the image.
            # So we follow up with a new message with the image.
            response, outgoing = agent.receive(
                "Here is the image the tool returned.",
                image_path=tool_responses[0]["content"],
                context=context,
                return_outgoing_message=True
            )
            context.append(outgoing)
            context.append(response)
            print(response)
        else:
            raise ValueError("Tool response is None.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tester = TestOpenAIAPI()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_openai_api()
    tester.test_openai_api_with_image()
