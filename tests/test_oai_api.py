
import argparse
import logging
import os

import pytest
import numpy as np

from eaa.agents.openai import OpenAIAgent
from eaa.tools.base import ToolReturnType

import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        
        
        agent = OpenAIAgent(
            llm_config={
                "model": "openai/gpt-4o-2024-11-20",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "base_url": "https://openrouter.ai/api/v1",
            },
            system_message="You are a helpful assistant."
        )
        
        agent.register_tools(
            [
                {
                    "name": "list_sum",
                    "function": list_sum,
                    "return_type": ToolReturnType.NUMBER
                }
            ]
        )
        
        # `store_message=True` ensures the user message is saved in the message history.
        response = agent.receive("Can you sum these numbers: 2, 4, 6, 6, 7?", store_message=True)
        tool_response = agent.handle_tool_call(response)
        if tool_response is not None:
            # `store_message=True` ensures the tool response is saved in the message history.
            response = agent.receive(tool_response, role="tool", store_message=True)
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
        
        agent = OpenAIAgent(
            llm_config={
                "model": "openai/gpt-4o-2024-11-20",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "base_url": "https://openrouter.ai/api/v1",
            },
            system_message="You are a helpful assistant."
        )
        
        agent.register_tools(
            [
                {
                    "name": "get_image",
                    "function": get_image,
                    "return_type": ToolReturnType.IMAGE_PATH
                }
            ]
        )
        
        response = agent.receive(
            "Please use your tool to get the image, and tell me what you see.",
        )
        tool_response = agent.handle_tool_call(response)
        if tool_response is not None:
            agent.receive(tool_response, role="tool", request_response=False, store_message=True, store_response=True)
            # Tools are not allowed to return images; it only returns the path to the image.
            # So we follow up with a new message with the image.
            # `store_message=False` ensures the image message is not saved in the message history so it doesn't
            # get repeatedly sent in future conversations which would drive up the cost.
            response = agent.receive(
                "Here is the image the tool returned.",
                image_path=tool_response["content"],
                store_message=False,
                store_response=True
            )
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
