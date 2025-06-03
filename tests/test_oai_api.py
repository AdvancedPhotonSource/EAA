
import argparse
import logging
import os

import pytest
import numpy as np

from eaa.agents.openai import OpenAIAgent

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
            {"list_sum": list_sum}
        )
        
        response = agent.receive("Can you sum these numbers: 2, 4, 6, 6, 7?")
        tool_response = agent.handle_tool_call(response)
        if tool_response is not None:
            response = agent.receive(tool_response)
            print(response)
            
    @pytest.mark.local
    def test_openai_api_with_image(self):
        image_path = os.path.join(self.get_ci_input_data_dir(), "simulated_images", "cameraman.png")
        
        agent = OpenAIAgent(
            llm_config={
                "model": "openai/gpt-4o-2024-11-20",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "base_url": "https://openrouter.ai/api/v1",
            },
            system_message="You are a helpful assistant."
        )
        
        response = agent.receive(
            "Can you tell me what is in this image?",
            image_path=image_path
        )
        print(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tester = TestOpenAIAPI()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_openai_api()
    tester.test_openai_api_with_image()
