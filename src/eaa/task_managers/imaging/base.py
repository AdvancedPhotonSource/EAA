from textwrap import dedent
import logging

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool
from eaa.comms import get_api_key
from eaa.agents.openai import OpenAIAgent

logger = logging.getLogger(__name__)


class ImagingBaseTaskManager(BaseTaskManager):
        
    assistant_system_message = dedent(
        """\
        You are helping scientists at a microscopy facility to
        to calibrate their imaging system and set up their experiments.
        You are given the tools that adjust the imaging system, move
        the sample stage, and acquire images.
        When using tools, only make one call at a time. Do not make 
        multiple calls simultaneously.\
        """
    )
    
    def __init__(
        self,
        model_name: str = "gpt-4o", 
        model_base_url: str = None,
        tools: list[BaseTool] = [],
        *args, **kwargs
    ) -> None:
        """An agent that searches for the best setup parameters
        for an imaging system.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use.
        model_base_url : str, optional
            The base URL of the model. This is only needed for self-hosted models.
        tools : list[BaseTool], optional
            A list of tools given to the agent.
        """        
        super().__init__(
            model_name=model_name, 
            model_base_url=model_base_url,
            tools=tools, 
            *args, **kwargs
        )

    def build_agent(self, *args, **kwargs) -> None:
        """Build the assistant(s)."""
        llm_config = self.get_llm_config(*args, **kwargs)
        self.agent = OpenAIAgent(
            llm_config=llm_config,
            system_message=self.assistant_system_message,
        )
        
    def get_llm_config(self, *args, **kwargs):
        llm_config = {
            "model": self.model,
            "api_key": get_api_key(self.model, self.model_base_url),
        }
        if self.model_base_url:
            llm_config["base_url"] = self.model_base_url
        return llm_config
    
    def build_tools(self, *args, **kwargs):
        self.register_tools(self.tools)
            
    def prerun_check(self, *args, **kwargs) -> bool:
        if len(self.agent.tool_manager.tools) == 0:
            logger.warning("No tools registered for the main agent.")
        return super().prerun_check(*args, **kwargs)
        
    def run(self, *args, **kwargs) -> None:
        """Run the task manager."""
        super().run(*args, **kwargs)
        
        self.run_fov_search(*args, **kwargs)
        
    def run_conversation(self, *args, **kwargs) -> None:
        """Run a conversation with the assistant."""
        message = input("Enter a message: ")
        while True:
            if message.lower() == "exit":
                break
            _ = self.agent.receive(message, store_message=True, store_response=True)
            message = input("Enter a message: ")
