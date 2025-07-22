from textwrap import dedent
import logging
from typing import Optional

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool
from eaa.api.llm_config import LLMConfig

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
        llm_config: LLMConfig = None,
        tools: list[BaseTool] = (), 
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args, **kwargs
    ) -> None:
        """An agent that searches for the best setup parameters
        for an imaging system.

        Parameters
        ----------
        llm_config : LLMConfig
            The configuration for the LLM.
        tools : list[BaseTool]
            A list of tools provided to the agent.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        build : bool
            Whether to build the internal state of the task manager.
        """        
        super().__init__(
            llm_config=llm_config,
            tools=tools, 
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
    
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
