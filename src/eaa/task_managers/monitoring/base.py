from typing import Optional
import logging
import time
from textwrap import dedent

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool
from eaa.api.llm_config import LLMConfig
from eaa.agents.memory import MemoryManagerConfig

logger = logging.getLogger(__name__)


class BaseMonitoringTaskManager(BaseTaskManager):
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        tools: list[BaseTool] = (),
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args, **kwargs
    ) -> None:
        """An agent that monitors the state of an experiment.

        Parameters
        ----------
        llm_config : LLMConfig
            The configuration for the LLM.
        memory_config : MemoryManagerConfig, optional
            Memory configuration to forward to the underlying agent.
        tools : list[BaseTool]
            The tools to use to monitor the experiment.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        """
        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            tools=tools,
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
        
    def run(
        self,
        time_interval: float,
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None,
    ) -> None:
        """Run the monitoring task.

        Parameters
        ----------
        time_interval : float
            The time interval in seconds between each status check.
        initial_prompt : str, optional
            The prompt to initiate each status check. If provided, this prompt
            will override the default initial prompt.
        additional_prompt : str, optional
            If provided, this prompt will be added to the initial prompt (either
            the default one or the one provided by `initial_prompt`).
        """
        if initial_prompt is None:
            initial_prompt = dedent(
                """\
                You are monitoring the status of an experiment. You are given
                a set of tools to query the status of various aspects of the
                experiment. Use the tools to check if everything is right. If
                something looks abnormal, either attempt to fix it or alert the
                user. 
                
                Use proper trigger words in your response in the following
                scenarios:
                - You have checked all the statuses and everything is right - add \"TERMINATE\".
                - Something is wrong, but you have fixed it - add \"TERMINATE\".
                - Something is wrong, and you need immediate human input - add \"NEED HUMAN\".
                """
            )
            
        if additional_prompt is not None:
            initial_prompt += "\n" + additional_prompt
        
        while True:
            self.run_once(initial_prompt=initial_prompt)
            time.sleep(time_interval)
    
    def run_once(self, initial_prompt: str) -> None:
        """Run a single status check.
        """
        return self.run_feedback_loop(
            initial_prompt=initial_prompt,
            termination_behavior="return",
        )
