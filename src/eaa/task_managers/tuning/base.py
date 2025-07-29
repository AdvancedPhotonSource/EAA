from typing import Optional
import logging

from eaa.tools.imaging.param_tuning import SetParameters
from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool
from eaa.api.llm_config import LLMConfig

logger = logging.getLogger(__name__)


class BaseParameterTuningTaskManager(BaseTaskManager):
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        param_setting_tool: SetParameters = None,
        tools: list[BaseTool] = (),
        initial_parameters: dict[str, float] = None,
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]] = None,
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
        param_setting_tool : SetParameters
            The tool to use to set the parameters.
        initial_parameters : dict[str, float], optional
            The initial parameters given as a dictionary of 
            parameter names and values.
        parameter_ranges : list[tuple[float, ...], tuple[float, ...]]
            The ranges of the parameters. It should be given as a list of
            2 tuples, where the first tuple gives the lower bounds and the
            second tuple gives the upper bounds. The order of the parameters
            should match the order of the initial parameters.
        tools : list[BaseTool], optional
            Other tools provided to the agent.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        """
        self.param_setting_tool: SetParameters = param_setting_tool
        self.initial_parameters: dict[str, float] = initial_parameters
        self.parameter_names = list(initial_parameters.keys())
        self.parameter_ranges = parameter_ranges
        
        super().__init__(
            llm_config=llm_config,
            tools=[param_setting_tool, *tools],
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.initialize_parameter_setting_tool()
        
    def initialize_parameter_setting_tool(self):
        self.param_setting_tool.set_parameters(list(self.initial_parameters.values()))
        
    def prerun_check(self, *args, **kwargs) -> bool:
        if self.initial_parameters is None:
            raise ValueError("initial_parameters must be provided.")
        return super().prerun_check(*args, **kwargs)
        
    def run(self, *args, **kwargs) -> None:
        raise NotImplementedError