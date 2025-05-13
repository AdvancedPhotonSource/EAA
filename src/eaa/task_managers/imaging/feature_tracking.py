from typing import Literal
from textwrap import dedent

from eaa.tools.base import BaseTool
from eaa.task_managers.imaging.base import ImagingBaseTaskManager


class FeatureTrackingTaskManager(ImagingBaseTaskManager):
    
    def __init__(
        self,
        model_name: str = "gpt-4o", 
        model_base_url: str = None,
        tools: list[BaseTool] = [],
        speaker_selection_method: Literal["round_robin", "random", "auto"] = "auto",
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
        speaker_selection_method : Literal["round_robin", "random", "auto"], optional
            The method to select the next speaker in the group chat.
            - "round_robin": select the next speaker in a round-robin fashion.
            - "random": select the next speaker randomly.
            - "auto": let the LLM decide the next speaker. Some models might have issues
              with suggesting the speaker in the right format when used as the group chat
              manager. In that case, use "round_robin" or "random" instead.
        """        
        super().__init__(
            model_name=model_name, 
            model_base_url=model_base_url,
            tools=tools, 
            speaker_selection_method=speaker_selection_method,
            *args, **kwargs
        )
        
    def run_fov_search(
        self,
        feature_description: str,
        y_range: tuple[float, float],
        x_range: tuple[float, float],
        fov_size: tuple[float, float] = (100, 100),
        step_size: tuple[float, float] = (100, 100),
        *args, **kwargs
    ) -> None:
        """Run a search for the best field of view for the microscope.
        
        Parameters
        ----------
        feature_description : str
            A text description of the feature to search for.
        y_range : tuple[float, float]
            The range of y coordinates to search for the feature.
        x_range : tuple[float, float]
            The range of x coordinates to search for the feature.
        fov_size : tuple[float, float], optional
            The size of the field of view.
        step_size : float, optional
            The step size to move the field of view each time.
        """
        message = dedent(f"""\
            You are given a tool that acquires an image of a sub-region
            of a sample at given location and with given size (the field
            of view, or FOV). Use this tool to find a subregion that contains
            the following feature: {feature_description}.
            The feature should be roughly centered in the field of view. 
            The field of view size should always be {fov_size}. Start from 
            position (y={y_range[0]}, x={x_range[0]}), and gradually move the FOV 
            with a step size of {step_size[0]} in the y direction and 
            {step_size[1]} in the x direction, and examine the image until you 
            find the feature of interest. Positions should not go beyond 
            y={y_range[1]} and x={x_range[1]}. 
            When you find the feature of interest, report the coordinates of 
            the FOV. When calling tools, make only one call at a time. Do not make
            another call before getting the response of a previous one. When you 
            finish the search, say 'TERMINATE'.\
            """
        )
        
        self.agents.user_proxy.initiate_chat(
            self.agents.group_chat_manager,
            message=message
        )
