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
        feature_description: str = None,
        y_range: tuple[float, float] = None,
        x_range: tuple[float, float] = None,
        fov_size: tuple[float, float] = (100, 100),
        step_size: tuple[float, float] = (100, 100),
        max_rounds: int = 99,
        initial_prompt: str = None,
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
        max_rounds : int, optional
            The maximum number of rounds to search for the feature.
        initial_prompt : str, optional
            If given, this prompt will override the default prompt to
            be used as the initial message to the agent. `feature_description`,
            `y_range`, `x_range`, `fov_size`, and `step_size` should not be
            provided if this is given.
        """
        if initial_prompt is None:
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
        else:
            if (
                feature_description is not None or
                y_range is not None or
                x_range is not None or
                fov_size is not None or
                step_size is not None
            ):
                raise ValueError(
                    "`feature_description`, `y_range`, `x_range`, `fov_size`, and `step_size` "
                    "should not be provided if `initial_prompt` is given."
                )
            message = initial_prompt
        
        round = 0
        image_path = None
        while round < max_rounds:
            # Messages after the first one contain images so we don't store them.
            # However, AI responses contain tool calls so we need to store them.
            response = self.agent.receive(
                message,
                image_path=image_path,
                store_message=round == 0, 
                store_response=True
            )
            
            if "TERMINATE" in response["content"]:
                message = input(
                    "Termination condition triggered. What to do next? Type \"exit\" to exit. "
                )
                if message.lower() == "exit":
                    return
                else:
                    continue
            
            tool_response = self.agent.handle_tool_call(response)
            if tool_response:
                self.agent.receive(
                    tool_response, 
                    role="tool", 
                    request_response=False,
                    store_message=True, 
                    store_response=True
                )
                message = dedent("""\
                    Here is the image the tool returned. If the feature is there, 
                    report the coordinates of the FOV and include 'TERMINATE' in
                    your response. Otherwise, continue to call tools to run the search.
                    Include a brief description of what you see in the image in your response.
                    """
                )
                image_path = tool_response["content"]
            else:
                message = ""
                image_path = None
            
            round += 1

