from typing import Optional, Literal
from textwrap import dedent

from eaa.tools.base import BaseTool
from eaa.task_managers.imaging.base import ImagingBaseTaskManager
from eaa.api.llm_config import LLMConfig


class FeatureTrackingTaskManager(ImagingBaseTaskManager):
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        tools: list[BaseTool] = [], 
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args, **kwargs
    ) -> None:
        """An agent that searches for a described feature in the sample,
        or tracks the position of a feature when the FOV drifts.

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
        
    def run_fov_search(
        self,
        feature_description: str = None,
        y_range: tuple[float, float] = None,
        x_range: tuple[float, float] = None,
        fov_size: tuple[float, float] = None,
        step_size: tuple[float, float] = None,
        max_rounds: int = 99,
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None,
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
            The size of the field of view in (height, width).
        step_size : float, optional
            The step size to move the field of view each time (dy, dx).
        max_rounds : int, optional
            The maximum number of rounds to search for the feature.
        initial_prompt : str, optional
            If given, this prompt will override the default prompt to
            be used as the initial message to the agent. `feature_description`,
            `y_range`, `x_range`, `fov_size`, and `step_size` should not be
            provided if this is given.
        """
        if initial_prompt is None:
            initial_prompt = dedent(f"""\
                You are given a tool that acquires an image of a sub-region
                of a sample at given location and with given size (the field
                of view, or FOV). Each time you call the tool, you will see
                the image acquired. Use this tool to find a subregion that contains
                the following feature: {feature_description}.
                The feature should be centered in the field of view. Each time you
                see an acquired image, check if it is in the FOV; if not, move the
                FOV until you find it.
                Here are your detailed instructions:
                - At the beginning, use an FOV size of {fov_size} (height, width). 
                  You can change the FOV size during the process to see a larger area,
                  but go back to this size when you find the feature and acquire a 
                  final image of it.
                - Start from position (y={y_range[0]}, x={x_range[0]}), and gradually
                  move the FOV to find the feature. Positions should not go beyond
                  y={y_range[1]} and x={x_range[1]}. When moving the FOV, you can start
                  with the step size of {step_size[0]} in the y direction and {step_size[1]}
                  in the x direction. You can change the step sizes during the process.
                - When you find the feature, adjust the positions of the FOV to make the
                  feature centered in the FOV. If the feature is off to the left, move
                  the FOV to the left; if the feature is off to the top, move the FOV
                  to the top.
                - When you find the feature of interest, report the coordinates of the
                  FOV. When calling tools, make only one call at a time. Do not make
                  another call before getting the response of a previous one. When you 
                  finish the search or need user response, say 'TERMINATE'.\
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
        
        if additional_prompt is not None:
            initial_prompt = initial_prompt + "\nAdditional instructions:\n" + additional_prompt

        self.run_imaging_feedback_loop(
            initial_prompt=initial_prompt,
            initial_image_path=None,
            message_with_acquired_image=dedent("""\
                Here is the image the tool returned. If the feature is there, 
                report the coordinates of the FOV and include 'TERMINATE' in
                your response. Otherwise, continue to call tools to run the search.
                Include a brief description of what you see in the image in your response.
                """
            ),
            store_all_images_in_context=False,
            max_rounds=max_rounds
        )

    def run_feature_tracking(
        self,
        reference_image_path: str,
        initial_position: Optional[tuple[float, float]] = None,
        initial_fov_size: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        x_range: Optional[tuple[float, float]] = None,
        max_rounds: int = 99,
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None
    ):
        """Search for a feature that drifted out of the field of view
        given a reference image of it, and bring the feature back into
        the field of view.
        
        Parameters
        ----------
        reference_image_path : str
            The path to the reference image containing the feature
            to look for.
        initial_position : tuple[float, float], optional
            The initial position of the field of view.
        initial_fov_size : tuple[float, float], optional
            The size of the initial field of view.
        y_range : tuple[float, float], optional
            The range of y coordinates to search for the feature.
        x_range : tuple[float, float], optional
            The range of x coordinates to search for the feature.
        max_rounds : int, optional
            The maximum number of rounds to search for the feature.
        initial_prompt : str, optional
            If given, this prompt will override the default prompt to
            be used as the initial message to the agent.
            `initial_position`, `initial_fov_size`, `y_range`, and `x_range`
            should not be provided if this is given.
        """
        if initial_prompt is None:
            initial_prompt = dedent(f"""\
                You are given an image of a feature in the sample that
                was previously captured by the microscope. The feature
                drifted out of the field of view. Use the tool to acquire
                images at different places to find that feature again,
                and bring it back to the same location of the field of view
                as the reference image. Try zooming out the field of view,
                and move around to find the feature, then zoom back in.
                The new image you acquire might be blurrier than the reference.
                
                For your first attempt, start from the initial position of
                x = {initial_position[1]}, y = {initial_position[0]}, and
                use an initial field of view size of 
                {initial_fov_size[1]} in x and {initial_fov_size[0]} in y.
                
                After you find the feature, report the coordinates of the
                field of view and include 'TERMINATE' in your response.
                
                Make sure you only make one tool call at a time. Do not make
                multiple calls simultaneously.
                """
            )
        else:
            if (
                initial_position is not None or
                initial_fov_size is not None or
                y_range is not None or
                x_range is not None
            ):
                raise ValueError(
                    "`initial_position`, `initial_fov_size`, `y_range`, and `x_range` "
                    "should not be provided if `initial_prompt` is given."
                )

        if additional_prompt is not None:
            initial_prompt = initial_prompt + "\nAdditional instructions:\n" + additional_prompt

        self.run_imaging_feedback_loop(
            initial_prompt=initial_prompt,
            initial_image_path=reference_image_path,
            store_all_images_in_context=False,
            max_rounds=max_rounds
        )
