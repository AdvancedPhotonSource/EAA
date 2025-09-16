from typing import Optional, Callable, Literal
from textwrap import dedent

from eaa.tools.base import BaseTool
from eaa.tools.imaging.acquisition import AcquireImage
from eaa.tools.imaging.registration import ImageRegistration
from eaa.task_managers.imaging.base import ImagingBaseTaskManager
from eaa.api.llm_config import LLMConfig
from eaa.util import get_image_path_from_text


class FeatureTrackingTaskManager(ImagingBaseTaskManager):
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        image_acquisition_tool: AcquireImage = None,
        image_registration_tool: ImageRegistration = None,
        additional_tools: list[BaseTool] = (), 
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
        image_acquisition_tool : AcquireImage
            The tool to use to acquire images.
        image_registration_tool : ImageRegistration
            The tool to use to register images.
        additional_tools : list[BaseTool]
            Additional tools provided to the agent (not including the
            image acquisition tool and the image registration tool).
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        build : bool
            Whether to build the internal state of the task manager.
        """
        if image_acquisition_tool is None:
            raise ValueError("image_acquisition_tool must be provided.")
        
        self.image_acquisition_tool = image_acquisition_tool
        self.registration_tool = image_registration_tool
        
        self.last_acquisition_count_stitched = 0
        
        tools = []
        for t in [image_acquisition_tool, image_registration_tool, *additional_tools]:
            if t is not None:
                tools.append(t)
        super().__init__(
            llm_config=llm_config,
            tools=tools, 
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
        
    def image_path_tool_response_hook_factory(
        self,
        add_reference_image_to_images_acquired: bool,
        reference_image_path: str
    ) -> Callable:
        """Factory function that returns a hook function for the image path tool response.
        """
        def hook_function(image_path: str) -> None:
            message = ""
            if (
                add_reference_image_to_images_acquired
                and self.image_acquisition_tool.counter_acquire_image > self.last_acquisition_count_stitched
                ):
                image_path = self.add_reference_image_to_images_acquired(
                    image_path, reference_image_path
                )
                self.last_acquisition_count_stitched = self.image_acquisition_tool.counter_acquire_image
                message = (
                    "Here is the new image (left). "
                    "The reference image (right) is also shown for your reference."
                )
            
            response, outgoing = self.agent.receive(
                message,
                image_path=image_path,
                context=self.context,
                return_outgoing_message=True
            )
            return response, outgoing
        
        if not add_reference_image_to_images_acquired:
            return None
        else:
            return hook_function
        
    def run_fov_search(
        self,
        feature_description: str = None,
        y_range: tuple[float, float] = None,
        x_range: tuple[float, float] = None,
        fov_size: tuple[float, float] = None,
        step_size: tuple[float, float] = None,
        max_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_past_images_to_keep_in_context: Optional[int] = None,
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
        n_first_images_to_keep_in_context, n_past_images_to_keep_in_context : int, optional
            The number of first and last images to keep in the context. If both of
            them are None, all images will be kept.
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

        self.run_feedback_loop(
            initial_prompt=initial_prompt,
            initial_image_path=None,
            message_with_acquired_image=dedent("""\
                Here is the image the tool returned. If the feature is there, 
                report the coordinates of the FOV and include 'TERMINATE' in
                your response. Otherwise, continue to call tools to run the search.
                Include a brief description of what you see in the image in your response.
                """
            ),
            max_rounds=max_rounds,
            n_first_images_to_keep_in_context=n_first_images_to_keep_in_context,
            n_past_images_to_keep_in_context=n_past_images_to_keep_in_context
        )

    def run_feature_tracking(
        self,
        reference_image_path: Optional[str] = None,
        initial_position: Optional[tuple[float, float]] = None,
        initial_fov_size: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        x_range: Optional[tuple[float, float]] = None,
        add_reference_image_to_images_acquired: bool = False,
        max_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_past_images_to_keep_in_context: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None,
        termination_behavior: Literal["ask", "return"] = "ask"
    ):
        """Search for a feature that drifted out of the field of view
        given a reference image of it, and bring the feature back into
        the field of view.
        
        Parameters
        ----------
        reference_image_path : str
            The path to the reference image containing the feature
            to look for. You can also leave this argument as None and
            provide the reference image in terminal or WebUI when prompted.
        initial_position : tuple[float, float], optional
            The initial position of the field of view.
        initial_fov_size : tuple[float, float], optional
            The size of the initial field of view.
        y_range : tuple[float, float], optional
            The range of y coordinates to search for the feature.
        x_range : tuple[float, float], optional
            The range of x coordinates to search for the feature.
        add_reference_image_to_images_acquired : bool, optional
            If True, the reference image will be stitched side-by-side with
            2D microscopy images acquired. This allows the agent to always see
            the reference image in new messages when needed, instead of having
            the reference image only in the first message in the context. This
            may be particularly useful for inference endpoint providers that do
            not support images in the context.
        max_rounds : int, optional
            The maximum number of rounds to search for the feature.
        n_first_images_to_keep_in_context, n_past_images_to_keep_in_context : int, optional
            The number of first and last images to keep in the context. If both of
            them are None, all images will be kept.
        initial_prompt : str, optional
            If given, this prompt will override the default prompt to
            be used as the initial message to the agent.
            `initial_position`, `initial_fov_size`, `y_range`, and `x_range`
            should not be provided if this is given.
        additional_prompt : str, optional
            Additional instructions to the agent.
        termination_behavior : Literal["ask", "return"], optional
            Decides what to do when the agent sends termination signal ("TERMINATE")
            in the response. If "ask", the user will be asked to provide further
            instructions. If "return", the function will return directly.
        """
        if reference_image_path is None:
            user_image_input = self.get_user_input(
                prompt="Please provide the reference image as: <img /path/to/image.png>.",
                display_prompt_in_webui=True
            )
            reference_image_path = get_image_path_from_text(user_image_input)
        
        if initial_prompt is None:
            initial_prompt = dedent(f"""\
                You are given a reference image of a field of view (FOV) of a
                microscope. After that reference image was collected, the
                imaging system drifted, so the FOV at the same location is
                now at a different location. Use the tool to acquire
                images at different places to find that feature again,
                and bring it back to the same location of the field of view
                as the reference image. Try zooming out the field of view,
                and move around to find the feature, then zoom back in.
                The new image you acquire might be blurrier than the reference.
                
                - For your first attempt, start from the initial position of
                  x = {initial_position[1]}, y = {initial_position[0]}, and
                  use an initial field of view size of 
                  {initial_fov_size[1]} in x and {initial_fov_size[0]} in y.
                
                - After acquiring the first image, try zooming out and moving
                  the FOV around by calling the acquisition tool with different
                  locations and sizes/step sizes until the acquired image has
                  substantial overlap with the reference image.
                
                - When the acquired image has substantial overlap with the
                  reference, change the FOV size back to the initial size
                  **while keeping the location of the FOV the same as your
                  latest acquisition**.
                
                - Adjust the final field of view so that the final image you
                  see is as aligned with the reference image as possible.
                  If you have an image registration tool, use it to perform
                  the precise alignment. However, only use you tool if you see
                  **a large amount of overlap between the last acquired image
                  and the reference**, otherwise registration will not be accurate.
                  When calling the registration tool, always set `register_with`
                  to `"reference"`. The tool does not need you to collect any
                  reference or baseline images; they are already provided to the
                  tool. The offset returned by the registration tool should be **added**
                  to the positions of the image acquisition tool. For example,
                  if the last image is acquired at (y = 100, x = 100), and the
                  if the registration tool returns an offset of (dy, dx),
                  the next image should be acquired at (y = 100 + dy, x = 100 + dx).
                
                - After the last acquired image is aligned with the reference, 
                  report the coordinates of the field of view and include 'TERMINATE'
                  in your response.
                
                Other notes:
                
                - Unless specifically noted, all coordinates are given in
                  (y, x) order. When writing coordinates in your response,
                  do not just write two numbers; instead, explicitly specify
                  y/x axis and write them as (y = <y>, x = <x>).
                  
                - Always explain your actions when calling a tool.
                
                - Make sure you only make one tool call at a time. Do not make
                  multiple calls simultaneously.
                  
                - Do NOT acquire images at the initial location over and over again!
                  You do NOT need to acquire baseline images for registration.
                  The registration tool, if available, already has the reference
                  image.
                  
                - When the acquired image looks well aligned with the reference,
                  stop the process by adding 'TERMINATE' to your response.
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

        self.run_feedback_loop(
            initial_prompt=initial_prompt,
            initial_image_path=reference_image_path,
            max_rounds=max_rounds,
            n_first_images_to_keep_in_context=n_first_images_to_keep_in_context,
            n_past_images_to_keep_in_context=n_past_images_to_keep_in_context,
            allow_non_image_tool_responses=True,
            hook_functions={
                "image_path_tool_response": self.image_path_tool_response_hook_factory(
                    add_reference_image_to_images_acquired,
                    reference_image_path
                )
            },
            termination_behavior=termination_behavior
        )
