from typing import Literal, Optional

from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.tool.base import BaseTool
from eaa_core.util import get_image_path_from_text
from eaa_imaging.task_manager.imaging.base import ImagingBaseTaskManager
from eaa_imaging.tool.imaging.acquisition import AcquireImage
from eaa_imaging.tool.imaging.registration import ImageRegistration


def initialize_feature_tracking_task_manager(
    task_manager: ImagingBaseTaskManager,
    *,
    llm_config: LLMConfig = None,
    memory_config: Optional[MemoryManagerConfig] = None,
    image_acquisition_tool: AcquireImage = None,
    image_registration_tool: ImageRegistration = None,
    additional_tools: list[BaseTool] = (),
    session_db_path: Optional[str] = "session.sqlite",
    build: bool = True,
    args: tuple = (),
    kwargs: Optional[dict] = None,
) -> None:
    """Initialize common imaging state for ROI-search and feature tracking.

    Parameters
    ----------
    task_manager : ImagingBaseTaskManager
        Task manager instance being initialized.
    llm_config : LLMConfig, optional
        Configuration for the language model.
    memory_config : MemoryManagerConfig, optional
        Long-term memory configuration.
    image_acquisition_tool : AcquireImage
        Tool used to acquire microscope images.
    image_registration_tool : ImageRegistration, optional
        Optional registration tool used by the workflow.
    additional_tools : list[BaseTool], optional
        Additional tools to register alongside the imaging tools.
    session_db_path : str, optional
        SQLite path used for transcript persistence.
    build : bool, optional
        Whether to build the task manager immediately.
    args : tuple, optional
        Positional arguments forwarded to ``ImagingBaseTaskManager``.
    kwargs : dict, optional
        Keyword arguments forwarded to ``ImagingBaseTaskManager``.
    """
    if image_acquisition_tool is None:
        raise ValueError("image_acquisition_tool must be provided.")

    task_manager.image_acquisition_tool = image_acquisition_tool
    task_manager.registration_tool = image_registration_tool
    task_manager.last_acquisition_count_stitched = 0

    tools = [
        tool
        for tool in [image_acquisition_tool, image_registration_tool, *additional_tools]
        if tool is not None
    ]
    ImagingBaseTaskManager.__init__(
        task_manager,
        llm_config=llm_config,
        memory_config=memory_config,
        tools=tools,
        session_db_path=session_db_path,
        build=build,
        *(args or ()),
        **(kwargs or {}),
    )

class FeatureTrackingTaskManager(ImagingBaseTaskManager):
    """Track a previously seen feature back into the microscope field of view."""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        image_acquisition_tool: AcquireImage = None,
        image_registration_tool: ImageRegistration = None,
        additional_tools: list[BaseTool] = (),
        session_db_path: Optional[str] = "session.sqlite",
        build: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the feature-tracking task manager.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            Configuration for the language model.
        memory_config : MemoryManagerConfig, optional
            Long-term memory configuration.
        image_acquisition_tool : AcquireImage
            Tool used to acquire microscope images.
        image_registration_tool : ImageRegistration, optional
            Optional registration tool used during feature tracking.
        additional_tools : list[BaseTool], optional
            Additional tools to register alongside the imaging tools.
        session_db_path : str, optional
            SQLite path used for transcript persistence.
        build : bool, optional
            Whether to build the task manager immediately.
        *args
            Positional arguments forwarded to ``ImagingBaseTaskManager``.
        **kwargs
            Keyword arguments forwarded to ``ImagingBaseTaskManager``.
        """
        initialize_feature_tracking_task_manager(
            self,
            llm_config=llm_config,
            memory_config=memory_config,
            image_acquisition_tool=image_acquisition_tool,
            image_registration_tool=image_registration_tool,
            additional_tools=additional_tools,
            session_db_path=session_db_path,
            build=build,
            args=args,
            kwargs=kwargs,
        )

    def run(
        self,
        reference_image_path: Optional[str] = None,
        initial_position: Optional[tuple[float, float]] = None,
        initial_fov_size: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        x_range: Optional[tuple[float, float]] = None,
        add_reference_image_to_images_acquired: bool = False,
        max_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_last_images_to_keep_in_context: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None,
        termination_behavior: Literal["ask", "return"] = "ask",
        max_arounds_reached_behavior: Literal["return", "raise"] = "return",
    ) -> None:
        """Run the feature-tracking workflow.

        Parameters
        ----------
        reference_image_path : str, optional
            Path to the reference image containing the target feature. When
            omitted, the user is prompted to provide an image.
        initial_position : tuple[float, float], optional
            Initial field-of-view center in ``(y, x)`` order.
        initial_fov_size : tuple[float, float], optional
            Initial field-of-view size in ``(height, width)`` order.
        y_range : tuple[float, float], optional
            Search bounds for the vertical stage coordinate.
        x_range : tuple[float, float], optional
            Search bounds for the horizontal stage coordinate.
        add_reference_image_to_images_acquired : bool, optional
            Whether to stitch the reference image onto newly acquired images.
        max_rounds : int, optional
            Maximum number of feedback-loop rounds to allow.
        n_first_images_to_keep_in_context : int, optional
            Number of earliest images to keep in context when pruning.
        n_last_images_to_keep_in_context : int, optional
            Number of latest images to keep in context when pruning.
        initial_prompt : str, optional
            Explicit prompt override. When provided, geometry arguments must be
            omitted.
        additional_prompt : str, optional
            Additional instructions appended to the generated prompt.
        termination_behavior : {"ask", "return"}, optional
            Behavior when the model emits ``TERMINATE``.
        max_arounds_reached_behavior : {"return", "raise"}, optional
            Behavior when the workflow reaches ``max_rounds``.
        """
        self.prerun_check()

        if reference_image_path is None:
            user_image_input = self.get_user_input(
                prompt="Please provide the reference image as: <img /path/to/image.png>.",
                display_prompt_in_webui=True,
            )
            reference_image_path = get_image_path_from_text(user_image_input)

        if initial_prompt is None:
            initial_prompt = (
                f"You are given a reference image of a field of view (FOV) of a "
                f"microscope. After that reference image was collected, the "
                f"imaging system drifted, so the FOV at the same location is "
                f"now at a different location. Use the tool to acquire "
                f"images at different places to find that feature again, "
                f"and bring it back to the same location of the field of view "
                f"as the reference image. Try zooming out the field of view, "
                f"and move around to find the feature, then zoom back in. "
                f"The new image you acquire might be blurrier than the reference.\n\n"
                f"- For your first attempt, start from the initial position of "
                f"x = {initial_position[1]}, y = {initial_position[0]}, and "
                f"use an initial field of view size of "
                f"{initial_fov_size[1]} in x and {initial_fov_size[0]} in y.\n\n"
                f"- After acquiring the first image, try zooming out and moving "
                f"the FOV around by calling the acquisition tool with different "
                f"locations and sizes/step sizes until the acquired image has "
                f"substantial overlap with the reference image.\n\n"
                f"- When the acquired image has substantial overlap with the "
                f"reference, change the FOV size back to the initial size "
                f"**while keeping the location of the FOV the same as your "
                f"latest acquisition**.\n\n"
                f"- Adjust the final field of view so that the final image you "
                f"see is as aligned with the reference image as possible. "
                f"If you have an image registration tool, use it to perform "
                f"the precise alignment. However, only use you tool if you see "
                f"**a large amount of overlap between the last acquired image "
                f"and the reference**, otherwise registration will not be accurate. "
                f"When calling the registration tool, always set `register_with` "
                f'to `"reference"`. The tool does not need you to collect any '
                f"reference or baseline images; they are already provided to the "
                f"tool. The offset returned by the registration tool should be **subtracted** "
                f"to the positions of the image acquisition tool. For example, "
                f"if the last image is acquired at (y = 100, x = 100), and the "
                f"if the registration tool returns an offset of (dy, dx), "
                f"the next image should be acquired at (y = 100 - dy, x = 100 - dx).\n\n"
                f"- After the last acquired image is aligned with the reference, "
                f"report the coordinates of the field of view and include 'TERMINATE' "
                f"in your response.\n\n"
                f"Other notes:\n\n"
                f"- Unless specifically noted, all coordinates are given in "
                f"(y, x) order. When writing coordinates in your response, "
                f"do not just write two numbers; instead, explicitly specify "
                f"y/x axis and write them as (y = <y>, x = <x>).\n\n"
                f"- Always explain your actions when calling a tool.\n\n"
                f"- Make sure you only make one tool call at a time. Do not make "
                f"multiple calls simultaneously.\n\n"
                f"- Do NOT acquire images at the initial location over and over again! "
                f"You do NOT need to acquire baseline images for registration. "
                f"The registration tool, if available, already has the reference "
                f"image.\n\n"
                f"- When the acquired image looks well aligned with the reference, "
                f"stop the process by adding 'TERMINATE' to your response."
            )
        elif any(value is not None for value in [initial_position, initial_fov_size, y_range, x_range]):
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
            n_last_images_to_keep_in_context=n_last_images_to_keep_in_context,
            allow_non_image_tool_responses=True,
            termination_behavior=termination_behavior,
            max_arounds_reached_behavior=max_arounds_reached_behavior,
        )

    def run_from_checkpoint(self, checkpoint_db_path: Optional[str] = None) -> None:
        """Resume the feature-tracking workflow from a checkpoint.

        Parameters
        ----------
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and updates instead of
            ``self.session_db_path``.
        """
        self.prerun_check()
        self.run_feedback_loop_from_checkpoint(checkpoint_db_path=checkpoint_db_path)

__all__ = [
    "initialize_feature_tracking_task_manager",
    "FeatureTrackingTaskManager",
]
