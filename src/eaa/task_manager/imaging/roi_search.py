from typing import Optional

from eaa.api.llm_config import LLMConfig
from eaa.api.memory import MemoryManagerConfig
from eaa.core.tooling.base import BaseTool
from eaa.task_manager.imaging.base import ImagingBaseTaskManager
from eaa.task_manager.imaging.feature_tracking import initialize_feature_tracking_task_manager
from eaa.tool.imaging.acquisition import AcquireImage
from eaa.tool.imaging.registration import ImageRegistration


class ROISearchTaskManager(ImagingBaseTaskManager):
    """Search for a region of interest using the shared feedback-loop graph."""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        image_acquisition_tool: AcquireImage = None,
        image_registration_tool: ImageRegistration = None,
        additional_tools: list[BaseTool] = (),
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the ROI-search task manager.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            Configuration for the language model.
        memory_config : MemoryManagerConfig, optional
            Long-term memory configuration.
        image_acquisition_tool : AcquireImage
            Tool used to acquire microscope images.
        image_registration_tool : ImageRegistration, optional
            Optional registration tool available during ROI search.
        additional_tools : list[BaseTool], optional
            Additional tools to register alongside the imaging tools.
        message_db_path : str, optional
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
            message_db_path=message_db_path,
            build=build,
            args=args,
            kwargs=kwargs,
        )

    def run(
        self,
        feature_description: str = None,
        y_range: tuple[float, float] = None,
        x_range: tuple[float, float] = None,
        fov_size: tuple[float, float] = None,
        step_size: tuple[float, float] = None,
        max_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_last_images_to_keep_in_context: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """Run a field-of-view search workflow.

        Parameters
        ----------
        feature_description : str, optional
            Text description of the feature to search for. The text can include
            an ``<img /path/to/image.png>`` reference image tag.
        y_range : tuple[float, float], optional
            Inclusive search range for the vertical stage coordinate.
        x_range : tuple[float, float], optional
            Inclusive search range for the horizontal stage coordinate.
        fov_size : tuple[float, float], optional
            Initial field-of-view size in ``(height, width)`` order.
        step_size : tuple[float, float], optional
            Initial grid-search step size in ``(dy, dx)`` order.
        max_rounds : int, optional
            Maximum number of feedback-loop rounds to allow.
        n_first_images_to_keep_in_context : int, optional
            Number of earliest images to keep in context when pruning.
        n_last_images_to_keep_in_context : int, optional
            Number of latest images to keep in context when pruning.
        initial_prompt : str, optional
            Explicit prompt override. When provided, the geometry arguments and
            feature description must be omitted.
        additional_prompt : str, optional
            Additional instructions appended to the generated prompt.
        *args
            Unused positional compatibility arguments.
        **kwargs
            Unused keyword compatibility arguments.
        """
        if initial_prompt is None:
            initial_prompt = (
                f"You are given a tool that acquires an image of a sub-region "
                f"of a sample at given location and with given size (the field "
                f"of view, or FOV). Each time you call the tool, you will see "
                f"the image acquired. Use this tool to find a subregion that contains "
                f"the following feature: {feature_description}. "
                f"The feature should be centered in the field of view. Each time you "
                f"see an acquired image, check if it is in the FOV; if not, move the "
                f"FOV until you find it.\n"
                f"Here are your detailed instructions:\n"
                f"- At the beginning, use an FOV size of {fov_size} (height, width). "
                f"You can change the FOV size during the process to see a larger area, "
                f"but go back to this size when you find the feature and acquire a "
                f"final image of it.\n"
                f"- Start from position (y={y_range[0]}, x={x_range[0]}), and gradually "
                f"move the FOV to find the feature. Positions should stay in the range of "
                f"y={y_range[0]} to {y_range[1]} and x={x_range[0]} to {x_range[1]}. \n"
                f"- Use a regular grid search pattern at the beginning. Use a step size of {step_size[0]} "
                f"in the y direction and {step_size[1]} in the x direction. When you see the\n"
                f"feature, you can move the FOV more arbitrarily to make it better centered.\n"
                f"- When you find the feature, adjust the positions of the FOV to make the "
                f"feature centered in the FOV. If the feature is off to the left, move "
                f"the FOV to the left; if the feature is off to the top, move the FOV "
                f"to the top.\n"
                f"- Do not acquire images at the same or close location over and over again. "
                f"If you find yourself calling the tool repeatedly at close locations, "
                f"stop the process.\n"
                f"- When you find the feature of interest, report the coordinates of the "
                f"FOV.\n"
                f"- Explain every tool call you make."
                f"- When calling tools, make only one call at a time. Do not make "
                f"another call before getting the response of a previous one. \n"
                f"- When you finish the search or need user response, say 'TERMINATE'.\n"
            )
        elif any(
            value is not None
            for value in [feature_description, y_range, x_range, fov_size, step_size]
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
            message_with_yielded_image=(
                "Here is the image the tool returned. If the feature is there, "
                "report the coordinates of the FOV and include 'TERMINATE' in "
                "your response. Otherwise, continue to call tools to run the search. "
                "Include a brief description of what you see in the image in your response."
            ),
            max_rounds=max_rounds,
            n_first_images_to_keep_in_context=n_first_images_to_keep_in_context,
            n_last_images_to_keep_in_context=n_last_images_to_keep_in_context,
        )
