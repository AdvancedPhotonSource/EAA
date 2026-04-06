from typing import Optional

from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.task_manager.prompts import render_prompt_template
from eaa_core.tool.base import BaseTool
from eaa_imaging.task_manager.imaging.base import ImagingBaseTaskManager
from eaa_imaging.task_manager.imaging.feature_tracking import initialize_feature_tracking_task_manager
from eaa_imaging.tool.imaging.acquisition import AcquireImage
from eaa_imaging.tool.imaging.registration import ImageRegistration


class ROISearchTaskManager(ImagingBaseTaskManager):
    """Search for a region of interest using the shared feedback-loop graph."""

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
        self.prerun_check()

        if initial_prompt is None:
            initial_prompt = render_prompt_template(
                "eaa_imaging.task_manager.prompts",
                "roi_search.md",
                {
                    "FEATURE_DESCRIPTION": feature_description,
                    "FOV_SIZE": fov_size,
                    "Y_MIN": y_range[0],
                    "Y_MAX": y_range[1],
                    "X_MIN": x_range[0],
                    "X_MAX": x_range[1],
                    "STEP_SIZE_Y": step_size[0],
                    "STEP_SIZE_X": step_size[1],
                },
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

    def run_from_checkpoint(self, checkpoint_db_path: Optional[str] = None) -> None:
        """Resume the ROI-search workflow from a checkpoint.

        Parameters
        ----------
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and updates instead of
            ``self.session_db_path``.
        """
        self.prerun_check()
        self.run_feedback_loop_from_checkpoint(checkpoint_db_path=checkpoint_db_path)
