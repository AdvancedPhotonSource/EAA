from typing import Optional, TYPE_CHECKING, Any
import logging
import copy

import numpy as np
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph

from eaa.core.message_proc import (
    generate_openai_message,
    get_tool_call_info,
)
from eaa.api.llm_config import LLMConfig
from eaa.api.memory import MemoryManagerConfig
from eaa.core.exceptions import MaxRoundsReached
from eaa.core.task_manager.nodes import NodeFactory
from eaa.core.task_manager.state import FeedbackLoopState
from eaa.core.tooling.base import BaseTool
from eaa.core.task_manager.base import load_latest_checkpoint_state_from_connection

from eaa.tool.imaging.acquisition import AcquireImage
from eaa.tool.param_tuning import SetParameters
from eaa.task_manager.tuning.base import BaseParameterTuningTaskManager
from eaa.task_manager.imaging.base import ImagingBaseTaskManager
from eaa.task_manager.imaging.feature_tracking import FeatureTrackingTaskManager
from eaa.tool.imaging.registration import ImageRegistration
import eaa.image_proc as ip

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from eaa.core.task_manager.base import BaseTaskManager


class FocusingNodeFactory(NodeFactory):
    """Node factory for the focusing workflow with registration routing."""

    def __init__(self, task_manager: "BaseTaskManager"):
        """Initialize the focusing node factory.

        Args:
            task_manager: Owning task manager.
        """
        super().__init__(task_manager)

    def route_after_tool_execution(self, state: FeedbackLoopState) -> str:
        """Route post-tool execution for the focusing workflow.

        Args:
            state: Active feedback-loop state.

        Returns:
            Next node name.
        """
        if self.task_manager.should_run_image_registration(state):
            return "image_registration"
        return super().route_after_tool_execution(state)

    def image_registration(self, state: FeedbackLoopState) -> dict[str, object]:
        """Handle image-acquisition follow-up with registration.

        Args:
            state: Active feedback-loop state.

        Returns:
            Updated graph state payload.
        """
        response = state.latest_response or {}
        tool_call_info_list = get_tool_call_info(response, index=None) or []
        tool_messages = state.latest_tool_messages
        followup_messages: list[dict[str, Any]] = []

        for index, tool_message in enumerate(tool_messages):
            tool_call_info = tool_call_info_list[index] if index < len(tool_call_info_list) else None
            tool_name = tool_call_info.get("function", {}).get("name") if tool_call_info else None
            image_paths = self.task_manager.tool_executor.extract_image_paths_from_tool_response(
                tool_message.get("content")
            )
            if (
                tool_name == "acquire_image"
                and len(image_paths) > 0
            ):
                for image_path in image_paths:
                    followup_messages.extend(
                        self.task_manager.build_acquisition_followup_messages(
                            image_path,
                            run_registration=True,
                        )
                    )
                continue

            followup_messages.extend(
                self.task_manager.tool_executor.build_tool_followup_messages(
                    tool_message,
                    skill_catalog=self.task_manager.skill_catalog,
                    message_with_yielded_image=state.message_with_yielded_image,
                    allow_non_image_tool_responses=state.allow_non_image_tool_responses,
                    tool_call_info=tool_call_info,
                )
            )

        self.apply_followup_messages_for_state(state, followup_messages)
        return state.model_dump()

    def image_followup(self, state: FeedbackLoopState) -> dict[str, object]:
        """Append focusing-specific follow-up messages after tool execution.

        Args:
            state: Active feedback-loop state.

        Returns:
            Updated graph state payload.
        """
        response = state.latest_response or {}
        tool_call_info_list = get_tool_call_info(response, index=None) or []
        tool_messages = state.latest_tool_messages
        followup_messages: list[dict[str, Any]] = []

        for index, tool_message in enumerate(tool_messages):
            tool_call_info = tool_call_info_list[index] if index < len(tool_call_info_list) else None
            tool_name = tool_call_info.get("function", {}).get("name") if tool_call_info else None
            image_paths = self.task_manager.tool_executor.extract_image_paths_from_tool_response(
                tool_message.get("content")
            )
            if (
                tool_name == "acquire_image"
                and len(image_paths) > 0
            ):
                for image_path in image_paths:
                    followup_messages.extend(
                        self.task_manager.build_acquisition_followup_messages(
                            image_path,
                            run_registration=False,
                        )
                    )
                continue

            followup_messages.extend(
                self.task_manager.tool_executor.build_tool_followup_messages(
                    tool_message,
                    skill_catalog=self.task_manager.skill_catalog,
                    message_with_yielded_image=state.message_with_yielded_image,
                    allow_non_image_tool_responses=state.allow_non_image_tool_responses,
                    tool_call_info=tool_call_info,
                )
            )

        self.apply_followup_messages_for_state(state, followup_messages)
        return state.model_dump()


class ScanningMicroscopeFocusingTaskManager(BaseParameterTuningTaskManager):
    
    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        param_setting_tool: SetParameters = None,
        acquisition_tool: AcquireImage = None,
        image_registration_tool: Optional[ImageRegistration] = None,
        additional_tools: list[BaseTool] = (),
        initial_parameters: dict[str, float] = None,
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]] = None,
        use_feature_tracking_subtask: bool = False,
        feature_tracking_kwargs: Optional[dict] = None,
        session_db_path: Optional[str] = "session.sqlite",
        build: bool = True,
        *args, **kwargs
    ):
        """A task manager for focusing a scanning microscope.
        
        The task manager assumes that the user has a test pattern that has
        thin lines that can be used to evaluate the focus. It expects a
        2D image acquisition tool, a line scan tool, and a parameter setting 
        tool. The workflow is as follows:
        
        1. The user provides a reference image that highlights the thin
           feature that should be used to evaluate the focus through line
           scan, or describe it verbally.
        2. The agent runs a line scan across the feature and obtain its
           line profile and the FWHM of its Gaussian fit.
        3. The agent uses the parameter setting tool to adjust the parameters
           controlling the focus.
        4. The agent runs a 2D image scan around the area to acquire a new image,
           which may have drifted due to the focus adjustment.
        5. The agent runs a new line scan across the same feature used previously
           and compare the FWHM of the Gaussian fit.
        6. The agent repeats the process until the FWHM of the Gaussian fit is
           minimized.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            The LLM configuration to use.
        memory_config : MemoryManagerConfig, optional
            Memory configuration forwarded to the agent.
        param_setting_tool : SetParameters
            The tool to use to set the parameters.
        acquisition_tool : AcquireImage
            The BaseTool object used to acquire data. It should contain a 2D
            image acquisition tool and a line scan tool.
        image_registration_tool : ImageRegistration, optional
            The image registration tool. This tool is optional and is only
            used for the feature tracking sub-task if `use_feature_tracking_subtask`
            is True. To use registration in the focusing task manager, refer to
            ``use_registration_in_workflow`` in the ``run`` method.
        additional_tools : list[BaseTool], optional
            Additional tools provided to the agent (not including the
            parameter setting tool and the acquisition tool).
        initial_parameters : dict[str, float], optional
            The initial parameters given as a dictionary of 
            parameter names and values.
        parameter_ranges : list[tuple[float, ...], tuple[float, ...]]
            The ranges of the parameters. It should be given as a list of
            2 tuples, where the first tuple gives the lower bounds and the
            second tuple gives the upper bounds. The order of the parameters
            should match the order of the initial parameters.
        use_feature_tracking_subtask : bool, optional
            If True, a feature tracking sub-task manager will be created and
            runs when a 2D image is acquired to restore drifted FOV.
        feature_tracking_kwargs : dict, optional
            The kwargs for the feature tracking sub-task manager. Required
            if `use_feature_tracking_subtask` is True. The dictionary should
            contain:
            - `y_range`: Tuple[float, float] The range of the y-coordinate of the feature.
            - `x_range`: Tuple[float, float] The range of the x-coordinate of the feature.

            Initial positions should not be included in the dictionary because
            they will be determined by the logic.
        session_db_path : Optional[str], optional
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        build : bool, optional
            Whether to build the internal state of the task manager.
        """
        if acquisition_tool is None:
            raise ValueError("`acquisition_tool` must be provided.")
        
        self.acquisition_tool = acquisition_tool
        self.image_registration_tool = image_registration_tool
        
        self.last_acquisition_count_registered = 0
        self.last_acquisition_count_stitched = 0
        self.active_use_registration_in_workflow = False
        self.active_add_reference_image_to_images_acquired = False
        self.active_reference_image_path: Optional[str] = None
        
        self.use_feature_tracking_subtask = use_feature_tracking_subtask
        self.feature_tracking_task_manager: Optional[FeatureTrackingTaskManager] = None
        self.feature_tracking_kwargs = feature_tracking_kwargs
        
        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            param_setting_tool=param_setting_tool,
            additional_tools=[acquisition_tool, *additional_tools],
            initial_parameters=initial_parameters,
            parameter_ranges=parameter_ranges,
            session_db_path=session_db_path,
            build=build,
            *args, **kwargs
        )
        
    def build_task_graph(self, checkpointer: Any = None) -> CompiledStateGraph:
        """Build the focusing task graph with image-registration routing.

        Parameters
        ----------
        checkpointer : Any, optional
            LangGraph checkpointer.

        Returns
        -------
        CompiledStateGraph
            Compiled focusing feedback-loop graph.
        """
        node_factory = FocusingNodeFactory(self)
        builder = StateGraph(FeedbackLoopState)
        builder.add_node(
            "handle_human_gate",
            node_factory.handle_human_gate,
        )
        builder.add_node(
            "reprompt_model",
            node_factory.reprompt_model,
        )
        builder.add_node(
            "execute_tools",
            node_factory.execute_tools,
            input_schema=FeedbackLoopState,
        )
        builder.add_node(
            "image_registration",
            node_factory.image_registration,
            input_schema=FeedbackLoopState,
        )
        builder.add_node(
            "image_followup",
            node_factory.image_followup,
            input_schema=FeedbackLoopState,
        )
        builder.add_node(
            "finalize_round",
            node_factory.finalize_round,
        )
        builder.add_node(
            "call_model",
            node_factory.call_model,
            input_schema=FeedbackLoopState,
        )
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            node_factory.route_after_feedback_response,
        )
        builder.add_conditional_edges(
            "handle_human_gate",
            node_factory.route_after_feedback_response,
        )
        builder.add_conditional_edges(
            "reprompt_model",
            node_factory.route_after_feedback_response,
        )
        builder.add_conditional_edges(
            "execute_tools",
            node_factory.route_after_tool_execution,
        )
        builder.add_edge("image_registration", "finalize_round")
        builder.add_edge("image_followup", "finalize_round")
        builder.add_conditional_edges(
            "finalize_round",
            node_factory.route_after_feedback_round,
        )
        return builder.compile(checkpointer=checkpointer)

    def should_run_image_registration(self, state: FeedbackLoopState) -> bool:
        """Return whether the registration node should run for the latest tool batch.

        Parameters
        ----------
        state : FeedbackLoopState
            Active feedback-loop state.

        Returns
        -------
        bool
            Whether the registration node should run.
        """
        if not self.active_use_registration_in_workflow:
            return False
        if self.acquisition_tool.image_km1 is None or self.acquisition_tool.image_k is None:
            return False
        if (
            self.acquisition_tool.counter_acquire_image
            <= self.last_acquisition_count_registered
        ):
            return False
        response = state.latest_response or {}
        tool_call_info_list = get_tool_call_info(response, index=None) or []
        tool_messages = state.latest_tool_messages
        for index, tool_message in enumerate(tool_messages):
            tool_call_info = tool_call_info_list[index] if index < len(tool_call_info_list) else None
            tool_name = tool_call_info.get("function", {}).get("name") if tool_call_info else None
            if tool_name != "acquire_image":
                continue
            image_paths = self.tool_executor.extract_image_paths_from_tool_response(
                tool_message.get("content")
            )
            if len(image_paths) > 0:
                return True
        return False

    def build_acquisition_followup_messages(
        self,
        image_path: str,
        *,
        run_registration: bool,
    ) -> list[dict[str, Any]]:
        """Build focusing-specific follow-up messages after image acquisition.

        Parameters
        ----------
        image_path : str
            Returned image path.
        run_registration : bool
            Whether to include the registration follow-up path.

        Returns
        -------
        list[dict[str, Any]]
            Follow-up messages for the acquired image.
        """
        message = ""
        if (
            self.active_add_reference_image_to_images_acquired
            and self.active_reference_image_path is not None
            and self.acquisition_tool.counter_acquire_image > self.last_acquisition_count_stitched
        ):
            image_path = ImagingBaseTaskManager.add_reference_image_to_images_acquired(
                image_path,
                self.active_reference_image_path,
            )
            self.last_acquisition_count_stitched = self.acquisition_tool.counter_acquire_image
            message = (
                "Here is the new image (left). "
                "The reference image (right) is also shown for your reference. "
            )

        run_feature_tracking = False
        if self.use_feature_tracking_subtask:
            run_feature_tracking = self.should_run_feature_tracking()
            if run_feature_tracking:
                temp_context = [
                    generate_openai_message(content="Image 1", image=self.acquisition_tool.image_km1),
                    generate_openai_message(content="Image 2", image=self.acquisition_tool.image_k),
                ]
                response, outgoing = self.invoke_model_raw(
                    "Does the last image have any overlap with the previous one? "
                    "Just answer with 'yes' or 'no'.",
                    image_path=image_path,
                    context=temp_context,
                    return_outgoing_message=True,
                )
                self.update_message_history(outgoing, update_context=False, update_full_history=True)
                self.update_message_history(response, update_context=False, update_full_history=True)

                if "no" in response["content"].lower():
                    feature_tracking_response = self.restore_fov(
                        self.acquisition_tool.image_km1,
                        add_target_image_to_images_acquired=(
                            self.active_add_reference_image_to_images_acquired
                        ),
                    )
                    self.last_acquisition_count_registered = self.acquisition_tool.counter_acquire_image
                    message += (
                        "Here is the image you just acquired. Since there is no overlap "
                        "with the last image, feature tracking has been performed. Here "
                        f"is the result: \n{feature_tracking_response}\n"
                        "Use the result to adjust the line scan positions."
                    )
                else:
                    run_feature_tracking = False

        if run_registration and not run_feature_tracking and self.should_run_feature_registration():
            shift = self.register_images(
                self.acquisition_tool.image_k,
                self.acquisition_tool.image_km1,
            )
            if len(message) == 0:
                message = "Here is the new image. "
            scan_pos_diff = [
                float(self.acquisition_tool.image_acquisition_call_history[-1][f"{direction}_center"])
                - float(self.acquisition_tool.image_acquisition_call_history[-2][f"{direction}_center"])
                for direction in ["y", "x"]
            ]
            offset_to_subtract = [float(shift[i] - scan_pos_diff[i]) for i in [0, 1]]
            message += (
                "Image registration has found the offset to apply to the new image for "
                f"alignment with the previous one to be {shift.tolist()} (y, x). Taking "
                f"into account the difference in scan positions ({scan_pos_diff}), the "
                f"offset to use is {offset_to_subtract} (y, x). Use this offset to "
                "adjust the line scan positions by **subtracting** it from both "
                "the x and y coordinates of the start and end points of the previous line scan. "
            )
            if len(self.acquisition_tool.line_scan_call_history[-1]) > 0:
                message += (
                    "For your reference, the last line scan tool call is "
                    f"{self.acquisition_tool.line_scan_call_history[-1]}."
                    "Also use this offset to update the argument when you perform 2D image acquisition "
                    "next time. The last 2D image acquisition call is "
                    f"{self.acquisition_tool.image_acquisition_call_history[-1]}."
                )
            self.last_acquisition_count_registered = self.acquisition_tool.counter_acquire_image

        if len(message) == 0:
            message = "Here is the image the tool returned."
        return [generate_openai_message(content=message, image_path=image_path)]

    def should_run_feature_tracking(self) -> bool:
        """Return whether overlap should be checked for feature tracking.

        Returns
        -------
        bool
            Whether overlap-check logic should run.
        """
        return (
            self.acquisition_tool.image_km1 is not None
            and self.acquisition_tool.image_k is not None
            and self.acquisition_tool.counter_acquire_image > self.last_acquisition_count_registered
        )

    def should_run_feature_registration(self) -> bool:
        """Return whether image registration can run on the latest acquisition.

        Returns
        -------
        bool
            Whether registration can run.
        """
        return (
            self.acquisition_tool.image_km1 is not None
            and self.acquisition_tool.image_k is not None
            and self.acquisition_tool.counter_acquire_image > self.last_acquisition_count_registered
        )

    def restore_fov(
        self, 
        target_image: np.ndarray, 
        add_target_image_to_images_acquired: bool = False
    ) -> str:
        """Run the feature tracking sub-task to make the FOV aligned with the target image.
        """
        fig = ip.plot_2d_image(target_image, add_axis_ticks=False)
        target_image_path = BaseTool.save_image_to_temp_dir(fig, "target_image.png", add_timestamp=True)
        
        self.feature_tracking_task_manager = FeatureTrackingTaskManager(
            llm_config=self.llm_config,
            memory_config=self.memory_config,
            image_acquisition_tool=copy.deepcopy(self.acquisition_tool),
            image_registration_tool=copy.deepcopy(self.image_registration_tool),
            session_db_path=self.session_db_path,
        )
        
        try:
            self.feature_tracking_task_manager.run(
                reference_image_path=target_image_path,
                initial_position=(
                    self.acquisition_tool.image_acquisition_call_history[-1]["y_center"],
                    self.acquisition_tool.image_acquisition_call_history[-1]["x_center"]
                ),
                initial_fov_size=(
                    self.acquisition_tool.image_acquisition_call_history[-1]["size_y"], 
                    self.acquisition_tool.image_acquisition_call_history[-1]["size_x"]
                ),
                add_reference_image_to_images_acquired=add_target_image_to_images_acquired,
                **self.feature_tracking_kwargs,
                max_rounds=20,
                termination_behavior="return",
                max_arounds_reached_behavior="raise"
            )
            feature_tracking_response = self.feature_tracking_task_manager.context[-1]["content"]
            feature_tracking_response = feature_tracking_response.replace("TERMINATE", "")
        except MaxRoundsReached:
            feature_tracking_response = (
                "Maximum number of rounds reached, but the feature tracking "
                "sub-task was not able to restore the FOV. To try again, re-collect "
                "the 2D image. Or ask for human intervention by adding \"NEED HUMAN\" "
                "to your response."
            )
        
        return feature_tracking_response
        
    def register_images(self, image_k: np.ndarray, image_km1: np.ndarray) -> np.ndarray:
        """Register the two images and return the offset.
        """
        # Recreate another object of the registration tool here instead of reusing
        # the one provided to avoid perturbing the internal state of the tool.
        if self.image_registration_tool is None:
            raise ValueError(
                "`image_registration_tool` should be provided in the class constructor."
            )
        registration_tool = self.image_registration_tool
        shift = registration_tool.register_images(
            image_t=registration_tool.process_image(image_k),
            image_r=registration_tool.process_image(image_km1),
            psize_t=self.acquisition_tool.psize_k,
            psize_r=self.acquisition_tool.psize_km1
        )
        return shift
    
    def run(
        self,
        reference_image_path: Optional[str] = None,
        reference_feature_description: Optional[str] = None,
        suggested_2d_scan_kwargs: dict = None,
        suggested_parameter_step_size: Optional[float] = None,
        line_scan_step_size: float = None,
        use_registration_in_workflow: bool = True,
        add_reference_image_to_images_acquired: bool = False,
        initial_prompt: Optional[str] = None,
        max_iters: int = 20,
        n_last_images_to_keep_in_context: Optional[int] = None,
        additional_prompt: Optional[str] = None,
        *args, **kwargs
    ):
        """Run the focusing task.
        
        Parameters
        ----------
        reference_image_path : Optional[str]
            The path to the reference image, which should show a 2D scan
            of the ROI with the desired line scan path indicated by a
            marker. ``reference_feature_description`` will be ignored if
            this argument is provided. You can also leave this argument 
            as None and provide the reference image in terminal or WebUI
            when prompted.
        reference_feature_description : Optional[str]
            The description of the feature across which line scans should
            be done. Ignored if ``reference_image_path`` is provided.
        suggested_2d_scan_kwargs : dict
            The suggested kwargs for the 2D scan. The argument should match
            the arguments of the 2D image acquisition tool.
        suggested_parameter_step_size : float
            The suggested step size for the parameter adjustment.
        line_scan_step_size : float
            The step size for the line scan.
        use_registration_in_workflow : bool
            If True, image registration will be performed when the 2D image
            acquisition tool is called, and the offset found will be given 
            to the agent along with the acquired image. Note that this is done
            with Python logic instead of agent tool call; if a registration
            tool is available for the agent to call, it should be provided through
            ``tools`` and you may want to set this argument to False.
        add_reference_image_to_images_acquired : bool
            If True, the reference image will be stitched side-by-side with
            2D microscopy images acquired. This allows the agent to always see
            the reference image in new messages when needed, instead of having
            the reference image only in the first message in the context. This
            may be particularly useful for inference endpoint providers that do
            not support images in the context.
        initial_prompt : Optional[str]
            If provided, this prompt will override the default initial prompt.
        max_iters : int, optional
            The maximum number of iterations to run.
        n_last_images_to_keep_in_context : int, optional
            The number of past images to keep in the context. If None, all images
            will be kept.
        additional_prompt : Optional[str]
            If provided, this prompt will be added to the initial prompt.
        """
        self.prerun_check()

        if reference_image_path is None and reference_feature_description is None:
            raise ValueError(
                "Either `reference_image_path` or `reference_feature_description` must be provided."
            )
            
        if use_registration_in_workflow and self.image_registration_tool is None:
            raise ValueError(
                "`image_registration_tool` should be provided in the class constructor "
                "if `use_registration_in_workflow` is True."
            )
            
        if reference_image_path is None:
            reference_image_prompts = ""
        else:
            reference_image_prompts = (
                f"<img {reference_image_path}>\n"
                f"You will see a reference 2D scan image in this message. "
                f"This image is acquired in the region of interest that "
                f"contains the thin feature to be line-scanned. The line scan path "
                f"across that feature is indicated by a marker."
            )

        if initial_prompt is None:
            feat_text_description = ""
            if reference_feature_description is not None:
                feat_text_description = (
                    f"Here is the description of the feature: **{reference_feature_description}**. "
                )
            param_step_size_prompt = ""
            if suggested_parameter_step_size is not None:
                param_step_size_prompt = (
                    f"- The suggested step size for adjusting the parameter is "
                    f"{suggested_parameter_step_size}. You can adjust the step size "
                    f"to a smaller value if you want to fine-tune the parameter."
                )
                
            line_scan_step_size_prompt = ""
            if line_scan_step_size is not None:
                line_scan_step_size_prompt = (
                    f"The suggested step size for the line scan is {line_scan_step_size}."
                )
                
            if use_registration_in_workflow:
                registration_prompt = (
                    "Along with this image, you will also be given the offset of "
                    "this image compared to the previous image found through image registration. "
                    "Use this offset by subtracting it from line-scan and image-acquisition coordinates. "
                    "Also use this offset to "
                    "update the positions of your next 2D image acquisition. Note that the offset "
                    "is just a suggestion. If the new image does not appear to have any overlap "
                    "with the previous one, the offset won't be reliable. In that case, try "
                    "adjusting the image acquisition tool's parameters to move the field of view "
                    "closer to the previous image."
                )
                line_scan_positioning_prompt = (
                    "Use the offset given by image registration and subtract it from the previous line-scan coordinates."
                )
            else:
                registration_prompt = (
                    "Use your registration tool to find the offset between the new image "
                    "and the previous one. Use this offset to adjust the line scan positions "
                    "by **subtracting** it from both the x and y coordinates of the start and end "
                    "points of the previous line scan. "
                )
                line_scan_positioning_prompt = (
                    "Read the coordinates of the line scan path from the axis ticks."
                )
            
            initial_prompt = (
                f"You will adjust the focus of a scanning microscope by adjusting "
                f"the parameters of its optics. The focusing quality can be evalutated "
                f"by performing a line scan across a thin feature and observe the FWHM "
                f"of its Gaussian fit. The smaller the FWHM, the sharper the image. "
                f"But each time you adjust the focus, the image may drift due to "
                f"the change of the optics. You will need to perform a 2D scan "
                f"prior to the line scan to locate the feature that is line-scanned.\n"
                f"{reference_image_prompts}\n"
                f"{feat_text_description}\n\n"
                f"Follow the procedure below to focus the microscope:\n\n"
                f"1. First, perform a 2D scan of the region of interest using the "
                f"\"acquire_image\" tool and the following arguments: "
                f"{suggested_2d_scan_kwargs}. "
                f"The image should look similar to the reference image. "
                f"Determine the coordinates of the line scan path across the feature, "
                f"and use the \"scan_line\" tool to perform a line scan across the feature. "
                f"{line_scan_step_size_prompt}\n"
                f"2. The line scan tool will return a plot along the scan line. You should "
                f"see a peak in the plot. A Gaussian fit will be included in the plot "
                f"and the FWHM of the Gaussian fit will be shown.\n"
                f"3. Adjust the optics parameters using the parameter setting tool. "
                f"The initial parameter values are {self.initial_parameters}.\n"
                f"4. Acquire an image of the region using the image acquisition tool. "
                f"Here are the suggested arguments: {suggested_2d_scan_kwargs}. The "
                f"image acquired may have drifted compared to the last one you saw, "
                f"but you should still see the line-scanned feature there. If not, "
                f"try adjusting the image acquisition tool's parameters to locate that "
                f"feature. {registration_prompt}\n"
                f"5. Once you find the line-scanned feature, perform a new line scan across "
                f"it again. Due to the drift, the start/end points' coordinates may need to "
                f"be changed. {line_scan_positioning_prompt}\n"
                f"6. You will be presented with the new line scan plot and the FWHM of the "
                f"Gaussian fit.\n"
                f"7. Compare the new FWHM with the last one. If it is smaller, you are on the "
                f"right track. Keep adjusting the parameters to the same direction. Otherwise, "
                f"adjust the parameters in the opposite direction.\n"
                f"8. Repeat the process from step 4.\n"
                f"9. When you find the FWHM is minimized, you are done. Add \"TERMINATE\" to "
                f"your response to hand over control back to the user.\n\n"
                f"Important notes:\n\n"
                f"- Your line scan should cross only one line feature, and you should see "
                f"**exactly one peak** in the line scan plot. If there isn't one, or if "
                f"the Gaussian fit looks bad, check your arguments "
                f"to the line scan tool and run it again. Make sure your line scan strictly "
                f"follow the marker in the reference image. Do not trust the FWHM value "
                f"in the line plot if there is no peak, or if the peak is incomplete!\n"
                f"- The line scan plot should show a complete peak. If the peak is incomplete, "
                f"adjust the line scan tool's arguments to make it complete.\n"
                f"- The minimal point of the FWHM is indicated by an inflection of the trend "
                f"of the FWHM with regards to the optics parameters. For example, if the FWHM "
                f"is 3 with a parameter value of 10, then 1 with a parameter value of 11, then "
                f"3 with a parameter value of 12, this means the optimal parameter value is around "
                f"11.\n"
                f"{param_step_size_prompt}\n"
                f"- When calling a tool, explain what you are doing.\n"
                f"- When making a tool call, only call one tool at a time. Do not call multiple "
                f"tools in one response.\n"
                f"- Remember that when coordinates are given in (y, x) order, the first coordinate "
                f"is the row index (vertical) and the second is the column index (horizontal). "
                f"When you write coordinates in your response, do not just write two numbers; "
                f"instead, explicitly specify y/x axis and write them as (y = <y>, x = <x>).\n\n"
                f"When you finish or when you need human input, add \"TERMINATE\" to your response."
            )
        if additional_prompt is not None:
            initial_prompt += "\nAdditional instructions:\n" + additional_prompt

        self.active_use_registration_in_workflow = use_registration_in_workflow
        self.active_add_reference_image_to_images_acquired = (
            add_reference_image_to_images_acquired
        )
        self.active_reference_image_path = reference_image_path
        
        self.state = FeedbackLoopState(
            messages=list(self.context),
            full_history=list(self.full_history),
            round_index=0,
            await_user_input=False,
            initial_prompt=initial_prompt,
            initial_image_path=reference_image_path,
            message_with_yielded_image="Here is the image the tool returned.",
            max_rounds=max_iters,
            n_first_images_to_keep_in_context=1,
            n_last_images_to_keep_in_context=n_last_images_to_keep_in_context,
            allow_non_image_tool_responses=True,
            allow_multiple_tool_calls=False,
            expected_tool_call_sequence=[
                "scan_line",
                "set_parameters",
                "acquire_image",
            ],
            expected_tool_call_sequence_tolerance=1,
            termination_behavior=kwargs.pop("termination_behavior", "ask"),
            max_arounds_reached_behavior=kwargs.pop("max_arounds_reached_behavior", "return"),
        )
        super().run()

    def run_from_checkpoint(
        self,
        checkpoint_db_path: Optional[str] = None,
        use_registration_in_workflow: Optional[bool] = None,
        add_reference_image_to_images_acquired: Optional[bool] = None,
        reference_image_path: Optional[str] = None,
    ) -> None:
        """Resume the scanning-microscope focusing workflow from a checkpoint.

        Parameters
        ----------
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and updates instead of
            ``self.session_db_path``.
        use_registration_in_workflow : Optional[bool], optional
            Override for whether registration follow-up should be routed
            through the dedicated node.
        add_reference_image_to_images_acquired : Optional[bool], optional
            Override for whether the reference image should be stitched onto
            newly acquired images.
        reference_image_path : Optional[str], optional
            Override for the reference image path used for stitched follow-up
            images.
        """
        self.prerun_check()
        if use_registration_in_workflow is not None:
            self.active_use_registration_in_workflow = use_registration_in_workflow
        if add_reference_image_to_images_acquired is not None:
            self.active_add_reference_image_to_images_acquired = (
                add_reference_image_to_images_acquired
            )
        if reference_image_path is not None:
            self.active_reference_image_path = reference_image_path
        graph, checkpoint_config, _ = self.get_checkpointed_graph(
            "task_graph",
            checkpoint_db_path=checkpoint_db_path,
            load_state=False,
        )
        snapshot = graph.get_state(checkpoint_config)
        fallback_loaded = (
            snapshot.created_at is None
            or snapshot.values is None
            or len(snapshot.values) == 0
        )
        if not fallback_loaded and snapshot.values is not None:
            resumed_state = FeedbackLoopState.model_validate(snapshot.values)
        else:
            resolved_checkpoint_path, _ = self.resolve_checkpoint_storage(
                "task_graph",
                checkpoint_db_path=checkpoint_db_path,
            )
            latest_state = load_latest_checkpoint_state_from_connection(
                connection=self.checkpoint_connections[("task_graph", resolved_checkpoint_path)],
                prune_checkpoints=self.prune_checkpoints,
            )
            if latest_state is None:
                raise ValueError(
                    f"No task-graph checkpoint found in shared checkpoint DB "
                    f"{resolved_checkpoint_path}."
                )
            resumed_state = FeedbackLoopState.model_validate(latest_state)
        restart_from_human_gate = (
            fallback_loaded
            or resumed_state.exit_requested
            or resumed_state.return_requested
        )
        if restart_from_human_gate:
            resumed_state.exit_requested = False
            resumed_state.return_requested = False
            resumed_state.await_user_input = True
        self.state = resumed_state
        self.task_graph = graph
        if restart_from_human_gate:
            checkpoint_config = graph.update_state(
                checkpoint_config,
                resumed_state.model_dump(),
                as_node="handle_human_gate",
            )
        final_state = graph.invoke(None, config=checkpoint_config)
        self.state = FeedbackLoopState.model_validate(final_state)
        if self.state.chat_requested:
            self.handoff_to_chat()
