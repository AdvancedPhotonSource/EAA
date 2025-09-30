from typing import Optional, Callable
import logging
import copy

import numpy as np

from eaa.tools.imaging.acquisition import AcquireImage
from eaa.tools.imaging.param_tuning import SetParameters
from eaa.task_managers.tuning.base import BaseParameterTuningTaskManager
from eaa.task_managers.imaging.base import ImagingBaseTaskManager
from eaa.task_managers.imaging.feature_tracking import FeatureTrackingTaskManager
from eaa.tools.base import ToolReturnType, BaseTool
from eaa.tools.imaging.registration import ImageRegistration
from eaa.agents.base import print_message
from eaa.api.llm_config import LLMConfig
from eaa.agents.memory import MemoryManagerConfig
from eaa.util import get_image_path_from_text
import eaa.image_proc as ip

logger = logging.getLogger(__name__)


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
        message_db_path: Optional[str] = None,
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
        message_db_path : Optional[str], optional
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
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
        
    def image_path_tool_response_hook_factory(
        self,
        use_registration_in_workflow: bool,
        add_reference_image_to_images_acquired: bool,
        use_feature_tracking_subtask: bool,
        reference_image_path: str
    ) -> Callable:
        """Factory function that returns a hook function for the image path tool response.
        If none of the flags are True, it returns None.
        """
        def hook_function(image_path: str) -> None:
            message = ""
            if (
                add_reference_image_to_images_acquired
                and self.acquisition_tool.counter_acquire_image > self.last_acquisition_count_stitched
            ):
                image_path = ImagingBaseTaskManager.add_reference_image_to_images_acquired(
                    image_path, reference_image_path
                )
                self.last_acquisition_count_stitched = self.acquisition_tool.counter_acquire_image
                message = (
                    "Here is the new image (left). "
                    "The reference image (right) is also shown for your reference. "
                )
            
            run_feature_tracking = use_feature_tracking_subtask
            if use_feature_tracking_subtask:
                # Let agent decide whether feature tracking is needed.
                if (
                    self.acquisition_tool.image_km1 is not None
                    and self.acquisition_tool.image_k is not None
                    and self.acquisition_tool.counter_acquire_image > self.last_acquisition_count_registered
                ):
                    response, outgoing = self.agent.receive(
                        "Here is the collected image. Does it have any overlap with the reference image? "
                        "Just answer with 'yes' or 'no'.",
                        image_path=image_path,
                        context=self.context,
                        return_outgoing_message=True
                    )
                    self.update_message_history(outgoing, update_context=False, update_full_history=True)
                    self.update_message_history(response, update_context=True, update_full_history=True)
                    
                    if "no" in response["content"].lower():
                        # If there is no overlap, run feature tracking to restore the FOV.
                        run_feature_tracking = True
                        feature_tracking_response = self.restore_fov(
                            self.acquisition_tool.image_km1,
                            add_target_image_to_images_acquired=add_reference_image_to_images_acquired
                        )
                        self.last_acquisition_count_registered = self.acquisition_tool.counter_acquire_image
                        
                        if len(message) == 0:
                            message = ""
                        message += (
                            f"Here is the image you just acquired. Since there is no overlap "
                            f"with the last image, feature tracking has been performed. Here "
                            f"is the result: \n{feature_tracking_response}\n"
                            f"Use the result to adjust the line scan positions."
                        )
                    else:
                        run_feature_tracking = False
                else:
                    run_feature_tracking = False
            
            if use_registration_in_workflow and not run_feature_tracking:
                image_k = self.acquisition_tool.image_k
                image_km1 = self.acquisition_tool.image_km1
                if (
                    image_km1 is not None
                    and image_k is not None
                    and self.acquisition_tool.counter_acquire_image > self.last_acquisition_count_registered
                ):
                    shift = self.register_images(image_k, image_km1)
                    
                    if len(message) == 0:
                        message = "Here is the new image. "
                    scan_pos_diff = [
                        float(self.acquisition_tool.image_acquisition_call_history[-1][f"loc_{dir}"])
                        - float(self.acquisition_tool.image_acquisition_call_history[-2][f"loc_{dir}"])
                        for dir in ["y", "x"]
                    ]
                    message += (
                        f"Phase correlation has found the offset between "
                        f"the new image and the previous one to be {shift.tolist()} (y, x). Taking "
                        f"into account the difference in scan positions ({scan_pos_diff}), the net "
                        f"drift is {[float(shift[i] + scan_pos_diff[i]) for i in [0, 1]]} (y, x). Use this offset to "
                        f"to adjust the line scan positions by **adding** it to both "
                        f"the x and y coordinates of the start and end points of the previous line scan. "
                        f"For your reference, the last line scan tool call is {self.acquisition_tool.line_scan_call_history[-1]}."
                        f"Also use this offset to update the argument when you perform 2D image acquisition "
                        f"next time. The last 2D image acquisition call is {self.acquisition_tool.image_acquisition_call_history[-1]}."
                    )
                    self.last_acquisition_count_registered = self.acquisition_tool.counter_acquire_image
            response, outgoing = self.agent.receive(
                message,
                image_path=image_path,
                context=self.context,
                return_outgoing_message=True
            )
            return response, outgoing
        
        if (
            (not use_registration_in_workflow)
            and (not add_reference_image_to_images_acquired)
            and (not use_feature_tracking_subtask)
        ):
            return None
        else:
            return hook_function
        
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
            message_db_path=self.message_db_path,
            memory_vector_store=self._memory_vector_store,
            memory_notability_filter=self._memory_notability_filter,
            memory_formatter=self._memory_formatter,
            memory_embedder=self._memory_embedder,
        )
        
        self.feature_tracking_task_manager.run_feature_tracking(
            reference_image_path=target_image_path,
            initial_position=(
                self.acquisition_tool.image_acquisition_call_history[-1]["loc_y"], 
                self.acquisition_tool.image_acquisition_call_history[-1]["loc_x"]
            ),
            initial_fov_size=(
                self.acquisition_tool.image_acquisition_call_history[-1]["size_y"], 
                self.acquisition_tool.image_acquisition_call_history[-1]["size_x"]
            ),
            add_reference_image_to_images_acquired=add_target_image_to_images_acquired,
            **self.feature_tracking_kwargs,
            termination_behavior="return"
        )
        feature_tracking_response = self.feature_tracking_task_manager.context[-1]["content"]
        feature_tracking_response = feature_tracking_response.replace("TERMINATE", "")
        
    def register_images(self, image_k: np.ndarray, image_km1: np.ndarray) -> np.ndarray:
        """Register the two images and return the offset.
        """
        # Recreate another object of the registration tool here instead of reusing
        # the one provided to avoid perturbing the internal state of the tool.
        if self.image_registration_tool is None:
            raise ValueError(
                "`image_registration_tool` should be provided in the class constructor."
            )
        registration_tool = copy.deepcopy(self.image_registration_tool)
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
        n_past_images_to_keep_in_context: Optional[int] = None,
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
        n_past_images_to_keep_in_context : int, optional
            The number of past images to keep in the context. If None, all images
            will be kept.
        additional_prompt : Optional[str]
            If provided, this prompt will be added to the initial prompt.
        """
        if reference_image_path is None:
            user_image_input = self.get_user_input(
                prompt="Please provide the reference image as: <img /path/to/image.png>.",
                display_prompt_in_webui=True
            )
            reference_image_path = get_image_path_from_text(user_image_input)
        
        if reference_image_path is None and reference_feature_description is None:
            raise ValueError(
                "Either `reference_image_path` or `reference_feature_description` must be provided."
            )
            
        if use_registration_in_workflow and self.image_registration_tool is None:
            raise ValueError(
                "`image_registration_tool` should be provided in the class constructor "
                "if `use_registration_in_workflow` is True."
            )

        if initial_prompt is None:
            feat_text_description = ""
            if reference_feature_description is not None:
                feat_text_description = (
                    f"Also, here is the description of the feature: **{reference_feature_description}**. "
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
                    "Use this offset to adjust the line scan positions. Also use this offset to "
                    "update the positions of your next 2D image acquisition. Note that the offset "
                    "is just a suggestion. If the new image does not appear to have any overlap "
                    "with the previous one, the offset won't be reliable. In that case, try "
                    "adjusting the image acquisition tool's parameters to move the field of view "
                    "closer to the previous image."
                )
                line_scan_positioning_prompt = (
                    "Use the offset given by image registration to adjust the line scan positions."
                )
            else:
                registration_prompt = ""
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
                f"<img {reference_image_path}>\n"
                f"You will see a reference 2D scan image in this message. "
                f"This image is acquired in the region of interest that "
                f"contains the thin feature to be line-scanned. The line scan path "
                f"across that feature is indicated by a marker. {feat_text_description}\n\n"
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
                f"**exactly one peak** in the line scan plot. If there isn't one, or if there "
                f"are multiple peaks, or if the Gaussian fit looks bad, check your arguments "
                f"to the line scan tool and run it again. Make sure your line scan strictly "
                f"follow the marker in the reference image. Do not trust the FWHM value "
                f"in the line plot if there is no peak, if the peak is incomplete, or if "
                f"there are multiple peaks!\n"
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
        
        # Always keep the first (reference) image.
        self.run_feedback_loop(
            initial_prompt=initial_prompt,
            initial_image_path=reference_image_path,
            allow_non_image_tool_responses=True,
            n_first_images_to_keep_in_context=1,
            n_past_images_to_keep_in_context=n_past_images_to_keep_in_context,
            max_rounds=max_iters,
            hook_functions={
                "image_path_tool_response": self.image_path_tool_response_hook_factory(
                    use_registration_in_workflow=use_registration_in_workflow,
                    add_reference_image_to_images_acquired=add_reference_image_to_images_acquired,
                    use_feature_tracking_subtask=self.use_feature_tracking_subtask,
                    reference_image_path=reference_image_path
                )
            },
            *args, **kwargs
        )


class ParameterTuningTaskManager(BaseParameterTuningTaskManager):
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        param_setting_tool: SetParameters = None,
        acquisition_tool: AcquireImage = None,
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
        memory_config : MemoryManagerConfig, optional
            Memory configuration forwarded to the agent.
        param_setting_tool : SetParameters
            The tool to use to set the parameters.
        acquisition_tool : SimulatedAcquireImage, optional
            The tool to use to acquire images. This tool will 
            not be called by AI; it is executed automatically 
            following each parameter adjustment.
        initial_parameters : dict[str, float], optional
            The initial parameters given as a dictionary of 
            parameter names and values.
        parameter_ranges : list[tuple[float, ...], tuple[float, ...]]
            The ranges of the parameters. It should be given as a list of
            2 tuples, where the first tuple gives the lower bounds and the
            second tuple gives the upper bounds. The order of the parameters
            should match the order of the initial parameters.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        """
        if "tools" in kwargs.keys():
            raise ValueError(
                "`tools` should not be provided to `ParameterTuningTaskManager`. Instead, "
                "provide the `param_setting_tool` and `acquisition_tool`."
            )
            
        self.acquisition_tool = acquisition_tool
        
        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            param_setting_tool=param_setting_tool,
            additional_tools=[param_setting_tool],
            initial_parameters=initial_parameters,
            parameter_ranges=parameter_ranges,
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
        
    def prerun_check(self, *args, **kwargs) -> bool:
        if self.initial_parameters is None:
            raise ValueError("initial_parameters must be provided.")
        return super().prerun_check(*args, **kwargs)
        
    def run(
        self,
        acquisition_tool_kwargs: dict = {},
        n_past_images_to_keep: int = 3,
        max_iters: int = 10, 
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None,
    ) -> None:
        """Run the parameter tuning task.
        
        Parameters
        ----------
        acquisition_tool_kwargs : dict
            The arguments for the acquisition tool. These arguments will be
            used to acquire images for evaluation.
        n_past_images_to_keep : int, optional
            The number of most recent images to keep in the context. Having past
            images in the context allows to agent to "remember" images it
            has seen before; however, it also increases the context size
            and inference cost.
        max_iters : int, optional
            The maximum number of iterations to run.
        initial_prompt : str, optional
            If provided, this prompt will override the default initial prompt.
        additional_prompt : str, optional
            If provided, this prompt will be added to the initial prompt (either
            the default one or the one provided by `initial_prompt`).
        """
        self.prerun_check()
        
        initial_parameter_values = list(self.initial_parameters.values())
        self.param_setting_tool.set_parameters(initial_parameter_values)
        last_img_path = self.acquisition_tool.acquire_image(**acquisition_tool_kwargs)
        
        bounds_str = ""
        for i, param in enumerate(self.parameter_names):
            bounds_str += f"{param}: {self.parameter_ranges[0][i]} to {self.parameter_ranges[1][i]}\n"
        
        if initial_prompt is None:
            initial_prompt = (
                f"You are tuning the parameters of a microscope to attain the best "
                f"image sharpness. The parameters are {list(self.parameter_names)}, "
                f"and their current values are {initial_parameter_values}. An image acquired "
                f"with the current parameters is shown below.\n\n"
                f"<img {last_img_path}>\n\n"
                f"Here are the tunable ranges of the parameters:\n"
                f"{bounds_str}\n"
                f"You can change the parameters using your parameter setting tool. "
                f"An image acquired with the new parameters will be given to you "
                f"after each parameter change. Here are some detailed instructions:\n\n"
                f"- Tune parameters one by one. Start with the first parameter, tweak it "
                f"to attain the sharpest possible image, then move on to the next parameter. "
                f"Do not change more than one parameter at a time.\n"
                f"- The sharpness of the image is convex with regards to the parameters. There "
                f"is only one optimal point; assume there is no local maximum. As such, if "
                f"you find the image comes more blurry when changing a parameter in a direction, "
                f"you should consider changing it the other way; if you find the image comes "
                f"sharper when changing a parameter in a direction, you are on the right track.\n"
                f"- For each parameter, first get a coarse estimate of the optimal value, then "
                f"fine-tune it. To get a coarse estimate, look for a peak in the sharpness. In "
                f"other words, find a parameter value that gives a sharper image than the value "
                f"immediately before and after it. For example, if the image becomes sharper when "
                f"you increase the parameter from 4 to 5, but becomes blurrier when you increase "
                f"it from 5 to 6, then the optimal value is around 5.\n"
                f"- Choose the step size for changing parameters wisely. For each parameter, start "
                f"with a large step size, and decrease it as you get closer to the optimal point.\n"
                f"- Only call the parameter setting tool one at a time. Do not call it multiple times "
                f"in one response.\n\n"
                f"When you finish or when you need human input, add \"TERMINATE\" to your response."
            )
        if additional_prompt is not None:
            initial_prompt += "\nAdditional instructions:\n" + additional_prompt
        
        round = 0
        response, outgoing = self.agent.receive(
            initial_prompt,
            context=self.context,
            image_path=last_img_path,
            return_outgoing_message=True
        )
        self.update_message_history(outgoing, update_context=True, update_full_history=True)
        self.update_message_history(response, update_context=True, update_full_history=True)
        while round < max_iters:
            if response["content"] is not None and "TERMINATE" in response["content"]:
                message = self.get_user_input(
                    "Termination condition triggered. What to do next? Type \"\\exit\" to exit. "
                )
                if message.lower() == "\\exit":
                    return
                else:
                    response, outgoing = self.agent.receive(
                        message,
                        context=self.context,
                        image_path=None,
                        return_outgoing_message=True
                    )
                    self.update_message_history(outgoing, update_context=True, update_full_history=True)
                    self.update_message_history(response, update_context=True, update_full_history=True)
                    continue
            
            tool_responses, tool_response_types = self.agent.handle_tool_call(response, return_tool_return_types=True)
            if len(tool_responses) == 1:
                tool_response = tool_responses[0]
                tool_response_type = tool_response_types[0]
                # Just save the tool response, but don't send yet. We will send it
                # together with the image later.
                print_message(tool_response)
                self.update_message_history(tool_response, update_context=True, update_full_history=True)
                
                if tool_response_type == ToolReturnType.EXCEPTION:
                    response, outgoing = self.agent.receive(
                        "The tool returned an exception. Please fix the exception and try again.",
                        image_path=None,
                        context=self.context,
                        return_outgoing_message=True
                    )
                else:
                    # Acquire an image with the new parameters.
                    last_img_path = self.acquisition_tool.acquire_image(**acquisition_tool_kwargs)
                    response, outgoing = self.agent.receive(
                        "An image acquired with the new parameters is shown below.",
                        image_path=last_img_path,
                        context=self.context,
                        return_outgoing_message=True
                    )
                self.purge_context_images(keep_first_n=1, keep_last_n=n_past_images_to_keep - 1)
                self.update_message_history(outgoing, update_context=True, update_full_history=True)
                self.update_message_history(response, update_context=True, update_full_history=True)
            elif len(tool_responses) > 1:
                response, outgoing = self.agent.receive(
                    "There are more than one tool calls in your response. "
                    "Make sure you only make one call at a time. Please redo "
                    "your tool calls.",
                    image_path=None,
                    context=self.context,
                    return_outgoing_message=True
                )
                self.update_message_history(outgoing, update_context=True, update_full_history=True)
                self.update_message_history(response, update_context=True, update_full_history=True)
            else:
                response, outgoing = self.agent.receive(
                    "There is no tool call in the response. Make sure you call the tool correctly.",
                    image_path=None,
                    context=self.context,
                    return_outgoing_message=True
                )
                self.update_message_history(outgoing, update_context=True, update_full_history=True)
                self.update_message_history(response, update_context=True, update_full_history=True)
            round += 1
