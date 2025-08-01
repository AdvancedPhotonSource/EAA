from typing import Optional
from textwrap import dedent
import logging

from eaa.tools.imaging.acquisition import AcquireImage
from eaa.tools.imaging.param_tuning import SetParameters
from eaa.task_managers.tuning.base import BaseParameterTuningTaskManager
from eaa.tools.base import ToolReturnType, BaseTool
from eaa.agents.base import print_message
from eaa.api.llm_config import LLMConfig
from eaa.image_proc import windowed_phase_cross_correlation

logger = logging.getLogger(__name__)


class ScanningMicroscopeFocusingTaskManager(BaseParameterTuningTaskManager):
    
    def __init__(
        self,
        llm_config: LLMConfig = None,
        param_setting_tool: SetParameters = None,
        acquisition_tool: AcquireImage = None,
        tools: list[BaseTool] = (),
        initial_parameters: dict[str, float] = None,
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]] = None,
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
        param_setting_tool : SetParameters
            The tool to use to set the parameters.
        acquisition_tool : AcquireImage
            The BaseTool object used to acquire data. It should contain a 2D
            image acquisition tool and a line scan tool.
        tools : list[BaseTool], optional
            Other tools provided to the agent.
        initial_parameters : dict[str, float], optional
            The initial parameters given as a dictionary of 
            parameter names and values.
        parameter_ranges : list[tuple[float, ...], tuple[float, ...]]
            The ranges of the parameters. It should be given as a list of
            2 tuples, where the first tuple gives the lower bounds and the
            second tuple gives the upper bounds. The order of the parameters
            should match the order of the initial parameters.
        message_db_path : Optional[str], optional
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        build : bool, optional
            Whether to build the internal state of the task manager.
        """
        self.acquisition_tool = acquisition_tool
        
        self.last_acquisition_count_registered = -1
        
        super().__init__(
            llm_config=llm_config,
            param_setting_tool=param_setting_tool,
            tools=[acquisition_tool, *tools],
            initial_parameters=initial_parameters,
            parameter_ranges=parameter_ranges,
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
        
    def run_registration_and_send_image(self, image_path: str) -> None:
        """Register the new image with the previous one and
        send the offset and the new image to the agent.
        
        This routine assumes `self.image_km1` and `self.image_k` of
        `self.acquisition_tool` are already set.
        """
        image_k = self.acquisition_tool.image_k
        image_km1 = self.acquisition_tool.image_km1
        
        if (
            image_km1 is None 
            or self.acquisition_tool.counter == self.last_acquisition_count_registered
        ):
            response, outgoing = self.agent.receive(
                "Here is the new image.",
                image_path=image_path,
                context=self.context,
                return_outgoing_message=True
            )
        else:
            # Run registration.
            image_k = image_k if image_k.ndim == 2 else image_k.mean(-1)
            image_km1 = image_km1 if image_km1.ndim == 2 else image_km1.mean(-1)
            shift = windowed_phase_cross_correlation(image_k, image_km1)
            shift = shift * self.acquisition_tool.psize_k
            
            response, outgoing = self.agent.receive(
                f"Here is the new image. Phase correlation has found the offset between "
                f"the new image and the previous one to be {shift.tolist()} (y, x). Use "
                f"this offset to adjust the line scan positions by **adding** it to both "
                f"the x and y coordinates of the start and end points of the previous line scan.",
                image_path=image_path,
                context=self.context,
                return_outgoing_message=True
            )
            self.last_acquisition_count_registered = self.acquisition_tool.counter
        return response, outgoing
        
    def run(
        self,
        reference_image_path: str,
        reference_feature_description: Optional[str] = None,
        suggested_2d_scan_kwargs: dict = None,
        suggested_parameter_step_size: Optional[float] = None,
        line_scan_step_size: float = None,
        initial_prompt: Optional[str] = None,
        max_iters: int = 20,
        n_past_images_to_keep: Optional[int] = None,
        additional_prompt: Optional[str] = None,
        *args, **kwargs
    ):
        """Run the focusing task.
        
        Parameters
        ----------
        reference_image_path : Optional[str]
            The path to the reference image, which should show a 2D scan
            of the ROI with the desired line scan path indicated by a
            marker. `reference_feature_description` will be ignored if
            this argument is provided.
        reference_feature_description : Optional[str]
            The description of the feature across which line scans should
            be done. Ignored if `reference_image_path` is provided.
        suggested_2d_scan_kwargs : dict
            The suggested kwargs for the 2D scan. The argument should match
            the arguments of the 2D image acquisition tool.
        suggested_parameter_step_size : float
            The suggested step size for the parameter adjustment.
        line_scan_step_size : float
            The step size for the line scan.
        initial_prompt : Optional[str]
            If provided, this prompt will override the default initial prompt.
        max_iters : int, optional
            The maximum number of iterations to run.
        n_past_images_to_keep : int, optional
            The number of past images to keep in the context. If None, all images
            will be kept.
        additional_prompt : Optional[str]
            If provided, this prompt will be added to the initial prompt.
        """
        if reference_image_path is None and reference_feature_description is None:
            raise ValueError(
                "Either `reference_image_path` or `reference_feature_description` must be provided."
            )

        if initial_prompt is None:
            feat_text_description = ""
            if reference_feature_description is not None:
                feat_text_description = f"Also, here is the description of the feature: {reference_feature_description}. "
            param_step_size_prompt = ""
            if suggested_parameter_step_size is not None:
                param_step_size_prompt = dedent(
                    f"""\
                    - The suggested step size for adjusting the parameter is 
                      {suggested_parameter_step_size}. You can adjust the step size
                      to a smaller value if you want to fine-tune the parameter.
                    """
                )
            
            initial_prompt = dedent(
                f"""\
                You will adjust the focus of a scanning microscope by adjusting
                the parameters of its optics. The focusing quality can be evalutated
                by performing a line scan across a thin feature and observe the FWHM
                of its Gaussian fit. The smaller the FWHM, the sharper the image.
                But each time you adjust the focus, the image may drift due to
                the change of the optics. You will need to perform a 2D scan
                prior to the line scan to locate the feature that is line-scanned.
                <img {reference_image_path}> 
                
                You will see a reference 2D scan image in this message. 
                This image is acquired in the region of interest that
                contains the thin feature to be line-scanned. The line scan path
                across that feature is indicated by a marker. {feat_text_description}
                
                Follow the procedure below to focus the microscope:
                
                1. First, perform a 2D scan of the region of interest using the
                   "acquire_image" tool and the following arguments:
                   {suggested_2d_scan_kwargs}. 
                   The image should look similar to the reference image.
                   Determine the coordinates of the line scan path across the feature,
                   and use the "scan_line" tool to perform a line scan across the feature.
                2. The line scan tool will return a plot along the scan line. You should
                   see a peak in the plot. A Gaussian fit will be included in the plot
                   and the FWHM of the Gaussian fit will be shown.
                3. Adjust the optics parameters using the parameter setting tool.
                   The initial parameter values are {self.initial_parameters}.
                4. Acquire an image of the region using the image acquisition tool.
                   Here are the suggested arguments: {suggested_2d_scan_kwargs}. The 
                   image acquired may have drifted compared to the last one you saw,
                   but you should still see the line-scanned feature there. If not,
                   try adjusting the image acquisition tool's parameters to locate that
                   feature. Along with this image, you will also be given the offset of
                   this image compared to the previous image found through phase correlation.
                   Use this offset to adjust the line scan positions. Note that the offset
                   is just a suggestion. If the new image does not appear to have any overlap
                   with the previous one, the offset won't be reliable. In that case, try
                   adjusting the image acquisition tool's parameters to move the field of view
                   closer to the previous image.
                5. Once you find the line-scanned feature, perform a new line scan across
                   it again. Due to the drift, the start/end points' coordinates may need to
                   be changed. Read the coordinates from the axis ticks.
                6. You will be presented with the new line scan plot and the FWHM of the
                   Gaussian fit.
                7. Compare the new FWHM with the last one. If it is smaller, you are on the
                   right track. Keep adjusting the parameters to the same direction. Otherwise,
                   adjust the parameters in the opposite direction.
                8. Repeat the process from step 4.
                9. When you find the FWHM is minimized, you are done. Add "TERMINATE" to 
                   your response to hand over control back to the user.
                   
                Other notes:
                
                - Your line scan should cross only one line feature, and you should see
                  **exactly one peak** in the line scan plot. If there isn't one, or if there
                  are multiple peaks, or if the Gaussian fit looks bad, check your arguments
                  to the line scan tool and run it again. Make sure your line scan strictly
                  follow the marker in the reference image.
                - The line scan plot should show a complete peak. If the peak is incomplete,
                  adjust the line scan tool's arguments to make it complete.
                - The minimal point of the FWHM is indicated by an inflection of the trend
                  of the FWHM with regards to the optics parameters. For example, if the FWHM
                  is 3 with a parameter value of 10, then 1 with a parameter value of 11, then
                  3 with a parameter value of 12, this means the optimal parameter value is around
                  11.
                {param_step_size_prompt}
                - When calling a tool, explain what you are doing.
                - When making a tool call, only call one tool at a time. Do not call multiple
                  tools in one response.
                
                When you finish or when you need human input, add "TERMINATE" to your response.\
                """
            )
        if additional_prompt is not None:
            initial_prompt += "\nAdditional instructions:\n" + additional_prompt
        
        # Always keep the first (reference) image.
        self.run_feedback_loop(
            initial_prompt=initial_prompt,
            initial_image_path=reference_image_path,
            store_all_images_in_context=True,
            allow_non_image_tool_responses=True,
            n_first_images_to_keep=1,
            n_past_images_to_keep=n_past_images_to_keep,
            max_rounds=max_iters,
            hook_functions={
                "image_path_tool_response": self.run_registration_and_send_image
            },
            *args, **kwargs
        )


class ParameterTuningTaskManager(BaseParameterTuningTaskManager):
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
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
            param_setting_tool=param_setting_tool,
            tools=[param_setting_tool],
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
            initial_prompt = dedent(
                f"""\
                You are tuning the parameters of a microscope to attain the best
                image sharpness. The parameters are {list(self.parameter_names)},
                and their current values are {initial_parameter_values}. An image acquired
                with the current parameters is shown below. 
                
                <img {last_img_path}>
                
                Here are the tunable ranges of the parameters:
                {bounds_str}
                
                You can change the parameters using your parameter setting tool. 
                An image acquired with the new parameters will be given to you
                after each parameter change. Here are some detailed instructions:
                
                - Tune parameters one by one. Start with the first parameter, tweak it
                to attain the sharpest possible image, then move on to the next parameter.
                Do not change more than one parameter at a time.
                - The sharpness of the image is convex with regards to the parameters. There
                is only one optimal point; assume there is no local maximum. As such, if
                you find the image comes more blurry when changing a parameter in a direction,
                you should consider changing it the other way; if you find the image comes
                sharper when changing a parameter in a direction, you are on the right track.
                - For each parameter, first get a coarse estimate of the optimal value, then
                fine-tune it. To get a coarse estimate, look for a peak in the sharpness. In
                other words, find a parameter value that gives a sharper image than the value
                immediately before and after it. For example, if the image becomes sharper when
                you increase the parameter from 4 to 5, but becomes blurrier when you increase
                it from 5 to 6, then the optimal value is around 5.
                - Choose the step size for changing parameters wisely. For each parameter, start
                with a large step size, and decrease it as you get closer to the optimal point.
                - Only call the parameter setting tool one at a time. Do not call it multiple times
                in one response.
                
                When you finish or when you need human input, add "TERMINATE" to your response.\
                """
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
                    "Termination condition triggered. What to do next? Type \"exit\" to exit. "
                )
                if message.lower() == "exit":
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
