from typing import Optional, Tuple, Sequence, Literal
import logging
import copy
import json

import numpy as np
import botorch.acquisition

from sciagent.api.llm_config import LLMConfig
from sciagent.api.memory import MemoryManagerConfig

from eaa.tool.imaging.acquisition import AcquireImage
from eaa.tool.imaging.param_tuning import SetParameters
from eaa.task_manager.imaging.analytical_feature_tracking import AnalyticalFeatureTrackingTaskManager
from eaa.task_manager.tuning.base import BaseParameterTuningTaskManager
from eaa.tool.imaging.registration import ImageRegistration
from eaa.tool.bo import BayesianOptimizationTool
from eaa.util import to_numpy
from eaa.image_proc import check_feature_presence_llm

logger = logging.getLogger(__name__)


class AnalyticalScanningMicroscopeFocusingTaskManager(BaseParameterTuningTaskManager):
    
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
        line_scan_tool_x_coordinate_args: Tuple[str, ...] = ("x_center",),
        line_scan_tool_y_coordinate_args: Tuple[str, ...] = ("y_center",),
        image_acquisition_tool_x_coordinate_args: Tuple[str, ...] = ("x_center",),
        image_acquisition_tool_y_coordinate_args: Tuple[str, ...] = ("y_center",),
        *args, **kwargs
    ):
        """Analytical scanning microscope focusing task manager driven
        by logic instead of LLM.
        
        The workflow is as follows:
        1. Acquire a 2D image in the user-specified region of interest.
        2. Run a line scan at user-specified coordinates and record the FWHM of the Gaussian fit.
        3. Change parameter and acquire a new 2D image.
        4.1. If the same feature remains in the FOV, run image registration to get the offset and
           adjust 1D/2D scan coordinates.
        4.2. If the feature is no longer in the FOV, run a spiral feature tracking to find the feature.
        5. Repeat 1 - 3 a few times to collect initial data for Bayesian optimization.
        6. Use Bayesian optimization to suggest new parameters.
        7. Change parameter. 
        8. Run image registration or feature tracking as in 4.
        9. Run line scan and record the FWHM of the Gaussian fit, update Gaussian process model.
        10. Repeat 6 - 9 until the FWHM is minimized.

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
        bo_tool : BayesianOptimizationTool, optional
            The Bayesian optimization tool to use.
        image_registration_tool : ImageRegistration, optional
            The image registration tool. This tool is optional and is only
            used for the feature tracking sub-task if `use_feature_tracking_subtask`
            is True. To use registration in the focusing task manager, refer to
            ``use_registration_in_workflow`` in the ``run`` method.
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
        line_scan_tool_x_coordinate_args: Tuple[str, ...]
            The names of the arguments of the line scan tool that specify x-coordinates.
            When the lab-frame coordinates drift and offsets are found, these arguments
            will be updated using the offsets.
        line_scan_tool_y_coordinate_args: Tuple[str, ...]
            See `line_scan_tool_x_coordinate_args`.
        image_acquisition_tool_x_coordinate_args: Tuple[str, ...]
            See `line_scan_tool_x_coordinate_args`.
        image_acquisition_tool_y_coordinate_args: Tuple[str, ...]
            See `line_scan_tool_y_coordinate_args`.
        """
        if acquisition_tool is None:
            raise ValueError("`acquisition_tool` must be provided.")
        
        self.acquisition_tool = acquisition_tool
        self.bo_tool = self.create_bo_tool(parameter_ranges)
        self.image_registration_tool = self.create_image_registration_tool(acquisition_tool)
        
        if hasattr(acquisition_tool, "line_scan_return_gaussian_fit"):
            acquisition_tool.line_scan_return_gaussian_fit = True
        else:
            logger.warning(
                "`line_scan_return_gaussian_fit` attribute is not found in the acquisition tool."
            )
        
        self.last_acquisition_count_registered = 0
        self.last_acquisition_count_stitched = 0
        
        self.feature_tracking_task_manager: Optional[AnalyticalFeatureTrackingTaskManager] = None
        
        self.line_scan_tool_x_coordinate_args = line_scan_tool_x_coordinate_args
        self.line_scan_tool_y_coordinate_args = line_scan_tool_y_coordinate_args
        self.image_acquisition_tool_x_coordinate_args = image_acquisition_tool_x_coordinate_args
        self.image_acquisition_tool_y_coordinate_args = image_acquisition_tool_y_coordinate_args
        
        self.line_scan_kwargs = {}
        self.image_acquisition_kwargs = {}
        
        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            param_setting_tool=param_setting_tool,
            initial_parameters=initial_parameters,
            parameter_ranges=parameter_ranges,
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
        
    def create_bo_tool(self, parameter_ranges: list[tuple[float, ...], tuple[float, ...]]):
        bo_tool = BayesianOptimizationTool(
            bounds=parameter_ranges,
            n_observations=1,
            kernel_lengthscales=None,
            acquisition_function_class=botorch.acquisition.UpperConfidenceBound,
            acquisition_function_kwargs={"beta": 1.0},
        )
        return bo_tool
    
    def create_image_registration_tool(self, acquisition_tool: AcquireImage):
        image_registration_tool = ImageRegistration(
            image_acquisition_tool=acquisition_tool,
            reference_image=None,
            reference_pixel_size=1.0,
            image_coordinates_origin="top_left",
        )
        return image_registration_tool
    
    def prerun_check(
        self, 
        initial_sampling_range: Optional[Tuple[float, float]], 
        parameter_change_step_limit: Optional[float | Tuple[float, ...]]
    ) -> bool:
        if initial_sampling_range is None:
            raise ValueError("initial_sampling_range must be provided.")
        if len(initial_sampling_range) != len(self.parameter_names):
            raise ValueError(
                f"The length of initial_sampling_range must be the same as the number of parameters, "
                f"but got {len(initial_sampling_range)} and {len(self.parameter_names)}."
            )
        if isinstance(parameter_change_step_limit, Sequence):
            if len(parameter_change_step_limit) != len(self.parameter_names):
                raise ValueError(
                    f"The length of parameter_change_step_limit must be the same as the number of parameters, "
                    f"but got {len(parameter_change_step_limit)} and {len(self.parameter_names)}."
                )
        return True
        
    def run(
        self, 
        initial_2d_scan_kwargs: dict = None,
        initial_line_scan_kwargs: dict = None,
        n_initial_points: int = 5,
        initial_sampling_window_size: Optional[Tuple[float, ...]] = None,
        n_max_bo_iterations: int = 99,
        parameter_change_step_limit: Optional[float | Tuple[float, ...]] = None,
        termination_behavior: Literal["ask", "return"] = "ask",
        *args, **kwargs
    ):
        """Run the focusing task.
        
        Parameters
        ----------
        initial_line_scan_kwargs: dict
            The keyword arguments for the initial line scan. The argument should
            match the signature of the `scan_line` method of the acquisition tool.
        initial_2d_scan_kwargs: dict
            The keyword arguments for the initial 2D scan. The argument should
            match the signature of the `acquire_image` method of the acquisition tool.
        n_initial_line_scans: int
            The number of initial points to prime the Gaussian process model.
        initial_sampling_range: Optional[Tuple[float, float]]
            The range over which the initial measurements for Bayesian optimization
            are sampled. Should be a tuple with the same length as the number of parameters.
        n_max_bo_iterations: int
            The maximum number of Bayesian optimization iterations.
        parameter_change_step_limit: float
            The limit on the step size of the parameter change. Parameter changes
            are clipped to this limit if the absolute difference between the one
            suggested by BO and the current parameter value is larger than this limit.
            If None, no limit is applied.
        termination_behavior: Literal["ask", "return"]
            The behavior when the task manager reaches the maximum number of Bayesian
            optimization iterations. If "ask", the task manager will ask the user for
            input. If "return", the task manager will return.
        """
        try:
            self.prerun_check(initial_sampling_window_size, parameter_change_step_limit)
            self.initialize_kwargs_buffers(initial_line_scan_kwargs, initial_2d_scan_kwargs)
            
            self.record_system_message(
                "Interrupt the process and give correction offset anytime by pressing "
                "Ctrl+C in terminal."
            )
            
            # Initial 2D scan to populate image buffer of acquisition tool.
            self.run_2d_scan()
            
            # Initialize BO tool.
            self.collect_initial_data_for_bo(
                current_x=np.array(list(self.initial_parameters.values())),
                sampling_range=initial_sampling_window_size,
                n=n_initial_points,
            )
            self.bo_tool.build(acquisition_function_kwargs=None)
            
            # Run Bayesian optimization.
            for i_iter in range(n_max_bo_iterations):
                iter_message = f"Running Bayesian optimization iteration {i_iter}..."
                logger.info(iter_message)
                self.record_system_message(iter_message)
                p_suggested = self.get_suggested_next_parameters(parameter_change_step_limit)
                suggestion_message = f"Suggested parameter: {p_suggested}"
                logger.info(suggestion_message)
                self.record_system_message(suggestion_message)
                try:
                    self.run_tuning_iteration(p_suggested)
                except KeyboardInterrupt:
                    if not self.apply_user_correction_offset():
                        raise
                    continue
            report = self.generate_report_csv()
            final_report_message = f"Final report:\n{report}"
            logger.info(final_report_message)
            self.record_system_message(final_report_message, update_context=True)
        except KeyboardInterrupt:
            pass
        
        if termination_behavior == "ask":
            self.run_conversation()
        elif termination_behavior == "return":
            return
        else:
            raise ValueError(
                f"Invalid termination behavior: {termination_behavior}. "
                "Must be one of 'ask' or 'return'."
            )
        
    def initialize_kwargs_buffers(
        self, initial_line_scan_kwargs: dict, initial_2d_scan_kwargs: dict
    ):
        self.line_scan_kwargs = copy.deepcopy(initial_line_scan_kwargs)
        self.image_acquisition_kwargs = copy.deepcopy(initial_2d_scan_kwargs)
        
    def run_line_scan(self) -> float:
        """Run a line scan and return the FWHM of the Gaussian fit.

        Returns
        -------
        float
            The FWHM of the Gaussian fit.
        """
        res = self.acquisition_tool.acquire_line_scan(**self.line_scan_kwargs)
        try:
            res = json.loads(res)
        except json.JSONDecodeError:
            raise ValueError(
                f"The line scan tool should return a stringified JSON object, but got {res}."
            )
        if "fwhm" not in res:
            raise ValueError(
                f"The stringified JSON object should contain the 'fwhm' key, but got {res}."
            )
        content = f"Line scan completed with kwargs {self.line_scan_kwargs}. FWHM = {res['fwhm']:.4f}"
        image_path = res.get("image_path")
        if isinstance(image_path, str):
            self.record_system_message(content, image_path=image_path)
        else:
            self.record_system_message(content)
        return res["fwhm"]
    
    def update_bo_model(self, fwhm: float):
        x = self.param_setting_tool.get_parameter_at_iteration(-1)
        x = np.array(x).reshape(1, -1)
        # Use negative FWHM because we want to minimize the FWHM.
        self.bo_tool.update(x, -np.array([[fwhm]]))
        
    def run_2d_scan(self):
        image_path = self.acquisition_tool.acquire_image(**self.image_acquisition_kwargs)
        content = f"Acquired 2D scan with kwargs: {self.image_acquisition_kwargs}"
        if isinstance(image_path, str):
            self.record_system_message(content, image_path=image_path)
        else:
            self.record_system_message(content)

    def get_suggested_next_parameters(self, step_size_limit: Optional[float | Tuple[float, ...]] = None):
        p_suggested = to_numpy(self.bo_tool.suggest(n_suggestions=1)[0])
        p_current = to_numpy(self.param_setting_tool.get_parameter_at_iteration(-1))
        if step_size_limit is not None:
            signs = np.sign(p_suggested - p_current)
            step_sizes = np.abs(p_suggested - p_current)
            step_sizes = np.clip(step_sizes, min=None, max=step_size_limit)
            p_suggested = p_current + signs * step_sizes
        return p_suggested
    
    def find_offset_and_feature_presence(self) -> Tuple[np.ndarray, bool]:
        """Find the offset between the latest image and the previous image 
        and check if the feature is present.

        Returns
        -------
        np.ndarray
            The offset between the latest image and the previous image.
            Offset is in physical units, i.e., pixel size is already accounted for.
        bool
            Whether the feature is present in the current image.
        """
        image_k = self.acquisition_tool.image_k
        image_km1 = self.acquisition_tool.image_km1
        
        if self.llm_config is None:
            logger.warning("`llm_config` is not provided. Unable to check if the feature is present.")
            is_present = True
        else:
            is_present = check_feature_presence_llm(
                task_manager=self,
                image=image_k,
                reference_image=image_km1,
            )
        
        shift = self.image_registration_tool.register_images(
            image_t=self.image_registration_tool.process_image(image_k),
            image_r=self.image_registration_tool.process_image(image_km1),
            psize_t=self.acquisition_tool.psize_k,
            psize_r=self.acquisition_tool.psize_km1,
            return_correlation_value=False,
        ).astype(float)
        
        # Count in the difference of scan positions.
        scan_pos_diff = np.array([
            float(self.acquisition_tool.image_acquisition_call_history[-1][f"loc_{dir}"])
            - float(self.acquisition_tool.image_acquisition_call_history[-2][f"loc_{dir}"])
            for dir in ["y", "x"]
        ]).astype(float)
        shift += scan_pos_diff
        return shift, is_present
    
    def apply_offset_to_kwargs_buffers(self, offset: np.ndarray):
        for arg in self.line_scan_tool_x_coordinate_args:
            self.line_scan_kwargs[arg] += offset[1]
        for arg in self.line_scan_tool_y_coordinate_args:
            self.line_scan_kwargs[arg] += offset[0]
        for arg in self.image_acquisition_tool_x_coordinate_args:
            self.image_acquisition_kwargs[arg] += offset[1]
        for arg in self.image_acquisition_tool_y_coordinate_args:
            self.image_acquisition_kwargs[arg] += offset[0]
            
    def collect_initial_data_for_bo(
        self, 
        current_x: np.ndarray,
        sampling_range: np.ndarray,
        n: int = 5,
    ):
        if len(sampling_range) != len(self.parameter_names):
            raise ValueError(
                f"The length of sampling_range must be the same as the number of parameters, "
                f"but got {len(sampling_range)} and {len(self.parameter_names)}."
            )
        sampling_range = np.array(sampling_range)
        if len(current_x) != len(self.parameter_names):
            raise ValueError(
                f"The length of current_x must be the same as the number of parameters, "
                f"but got {len(current_x)} and {len(self.parameter_names)}."
            )
        current_x = np.array(current_x)
        
        xs = np.linspace(current_x - sampling_range, current_x + sampling_range, n)
        for x in xs:
            while True:
                try:
                    self.run_tuning_iteration(x)
                    break
                except KeyboardInterrupt:
                    if not self.apply_user_correction_offset():
                        raise

    def run_tuning_iteration(self, x: np.ndarray):
        if len(x) != len(self.parameter_names):
            raise ValueError(
                f"The length of x must be the same as the number of parameters, "
                f"but got {len(x)} and {len(self.parameter_names)}."
            )
        x = np.array(x)
        self.param_setting_tool.set_parameters(x)
        self.run_2d_scan()
        offset, is_present = self.find_offset_and_feature_presence()
        if not is_present:
            msg = "Feature is not present in the current image. Running feature tracking sub-task."
            logger.info(msg)
            self.record_system_message(msg)
            offset = self.run_feature_tracking_subtask()
        self.apply_offset_to_kwargs_buffers(offset)
        fwhm = self.run_line_scan()
        self.update_bo_model(fwhm)

    def apply_user_correction_offset(self) -> bool:
        message = (
            "Manual correction requested. Enter offset as 'y,x' (blank to stop): "
        )
        while True:
            response = self.get_user_input(message, display_prompt_in_webui=True).strip()
            if response == "":
                return False
            parts = [part for part in response.replace(",", " ").split() if part]
            if len(parts) != 2:
                logger.info("Invalid offset format. Use two numbers like 'y,x'.")
                continue
            try:
                offset = np.array([float(parts[0]), float(parts[1])], dtype=float)
            except ValueError:
                logger.info("Invalid offset values. Use numeric values like 'y,x'.")
                continue
            self.apply_offset_to_kwargs_buffers(offset)
            correction_message = f"Applied user correction offset: {offset.tolist()}"
            logger.info(correction_message)
            self.record_system_message(correction_message)
            return True

    def generate_report_csv(self) -> str:
        xs = self.bo_tool.xs_untransformed.tolist()
        fwhms = self.bo_tool.ys_untransformed.tolist()
        report = "Parameters,FWHM\n"
        for x, fwhm in zip(xs, fwhms):
            report += f"{x[0]},{fwhm[0]}\n"
        return report

    def run_feature_tracking_subtask(self):
        if self.feature_tracking_task_manager is None:
            self.feature_tracking_task_manager = AnalyticalFeatureTrackingTaskManager(
                llm_config=self.llm_config,
                image_acquisition_tool=self.acquisition_tool,
                image_acquisition_tool_x_coordinate_args=self.image_acquisition_tool_x_coordinate_args,
                image_acquisition_tool_y_coordinate_args=self.image_acquisition_tool_y_coordinate_args,
                message_db_path=self.message_db_path,
            )
        offset = self.feature_tracking_task_manager.run(
            current_acquisition_kwargs=self.image_acquisition_kwargs,
            reference_image=self.acquisition_tool.image_km1,
            step_size=[
                self.acquisition_tool.image_acquisition_call_history[-1]["size_y"] * 0.8, 
                self.acquisition_tool.image_acquisition_call_history[-1]["size_x"] * 0.8
            ],
            reference_image_pixel_size=self.acquisition_tool.psize_km1,
            n_max_rounds=20,
        )
        return offset
