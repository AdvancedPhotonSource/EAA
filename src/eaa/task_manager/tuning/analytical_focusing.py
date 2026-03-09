from typing import Optional, Tuple, Sequence, Literal, Any
import logging
import copy
import json
import re
from pathlib import Path

import numpy as np
import botorch.acquisition
import matplotlib.pyplot as plt
from sciagent.message_proc import generate_openai_message
from sciagent.skill import SkillMetadata

from sciagent.api.llm_config import LLMConfig
from sciagent.api.memory import MemoryManagerConfig
from sciagent.task_manager.base import BaseTaskManager
from sciagent.tool.base import BaseTool
from sciagent.message_proc import print_message

from eaa.tool.imaging.acquisition import AcquireImage
from eaa.tool.imaging.line_scan_predictor import LineScanPredictor
from eaa.tool.imaging.param_tuning import SetParameters
from eaa.task_manager.tuning.base import BaseParameterTuningTaskManager
from eaa.tool.imaging.registration import ImageRegistration
from eaa.tool.optimization import (
    BaseSequentialOptimizationTool,
    BayesianOptimizationTool,
)
from eaa.tool.regression import MultivariateLinearRegression
from eaa.util import to_numpy

logger = logging.getLogger(__name__)


class LineScanValidationFailed(RuntimeError):
    pass


class RegistrationFailed(RuntimeError):
    pass


class AnalyticalScanningMicroscopeFocusingTaskManager(BaseParameterTuningTaskManager):
    
    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        param_setting_tool: SetParameters = None,
        acquisition_tool: AcquireImage = None,
        optimization_tool: Optional[BaseSequentialOptimizationTool] = None,
        initial_parameters: dict[str, float] = None,
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]] = None,
        message_db_path: Optional[str] = None,
        build: bool = True,
        line_scan_tool_x_coordinate_args: Tuple[str, ...] = ("x_center",),
        line_scan_tool_y_coordinate_args: Tuple[str, ...] = ("y_center",),
        image_acquisition_tool_x_coordinate_args: Tuple[str, ...] = ("x_center",),
        image_acquisition_tool_y_coordinate_args: Tuple[str, ...] = ("y_center",),
        registration_method: Literal["phase_correlation", "sift", "mutual_information", "llm"] = "phase_correlation",
        registration_algorithm_kwargs: Optional[dict[str, Any]] = None,
        registration_target: Literal["previous", "initial"] = "previous",
        run_line_scan_checker: bool = True,
        run_offset_calibration: bool = True,
        line_scan_predictor_tool: Optional[LineScanPredictor] = None,
        dual_drift_selection_priming_iterations: int = 3,
        dual_drift_estimation_primary_source: Literal["registration", "line_scan_predictor"] = "line_scan_predictor",
        *args, **kwargs
    ):
        """Analytical scanning microscope focusing task manager driven
        by logic instead of LLM.
        
        The workflow is as follows:
        1. Acquire a 2D image in the user-specified region of interest.
        2. Run a line scan at user-specified coordinates and record the FWHM of the Gaussian fit.
        3. Change parameter and acquire a new 2D image. The change of parameter causes the sample
           to drift relative to the beam.
        4. Register the acquired image with the reference image (previous or initial) to estimate
           the drift correction that should be applied to the line scan and image acquisition tools.
           Update the positions for line scan and image acquisition.
        5. Repeat 1 - 4 a few times to collect initial data for Bayesian optimization.
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
        optimization_tool : BaseSequentialOptimizationTool, optional
            The optimization tool to use. Supported options include
            `BayesianOptimizationTool` and `QuadraticOptimizationTool`.
        image_registration_tool : ImageRegistration, optional
            The image registration tool.
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
        registration_algorithm_kwargs : Optional[dict[str, Any]]
            Optional keyword arguments forwarded to the selected image
            registration algorithm when aligning consecutive 2D scans.
        registration_target : Literal["previous", "initial"], optional
            The reference image used by the registration branch of drift
            correction.  "previous" (default) registers each new 2D scan
            against the immediately preceding one; small registration errors
            therefore accumulate over many iterations.  "initial" registers
            every new 2D scan against the very first scan, which prevents
            error accumulation at the cost of requiring sufficient overlap
            between the current and the initial image.
        run_line_scan_checker : bool, optional
            If True, run the LLM-based line-scan quality checker and allow it
            to request scan-argument adjustments before accepting a line scan.
        run_offset_calibration : bool, optional
            If True, run 2D image acquisition and image-registration-based offset
            calibration. If False, the loop only performs parameter setting,
            line scan, and optimization updates/suggestions.
        dual_drift_selection_priming_iterations : int, optional
            Number of parameter–drift samples to collect (warm-up phase) before
            the linear model is used to arbitrate between image registration and
            line scan predictor drift estimates.  Only relevant when
            ``line_scan_predictor_tool`` is provided.
        line_scan_predictor_tool : LineScanPredictor, optional
            If provided, both image registration and the line scan predictor
            are run after each 2D acquisition to estimate drift.  For the
            first ``n_parameter_drift_points_before_prediction`` parameter
            adjustments only the source selected by
            ``dual_drift_estimation_primary_source`` is used; concurrently a
            linear model is fit mapping optics parameters to cumulative drift
            from the initial position.  Afterwards the linear model prediction
            is used to arbitrate: the drift estimate (from either method)
            that is closer to the linear model prediction is chosen as the
            correct drift, and the linear model is then updated with that
            chosen value.  Requires ``run_offset_calibration=True``.
        dual_drift_estimation_primary_source : {"registration", "line_scan_predictor"}
            Which drift-estimation method to trust exclusively during the first
            ``n_parameter_drift_points_before_prediction`` iterations when
            ``line_scan_predictor_tool`` is provided.  After that point both
            methods are evaluated and the linear model arbitrates.  Defaults
            to ``"line_scan_predictor"``.
        """
        if acquisition_tool is None:
            raise ValueError("`acquisition_tool` must be provided.")
        
        self.acquisition_tool = acquisition_tool
        if optimization_tool is None:
            self.optimization_tool = self.create_bo_tool(parameter_ranges)
        else:
            self.optimization_tool = optimization_tool
        self.image_registration_tool = self.create_image_registration_tool(
            acquisition_tool,
            llm_config=llm_config,
            registration_method=registration_method,
        )
        self.registration_algorithm_kwargs = copy.deepcopy(
            registration_algorithm_kwargs or {}
        )
        if registration_target not in ("previous", "initial"):
            raise ValueError(
                f"`registration_target` must be 'previous' or 'initial', got {registration_target!r}."
            )
        self.registration_target = registration_target
        
        if hasattr(acquisition_tool, "line_scan_return_gaussian_fit"):
            acquisition_tool.line_scan_return_gaussian_fit = True
        else:
            logger.warning(
                "`line_scan_return_gaussian_fit` attribute is not found in the acquisition tool."
            )
        
        self.last_acquisition_count_registered = 0
        self.last_acquisition_count_stitched = 0

        self.line_scan_tool_x_coordinate_args = line_scan_tool_x_coordinate_args
        self.line_scan_tool_y_coordinate_args = line_scan_tool_y_coordinate_args
        self.image_acquisition_tool_x_coordinate_args = image_acquisition_tool_x_coordinate_args
        self.image_acquisition_tool_y_coordinate_args = image_acquisition_tool_y_coordinate_args
        
        self.line_scan_kwargs = {}
        self.image_acquisition_kwargs = {}

        self.run_line_scan_checker = run_line_scan_checker
        self.run_offset_calibration = run_offset_calibration
        if line_scan_predictor_tool is not None and not run_offset_calibration:
            raise ValueError(
                "`line_scan_predictor_tool` requires `run_offset_calibration=True` "
                "because a 2D image must be acquired before the predictor can run."
            )
        
        self.line_scan_predictor_tool = line_scan_predictor_tool
        self.dual_drift_estimation_primary_source = dual_drift_estimation_primary_source
        self.dual_drift_selection_priming_iterations = (
            dual_drift_selection_priming_iterations
        )
        if self.dual_drift_selection_priming_iterations < 1:
            raise ValueError(
                "`dual_drift_selection_priming_iterations` must be >= 1."
            )
        self.drift_model_y = MultivariateLinearRegression()
        self.drift_model_x = MultivariateLinearRegression()
        self.initial_image_acquisition_position: np.ndarray | None = None
        self.initial_line_scan_position: np.ndarray | None = None
        
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
            acquisition_function_kwargs={"beta": 3.0},
        )
        return bo_tool
    
    def create_image_registration_tool(
        self,
        acquisition_tool: AcquireImage,
        llm_config: Optional[LLMConfig] = None,
        registration_method: Literal["phase_correlation", "sift", "mutual_information", "llm"] = "llm",
    ):
        image_registration_tool = ImageRegistration(
            image_acquisition_tool=acquisition_tool,
            llm_config=llm_config,
            reference_image=None,
            reference_pixel_size=1.0,
            image_coordinates_origin="top_left",
            registration_method=registration_method,
            log_scale=True
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
        n_max_iterations: int = 99,
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
            The range over which the initial measurements for sequential optimization
            are sampled. Should be a tuple with the same length as the number of parameters.
        n_max_iterations: int
            The maximum number of sequential optimization iterations.
        parameter_change_step_limit: float
            The limit on the step size of the parameter change. Parameter changes
            are clipped to this limit if the absolute difference between the one
            suggested by the optimization tool and the current parameter value is larger than this limit.
            If None, no limit is applied.
        termination_behavior: Literal["ask", "return"]
            The behavior when the task manager reaches the maximum number of sequential
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
            
            # Initial measurements before optimization starts.
            if self.run_offset_calibration:
                self.run_2d_scan()
            self.run_line_scan()
            
            # Initialize optimization tool.
            self.collect_initial_data_for_optimization_tool(
                current_x=np.array(list(self.initial_parameters.values())),
                sampling_range=initial_sampling_window_size,
                n=n_initial_points,
            )
            self.optimization_tool.build(acquisition_function_kwargs=None)
            
            # Run sequential optimization.
            for i_iter in range(n_max_iterations):
                iter_message = f"Running sequential optimization iteration {i_iter}..."
                logger.info(iter_message)
                self.record_system_message(iter_message)
                p_suggested = self.get_suggested_next_parameters(parameter_change_step_limit)
                suggestion_message = f"Parameter suggested by optimization tool: {p_suggested}"
                logger.info(suggestion_message)
                self.record_system_message(suggestion_message)
                try:
                    self.run_tuning_iteration(p_suggested)
                except (KeyboardInterrupt, RuntimeError) as e:
                    if isinstance(e, RuntimeError):
                        message = f"A runtime error occurred: {e}. Please set offset manually."
                        logger.error(message)
                        self.record_system_message(message)
                    if not self.run_offset_calibration:
                        if isinstance(e, KeyboardInterrupt):
                            raise KeyboardInterrupt(
                                "Process interrupted while offset calibration is disabled."
                            ) from e
                    if not self.apply_user_correction_offset():
                        if isinstance(e, KeyboardInterrupt):
                            raise KeyboardInterrupt(
                                "Process interrupted by user during manual offset correction."
                            ) from e
                        raise RuntimeError(
                            "Manual offset correction was required but not provided."
                        ) from e
                    continue
            report = self.generate_report_csv()
            final_report_message = f"Final report:\n{report}"
            logger.info(final_report_message)
            self.record_system_message(final_report_message, update_context=True)
        except KeyboardInterrupt:
            pass
        
        if termination_behavior == "ask":
            logger.info("Entering chat mode...")
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
        if self.line_scan_kwargs is not None:
            self.initial_line_scan_position = self.extract_line_scan_position(self.line_scan_kwargs)
        
    def run_line_scan(self) -> float:
        """Run a line scan and return the FWHM of the Gaussian fit.

        Returns
        -------
        float
            The FWHM of the Gaussian fit.
        """
        while True:
            self.record_system_message(f"Acquiring line scan with ```{self.line_scan_kwargs}```")
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
            content = f"Line scan completed with kwargs: ```{self.line_scan_kwargs}```\nFWHM = {res['fwhm']:.4f}"
            image_path = res.get("image_path")
            if isinstance(image_path, str):
                self.record_system_message(content, image_path=image_path)
            else:
                self.record_system_message(content)

            if self.run_line_scan_checker:
                check_res = (
                    self.check_line_scan(
                        image_path,
                        res,
                        line_scan_residual_warning_threshold=0.015,
                    )
                    if isinstance(image_path, str)
                    else {"result": "ok"}
                )
                self.record_system_message(
                    f"Line scan validation result:```{check_res}```"
                )
            else:
                check_res = {"result": "ok"}

            if check_res["result"] == "ok":
                return res["fwhm"]
            if check_res["result"] == "adjusted":
                self.record_system_message(
                    f"Line scan adjusted. New line scan kwargs:```{self.line_scan_kwargs}```"
                )
                continue
            if check_res["result"] == "failed":
                raise LineScanValidationFailed("Line scan validation failed after retries.")
            raise ValueError(f"Unknown check_line_scan result: {check_res}")

    def parse_json_from_response(self, response_text: str) -> dict[str, Any]:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
            if match is None:
                raise ValueError(f"Unable to parse JSON from response: {response_text}")
            return json.loads(match.group(0))

    def extract_line_scan_position(self, kwargs: dict[str, float]) -> np.ndarray:
        if len(self.line_scan_tool_x_coordinate_args) == 0 or len(self.line_scan_tool_y_coordinate_args) == 0:
            raise ValueError("Line scan coordinate args must not be empty.")
        x_arg = self.line_scan_tool_x_coordinate_args[0]
        y_arg = self.line_scan_tool_y_coordinate_args[0]
        if x_arg not in kwargs or y_arg not in kwargs:
            raise ValueError(
                f"Cannot extract line scan position from kwargs. Missing {x_arg} or {y_arg} in {kwargs}."
            )
        return np.array([float(kwargs[y_arg]), float(kwargs[x_arg])], dtype=float)

    def extract_image_acquisition_position(self, kwargs: dict[str, float]) -> np.ndarray:
        if (
            len(self.image_acquisition_tool_x_coordinate_args) == 0
            or len(self.image_acquisition_tool_y_coordinate_args) == 0
        ):
            raise ValueError("Image acquisition coordinate args must not be empty.")
        x_arg = self.image_acquisition_tool_x_coordinate_args[0]
        y_arg = self.image_acquisition_tool_y_coordinate_args[0]
        if x_arg not in kwargs or y_arg not in kwargs:
            raise ValueError(
                "Cannot extract image acquisition position from kwargs. "
                f"Missing {x_arg} or {y_arg} in {kwargs}."
            )
        return np.array([float(kwargs[y_arg]), float(kwargs[x_arg])], dtype=float)

    def update_linear_drift_models(
        self,
        parameters: np.ndarray,
        current_position_yx: np.ndarray | None = None,
    ):
        if self.line_scan_predictor_tool is None:
            return
        if self.initial_line_scan_position is None:
            return

        if current_position_yx is None:
            current_position_yx = self.extract_line_scan_position(
                self.line_scan_kwargs
            )

        delta_yx = current_position_yx - self.initial_line_scan_position
        x_train = np.array(parameters, dtype=float).reshape(1, -1).tolist()
        self.drift_model_y.update(x=x_train, y=[[float(delta_yx[0])]])
        self.drift_model_x.update(x=x_train, y=[[float(delta_yx[1])]])
        self.record_system_message(
            "Updated linear drift model with parameter-drift sample:```"
            f"parameters={np.array(parameters, dtype=float).tolist()}, "
            f"delta_yx={delta_yx.tolist()}, "
            f"n_samples={self.drift_model_y.get_n_parameter_drift_points_collected()}.```"
        )

    def record_linear_drift_model_visualizations(self) -> None:
        image_paths = []
        for axis_name, model in [("y", self.drift_model_y), ("x", self.drift_model_x)]:
            fig = None
            try:
                fig = model.visualize_status()
                if fig is None:
                    continue
                fig_path = BaseTool.save_image_to_temp_dir(
                    fig=fig,
                    filename=f"linear_drift_model_{axis_name}_status.png",
                    add_timestamp=True,
                )
                image_paths.append(fig_path)
            except Exception as exc:
                logger.warning(
                    "Failed to visualize linear drift model status for %s-axis: %s",
                    axis_name,
                    exc,
                )
            finally:
                if fig is not None:
                    plt.close(fig)

        if len(image_paths) == 0:
            return
        self.record_system_message(
            content="Linear drift model status updated. Current status (y and x):",
            image_path=image_paths,
        )

    def build_line_scan_precheck_message(
        self,
        line_scan_result: dict[str, Any],
        line_scan_residual_warning_threshold: float,
    ) -> str:
        warnings = []
        nan_check_keys = [
            "fwhm",
            "a",
            "mu",
            "sigma",
            "c",
            "normalized_residual",
            "x_min",
            "x_max",
        ]
        nan_keys = []
        for key in nan_check_keys:
            value = line_scan_result.get(key)
            if isinstance(value, (int, float)) and np.isnan(value):
                nan_keys.append(key)
        if len(nan_keys) > 0:
            warnings.append(
                "Warning: Gaussian fitting contains NaN values in: "
                f"{', '.join(nan_keys)}. Check the fitting and scan line position carefully."
            )

        residual = line_scan_result.get("normalized_residual")
        if isinstance(residual, (int, float)) and not np.isnan(residual):
            if residual > line_scan_residual_warning_threshold:
                warnings.append(
                    "Warning: the noramlized residual [mean(((y_data - y_fit) / a)^2)] "
                    f"is high ({residual}). Check the fitting and scan line position carefully."
                )

        mu = line_scan_result.get("mu")
        sigma = line_scan_result.get("sigma")
        x_min = line_scan_result.get("x_min")
        x_max = line_scan_result.get("x_max")
        if (
            isinstance(mu, (int, float))
            and isinstance(sigma, (int, float))
            and isinstance(x_min, (int, float))
            and isinstance(x_max, (int, float))
            and not np.isnan(mu)
            and not np.isnan(sigma)
            and not np.isnan(x_min)
            and not np.isnan(x_max)
            and x_max > x_min
        ):
            sigma_abs = abs(float(sigma))
            lower_bound = float(x_min) + 3 * sigma_abs
            upper_bound = float(x_max) - 3 * sigma_abs
            if float(mu) < lower_bound or float(mu) > upper_bound:
                warnings.append(
                    "Warning: fitted Gaussian peak location is near scan-range edge "
                    f"(mu={mu}, x_min={x_min}, x_max={x_max}, sigma={sigma}). "
                    "Check the fitting and scan line position carefully."
                )
        if len(warnings) == 0:
            return "Pre-check result: no warnings."
        return "Pre-check result:\n" + "\n".join(warnings)

    def check_line_scan(
        self,
        image_path: str,
        line_scan_result: dict[str, Any],
        line_scan_residual_warning_threshold: float = 0.015,
    ) -> dict[str, Any]:
        if self.llm_config is None:
            logger.warning("LLM is unavailable. Skipping line scan check.")
            return {"result": "ok"}

        skill_path = (
            Path(__file__).resolve().parents[2]
            / "private_skills"
            / "check-line-scan"
            / "SKILL.md"
        )
        if not skill_path.exists():
            raise FileNotFoundError(f"Line scan check skill file not found: {skill_path}")

        skill_text = skill_path.read_text(encoding="utf-8")
        checker_task_manager = BaseTaskManager(
            llm_config=self.llm_config,
            memory_config=self.memory_config,
            tools=[self.acquisition_tool],
            message_db_path=None,
            use_coding_tools=False,
            build=True,
        )
        skill_tool_name = "skill-check-line-scan"
        checker_task_manager.skill_catalog = [
            SkillMetadata(
                name="check-line-scan",
                description="Instructions for checking a line scan.",
                tool_name=skill_tool_name,
                path=str(skill_path.parent),
            )
        ]
        checker_task_manager._inject_skill_doc_messages_to_context(
            tool_response={
                "content": {
                    "path": str(skill_path.parent),
                    "files": {"SKILL.md": skill_text},
                }
            },
            tool_call_info={"function": {"name": skill_tool_name}},
        )
        fit_summary = (
            "Line-scan fit summary:\n"
            f"- fwhm: {line_scan_result.get('fwhm')}\n"
            f"- a: {line_scan_result.get('a')}\n"
            f"- mu: {line_scan_result.get('mu')}\n"
            f"- sigma: {line_scan_result.get('sigma')}\n"
            f"- c: {line_scan_result.get('c')}\n"
            f"- normalized_residual: {line_scan_result.get('normalized_residual')}\n"
            f"- x_range: [{line_scan_result.get('x_min')}, {line_scan_result.get('x_max')}]"
        )
        precheck_summary = self.build_line_scan_precheck_message(
            line_scan_result,
            line_scan_residual_warning_threshold=line_scan_residual_warning_threshold,
        )

        self.record_system_message(
            f"Starting line scan quality check and adjustment with LLM. Below are the findings of "
            f"analytical pre-check based on Gaussian fitting result:\n"
            f"```{fit_summary}\n\n{precheck_summary}```"
        )

        starter_message = generate_openai_message(
            content=(
                "Use the instructions already in context and the provided image to check the line scan. "
                "Return exactly one JSON object as specified by the instructions.\n\n"
                f"{fit_summary}\n\n{precheck_summary}\n\n"
                "IMPORTANT: use your own judgement. The line scan might still be bad "
                "even if precheck didn't give any warnings!"
            ),
            role="user",
            image_path=image_path,
        )
        print_message(starter_message)
        checker_task_manager.run_conversation(
            message=starter_message,
            termination_behavior="return",
        )
        response_text = None
        for message in reversed(checker_task_manager.context):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                response_text = message["content"]
                break
        if response_text is None:
            raise RuntimeError("No assistant response found in line scan checker conversation.")

        result = self.parse_json_from_response(response_text)
        if "result" not in result:
            raise ValueError(f"`result` key is missing in check_line_scan response: {result}")

        if result["result"] == "ok":
            return result
        if result["result"] == "adjusted":
            new_line_scan_kwargs = result.get("new_line_scan_kwargs")
            if not isinstance(new_line_scan_kwargs, dict):
                raise ValueError(
                    f"`new_line_scan_kwargs` must be a dictionary when result='adjusted', got {result}."
                )
            old_line_scan_kwargs = copy.deepcopy(self.line_scan_kwargs)
            merged_line_scan_kwargs = copy.deepcopy(self.line_scan_kwargs)
            merged_line_scan_kwargs.update(new_line_scan_kwargs)
            old_scan_position = self.extract_line_scan_position(old_line_scan_kwargs)
            new_scan_position = self.extract_line_scan_position(merged_line_scan_kwargs)
            offset = new_scan_position - old_scan_position
            self.line_scan_kwargs = merged_line_scan_kwargs
            if self.run_offset_calibration:
                self.apply_offset_to_image_acquisition_kwargs(offset)
            return result
        if result["result"] == "failed":
            return result
        raise ValueError(f"Unsupported check_line_scan result: {result}")
    
    def update_optimization_model(self, fwhm: float):
        x = self.param_setting_tool.get_parameter_at_iteration(-1)
        x = np.array(x).reshape(1, -1)
        # Use negative FWHM because we want to minimize the FWHM.
        self.optimization_tool.update(x, -np.array([[fwhm]]))
        try:
            fig = self.optimization_tool.visualize_status()
            fig_path = BaseTool.save_image_to_temp_dir(
                fig=fig,
                filename="optimization_status.png",
                add_timestamp=True,
            )
            self.record_system_message(
                "Optimization status updated.",
                image_path=fig_path,
            )
        except Exception as exc:
            logger.warning("Failed to visualize optimization status: %s", exc)
        
    def run_2d_scan(self):
        self.record_system_message(f"Acquiring 2D scan...```{self.image_acquisition_kwargs}```")
        image_path = self.acquisition_tool.acquire_image(**self.image_acquisition_kwargs)
        if self.initial_image_acquisition_position is None:
            self.initial_image_acquisition_position = self.extract_image_acquisition_position(
                self.image_acquisition_kwargs
            )
        content = f"Acquired 2D scan with kwargs: ```{self.image_acquisition_kwargs}```"
        if isinstance(image_path, str):
            self.record_system_message(content, image_path=image_path)
        else:
            self.record_system_message(content)

    def get_suggested_next_parameters(self, step_size_limit: Optional[float | Tuple[float, ...]] = None):
        p_suggested = to_numpy(self.optimization_tool.suggest(n_suggestions=1)[0])
        p_current = to_numpy(self.param_setting_tool.get_parameter_at_iteration(-1))
        if step_size_limit is not None:
            signs = np.sign(p_suggested - p_current)
            step_sizes = np.abs(p_suggested - p_current)
            step_sizes = np.clip(step_sizes, min=None, max=step_size_limit)
            p_suggested = p_current + signs * step_sizes
        return p_suggested
    
    def find_position_correction(
        self, target: Literal["previous", "initial"] = "previous"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the correction that should be applied (added) to image acquisition and line scan positions.

        Parameters
        ----------
        target : Literal["previous", "initial"]
            The reference image to register against.
            "previous": register the current image with the immediately preceding image.
            "initial": register the current image with the very first image to prevent
            error accumulation.

        Returns
        -------
        np.ndarray
            The correction that should be applied to the line scan positions, which includes
            the pure image registration offset and the scan-position difference.
            When target is "previous", this is the correction relative to the previous step.
            When target is "initial", this is the cumulative correction from the initial position.
            The correction is in physical units, i.e., pixel size is already accounted for.
        np.ndarray
            The correction that should be applied to the image acquisition positions, which
            includes only the pure image registration offset.
            When target is "previous", this is the correction relative to the previous step.
            When target is "initial", this is the cumulative correction from the initial position.
        """
        if target == "previous":
            image_r = self.image_registration_tool.process_image(self.acquisition_tool.image_km1)
            psize_r = self.acquisition_tool.psize_km1
            scan_pos_diff = np.array([
                float(self.acquisition_tool.image_acquisition_call_history[-1][f"{dir}_center"])
                - float(self.acquisition_tool.image_acquisition_call_history[-2][f"{dir}_center"])
                for dir in ["y", "x"]
            ]).astype(float)
        elif target == "initial":
            image_r = self.image_registration_tool.process_image(self.acquisition_tool.image_0)
            psize_r = self.acquisition_tool.psize_0
            scan_pos_diff = np.array([
                float(self.acquisition_tool.image_acquisition_call_history[-1][f"{dir}_center"])
                - float(self.acquisition_tool.image_acquisition_call_history[0][f"{dir}_center"])
                for dir in ["y", "x"]
            ]).astype(float)
        else:
            raise ValueError(f"`target` must be 'previous' or 'initial', got {target!r}.")

        alignment_offset = np.array(
            self.image_registration_tool.register_images(
                image_t=self.image_registration_tool.process_image(self.acquisition_tool.image_k),
                image_r=image_r,
                psize_t=self.acquisition_tool.psize_k,
                psize_r=psize_r,
                registration_algorithm_kwargs=self.registration_algorithm_kwargs,
            ),
            dtype=float,
        )
        # Image registration offset is the offset by which the moving image should be rolled to match
        # the reference. We want acquisition position correction here, which is the negation of it.
        registration_correction = -alignment_offset

        # Count in the difference of scan positions.
        line_scan_correction = registration_correction + scan_pos_diff
        return line_scan_correction, registration_correction
    
    def apply_offset_to_line_scan_kwargs(self, offset: np.ndarray):
        for arg in self.line_scan_tool_x_coordinate_args:
            self.line_scan_kwargs[arg] += offset[1]
        for arg in self.line_scan_tool_y_coordinate_args:
            self.line_scan_kwargs[arg] += offset[0]

    def apply_offset_to_image_acquisition_kwargs(self, offset: np.ndarray):
        for arg in self.image_acquisition_tool_x_coordinate_args:
            self.image_acquisition_kwargs[arg] += offset[1]
        for arg in self.image_acquisition_tool_y_coordinate_args:
            self.image_acquisition_kwargs[arg] += offset[0]
            
    def collect_initial_data_for_optimization_tool(
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
        
        xs = np.linspace(current_x - sampling_range / 2, current_x + sampling_range / 2, n)
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
        x_original = np.array(self.param_setting_tool.get_parameter_at_iteration(-1), dtype=float)
        x_current = np.array(x, dtype=float)
        line_scan_kwargs_before = copy.deepcopy(self.line_scan_kwargs)
        image_acquisition_kwargs_before = copy.deepcopy(self.image_acquisition_kwargs)

        def rollback_and_shrink_delta(message_prefix: str) -> np.ndarray:
            for parameter_name in self.param_setting_tool.parameter_names:
                if len(self.param_setting_tool.parameter_history[parameter_name]) > 0:
                    self.param_setting_tool.parameter_history[parameter_name].pop()
            self.line_scan_kwargs = copy.deepcopy(line_scan_kwargs_before)
            self.image_acquisition_kwargs = copy.deepcopy(image_acquisition_kwargs_before)
            delta = x_current - x_original
            x_next = x_original + delta / 2
            self.record_system_message(
                f"{message_prefix} Retrying by shrinking parameter delta from {delta.tolist()} "
                f"to {(delta / 2).tolist()}, new parameters: {x_next.tolist()}."
            )
            if np.allclose(x_next, x_original):
                raise RuntimeError(f"{message_prefix} Parameter delta is too small to continue.")
            return x_next

        while True:
            self.record_system_message(f"Setting parameters to new value:```{x_current}```")
            self.param_setting_tool.set_parameters(x_current)
            if self.run_offset_calibration:
                self.run_2d_scan()
                self.apply_drift_correction(x_current)
            try:
                fwhm = self.run_line_scan()
                if np.isnan(fwhm):
                    raise LineScanValidationFailed("FWHM is NaN.")
            except LineScanValidationFailed:
                x_current = rollback_and_shrink_delta("Line scan validation failed.")
                continue
            self.update_optimization_model(fwhm)
            return

    def _select_drift(
        self,
        lsp_drift: np.ndarray,
        reg_drift: np.ndarray | None,
        x_current: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Select the drift estimate to use, logging the decision.

        During the warm-up phase (fewer than ``dual_drift_selection_priming_iterations``
        samples collected) the primary source is used unconditionally.
        Afterwards the linear model prediction arbitrates: the drift estimate
        closer in Euclidean distance to the model prediction is chosen.

        Parameters
        ----------
        lsp_drift : np.ndarray
            Cumulative drift from the line scan predictor.
        reg_drift : np.ndarray or None
            Cumulative drift from image registration, or ``None`` on failure.
        x_current : np.ndarray
            Current optics parameter vector (used in the arbitration phase).

        Returns
        -------
        chosen_drift : np.ndarray
        chosen_source : str
            ``"line_scan_predictor"`` or ``"registration"``.
        """
        n_collected = self.drift_model_y.get_n_parameter_drift_points_collected()
        n_needed = self.dual_drift_selection_priming_iterations

        if n_collected < n_needed:
            if self.dual_drift_estimation_primary_source == "line_scan_predictor" or reg_drift is None:
                chosen_drift, chosen_source = lsp_drift, "line_scan_predictor"
            else:
                chosen_drift, chosen_source = reg_drift, "registration"
            if reg_drift is None:
                self.record_system_message(
                    "Dual drift estimation: image registration returned NaN; "
                    "falling back to line scan predictor."
                )
            self.record_system_message(
                f"Dual drift estimation (primary phase, n={n_collected}/{n_needed}): "
                f"```using {chosen_source}\n"
                f"lsp_drift={lsp_drift.tolist()}\n"
                f"reg_drift={reg_drift.tolist() if reg_drift is not None else 'NaN'}```"
            )
        else:
            x_in = np.array(x_current, dtype=float).reshape(1, -1).tolist()
            model_drift = np.array(
                [
                    float(self.drift_model_y.predict(x_in)[0][0]),
                    float(self.drift_model_x.predict(x_in)[0][0]),
                ],
                dtype=float,
            )
            dist_lsp = float(np.linalg.norm(lsp_drift - model_drift))
            dist_reg = (
                float(np.linalg.norm(reg_drift - model_drift))
                if reg_drift is not None
                else np.inf
            )
            if dist_lsp <= dist_reg:
                chosen_drift, chosen_source = lsp_drift, "line_scan_predictor"
            else:
                chosen_drift, chosen_source = reg_drift, "registration"
            dist_reg_str = f"{dist_reg:.4f}" if reg_drift is not None else "inf"
            self.record_system_message(
                f"Dual drift estimation (arbitration phase):\n"
                f"```model_drift={model_drift.tolist()}\n"
                f"lsp_drift={lsp_drift.tolist()} (dist={dist_lsp:.4f})\n"
                f"reg_drift={reg_drift.tolist() if reg_drift is not None else 'NaN'} (dist={dist_reg_str})\n"
                f"Chosen: {chosen_source}```"
            )

        return chosen_drift, chosen_source

    def apply_drift_correction(self, x_current: np.ndarray) -> None:
        """Run both drift estimators, select the best estimate, and apply it.

        Parameters
        ----------
        x_current : np.ndarray
            Current optics parameter vector.
        """
        if self.initial_line_scan_position is None or self.initial_image_acquisition_position is None:
            raise RuntimeError(
                "apply_drift_correction requires initial scan positions to be set. "
                "Ensure initialize_kwargs_buffers and run_2d_scan have been called first."
            )

        # Get the offset from the registration.
        # line_scan_pos_offset_reg: the offset counting in BOTH the pure image registration and the scan-position difference
        # -> to be applied to the line scan position.
        # pure_registration_offset: the offset counting in ONLY the pure image registration
        # -> to be applied to the image acquisition position.
        line_scan_correction_reg, image_acq_correction_reg = self.find_position_correction(
            target=self.registration_target
        )
        if self.registration_target == "previous":
            # find_position_correction returns corrections relative to the previous step.
            line_scan_correction_reg_wrt_prev = line_scan_correction_reg
            line_scan_correction_reg_wrt_initial = (
                self.extract_line_scan_position(self.line_scan_kwargs)
                + line_scan_correction_reg
                - self.initial_line_scan_position
            )
            image_acq_correction_reg_wrt_prev = image_acq_correction_reg
            image_acq_correction_reg_wrt_initial = (
                image_acq_correction_reg
                + self.extract_image_acquisition_position(self.image_acquisition_kwargs)
                - self.initial_image_acquisition_position
            )
        else:
            # find_position_correction returns cumulative corrections from the initial position.
            # Convert to wrt_prev so the apply helpers can add them to the current kwargs.
            line_scan_correction_reg_wrt_initial = line_scan_correction_reg
            line_scan_correction_reg_wrt_prev = (
                self.initial_line_scan_position
                + line_scan_correction_reg
                - self.extract_line_scan_position(self.line_scan_kwargs)
            )
            image_acq_correction_reg_wrt_initial = image_acq_correction_reg
            image_acq_correction_reg_wrt_prev = (
                self.initial_image_acquisition_position
                + image_acq_correction_reg
                - self.extract_image_acquisition_position(self.image_acquisition_kwargs)
            )
        
        # Get the offset from the line scan predictor. Note that the offset is with regards to the initial image.
        if self.line_scan_predictor_tool is not None:
            lsp_json = json.loads(self.line_scan_predictor_tool.predict_line_scan_position())
            predicted_center = np.array(
                [lsp_json["center_y"], lsp_json["center_x"]], dtype=float
            )
            line_scan_correction_lsp_wrt_initial = predicted_center - self.initial_line_scan_position
            line_scan_correction_lsp_wrt_prev = (
                predicted_center - self.extract_line_scan_position(self.acquisition_tool.line_scan_call_history[-1])
            )
            scan_pos_diff = np.array([
                float(self.acquisition_tool.image_acquisition_call_history[-1][f"{dir}_center"])
                - float(self.acquisition_tool.image_acquisition_call_history[-2][f"{dir}_center"])
                for dir in ["y", "x"]
            ]).astype(float)
            image_acq_correction_lsp_wrt_initial = line_scan_correction_lsp_wrt_initial - scan_pos_diff
            image_acq_correction_lsp_wrt_prev = line_scan_correction_lsp_wrt_prev - scan_pos_diff

            # Select the best correction to use.
            chosen_line_scan_correction_wrt_initial, chosen_source = self._select_drift(
                line_scan_correction_lsp_wrt_initial, line_scan_correction_reg_wrt_initial, x_current
            )
            chosen_line_scan_correction_wrt_prev = (
                line_scan_correction_lsp_wrt_prev
                if chosen_source == "line_scan_predictor"
                else line_scan_correction_reg_wrt_prev
            )
            chosen_image_acq_correction_wrt_prev = (
                image_acq_correction_lsp_wrt_prev
                if chosen_source == "line_scan_predictor"
                else image_acq_correction_reg_wrt_prev
            )
            chosen_image_acq_correction_wrt_initial = (
                image_acq_correction_lsp_wrt_initial
                if chosen_source == "line_scan_predictor"
                else image_acq_correction_reg_wrt_initial
            )
        else:
            chosen_source = "registration"
            chosen_line_scan_correction_wrt_prev = line_scan_correction_reg_wrt_prev
            chosen_line_scan_correction_wrt_initial = line_scan_correction_reg_wrt_initial
            chosen_image_acq_correction_wrt_prev = image_acq_correction_reg_wrt_prev
            chosen_image_acq_correction_wrt_initial = image_acq_correction_reg_wrt_initial
            
        self.apply_offset_to_line_scan_kwargs(chosen_line_scan_correction_wrt_prev)
        self.apply_offset_to_image_acquisition_kwargs(chosen_image_acq_correction_wrt_prev)
        
        self.record_system_message(
            f"Applied drift correction:\n"
            f"```source = {chosen_source}\n"
            f"image_registration_offset = {(-chosen_image_acq_correction_wrt_prev).tolist()}\n"
            f"chosen_line_scan_correction_wrt_prev = {chosen_line_scan_correction_wrt_prev.tolist()}\n"
            f"chosen_line_scan_correction_wrt_initial = {chosen_line_scan_correction_wrt_initial.tolist()}\n"
            f"chosen_image_acq_correction_wrt_prev = {chosen_image_acq_correction_wrt_prev.tolist()}\n"
            f"chosen_image_acq_correction_wrt_initial = {chosen_image_acq_correction_wrt_initial.tolist()}```"
        )

        # Update model with (parameters -> chosen cumulative drift). Pass the
        # target position directly so the model is consistent regardless of
        # whether kwargs have been flushed to the underlying tool yet.
        self.update_linear_drift_models(x_current, current_position_yx=chosen_line_scan_correction_wrt_initial)
        self.record_linear_drift_model_visualizations()

    def apply_user_correction_offset(self) -> bool:
        message = (
            "Manual correction requested. Enter offset-to-subtract as 'y,x' (blank to stop): "
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
            self.apply_offset_to_line_scan_kwargs(-offset)
            self.apply_offset_to_image_acquisition_kwargs(-offset)
            correction_message = f"Applied user correction offset: {offset.tolist()}"
            logger.info(correction_message)
            self.record_system_message(correction_message)
            return True

    def generate_report_csv(self) -> str:
        xs = self.optimization_tool.xs_untransformed.tolist()
        fwhms = (-self.optimization_tool.ys_untransformed).tolist()
        report = "Parameters,FWHM\n"
        for x, fwhm in zip(xs, fwhms):
            report += f"{x[0]},{fwhm[0]}\n"
        return report
