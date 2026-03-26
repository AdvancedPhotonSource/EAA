from typing import Optional

import torch

from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.task_manager.tuning.bo import (
    BayesianOptimizationStoppingCriterion,
    BayesianOptimizationTaskManager,
)
from eaa_core.tool.base import BaseTool
from eaa_core.tool.optimization import BayesianOptimizationTool


class WeightedPosteriorStandardDeviationStoppingCriterion(
    BayesianOptimizationStoppingCriterion
):
    """Stop when the weighted posterior uncertainty drops below a threshold."""

    def __init__(
        self,
        threshold: float,
        n_updates_to_begin: int = 10,
        n_check_interval: int = 5,
        n_max_measurements: Optional[int] = None,
    ) -> None:
        """Initialize the stopping criterion.

        Parameters
        ----------
        threshold : float
            Threshold on the maximum weighted posterior standard deviation.
        n_updates_to_begin : int, optional
            Minimum number of adaptive updates before checking the criterion.
        n_check_interval : int, optional
            Check every ``n_check_interval`` updates thereafter.
        n_max_measurements : int, optional
            Optional hard cap on the total number of measurements.
        """
        self.threshold = threshold
        self.n_updates_to_begin = n_updates_to_begin
        self.n_check_interval = n_check_interval
        self.n_max_measurements = n_max_measurements

    def should_stop(self, task_manager: "XANESAdaptiveSamplingTaskManager") -> bool:
        """Return whether the weighted posterior uncertainty is sufficiently low.

        The criterion queries posterior uncertainty on a dense normalized grid
        with shape ``(n_query, 1)`` and compares the maximum weighted standard
        deviation against the configured threshold.
        """
        n_measurements = int(
            task_manager.bayesian_optimization_tool.xs_untransformed.shape[0]
        )
        if self.n_max_measurements is not None and n_measurements >= self.n_max_measurements:
            self.reason = "max_measurements_reached"
            return True

        n_updates = task_manager.get_num_adaptive_updates_completed()
        if n_updates < self.n_updates_to_begin:
            return False
        if (n_updates - self.n_updates_to_begin) % self.n_check_interval != 0:
            return False

        n_query = max(n_measurements * 5, 32)
        x_query = torch.linspace(
            0.0,
            1.0,
            n_query,
            dtype=task_manager.bayesian_optimization_tool.xs_transformed.dtype,
            device=task_manager.bayesian_optimization_tool.xs_transformed.device,
        ).view(-1, 1)
        _, sigma = task_manager.bayesian_optimization_tool.get_posterior_mean_and_std(
            x_query,
            transform=False,
            untransform=True,
        )
        sigma = sigma.squeeze()
        acquisition_function = task_manager.bayesian_optimization_tool.acquisition_function
        weight_func = getattr(acquisition_function, "weight_func", None)
        if weight_func is not None:
            weights = torch.clip(weight_func(x_query).squeeze(), 0, 1)
        else:
            weights = 1.0
        if torch.max(sigma * weights) < self.threshold:
            self.reason = "weighted_max_uncertainty_below_threshold"
            return True
        return False


class XANESAdaptiveSamplingTaskManager(BayesianOptimizationTaskManager):
    """Task manager for adaptive XANES sampling on top of Bayesian optimization."""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        measurement_tool: BaseTool = None,
        bayesian_optimization_tool: BayesianOptimizationTool = None,
        additional_tools: list[BaseTool] = (),
        initial_points: Optional[torch.Tensor] = None,
        n_initial_points: int = 10,
        stopping_criteria: Optional[list[BayesianOptimizationStoppingCriterion]] = None,
        session_db_path: Optional[str] = "session.sqlite",
        build: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the XANES adaptive-sampling task manager.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            Configuration for the LLM interface. This argument is currently
            accepted for API compatibility, but it does not affect behavior at
            this stage.
        memory_config : MemoryManagerConfig, optional
            Memory configuration forwarded to the base task manager.
        measurement_tool : BaseTool
            Tool used to acquire XANES measurements. The task manager calls its
            ``measure`` method to evaluate suggested energies.
        bayesian_optimization_tool : BayesianOptimizationTool, optional
            Bayesian optimization tool used to generate candidate energies and
            maintain the surrogate model.
        additional_tools : list[BaseTool], optional
            Additional tools exposed to the task manager alongside the
            measurement tool.
        initial_points : torch.Tensor, optional
            Initial measurement energies with shape ``(n_points, n_features)``.
            When omitted, random initial points are drawn from the optimization
            bounds.
        n_initial_points : int, optional
            Number of random initial points to draw when ``initial_points`` is
            not provided.
        stopping_criteria : list[BayesianOptimizationStoppingCriterion], optional
            Additional stopping criteria checked after initialization and after
            each adaptive update.
        session_db_path : str, optional
            Optional SQLite path used to persist the shared session state.
        build : bool, optional
            Whether to build the internal task-manager state during
            initialization.
        *args
            Positional arguments forwarded to the base task manager.
        **kwargs
            Keyword arguments forwarded to the base task manager.
        """
        if measurement_tool is None:
            raise ValueError("`measurement_tool` must be provided.")

        self.measurement_tool = measurement_tool
        self.initial_observation_count = 0

        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            bayesian_optimization_tool=bayesian_optimization_tool,
            additional_tools=additional_tools,
            initial_points=initial_points,
            n_initial_points=n_initial_points,
            objective_function=measurement_tool,
            objective_function_method="measure",
            stopping_criteria=stopping_criteria,
            session_db_path=session_db_path,
            build=build,
            *args,
            **kwargs,
        )

    def collect_initial_observations(self) -> None:
        """Collect initial XANES measurements and record the warm-start size."""
        super().collect_initial_observations()
        self.initial_observation_count = int(
            self.bayesian_optimization_tool.xs_untransformed.shape[0]
        )

    def get_num_adaptive_updates_completed(self) -> int:
        """Return the number of adaptive updates beyond initialization."""
        total_observations = int(
            self.bayesian_optimization_tool.xs_untransformed.shape[0]
        )
        return max(0, total_observations - self.initial_observation_count)

    def should_stop(self) -> bool:
        """Return whether the XANES workflow should stop."""
        if hasattr(self.bayesian_optimization_tool, "should_stop"):
            if self.bayesian_optimization_tool.should_stop():
                self.stop_reason = getattr(
                    self.bayesian_optimization_tool,
                    "stop_reason",
                    None,
                )
                return True
        return super().should_stop()
