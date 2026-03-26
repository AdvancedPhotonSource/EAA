import logging
from collections.abc import Callable
from typing import Optional

import torch

from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.tool.base import BaseTool
from eaa_core.tool.optimization import BayesianOptimizationTool
from eaa_core.util import to_tensor

logger = logging.getLogger(__name__)


class BayesianOptimizationStoppingCriterion:
    """Base stopping criterion for Bayesian optimization task managers."""

    reason: str = ""

    def should_stop(self, task_manager: "BayesianOptimizationTaskManager") -> bool:
        """Return whether the optimization loop should stop."""
        raise NotImplementedError


class MaxObservationsStoppingCriterion(BayesianOptimizationStoppingCriterion):
    """Stop once a maximum number of observations has been collected."""

    def __init__(self, max_observations: int) -> None:
        """Initialize the stopping criterion.

        Parameters
        ----------
        max_observations : int
            Maximum number of observed x/y pairs allowed.
        """
        self.max_observations = max_observations
        self.reason = "max_observations_reached"

    def should_stop(self, task_manager: "BayesianOptimizationTaskManager") -> bool:
        """Return whether the observation cap has been reached."""
        n_observations = int(task_manager.bayesian_optimization_tool.xs_untransformed.shape[0])
        return n_observations >= self.max_observations


class BayesianOptimizationTaskManager(BaseTaskManager):
    """Task manager that runs an outer-loop Bayesian optimization workflow."""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        bayesian_optimization_tool: BayesianOptimizationTool = None,
        additional_tools: list[BaseTool] = (),
        initial_points: Optional[torch.Tensor] = None,
        n_initial_points: int = 20,
        objective_function: BaseTool | Callable | None = None,
        objective_function_method: str | None = None,
        stopping_criteria: Optional[list[BayesianOptimizationStoppingCriterion]] = None,
        session_db_path: Optional[str] = "session.sqlite",
        build: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Bayesian optimization task manager.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            The configuration for the LLM.
        memory_config : MemoryManagerConfig, optional
            Memory configuration forwarded to the agent.
        additional_tools : list[BaseTool], optional
            Additional tools exposed to the task manager, excluding the
            Bayesian optimization tool and the objective tool.
        bayesian_optimization_tool : BayesianOptimizationTool
            The Bayesian optimization tool to use.
        initial_points : torch.Tensor, optional
            Initial measurement points with shape ``(n_points, n_features)``.
            When omitted, random points are drawn from the optimization bounds.
        n_initial_points : int, optional
            Number of random initial points to draw when ``initial_points`` is not
            provided.
        objective_function : BaseTool | Callable
            Callable or tool used to evaluate points. The returned observations
            must have shape ``(n_samples, n_observations)``.
        objective_function_method : str | None, optional
            Method name to call when ``objective_function`` is a tool. When
            omitted, ``measure`` is preferred, then ``evaluate``, then a single
            exposed tool method.
        stopping_criteria : list[BayesianOptimizationStoppingCriterion], optional
            Additional stopping criteria checked after initialization and each
            update.
        session_db_path : Optional[str]
            Optional SQLite path used by the shared chat/task-manager session.
        build : bool, optional
            Whether to build the internal state of the task manager.
        """
        if bayesian_optimization_tool is None:
            raise ValueError(
                "Bayesian optimization tool should be explicitly passed to "
                "`bayesian_optimization_tool`."
            )
        if objective_function is None:
            raise ValueError("`objective_function` is required.")

        self.bayesian_optimization_tool = bayesian_optimization_tool

        tools = list(additional_tools)
        if isinstance(objective_function, BaseTool):
            tools.append(objective_function)
        for tool in tools:
            if isinstance(tool, BayesianOptimizationTool):
                raise ValueError(
                    "`BayesianOptimizationTool` should not be included in `tools`. "
                    "Instead, pass it to `bayesian_optimization_tool`."
                )

        self.objective_function = objective_function
        self.objective_function_method = objective_function_method
        self.initial_points = initial_points
        self.n_initial_points = n_initial_points
        self.stopping_criteria = list(stopping_criteria or [])
        self.stop_reason: str | None = None

        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            tools=tools,
            session_db_path=session_db_path,
            build=build,
            *args,
            **kwargs,
        )

    def run(
        self,
        n_iterations: int = 50,
        *args,
        **kwargs,
    ) -> None:
        """Run Bayesian optimization.

        When the task manager already contains observations, the optimization
        continues from the current state.

        Parameters
        ----------
        n_iterations : int, optional
            Maximum number of BO iterations to execute in this call.
        """
        if len(self.bayesian_optimization_tool.xs_untransformed) == 0:
            self.collect_initial_observations()

        if self.should_stop():
            return

        for _ in range(n_iterations):
            candidates = self.bayesian_optimization_tool.suggest(n_suggestions=1)
            logger.info("Candidate suggested: %s", candidates[0])
            y = self.evaluate_objective(candidates)
            logger.info("Objective function value: %s", y.reshape(-1))
            self.bayesian_optimization_tool.update(candidates, y)
            self.configure_bayesian_optimization()
            if self.should_stop():
                break

    def collect_initial_observations(self) -> None:
        """Collect initial observations and build the GP model.

        Initial ``x`` points are expected to have shape ``(n_points, n_features)``
        and the evaluated observations must have shape
        ``(n_points, n_observations)``.
        """
        if self.initial_points is None:
            xs_init = self.bayesian_optimization_tool.get_random_initial_points(
                n_points=self.n_initial_points
            )
        else:
            xs_init = to_tensor(self.initial_points)
        logger.info("Initial points (shape: %s):\n%s", xs_init.shape, xs_init)

        for x in xs_init:
            x = x[None, :]
            y = self.evaluate_objective(x)
            self.bayesian_optimization_tool.update(x, y)
        self.bayesian_optimization_tool.build()
        self.configure_bayesian_optimization()

    def evaluate_objective(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the objective function or objective tool at input points.

        Parameters
        ----------
        x : torch.Tensor
            Candidate locations with shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Objective values with shape ``(n_samples, n_observations)``.
        """
        if isinstance(self.objective_function, BaseTool):
            objective_callable = self.resolve_objective_tool_callable()
            y = objective_callable(x)
        else:
            y = self.objective_function(x)
        y = to_tensor(y)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        if y.ndim == 1:
            y = y[:, None]
        self.bayesian_optimization_tool.check_y_data(y)
        return y

    def resolve_objective_tool_callable(self) -> Callable:
        """Resolve the callable used to evaluate the objective tool."""
        tool = self.objective_function
        if not isinstance(tool, BaseTool):
            raise TypeError("`objective_function` is not a tool instance.")

        candidate_names: list[str]
        if self.objective_function_method is not None:
            candidate_names = [self.objective_function_method]
        else:
            candidate_names = ["measure", "evaluate"]

        for method_name in candidate_names:
            method = getattr(tool, method_name, None)
            if callable(method):
                return method

        if len(tool.exposed_tools) == 1:
            return tool.exposed_tools[0].function
        raise ValueError(
            "Could not resolve a tool method for `objective_function`. "
            "Pass `objective_function_method` explicitly."
        )

    def configure_bayesian_optimization(self) -> None:
        """Hook for subclasses to update acquisition or stopping state."""
        return None

    def should_stop(self) -> bool:
        """Return whether any configured stopping criterion has triggered."""
        for criterion in self.stopping_criteria:
            if criterion.should_stop(self):
                self.stop_reason = criterion.reason
                logger.info("Stopping criterion triggered: %s", self.stop_reason)
                return True
        return False
