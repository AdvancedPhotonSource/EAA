import logging
from typing import Optional, Callable

import torch

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool
from eaa.tools.bo import BayesianOptimizationTool
from eaa.api.llm_config import LLMConfig
from eaa.api.memory import MemoryManagerConfig

logger = logging.getLogger(__name__)


class BayesianOptimizationTaskManager(BaseTaskManager):
    
    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        bayesian_optimization_tool: BayesianOptimizationTool = None,
        additional_tools: list[BaseTool] = (),
        initial_points: Optional[torch.Tensor] = None,
        n_initial_points: int = 20,
        objective_function: Callable = None,
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args, **kwargs
    ) -> None:
        """Bayesian optimization task manager.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            The configuration for the LLM.
        memory_config : MemoryManagerConfig, optional
            Memory configuration forwarded to the agent.
        additional_tools : list[BaseTool], optional
            A list of tools for the agent (not including the
            `BayesianOptimizationTool`).
        bayesian_optimization_tool : BayesianOptimizationTool
            The Bayesian optimization tool to use.
        initial_points : torch.Tensor, optional
            A (n_points, n_features) tensor giving the initial points where
            the objective function should be evaluated to initialize the
            Gaussian process model. If None, random initial points will be
            generated.
        n_initial_points : int, optional
            The number of initial points to generate if `initial_points` is None.
        objective_function : Callable
            The objective function to be maximized. This function should take
            a single argument, which is a (n_points, n_features) tensor of
            points to evaluate the objective function at. It should return
            a (n_points, n_objectives) tensor of objective function values.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
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
        
        for tool in additional_tools:
            if isinstance(tool, BayesianOptimizationTool):
                raise ValueError(
                    "`BayesianOptimizationTool` should not be included in `tools`. "
                    "Instead, pass it to `bayesian_optimization_tool`."
                )
            
        self.objective_function = objective_function
        
        self.initial_points = initial_points
        self.n_initial_points = n_initial_points
        
        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            tools=additional_tools,
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
        
    def run(
        self, 
        n_iterations: int = 50, 
        *args, **kwargs
    ) -> None:
        """Run Bayesian optimization. Upon the second or later call,
        this function continues from the last iteration.
        
        Parameters
        ----------
        n_iterations : int, optional
            The number of iterations to run.
        """
        if len(self.bayesian_optimization_tool.xs_untransformed) == 0:
            if self.initial_points is None:
                xs_init = self.bayesian_optimization_tool.get_random_initial_points(n_points=self.n_initial_points)
            else:
                xs_init = self.initial_points
            logger.info(f"Initial points (shape: {xs_init.shape}):\n{xs_init}")
                
            for x in xs_init:
                x = x[None, :]
                y = self.objective_function(x)
                self.bayesian_optimization_tool.update(x, y)
            self.bayesian_optimization_tool.build()
        
        for i in range(n_iterations):
            candidates = self.bayesian_optimization_tool.suggest(n_suggestions=1)
            logger.info(f"Candidate suggested: {candidates[0]}")
            y = self.objective_function(candidates)
            logger.info(f"Objective function value: {y.item()}")
            self.bayesian_optimization_tool.update(candidates, y)
