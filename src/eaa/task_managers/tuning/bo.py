import logging
from typing import Optional, Callable

import torch

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool
from eaa.tools.bo import BayesianOptimizationTool

logger = logging.getLogger(__name__)


class BayesianOptimizationTaskManager(BaseTaskManager):
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        model_base_url: str = None,
        tools: list[BaseTool] = [],
        bayesian_optimization_tool: BayesianOptimizationTool = None,
        initial_points: Optional[torch.Tensor] = None,
        n_initial_points: int = 20,
        objective_function: Callable = None,
        message_db_path: Optional[str] = None,
        *args, **kwargs
    ) -> None:
        """Bayesian optimization task manager.

        Parameters
        ----------
        model_name : str, optional
            The model name of the agent.
        model_base_url : str, optional
            The LLM inference endpoint's base URL.
        tools : list[BaseTool], optional
            A list of tools for the agent. This should NOT include the
            `BayesianOptimizationTool`.
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
        """
        if bayesian_optimization_tool is None:
            raise ValueError("`bayesian_optimization_tool` is required.")
        if objective_function is None:
            raise ValueError("`objective_function` is required.")
        
        self.bayesian_optimization_tool = bayesian_optimization_tool
        
        for tool in tools:
            if isinstance(tool, BayesianOptimizationTool):
                raise ValueError("`BayesianOptimizationTool` should not be included in `tools`.")
            
        self.objective_function = objective_function
        
        self.initial_points = initial_points
        self.n_initial_points = n_initial_points
        
        super().__init__(
            model_name=model_name,
            model_base_url=model_base_url,
            tools=tools,
            message_db_path=message_db_path,
            *args, **kwargs
        )
        
    def objective_function(self, *args, **kwargs) -> None:
        raise NotImplementedError
        
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
        if len(self.bayesian_optimization_tool.xs_raw) == 0:
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
        