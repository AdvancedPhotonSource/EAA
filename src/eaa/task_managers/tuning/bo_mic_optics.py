import logging
from typing import Optional

import torch
import numpy as np
from PIL import Image

from eaa.task_managers.tuning.bo import BayesianOptimizationTaskManager
from eaa.task_managers.imaging.feature_tracking import FeatureTrackingTaskManager
from eaa.tools.base import BaseTool
from eaa.tools.bo import BayesianOptimizationTool
from eaa.api.llm_config import LLMConfig

logger = logging.getLogger(__name__)


class MicroscopyOpticsTuningBOTaskManager(
    BayesianOptimizationTaskManager, 
    FeatureTrackingTaskManager
):
    def __init__(
        self,
        llm_config: LLMConfig = None,
        image_acquisition_tool: BaseTool = None,
        parameter_setting_tool: BaseTool = None,
        bayesian_optimization_tool: BayesianOptimizationTool = None,
        additional_tools: list[BaseTool] = (),
        initial_points: Optional[torch.Tensor] = None,
        n_initial_points: int = 20,
        image_acquisition_kwargs: dict = {},
        feature_tracking_kwargs: dict = {},
        message_db_path: Optional[str] = None,
        *args, **kwargs
    ):
        """The Bayesian optimization task manager for microscopy optics tuning.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            The configuration for the LLM.
        bayesian_optimization_tool : BayesianOptimizationTool
            The Bayesian optimization tool to use.
        image_acquisition_tool : BaseTool
            The tool to use to acquire images.
        parameter_setting_tool : BaseTool
            The tool to use to set the parameters.
        additional_tools : list[BaseTool], optional
            Additional tools provided to the agent (not including the
            tools passed through explicit arguments).
        initial_points : torch.Tensor, optional
            A (n_points, n_features) tensor giving the initial points where
            the objective function should be evaluated to initialize the
            Gaussian process model. If None, random initial points will be
            generated.
        n_initial_points : int, optional
            The number of initial points to generate if `initial_points` is None.
        image_acquisition_kwargs : dict, optional
            The arguments of the image acquisition tool that should be used
            when acquiring images for evaluating the objective function.
        feature_tracking_kwargs : dict, optional
            The arguments of the feature tracking task manager's `run` method.
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
        if image_acquisition_tool is None:
            raise ValueError(
                "Image acquisition tool should be explicitly passed to "
                "`image_acquisition_tool`."
            )
        if parameter_setting_tool is None:
            raise ValueError(
                "Parameter setting tool should be explicitly passed to "
                "`parameter_setting_tool`."
            )
        
        self.image_acquisition_tool = image_acquisition_tool
        self.parameter_setting_tool = parameter_setting_tool
        self.image_acquisition_kwargs = image_acquisition_kwargs
        self.feature_tracking_kwargs = feature_tracking_kwargs
        
        BayesianOptimizationTaskManager.__init__(
            self,
            llm_config=llm_config,
            bayesian_optimization_tool=bayesian_optimization_tool,
            initial_points=initial_points,
            n_initial_points=n_initial_points,
            objective_function=self.objective_function,
            message_db_path=message_db_path,
            build=False,
            *args, **kwargs
        )
        FeatureTrackingTaskManager.__init__(
            self, 
            llm_config=llm_config,
            image_acquisition_tool=image_acquisition_tool,
            additional_tools=additional_tools,
            build=True,
            *args, **kwargs
        )

    def objective_function(self, x: torch.Tensor, *args, **kwargs):
        """Calculate the objective function value.

        Parameters
        ----------
        x : torch.Tensor
            A (n_points, n_features) tensor of points to evaluate the
            objective function at.

        Returns
        -------
        torch.Tensor
            A (n_points, 1) tensor of objective function values.
        """
        if x.ndim != 2:
            raise ValueError(
                "`x` should be a 2D tensor of shape (n_points, n_features)."
        )
        
        objective_values = torch.zeros(x.shape[0], 1, device=x.device)
                
        for i, x_i in enumerate(x):
            # Acquire an image with the current parameters. It will be used
            # as the reference image for feature tracking.
            acquired_image_path = self.image_acquisition_tool.acquire_image(
                **self.image_acquisition_kwargs
            )
            
            # Apply parameters.
            self.parameter_setting_tool.set_parameters(x_i)
            
            # Now the original feature will have drifted. Run feature tracking
            # to bring it back.
            self.run_feature_tracking(
                **self.feature_tracking_kwargs
            )
            
            # Get a new image after feature tracking.
            acquired_image_path = self.image_acquisition_tool.acquire_image(
                **self.image_acquisition_kwargs
            )
            image = Image.open(acquired_image_path)
            image = np.array(image)
            
            objective_values[i, 0] = np.std(image)
        return objective_values
