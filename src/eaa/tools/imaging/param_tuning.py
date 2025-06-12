from typing import Annotated, Dict, List, Any

import numpy as np

from eaa.tools.base import BaseTool, check, ToolReturnType
from eaa.tools.imaging.acquisition import SimulatedAcquireImage


class TuneOpticsParameters(BaseTool):
    
    name: str = "tune_optics_parameters"
    
    @check
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_history = []
        
        self.exposed_tools: List[Dict[str, Any]] = [
            {
                "name": "set_parameters",
                "function": self.set_parameters,
                "return_type": ToolReturnType.TEXT
            }
        ]
        
    def set_parameters(*args, **kwargs):
        raise NotImplementedError
        
        
class BlueSkyTuneOpticsParameters(TuneOpticsParameters):
    
    name: str = "bluesky_tune_optics_parameters"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def set_parameters(*args, **kwargs):
        raise NotImplementedError
        
        
class SimulatedTuneOpticsParameters(TuneOpticsParameters):
    
    name: str = "simulated_tune_optics_parameters"
    
    def __init__(
        self, 
        acquisition_tool: SimulatedAcquireImage, 
        true_parameters: list[float], 
        parameter_ranges: list[list[float], list[float]],
        blur_factor: float = 5,
        drift_factor: float = 0,
        *args, **kwargs
    ):
        """Simulate the tuning of optics parameters.

        Parameters
        ----------
        acquisition_tool : SimulatedAcquireImage
            The simulated image acquisition tool. Upon setting the optics
            parameters, attributes of the acquisition tool such as blur and
            offset will be updated in-place accordingly to simulate the 
            degradation and drift due to optics parameter changes.
        true_parameters : list[float]
            The true values of the optics parameters.
        parameter_ranges : list[list[float], list[float]]
            The ranges of the parameters. It should be given as a list of 2 sub-list,
            with the first sub-list containing the lower bounds and the second
            sub-list containing the upper bounds. These ranges will be used to
            scale the parameter errors. As such, inf is not allowed.
        blur_factor : float
            The factor determining the amount of blurring of the acquisition tool
            due to deviation from the true parameters.
        drift_factor : float
            The factor determining the amount of drift of the acquisition tool
            due to deviation from the true parameters.
        """
        super().__init__(*args, **kwargs)
        
        parameter_ranges = np.array(parameter_ranges)
        if np.any(np.isinf(parameter_ranges)):
            raise ValueError("Inf is not allowed in parameter ranges.")
        if (
            len(parameter_ranges) != 2 
            or len(parameter_ranges[0]) != len(parameter_ranges[1])
            or len(parameter_ranges[0]) != len(true_parameters)
            or np.any(parameter_ranges[1] <= parameter_ranges[0])
        ):
            raise ValueError(
                "`parameter_ranges` must be a list of 2 sub-lists, with the "
                "first sub-list containing the lower bounds and the second "
                "sub-list containing the upper bounds."
            )
        
        self.acquisition_tool = acquisition_tool
        self.true_parameters = np.array(true_parameters)
        self.parameter_ranges = parameter_ranges
        self.blur_factor = blur_factor
        self.drift_factor = drift_factor
        self.parameter_history = []
        
    def set_parameters(
        self, 
        parameters: list[float],
    ) -> Annotated[str, "A confirmation message."]:
        """Set the optics parameters of the imaging system to given values.
        
        Parameters
        ----------
        parameters : list[float]
            The parameters to set the optics to.
        
        Returns
        -------
        str
            A confirmation message.
        """
        normalized_errors = []
        parameters = np.array(parameters)
        scalers = self.parameter_ranges[1] - self.parameter_ranges[0]
        
        # Set blur.
        normalized_errors = (parameters - self.true_parameters) / scalers
        total_error = np.abs(normalized_errors).sum()
        blur = total_error * self.blur_factor
        self.acquisition_tool.set_blur(blur)
        
        # Set drift.
        if len(self.parameter_history) > 0 and self.drift_factor > 0:
            mean_delta = ((self.parameter_history[-1] - parameters) / scalers).mean()
            drift = np.random.rand(2) * mean_delta * self.drift_factor
            self.acquisition_tool.set_offset(drift)
        
        # Update parameter history.
        self.parameter_history.append(parameters)
        
        return "Optics parameters set."
