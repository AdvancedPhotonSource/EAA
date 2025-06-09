from typing import Annotated, Dict, Callable

import numpy as np

from eaa.tools.base import BaseTool, check
from eaa.tools.imaging.acquisition import SimulatedAcquireImage


class TuneOpticsParameters(BaseTool):
    
    name: str = "tune_optics_parameters"
    
    @check
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_history = []
        
        self.exposed_tools: Dict[str, Callable] = {
            "set_parameters": self.set_parameters
        }
        
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
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.acquisition_tool = acquisition_tool
        self.true_parameters = true_parameters
        
    def set_parameters(
        self, 
        param1: float, 
        param2: float, 
        param3: float, 
    ) -> Annotated[str, "A confirmation message."]:
        """Set the optics parameters of the imaging system to given values.
        
        Parameters
        ----------
        param1, param2, param3 : float
            The parameters to set the optics to.
        
        Returns
        -------
        str
            A confirmation message.
        """
        normalized_errors = []
        input_parameters = [param1, param2, param3]
        for i, true_param in enumerate(self.true_parameters):
            normalized_error = (input_parameters[i] - true_param) / (true_param + 1e-6)
            normalized_errors.append(normalized_error)
        normalized_errors = np.array(normalized_errors)
        total_error = np.sqrt(np.sum(normalized_errors ** 2))
        blur = total_error * 5
        self.acquisition_tool.set_blur(blur)
        
        self.parameter_history.append({
            "param1": param1,
            "param2": param2,
            "param3": param3,
        })
        return "Optics parameters set."
