from typing import Annotated

import numpy as np
from eaa_core.tool.base import tool
from eaa_core.tool.param_tuning import SetParameters

from eaa_imaging.tool.imaging.acquisition import SimulatedAcquireImage

class SimulatedSetParameters(SetParameters):
    
    name: str = "simulated_tune_optics_parameters"
    
    def __init__(
        self, 
        acquisition_tool: SimulatedAcquireImage,
        parameter_names: list[str],
        parameter_ranges: list[list[float], list[float]],
        true_parameters: list[float], 
        blur_factor: float = 5,
        drift_factor: float = 0,
        require_approval: bool = False,
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
            The amount of blurring is determined as
            ``sum(abs(delta_params / range)) * blur_factor``, where ``delta_params``
            is the difference between the true parameters and the parameters to set.
        drift_factor : float
            The factor determining the amount of drift of the acquisition tool
            due to deviation from the initial parameters. The amount of drift is
            determined as
            ``mean(delta_params / range) * drift_factor * z``, 
            where ``z`` is a random variable from a uniform distribution between 
            0 and 1.
        """
        super().__init__(
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges,
            require_approval=require_approval,
            *args,
            **kwargs
        )
        self.acquisition_tool = acquisition_tool
        self.true_parameters = np.array(true_parameters)
        self.blur_factor = blur_factor
        self.drift_factor = drift_factor
        
    @tool(name="set_parameters")
    def set_parameters(
        self, 
        parameters: Annotated[list[float], "The parameters to set the optics to"],
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
        if self.len_parameter_history > 0 and self.drift_factor > 0:
            mean_delta = ((self.get_parameter_at_iteration(0) - parameters) / scalers).mean()
            drift = np.ones(2) * mean_delta * self.drift_factor
            self.acquisition_tool.set_offset(drift)
        
        # Update parameter history.
        self.update_parameter_history(parameters)
        
        return "Optics parameters set."
