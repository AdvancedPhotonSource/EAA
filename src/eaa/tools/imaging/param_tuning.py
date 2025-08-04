from typing import Annotated, Dict, List, Any

import numpy as np

from eaa.tools.base import BaseTool, check, ToolReturnType
from eaa.tools.imaging.acquisition import SimulatedAcquireImage


class SetParameters(BaseTool):
    
    name: str = "tune_optics_parameters"
    
    @check
    def __init__(
        self,
        parameter_names: list[str],
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]],
        *args, 
        **kwargs
    ):
        """Base parameter setting tool.
        
        Parameters
        ----------
        parameter_names : list[str]
            The names of the parameters.
        parameter_ranges : list[tuple[float, ...], tuple[float, ...]]
            The ranges of the parameters. It should be given as a list of 2 sub-lists,
            with the first sub-list containing the lower bounds and the second
            sub-list containing the upper bounds. These ranges will be used to
            scale the parameter errors. As such, inf is not allowed.
        """
        super().__init__(*args, **kwargs)
        
        parameter_ranges = np.array(parameter_ranges)
        if np.any(np.isinf(parameter_ranges)):
            raise ValueError("Inf is not allowed in parameter ranges.")
        if (
            len(parameter_ranges) != 2 
            or len(parameter_ranges[0]) != len(parameter_ranges[1])
            or np.any(parameter_ranges[1] <= parameter_ranges[0])
            or len(parameter_names) != len(parameter_ranges[0])
        ):
            raise ValueError(
                "`parameter_ranges` must be a list of 2 sub-lists, with the "
                "first sub-list containing the lower bounds and the second "
                "sub-list containing the upper bounds. The number of parameters "
                "must match the number of lower and upper bounds."
            )
        self.parameter_ranges = parameter_ranges
        self.parameter_history: dict[str, list[float]] = self.get_initial_parameter_history(
            parameter_names
        )
        
        self.exposed_tools: List[Dict[str, Any]] = [
            {
                "name": "set_parameters",
                "function": self.set_parameters,
                "return_type": ToolReturnType.TEXT
            }
        ]
        
    @property
    def parameter_names(self) -> list[str]:
        return list(self.parameter_history.keys())
        
    @property
    def len_parameter_history(self) -> int:
        return len(self.parameter_history[self.parameter_names[0]])
    
    def get_parameter_at_iteration(
        self, iteration: int, as_dict: bool = False
    ) -> list[float] | dict[str, float]:
        """Get the parameters at a certain index from the parameter history.

        Parameters
        ----------
        iteration : int
            The index of the iteration.
        as_dict : bool, optional
            Whether to return the parameters as a dictionary.

        Returns
        -------
        list[float] | dict[str, float]
            The parameters at the given iteration. If `as_dict` is True, the parameters
            are returned as a dictionary. Otherwise, they are returned as a list.
        """
        if as_dict:
            return {k: self.parameter_history[k][iteration] for k in self.parameter_history.keys()}
        else:
            return [self.parameter_history[k][iteration] for k in self.parameter_history.keys()]
        
    def get_initial_parameter_history(self, parameter_names: list[str]):
        return {name: [] for name in parameter_names}
        
    def set_parameters(*args, **kwargs):
        raise NotImplementedError
    
    def update_parameter_history(self, parameters: list[float] | dict[str, float]):
        if isinstance(parameters, dict):
            for name, param in parameters.items():
                self.parameter_history[name].append(param)
        else:
            for i, val in enumerate(parameters):
                self.parameter_history[self.parameter_names[i]].append(val)
        
        
class BlueSkySetParameters(SetParameters):
    
    name: str = "bluesky_tune_optics_parameters"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def set_parameters(*args, **kwargs):
        raise NotImplementedError
        
        
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
            due to deviation from the true parameters. The amount of drift is
            determined as
            ``mean(delta_params / range) * drift_factor * z``, 
            where ``z`` is a random variable from a uniform distribution between 
            0 and 1.
        """
        super().__init__(
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges,
            *args, **kwargs
        )
        self.acquisition_tool = acquisition_tool
        self.true_parameters = np.array(true_parameters)
        self.blur_factor = blur_factor
        self.drift_factor = drift_factor
        
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
            mean_delta = ((self.get_parameter_at_iteration(-1) - parameters) / scalers).mean()
            drift = np.ones(2) * mean_delta * self.drift_factor
            self.acquisition_tool.set_offset(drift)
        
        # Update parameter history.
        self.update_parameter_history(parameters)
        
        return "Optics parameters set."
