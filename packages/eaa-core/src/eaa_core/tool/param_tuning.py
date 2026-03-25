"""Core abstractions for parameter-setting tools."""

from typing import Annotated
import json

import numpy as np

from eaa_core.tool.base import BaseTool, check, tool


class SetParameters(BaseTool):
    """Base tool for setting named parameters on an experimental system."""

    name: str = "set_parameters"

    @check
    def __init__(
        self,
        parameter_names: list[str],
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]],
        require_approval: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the parameter-setting tool.

        Parameters
        ----------
        parameter_names : list[str]
            Ordered parameter names managed by the tool.
        parameter_ranges : list[tuple[float, ...], tuple[float, ...]]
            Lower and upper bounds for each parameter.
        require_approval : bool, optional
            Whether tool calls require approval before execution.
        """
        super().__init__(*args, require_approval=require_approval, **kwargs)

        ranges = np.array(parameter_ranges)
        if np.any(np.isinf(ranges)):
            raise ValueError("Inf is not allowed in parameter ranges.")
        if (
            len(ranges) != 2
            or len(ranges[0]) != len(ranges[1])
            or np.any(ranges[1] <= ranges[0])
            or len(parameter_names) != len(ranges[0])
        ):
            raise ValueError(
                "`parameter_ranges` must provide lower and upper bounds for "
                "each parameter, and those bounds must match the number of "
                "parameter names."
            )
        self.parameter_ranges = ranges
        self.parameter_history: dict[str, list[float]] = {
            name: [] for name in parameter_names
        }

    @property
    def parameter_names(self) -> list[str]:
        """Return the ordered parameter names."""
        return list(self.parameter_history.keys())

    @property
    def len_parameter_history(self) -> int:
        """Return the number of recorded parameter updates."""
        return len(self.parameter_history[self.parameter_names[0]])

    def get_parameter_at_iteration(
        self,
        iteration: int,
        as_dict: bool = False,
    ) -> list[float] | dict[str, float]:
        """Return parameter values at a given history index."""
        if as_dict:
            return {
                key: self.parameter_history[key][iteration]
                for key in self.parameter_history
            }
        return [self.parameter_history[key][iteration] for key in self.parameter_history]

    @tool(name="set_parameters")
    def set_parameters(
        self,
        parameters: Annotated[list[float], "The parameters to set."],
    ) -> Annotated[str, "A confirmation message."]:
        """Set parameter values on the target system."""
        raise NotImplementedError

    @tool(name="get_current_parameters")
    def get_current_parameters(self) -> str:
        """Return the most recent parameter values as JSON."""
        return json.dumps(self.get_parameter_at_iteration(-1, as_dict=True))

    def update_parameter_history(
        self,
        parameters: list[float] | dict[str, float],
    ) -> None:
        """Append parameter values to the history."""
        if isinstance(parameters, dict):
            for name, param in parameters.items():
                self.parameter_history[name].append(param)
            return
        for index, value in enumerate(parameters):
            self.parameter_history[self.parameter_names[index]].append(value)
