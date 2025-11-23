from typing import Annotated, Optional, Tuple
import logging

from eaa.tools.base import ToolReturnType, ExposedToolSpec
from eaa.tools.imaging.param_tuning import SetParameters
from eaa.tools.imaging.aps_mic.util import validate_position_in_range

logger = logging.getLogger(__name__)


class BlueskySetParameters(SetParameters):
    
    from bluesky.run_engine import RunEngine
    from typing import Callable
    from ophyd import EpicsMotor
    import bluesky.plan_stubs as bps

    name: str = "bluesky_set_parameters"
    samz_motor: EpicsMotor = None
    RE: RunEngine = None
    allowable_z_range: Optional[Tuple[float, float]] = None
    bps: Callable = bps
    
    def __init__(
        self, 
        parameter_names: list[str] = ['sample-z'],
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]] = [[-0.5], [0.5]],
        require_approval: bool = False,
        *args, **kwargs
    ):
        """Parameter tuning tool for the imaging system.
        
        Parameters
        ----------
        parameter_names: list[str]
            The names of the parameters.
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]]
            The ranges of the parameters. It should be given as a list of 2 sub-lists,
            with the first sub-list containing the lower bounds and the second
            sub-list containing the upper bounds. These ranges will be used to
            scale the parameter errors. As such, inf is not allowed.
            The unit is in millimeters.
        require_approval: bool
            Whether to require approval for the parameter tuning.
        """

        # self.parameter_names = parameter_names
        # self.parameter_ranges = parameter_ranges

        super().__init__(
            *args,
            parameter_names=parameter_names, 
            parameter_ranges=parameter_ranges, 
            require_approval=require_approval, 
            **kwargs
        )
        
        self.exposed_tools = [
            ExposedToolSpec(
                name="set_parameters",
                function=self.set_parameters,
                return_type=ToolReturnType.TEXT,
            )
        ]
        
    def set_parameters(
        self, 
        parameters: Annotated[
            list[float], 
            "The parameters to set the optics to. For this function, "
            "the list should only contain one element giving the z position."
        ]
    ) -> str:
        """Set the sample z motor position of the imaging system.
        
        Parameters
        ----------
        parameters: list[float]
            The parameters to set the optics to. For this function, 
            the list should only contain one element giving the z position.
        """
        if self.RE is None:
            raise ValueError("RunEngine is not set")
        if self.samz_motor is None:
            raise ValueError("samz_motor is not set")
        if self.parameter_ranges is not None:
            validate_position_in_range(
                parameters[0], 
                (self.parameter_ranges[0][0], self.parameter_ranges[1][0]), 
                "z")
            self.RE(self.bps.mv(self.samz_motor, parameters[0]))
            msg = f"Move sample z motor to position: {parameters[0]}"
            logger.info(msg)
            return msg
        else:
            raise ValueError("parameter_ranges is not set")
