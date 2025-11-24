#initiating and loading the bluesky environment for the ISN

# ruff: noqa: F403

from eaa.tools.base import BaseTool
from eaa.tools.imaging.aps_mic.acquisition import BlueSkyAcquireImage
from eaa.tools.imaging.aps_mic.param_tuning import BlueskySetParameters


class BlueskyScanControl(BaseTool):
    
    from typing import Callable
    from ophyd import EpicsMotor
    
    samz_motor: EpicsMotor = None
    scan2d_plan: Callable = None
    scan1d_plan: Callable = None
    acquire_image_tool: BlueSkyAcquireImage = None
    param_tuning_tool: BlueskySetParameters = None
    
    def __init__(self, require_approval: bool = False, *args, **kwargs):
        super().__init__(require_approval=require_approval, *args, **kwargs)
        
        try: 
            from s2idd_uprobe.startup import RE, oregistry, fly2d_scanrecord, step1d_scanrecord, bps
        except ImportError:
            raise ImportError(
                "Bluesky control initialization failed. Please check that the bluesky-mic package is installed "
                "and the motors can only be reached from private subnet computers."
            )

        self.acquire_image_tool = BlueSkyAcquireImage()
        self.acquire_image_tool.RE = RE
        self.acquire_image_tool.savedata = oregistry["savedata"]
        self.acquire_image_tool.scan2d_plan = fly2d_scanrecord
        self.acquire_image_tool.scan1d_plan = step1d_scanrecord
        self.acquire_image_tool.samy_motor = oregistry["samy"]

        self.param_tuning_tool = BlueskySetParameters()
        self.param_tuning_tool.RE = RE
        self.param_tuning_tool.zp_z_motor = oregistry["zp_z"]

        self.exposed_tools = self.acquire_image_tool.exposed_tools + self.param_tuning_tool.exposed_tools
