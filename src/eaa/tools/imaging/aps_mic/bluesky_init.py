#initiating and loading the bluesky environment for the ISN

# ruff: noqa: F403
# from isn.startup import *
from s2idd_uprobe.startup import *


def get_control_components(device_name: str):
    try:
        # ruff: noqa: F405
        return oregistry[device_name]
    except KeyError:
        raise KeyError(f"Device {device_name} not found in the oregistry")
