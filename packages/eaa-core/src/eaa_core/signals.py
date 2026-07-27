"""Shared control signals passed between EAA runtime components."""

from enum import Enum


class ControlSignal(Enum):
    """Out-of-band signals that must not be interpreted as user input."""

    BACKGROUND_TOOL_COMPLETION_WAKEUP = "background_tool_completion_wakeup"
