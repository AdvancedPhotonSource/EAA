"""Compatibility shim for the moved microscopy optics BO task manager."""

from warnings import warn

from eaa.task_manager.imaging.bo_mic_optics import MicroscopyOpticsTuningBOTaskManager

warn(
    (
        "`eaa.task_manager.tuning.bo_mic_optics` is deprecated. Import "
        "microscopy BO task managers from `eaa.task_manager.imaging.bo_mic_optics`."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MicroscopyOpticsTuningBOTaskManager"]
