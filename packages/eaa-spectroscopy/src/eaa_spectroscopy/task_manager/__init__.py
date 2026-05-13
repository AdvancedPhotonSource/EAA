"""Task managers for spectroscopy workflows."""

from eaa_spectroscopy.lazy_imports import list_lazy_exports, resolve_lazy_export

EXPORTS = {
    "WeightedPosteriorStandardDeviationStoppingCriterion": (
        "eaa_spectroscopy.task_manager.spectroscopy.xanes"
    ),
    "XANESAdaptiveSamplingTaskManager": (
        "eaa_spectroscopy.task_manager.spectroscopy.xanes"
    ),
}

__all__ = [
    "WeightedPosteriorStandardDeviationStoppingCriterion",
    "XANESAdaptiveSamplingTaskManager",
]


def __getattr__(name: str) -> object:
    """Lazily import optional task managers."""
    return resolve_lazy_export(EXPORTS, globals(), __name__, name)


def __dir__() -> list[str]:
    """Return module attributes including lazy exports."""
    return list_lazy_exports(globals(), __all__)
