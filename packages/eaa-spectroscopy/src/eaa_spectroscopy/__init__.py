"""Public exports for spectroscopy tools and task managers."""

from eaa_spectroscopy.lazy_imports import list_lazy_exports, resolve_lazy_export

EXPORTS = {
    "AdaptiveXANESBayesianOptimization": (
        "eaa_spectroscopy.tool.spectroscopy.optimization"
    ),
    "ComprehensiveAugmentedAcquisitionFunction": (
        "eaa_spectroscopy.acquisition_function.xanes"
    ),
    "FittingResiduePosteriorStandardDeviation": (
        "eaa_spectroscopy.acquisition_function.xanes"
    ),
    "GradientAwarePosteriorStandardDeviation": (
        "eaa_spectroscopy.acquisition_function.xanes"
    ),
    "PosteriorStandardDeviationDerivedAcquisition": (
        "eaa_spectroscopy.acquisition_function.xanes"
    ),
    "SimulatedSpectrumMeasurementTool": (
        "eaa_spectroscopy.tool.spectroscopy.measurement"
    ),
    "SpectrumMeasurementTool": "eaa_spectroscopy.tool.spectroscopy.measurement",
    "WeightedPosteriorStandardDeviationStoppingCriterion": (
        "eaa_spectroscopy.task_manager.spectroscopy.xanes"
    ),
    "XANESAdaptiveSamplingTaskManager": (
        "eaa_spectroscopy.task_manager.spectroscopy.xanes"
    ),
}

__all__ = [
    "AdaptiveXANESBayesianOptimization",
    "ComprehensiveAugmentedAcquisitionFunction",
    "FittingResiduePosteriorStandardDeviation",
    "GradientAwarePosteriorStandardDeviation",
    "PosteriorStandardDeviationDerivedAcquisition",
    "SimulatedSpectrumMeasurementTool",
    "SpectrumMeasurementTool",
    "WeightedPosteriorStandardDeviationStoppingCriterion",
    "XANESAdaptiveSamplingTaskManager",
]


def __getattr__(name: str) -> object:
    """Lazily import optional spectroscopy symbols."""
    return resolve_lazy_export(EXPORTS, globals(), __name__, name)


def __dir__() -> list[str]:
    """Return module attributes including lazy exports."""
    return list_lazy_exports(globals(), __all__)
