"""Acquisition functions for spectroscopy workflows."""

from eaa_spectroscopy.lazy_imports import list_lazy_exports, resolve_lazy_export

EXPORTS = {
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
}

__all__ = [
    "ComprehensiveAugmentedAcquisitionFunction",
    "FittingResiduePosteriorStandardDeviation",
    "GradientAwarePosteriorStandardDeviation",
    "PosteriorStandardDeviationDerivedAcquisition",
]


def __getattr__(name: str) -> object:
    """Lazily import optional acquisition functions."""
    return resolve_lazy_export(EXPORTS, globals(), __name__, name)


def __dir__() -> list[str]:
    """Return module attributes including lazy exports."""
    return list_lazy_exports(globals(), __all__)
