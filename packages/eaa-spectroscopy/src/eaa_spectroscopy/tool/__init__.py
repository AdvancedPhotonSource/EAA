"""Tools for spectroscopy workflows."""

from eaa_spectroscopy.lazy_imports import list_lazy_exports, resolve_lazy_export

EXPORTS = {
    "AdaptiveXANESBayesianOptimization": (
        "eaa_spectroscopy.tool.spectroscopy.optimization"
    ),
    "SimulatedSpectrumMeasurementTool": (
        "eaa_spectroscopy.tool.spectroscopy.measurement"
    ),
    "SpectrumMeasurementTool": "eaa_spectroscopy.tool.spectroscopy.measurement",
}

__all__ = [
    "AdaptiveXANESBayesianOptimization",
    "SimulatedSpectrumMeasurementTool",
    "SpectrumMeasurementTool",
]


def __getattr__(name: str) -> object:
    """Lazily import optional spectroscopy tools."""
    return resolve_lazy_export(EXPORTS, globals(), __name__, name)


def __dir__() -> list[str]:
    """Return module attributes including lazy exports."""
    return list_lazy_exports(globals(), __all__)
