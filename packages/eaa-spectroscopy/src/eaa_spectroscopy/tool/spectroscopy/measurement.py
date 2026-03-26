from typing import Annotated, Callable, Optional

import numpy as np
import scipy.interpolate
import torch

from eaa_core.tool.base import BaseTool, check, tool
from eaa_core.util import to_numpy, to_tensor


class SpectrumMeasurementTool(BaseTool):
    """Base tool for measuring spectrum values at requested energies."""

    name: str = "spectrum_measurement"

    @check
    def __init__(
        self,
        *args,
        require_approval: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the spectrum measurement tool."""
        super().__init__(*args, require_approval=require_approval, **kwargs)

    @tool(name="measure")
    def measure(
        self,
        x: Annotated[torch.Tensor | np.ndarray, "Measurement locations."],
    ) -> Annotated[torch.Tensor, "Measured spectrum values."]:
        """Measure the spectrum at the requested locations."""
        raise NotImplementedError


class SimulatedSpectrumMeasurementTool(SpectrumMeasurementTool):
    """Simulated spectrum measurement tool backed by interpolation."""

    name: str = "simulated_spectrum_measurement"

    def __init__(
        self,
        f: Optional[Callable[[torch.Tensor], torch.Tensor | np.ndarray]] = None,
        data: Optional[tuple[np.ndarray, np.ndarray] | list[np.ndarray]] = None,
        noise_std: float = 0.0,
        *args,
        require_approval: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the simulated measurement tool.

        Parameters
        ----------
        f : Callable, optional
            Callable that maps ``x`` with shape ``(n_samples, n_features)`` to
            observations with shape ``(n_samples, n_observations)``.
        data : tuple[np.ndarray, np.ndarray] | list[np.ndarray], optional
            Pair of ``(x_grid, y_grid)`` used to build an interpolation-based
            simulator. ``x_grid`` must be one-dimensional.
        noise_std : float, optional
            Standard deviation of Gaussian measurement noise.
        """
        if f is None and data is None:
            raise ValueError("Either `f` or `data` must be provided.")
        self.f = f
        self.data = data
        self.noise_std = noise_std
        self.interpolator = None
        self.n_features = 1
        self.n_observations = 1
        super().__init__(*args, require_approval=require_approval, **kwargs)

    def build(self) -> None:
        """Build the interpolation backend when tabulated data are provided.

        The tabulated input grid must have shape ``(n_grid,)`` and the values
        must have shape ``(n_grid,)`` or ``(n_grid, n_observations)``.
        """
        if self.data is None:
            return
        x_data, y_data = self.data
        x_array = np.asarray(x_data, dtype=float).reshape(-1)
        y_array = np.asarray(y_data, dtype=float)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        if y_array.shape[0] != x_array.shape[0]:
            raise ValueError(
                "`data` must provide matching x/y sample counts, but got "
                f"{x_array.shape[0]} and {y_array.shape[0]}."
            )
        self.interpolator = scipy.interpolate.CubicSpline(
            x_array,
            y_array,
            axis=0,
            extrapolate=True,
        )
        self.n_observations = int(y_array.shape[1])

    def validate_x(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and normalize the x input shape.

        Parameters
        ----------
        x : torch.Tensor
            Candidate measurement locations with shape
            ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            The validated tensor with the same shape.
        """
        if x.ndim != 2:
            raise ValueError(
                "`x` must have shape (n_samples, n_features), "
                f"but got {tuple(x.shape)}."
            )
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"`x` must have {self.n_features} feature(s), but got {x.shape[1]}."
            )
        return x

    def normalize_y(self, y: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Normalize output values to ``(n_samples, n_observations)``.

        Parameters
        ----------
        y : torch.Tensor | np.ndarray
            Raw simulator output with shape ``(n_samples,)`` or
            ``(n_samples, n_observations)``.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(n_samples, n_observations)``.
        """
        y_tensor = to_tensor(y)
        if not isinstance(y_tensor, torch.Tensor):
            y_tensor = torch.as_tensor(y_tensor)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor[:, None]
        if y_tensor.ndim != 2:
            raise ValueError(
                "`y` must have shape (n_samples, n_observations), "
                f"but got {tuple(y_tensor.shape)}."
            )
        return y_tensor

    @tool(name="measure")
    def measure(
        self,
        x: Annotated[torch.Tensor | np.ndarray, "Measurement locations."],
        add_noise: Annotated[bool, "Whether to add simulated measurement noise."] = True,
    ) -> Annotated[torch.Tensor, "Measured spectrum values."]:
        """Measure the simulated spectrum at the requested locations.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            Measurement locations with shape ``(n_samples, n_features)``.
        add_noise : bool, optional
            Whether to add Gaussian noise to the simulated observations.

        Returns
        -------
        torch.Tensor
            Simulated observations with shape ``(n_samples, n_observations)``.
        """
        x_tensor = to_tensor(x)
        if not isinstance(x_tensor, torch.Tensor):
            x_tensor = torch.as_tensor(x_tensor)
        x_tensor = self.validate_x(x_tensor)

        if self.f is not None:
            y_tensor = self.normalize_y(self.f(x_tensor))
        else:
            y_interp = self.interpolator(to_numpy(x_tensor[:, 0]))
            y_tensor = self.normalize_y(y_interp)

        y_tensor = y_tensor.to(dtype=x_tensor.dtype, device=x_tensor.device)
        if add_noise and self.noise_std > 0:
            y_tensor = y_tensor + torch.randn_like(y_tensor) * self.noise_std
        return y_tensor
