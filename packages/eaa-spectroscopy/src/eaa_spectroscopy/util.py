from typing import Callable, Optional

import numpy as np
import scipy.interpolate
import scipy.signal
import torch

from eaa_core.util import to_numpy


def interp1d_tensor(x0: torch.Tensor, y0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate one-dimensional tensor data.

    Parameters
    ----------
    x0 : torch.Tensor
        Sorted sample locations with shape ``(n_points,)``.
    y0 : torch.Tensor
        Sample values with shape ``(n_points,)``.
    x : torch.Tensor
        Query locations with shape ``(n_queries,)``.

    Returns
    -------
    torch.Tensor
        Interpolated values with shape ``(n_queries,)``.
    """
    indices = torch.bucketize(x, x0)
    y_floor = y0[torch.clamp(indices - 1, 0, len(y0) - 1)]
    y_ceil = y0[torch.clamp(indices, 0, len(y0) - 1)]
    x_floor = x0[torch.clamp(indices - 1, 0, len(x0) - 1)]
    x_ceil = x0[torch.clamp(indices, 0, len(x0) - 1)]
    weights = ((x - x_floor) / (x_ceil - x_floor + 1e-8)).clamp(0, 1)
    return y_ceil * weights + y_floor * (1 - weights)


def elementwise_derivative_analytical(
    function: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    order: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return elementwise first- or second-order analytical derivatives.

    Parameters
    ----------
    function : Callable[[torch.Tensor], torch.Tensor]
        Differentiable callable whose output is interpreted elementwise.
    x : torch.Tensor
        Input tensor.
    order : int, optional
        Derivative order. Supported values are 1 and 2.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        First derivative, or first and second derivatives when ``order == 2``.
    """

    def differentiate(values: torch.Tensor) -> torch.Tensor:
        jacobian = torch.autograd.functional.jacobian(
            function,
            values,
            create_graph=True,
        )
        diagonal = jacobian[torch.arange(jacobian.shape[0]), torch.arange(jacobian.shape[0])]
        return diagonal

    gradient = differentiate(x)
    if order == 1:
        return gradient
    if order == 2:
        hessian = torch.autograd.functional.jacobian(
            differentiate,
            x,
            create_graph=True,
        )
        second = hessian[torch.arange(hessian.shape[0]), torch.arange(hessian.shape[0])]
        return gradient, second
    raise ValueError(f"Unsupported derivative order: {order}.")


def elementwise_derivative_finite_difference(
    function: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    order: int = 1,
    step_size: float = 1e-2,
    dims: Optional[list[int]] = None,
) -> torch.Tensor:
    """Return elementwise finite-difference derivatives along selected axes.

    Parameters
    ----------
    function : Callable[[torch.Tensor], torch.Tensor]
        Callable evaluated on perturbed versions of ``x``. The output is
        expected to preserve the leading batch structure of ``x``.
    x : torch.Tensor
        Input tensor, typically with shape ``batch_shape x q x d``.
    order : int, optional
        Derivative order.
    step_size : float, optional
        Central-difference step size.
    dims : list[int], optional
        Feature dimensions to differentiate along. Defaults to all trailing
        feature dimensions.

    Returns
    -------
    torch.Tensor
        Finite-difference derivatives with shape
        ``batch_shape x q x n_gradient_dims`` before any caller-side squeezing.
    """

    def differentiate(
        values: torch.Tensor,
        gradient_dim: int,
        derivative_order: int = 1,
    ) -> torch.Tensor:
        step = torch.zeros_like(values)
        step[..., gradient_dim] = step_size
        if derivative_order == 1:
            return (function(values + step) - function(values - step)) / (2 * step_size)
        return (
            differentiate(values + step, gradient_dim, derivative_order=derivative_order - 1)
            - differentiate(values - step, gradient_dim, derivative_order=derivative_order - 1)
        ) / (2 * step_size)

    gradient_dims = dims or list(range(x.shape[-1]))
    gradients = [
        differentiate(x, gradient_dim, derivative_order=order)
        for gradient_dim in gradient_dims
    ]
    return torch.cat(gradients, dim=-1)


def sigmoid(x: torch.Tensor | np.ndarray | float, r: float = 1.0, d: float = 0.0):
    """Return a sigmoid function value for tensor or NumPy inputs."""
    module = torch if isinstance(x, torch.Tensor) else np
    exponent = module.clip(-r * (x - d), None, 1e2)
    return 1.0 / (1.0 + module.exp(exponent))


def gaussian(
    x: torch.Tensor | np.ndarray | float,
    a: float,
    mu: float,
    sigma: float,
    c: float,
):
    """Return a Gaussian function value for tensor or NumPy inputs."""
    if isinstance(x, torch.Tensor):
        return a * torch.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c


def estimate_edge_location_and_width(
    data_x: np.ndarray,
    data_y: np.ndarray,
    x_dense: Optional[np.ndarray] = None,
    return_in_pixel_unit: bool = False,
) -> tuple[float | int, float]:
    """Estimate XANES absorption-edge location and width from sampled data.

    Parameters
    ----------
    data_x : np.ndarray
        Measured energies.
    data_y : np.ndarray
        Measured spectrum values.
    x_dense : np.ndarray, optional
        Dense query grid used for interpolation and gradient analysis.
    return_in_pixel_unit : bool, optional
        Whether to return the edge location in the dense-grid pixel index and
        the width in dense-grid pixels.

    Returns
    -------
    tuple[float | int, float]
        Edge location and width.
    """
    if x_dense is None:
        x_dense = np.linspace(data_x[0], data_x[-1], len(data_x) * 10)
    y_dense = scipy.interpolate.CubicSpline(data_x, data_y)(x_dense)

    gradient = (y_dense[2:] - y_dense[:-2]) / (x_dense[2:] - x_dense[:-2])
    dense_step = x_dense[1] - x_dense[0]

    peak_locations, peak_properties = scipy.signal.find_peaks(
        gradient,
        height=gradient.max() * 0.5,
        width=1,
    )
    peak_index = int(np.argmax(peak_properties["peak_heights"]))
    edge_location = peak_locations[peak_index]
    edge_width = peak_properties["widths"][peak_index]

    if return_in_pixel_unit:
        return edge_location, edge_width
    return float(x_dense[edge_location]), float(edge_width * dense_step)


def sort_unique_xy(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique x/y observations in NumPy form."""
    x_numpy = to_numpy(x.reshape(-1))
    y_numpy = to_numpy(y)
    x_unique, unique_indices = np.unique(x_numpy, return_index=True)
    y_unique = y_numpy[unique_indices]
    sort_indices = np.argsort(x_unique)
    return x_unique[sort_indices], y_unique[sort_indices]
