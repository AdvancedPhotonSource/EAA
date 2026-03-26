from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import scipy.ndimage
import scipy.signal
import torch
from botorch.acquisition import PosteriorStandardDeviation
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from torch import Tensor

from eaa_spectroscopy.util import (
    elementwise_derivative_analytical,
    elementwise_derivative_finite_difference,
    estimate_edge_location_and_width,
    gaussian,
    interp1d_tensor,
    sigmoid,
    sort_unique_xy,
)

logger = logging.getLogger(__name__)


class DummyAcquisition:
    """Fallback acquisition that always evaluates to zero."""

    def __getattr__(self, name: str):
        """Return a no-op callable for any missing attribute access."""
        def dummy_function(*args, **kwargs):
            return None

        return dummy_function

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Return a scalar zero acquisition value."""
        return torch.tensor(0.0, device=x.device)

    def __call__(self, *args, **kwargs):
        """Evaluate the dummy acquisition function."""
        return self.forward(*args, **kwargs)


class PosteriorStandardDeviationDerivedAcquisition(PosteriorStandardDeviation):
    """Posterior-std acquisition with optional XANES-specific weighting.

    This is the shared base for the XANES acquisition functions in EAA. It
    augments BoTorch's :class:`PosteriorStandardDeviation` with:

    - access to the owning BO tool and its observed data,
    - an optional acquisition-weighting mask in normalized x-space,
    - optional spline-interpolated posterior means, and
    - simple hyperparameter decay scheduling for additive terms.
    """

    def __init__(
        self,
        model: Model,
        bo_tool,
        maximize: bool = True,
        beta: float = 0.95,
        gamma: float = 0.95,
        acqf_weight_func_floor_value: float = 0.1,
        acqf_weight_func_edge_offset: float = -1.6,
        acqf_weight_func_post_edge_gain: float = 5.0,
        acqf_weight_func_post_edge_offset: float = 1.0,
        acqf_weight_func_post_edge_width: float = 0.5,
        acqf_weight_func_post_edge_decay_location: float = 50.0,
        add_posterior_stddev: bool = True,
        estimate_posterior_mean_by_interpolation: bool = False,
    ) -> None:
        """Initialize the acquisition function.

        Parameters
        ----------
        model : Model
            Underlying GP model.
        bo_tool : BayesianOptimizationTool
            Optimization tool that owns the GP model, observed data, and data
            transforms.
        maximize : bool, optional
            Forwarded to the BoTorch parent class.
        beta : float, optional
            Decay factor for additive-term weights.
        gamma : float, optional
            Decay factor for the acquisition weighting mask amplitude.
        acqf_weight_func_floor_value : float, optional
            Minimum value applied to the acquisition weighting function.
        acqf_weight_func_edge_offset : float, optional
            Offset, in units of estimated edge width, applied to the edge
            location when constructing the weighting function.
        acqf_weight_func_post_edge_gain : float, optional
            Gain controlling the post-edge Gaussian boost.
        acqf_weight_func_post_edge_offset : float, optional
            Offset applied to the center of the post-edge Gaussian term.
        acqf_weight_func_post_edge_width : float, optional
            Width multiplier for the post-edge Gaussian term.
        acqf_weight_func_post_edge_decay_location : float, optional
            Decay location, in units of edge width, for the trailing sigmoid.
        add_posterior_stddev : bool, optional
            Whether posterior standard deviation contributes directly.
        estimate_posterior_mean_by_interpolation : bool, optional
            Whether to replace the GP posterior mean with spline interpolation
            of observed data.
        """
        super().__init__(
            model=model,
            posterior_transform=bo_tool.get_unstandardize_posterior_transform(),
            maximize=maximize,
        )
        self.bo_tool = bo_tool
        self.add_posterior_stddev = add_posterior_stddev
        self.weight_func: Optional[Callable] = None
        self.alpha = 1.0
        self.phi = 0.0
        self.beta = beta
        self.gamma = gamma
        self.acqf_weight_func_floor_value = acqf_weight_func_floor_value
        self.acqf_weight_func_edge_offset = acqf_weight_func_edge_offset
        self.acqf_weight_func_post_edge_gain = acqf_weight_func_post_edge_gain
        self.acqf_weight_func_post_edge_offset = acqf_weight_func_post_edge_offset
        self.acqf_weight_func_post_edge_width = acqf_weight_func_post_edge_width
        self.acqf_weight_func_post_edge_decay_location = (
            acqf_weight_func_post_edge_decay_location
        )
        self.intermediate_data: dict[str, Any] = {}
        self.estimate_posterior_mean_by_interpolation = (
            estimate_posterior_mean_by_interpolation
        )

    def set_weight_func(self, function: Callable) -> None:
        """Attach an x-space weighting function to the acquisition.

        Parameters
        ----------
        function : Callable
            Callable that accepts normalized query locations with shape
            ``batch_shape x q x d`` and returns multiplicative weights that are
            broadcast-compatible with the acquisition value.
        """
        self.weight_func = function

    def apply_weight_func(self, x: Tensor, acquisition_value: Tensor) -> Tensor:
        """Apply the configured XANES weighting function, if present.

        Parameters
        ----------
        x : Tensor
            Query locations in normalized BoTorch format with shape
            ``batch_shape x q x d``.
        acquisition_value : Tensor
            Acquisition values with shape ``batch_shape`` or ``batch_shape x q``.

        Returns
        -------
        Tensor
            Weighted acquisition values with the same shape as
            ``acquisition_value``.
        """
        if self.weight_func is None:
            return acquisition_value
        weights = self.weight_func(x).squeeze(-2).squeeze(-1)
        return (self.alpha * weights + 1 - self.alpha) * acquisition_value

    def estimate_edge_location_and_width(
        self,
        x: Tensor,
        y: Tensor,
        input_is_transformed: bool = True,
        run_in_transformed_space: bool = True,
        return_normalized_values: bool = True,
    ) -> tuple[float, float]:
        """Estimate XANES edge location and width from observed data.

        Parameters
        ----------
        x : Tensor
            Observed energies with shape ``(n_samples, 1)``.
        y : Tensor
            Observed spectrum values with shape ``(n_samples, 1)``.
        input_is_transformed : bool, optional
            Whether ``x`` and ``y`` are already in normalized / standardized
            space.
        run_in_transformed_space : bool, optional
            Whether the edge estimation should operate in normalized x-space.
        return_normalized_values : bool, optional
            Whether the returned edge location and width should be normalized.

        Returns
        -------
        tuple[float, float]
            Estimated edge location and width.
        """
        if not input_is_transformed and run_in_transformed_space:
            x, y = self.bo_tool.transform_data(
                x=x,
                y=y,
                train_x=False,
                train_y=False,
            )

        x_data, y_data = sort_unique_xy(x, y)
        y_data = np.asarray(y_data).reshape(-1)
        if run_in_transformed_space:
            x_dense = np.linspace(0.0, 1.0, len(x_data) * 10)
        else:
            x_dense = np.linspace(x_data[0], x_data[-1], len(x_data) * 10)
        dense_step = x_dense[1] - x_dense[0]
        peak_location, peak_width = estimate_edge_location_and_width(
            x_data,
            y_data,
            x_dense=x_dense,
            return_in_pixel_unit=True,
        )

        if return_normalized_values and not run_in_transformed_space:
            return (
                float(peak_location) / len(x_dense) + dense_step,
                float(peak_width) / len(x_dense),
            )
        if not return_normalized_values and not run_in_transformed_space:
            return float(x_dense[peak_location]), float(peak_width * dense_step)
        return float(x_dense[peak_location]), float(peak_width * dense_step)

    def build_acquisition_weight_function(self) -> None:
        """Build the XANES acquisition weighting function in normalized x-space.
        """
        n_query = max(int(self.bo_tool.xs_untransformed.shape[0]) * 10, 32)
        x_query = torch.linspace(
            0.0,
            1.0,
            n_query,
            dtype=self.bo_tool.xs_transformed.dtype,
            device=self.bo_tool.xs_transformed.device,
        ).reshape(-1, 1)
        mu, _ = self.bo_tool.get_posterior_mean_and_std(
            x_query,
            transform=False,
            untransform=True,
        )
        energy_span = float(self.bo_tool.bounds[1][0] - self.bo_tool.bounds[0][0])
        mu_smoothed = scipy.ndimage.gaussian_filter(
            mu.squeeze().detach().cpu().numpy(),
            sigma=max(3.0 / energy_span * len(x_query), 1.0),
        )
        mu_gradient = scipy.signal.convolve(
            np.pad(mu_smoothed, [1, 1], mode="edge"),
            [0.5, 0, -0.5],
            mode="valid",
        )

        peak_locations, peak_properties = scipy.signal.find_peaks(
            mu_gradient,
            height=0.01,
            width=1,
        )
        if len(peak_locations) == 0:
            peak_location, peak_width = self.estimate_edge_location_and_width(
                self.bo_tool.xs_transformed,
                self.bo_tool.ys_transformed,
                input_is_transformed=True,
                run_in_transformed_space=True,
                return_normalized_values=True,
            )
        else:
            peak_index = int(np.argmax(peak_properties["peak_heights"]))
            peak_location = float(peak_locations[peak_index]) / len(x_query)
            peak_width = float(peak_properties["widths"][peak_index]) / len(x_query)
        peak_width = max(float(peak_width), 1.0 / len(x_query))

        def weight_function(x: Tensor) -> Tensor:
            transition_rate = 3200.0 / energy_span / peak_width
            weight = sigmoid(
                x,
                r=transition_rate,
                d=peak_location + self.acqf_weight_func_edge_offset * peak_width,
            )
            weight = weight + gaussian(
                x,
                a=self.acqf_weight_func_post_edge_gain,
                mu=peak_location + self.acqf_weight_func_post_edge_offset * peak_width,
                sigma=peak_width * self.acqf_weight_func_post_edge_width,
                c=0.0,
            )
            weight = weight - sigmoid(
                x,
                r=20.0 / peak_width,
                d=peak_location
                + self.acqf_weight_func_post_edge_decay_location * peak_width,
            )
            floor_value = self.acqf_weight_func_floor_value
            return weight * (1.0 - floor_value) + floor_value

        self.set_weight_func(weight_function)

    def update_hyperparams_following_schedule(self) -> None:
        """Decay auxiliary acquisition weights after each model update.

        The base implementation decays the generic additive-term weight
        ``phi`` and the weighting-mask mixing factor ``alpha``.
        """
        self.phi = self.phi * self.beta
        self.alpha = self.alpha * self.gamma

    def _mean_and_sigma(
        self,
        X: Tensor,
        compute_sigma: bool = True,
        min_var: float = 1e-12,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Return posterior mean/std, optionally using spline interpolation.

        Parameters
        ----------
        X : Tensor
            Query locations with shape ``batch_shape x q x d``.
        compute_sigma : bool, optional
            Whether to compute posterior standard deviation.
        min_var : float, optional
            Minimum posterior variance used by the BoTorch parent class.

        Returns
        -------
        tuple[Tensor, Optional[Tensor]]
            Posterior mean and optional standard deviation, both squeezed to
            shape ``batch_shape x q`` for the single-output XANES setting.
        """
        if not self.estimate_posterior_mean_by_interpolation:
            mean, sigma = super()._mean_and_sigma(
                X,
                compute_sigma=compute_sigma,
                min_var=min_var,
            )
            mean = mean.squeeze(-1)
            if sigma is not None:
                sigma = sigma.squeeze(-1)
            return mean, sigma

        sigma = None
        if compute_sigma:
            _, sigma = super()._mean_and_sigma(X, compute_sigma=True, min_var=min_var)
            sigma = sigma.squeeze(-1)
        x_query = X.squeeze(-2)
        mu = self.bo_tool.get_estimated_data_by_interpolation(
            x_query,
            input_is_transformed=True,
        ).squeeze(-1)
        return mu, sigma


class FittingResiduePosteriorStandardDeviation(
    PosteriorStandardDeviationDerivedAcquisition
):
    """Posterior-std acquisition augmented with reference-spectrum residue.

    The posterior mean over the reference-energy grid is fit as a linear
    combination of reference spectra. The residual of that fit is then
    interpolated back to query locations and combined with posterior
    uncertainty.
    """

    def __init__(
        self,
        model: Model,
        bo_tool,
        maximize: bool = True,
        beta: float = 0.99,
        gamma: float = 0.95,
        acqf_weight_func_floor_value: float = 0.1,
        acqf_weight_func_edge_offset: float = -1.6,
        acqf_weight_func_post_edge_gain: float = 5.0,
        acqf_weight_func_post_edge_offset: float = 1.0,
        acqf_weight_func_post_edge_width: float = 0.5,
        acqf_weight_func_post_edge_decay_location: float = 50.0,
        reference_spectra_x: Tensor | None = None,
        reference_spectra_y: Tensor | None = None,
        phi: Optional[float] = 0.1,
        add_posterior_stddev: bool = True,
        estimate_posterior_mean_by_interpolation: bool = False,
    ) -> None:
        """Initialize the residue-based acquisition function.

        Parameters
        ----------
        model : Model
            Underlying GP model.
        bo_tool : BayesianOptimizationTool
            Optimization tool that owns the GP model, observed data, and data
            transforms.
        maximize : bool, optional
            Forwarded to the BoTorch parent class.
        beta : float, optional
            Decay factor for additive-term weights.
        gamma : float, optional
            Decay factor for the acquisition weighting mask amplitude.
        acqf_weight_func_floor_value : float, optional
            Minimum value applied to the acquisition weighting function.
        acqf_weight_func_edge_offset : float, optional
            Offset, in units of estimated edge width, applied to the edge
            location when constructing the weighting function.
        acqf_weight_func_post_edge_gain : float, optional
            Gain controlling the post-edge Gaussian boost.
        acqf_weight_func_post_edge_offset : float, optional
            Offset applied to the center of the post-edge Gaussian term.
        acqf_weight_func_post_edge_width : float, optional
            Width multiplier for the post-edge Gaussian term.
        acqf_weight_func_post_edge_decay_location : float, optional
            Decay location, in units of edge width, for the trailing sigmoid.
        reference_spectra_x : Tensor, optional
            Reference-energy grid with shape ``(n_reference_points,)``.
        reference_spectra_y : Tensor, optional
            Reference spectra with shape
            ``(n_reference_spectra, n_reference_points)``.
        phi : float, optional
            Weight of the fitting-residue term. When ``None``, it is estimated
            automatically from posterior uncertainty.
        add_posterior_stddev : bool, optional
            Whether posterior standard deviation contributes directly.
        estimate_posterior_mean_by_interpolation : bool, optional
            Whether to replace the GP posterior mean with spline interpolation
            of observed data.
        """
        super().__init__(
            model=model,
            bo_tool=bo_tool,
            maximize=maximize,
            beta=beta,
            gamma=gamma,
            acqf_weight_func_floor_value=acqf_weight_func_floor_value,
            acqf_weight_func_edge_offset=acqf_weight_func_edge_offset,
            acqf_weight_func_post_edge_gain=acqf_weight_func_post_edge_gain,
            acqf_weight_func_post_edge_offset=acqf_weight_func_post_edge_offset,
            acqf_weight_func_post_edge_width=acqf_weight_func_post_edge_width,
            acqf_weight_func_post_edge_decay_location=(
                acqf_weight_func_post_edge_decay_location
            ),
            add_posterior_stddev=add_posterior_stddev,
            estimate_posterior_mean_by_interpolation=estimate_posterior_mean_by_interpolation,
        )
        if reference_spectra_x is None or reference_spectra_y is None:
            raise ValueError(
                "`reference_spectra_x` and `reference_spectra_y` are required "
                "for residue-based XANES acquisition."
            )
        self.reference_spectra_x = self.bo_tool.transform_data(
            x=reference_spectra_x.reshape(-1, 1),
            y=None,
            train_x=False,
            train_y=False,
        )[0].squeeze()
        self.reference_spectra_y = torch.as_tensor(
            reference_spectra_y,
            dtype=self.reference_spectra_x.dtype,
            device=self.reference_spectra_x.device,
        )
        self.phi = phi
        if self.phi is None:
            self.estimate_weights()

    def estimate_weights(self) -> None:
        """Auto-scale the residue term relative to posterior uncertainty.

        The method evaluates the acquisition on a dense normalized grid and
        chooses ``phi`` so the residue contribution has a comparable scale to
        posterior standard deviation.
        """
        self.phi = 1.0
        x = torch.linspace(0, 1, 100).view(-1, 1, 1)
        original_add_posterior_stddev = self.add_posterior_stddev
        self.add_posterior_stddev = True
        self.forward(x)
        self.add_posterior_stddev = original_add_posterior_stddev

        sigma = self.intermediate_data["sigma"]
        residue = self.intermediate_data["r"]
        self.phi = float(sigma.max() / residue.max() * 5.0)
        logger.info(
            "Automatically determined fitting residue weight: phi=%s.",
            self.phi,
        )

    @t_batch_mode_transform()
    def forward(
        self,
        x: Tensor,
        mu_x: Tensor | None = None,
        sigma_x: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Return the residue-augmented acquisition value.

        Parameters
        ----------
        x : Tensor
            Query locations with shape ``batch_shape x q x d``.
        mu_x : Tensor, optional
            Precomputed posterior means with shape ``batch_shape x q``.
        sigma_x : Tensor, optional
            Precomputed posterior standard deviations with shape
            ``batch_shape x q`` or ``batch_shape x q x 1``.

        Returns
        -------
        Tensor
            Acquisition values with shape ``batch_shape`` for ``q == 1``.
        """
        mu_reference, _ = self._mean_and_sigma(
            self.reference_spectra_x.view(-1, 1, 1),
            compute_sigma=False,
        )
        matrix = self.reference_spectra_y.T
        coeffs = torch.matmul(torch.linalg.pinv(matrix), mu_reference.reshape(-1, 1))
        y_fit = torch.matmul(matrix, coeffs).view(-1)
        residue = (y_fit - mu_reference) ** 2

        sigma = 0.0
        if self.add_posterior_stddev:
            if sigma_x is None:
                _, sigma = self._mean_and_sigma(x)
            else:
                sigma = sigma_x.squeeze(-1)
        residue_interp = interp1d_tensor(
            self.reference_spectra_x.squeeze(),
            residue,
            x.squeeze(),
        )
        acquisition_value = sigma + self.phi * residue_interp
        acquisition_value = self.apply_weight_func(x, acquisition_value)
        self.intermediate_data = {"sigma": sigma, "r": residue_interp}
        return acquisition_value


class GradientAwarePosteriorStandardDeviation(
    PosteriorStandardDeviationDerivedAcquisition
):
    """Posterior-std acquisition augmented with spectral gradient magnitude.

    The acquisition combines posterior uncertainty with the norm of the first-
    or higher-order derivative of the posterior mean along selected feature
    dimensions.
    """

    def __init__(
        self,
        model: Model,
        bo_tool,
        maximize: bool = True,
        beta: float = 0.99,
        gamma: float = 0.95,
        acqf_weight_func_floor_value: float = 0.1,
        acqf_weight_func_edge_offset: float = -1.6,
        acqf_weight_func_post_edge_gain: float = 5.0,
        acqf_weight_func_post_edge_offset: float = 1.0,
        acqf_weight_func_post_edge_width: float = 0.5,
        acqf_weight_func_post_edge_decay_location: float = 50.0,
        gradient_dims: Optional[list[int]] = None,
        phi: Optional[float] = 0.1,
        phi2: Optional[float] = 0.001,
        method: str = "analytical",
        order: int = 1,
        finite_difference_interval: float = 1e-2,
        add_posterior_stddev: bool = True,
        estimate_posterior_mean_by_interpolation: bool = False,
        subtract_background_gradient: bool = True,
    ) -> None:
        """Initialize the gradient-aware acquisition function.

        Parameters
        ----------
        model : Model
            Underlying GP model.
        bo_tool : BayesianOptimizationTool
            Optimization tool that owns the GP model, observed data, and data
            transforms.
        maximize : bool, optional
            Forwarded to the BoTorch parent class.
        beta : float, optional
            Decay factor for additive-term weights.
        gamma : float, optional
            Decay factor for the acquisition weighting mask amplitude.
        acqf_weight_func_floor_value : float, optional
            Minimum value applied to the acquisition weighting function.
        acqf_weight_func_edge_offset : float, optional
            Offset, in units of estimated edge width, applied to the edge
            location when constructing the weighting function.
        acqf_weight_func_post_edge_gain : float, optional
            Gain controlling the post-edge Gaussian boost.
        acqf_weight_func_post_edge_offset : float, optional
            Offset applied to the center of the post-edge Gaussian term.
        acqf_weight_func_post_edge_width : float, optional
            Width multiplier for the post-edge Gaussian term.
        acqf_weight_func_post_edge_decay_location : float, optional
            Decay location, in units of edge width, for the trailing sigmoid.
        gradient_dims : list[int], optional
            Feature dimensions along which derivatives are computed. Defaults to
            all feature dimensions.
        phi : float, optional
            Weight of the first-derivative term. When ``None``, it is estimated
            automatically.
        phi2 : float, optional
            Weight of the second- or higher-order derivative terms. When
            ``None``, it is estimated automatically.
        method : str, optional
            Derivative method, either ``"analytical"`` or ``"numerical"``.
        order : int, optional
            Highest derivative order to include.
        finite_difference_interval : float, optional
            Step size used by numerical differentiation in normalized x-space.
        add_posterior_stddev : bool, optional
            Whether posterior standard deviation contributes directly.
        estimate_posterior_mean_by_interpolation : bool, optional
            Whether to replace the GP posterior mean with spline interpolation
            of observed data.
        subtract_background_gradient : bool, optional
            Whether to subtract a background gradient estimate from the first
            derivative term when using numerical gradients.
        """
        super().__init__(
            model=model,
            bo_tool=bo_tool,
            maximize=maximize,
            beta=beta,
            gamma=gamma,
            acqf_weight_func_floor_value=acqf_weight_func_floor_value,
            acqf_weight_func_edge_offset=acqf_weight_func_edge_offset,
            acqf_weight_func_post_edge_gain=acqf_weight_func_post_edge_gain,
            acqf_weight_func_post_edge_offset=acqf_weight_func_post_edge_offset,
            acqf_weight_func_post_edge_width=acqf_weight_func_post_edge_width,
            acqf_weight_func_post_edge_decay_location=(
                acqf_weight_func_post_edge_decay_location
            ),
            add_posterior_stddev=add_posterior_stddev,
            estimate_posterior_mean_by_interpolation=estimate_posterior_mean_by_interpolation,
        )
        self.gradient_dims = gradient_dims
        self.phi = phi
        self.phi2 = phi2
        self.method = method
        self.order = order
        self.finite_difference_interval = finite_difference_interval
        self.subtract_background_gradient = subtract_background_gradient
        self.background_gradient = 0.0

        if method == "analytical" and order > 1:
            raise ValueError("Analytical gradients only support first-order derivatives.")
        if method not in {"analytical", "numerical"}:
            raise ValueError("`method` must be 'analytical' or 'numerical'.")
        if self.phi is None or self.phi2 is None:
            self.estimate_weights()

    def update_hyperparams_following_schedule(self) -> None:
        """Decay gradient-term weights after each update.

        This decays the first- and higher-order derivative weights as well as
        the weighting-mask mixing factor ``alpha``.
        """
        self.phi = self.phi * self.beta
        self.phi2 = self.phi2 * self.beta
        self.alpha = self.alpha * self.gamma

    def estimate_weights(self) -> None:
        """Auto-scale gradient terms relative to posterior uncertainty.

        The method evaluates the acquisition on a dense normalized grid and
        chooses derivative weights so gradient terms have scales comparable to
        posterior standard deviation.
        """
        self.phi = 1.0
        self.phi2 = 1.0
        x = torch.linspace(0, 1, 100).view(-1, 1, 1)
        original_add_posterior_stddev = self.add_posterior_stddev
        self.add_posterior_stddev = True
        self.forward(x)
        self.add_posterior_stddev = original_add_posterior_stddev

        sigma = self.intermediate_data["sigma"]
        gradients_all_orders = self.intermediate_data["gradients_all_orders"]
        self.phi = float(sigma.max() / gradients_all_orders[0].max() * 0.5)
        if len(gradients_all_orders) > 1:
            self.phi2 = float(sigma.max() / gradients_all_orders[1].max() * 0.5)
        logger.info(
            "Automatically determined gradient weights: phi=%s, phi2=%s.",
            self.phi,
            self.phi2,
        )

    @t_batch_mode_transform()
    def forward(
        self,
        x: Tensor,
        mu_x: Tensor | None = None,
        sigma_x: Tensor | None = None,
    ) -> Tensor:
        """Return the gradient-aware acquisition value.

        Parameters
        ----------
        x : Tensor
            Query locations with shape ``batch_shape x q x d``.
        mu_x : Tensor, optional
            Precomputed posterior means with shape ``batch_shape x q``.
        sigma_x : Tensor, optional
            Precomputed posterior standard deviations with shape
            ``batch_shape x q`` or ``batch_shape x q x 1``.

        Returns
        -------
        Tensor
            Acquisition values with shape ``batch_shape`` for ``q == 1``.
        """
        if mu_x is None or sigma_x is None:
            _, sigma = self._mean_and_sigma(x)
        else:
            sigma = sigma_x

        gradients_all_orders: list[Tensor | float] = [0.0] * self.order
        if self.method == "analytical":
            gradient = self.calculate_gradients_analytical(x)
            gradients_all_orders[0] = torch.linalg.norm(gradient, dim=-1)
        else:
            for derivative_order in range(1, self.order + 1):
                gradient = self.calculate_gradients_numerical(x, order=derivative_order)
                if derivative_order == 1 and self.subtract_background_gradient:
                    gradient = gradient - self.background_gradient
                gradients_all_orders[derivative_order - 1] = torch.linalg.norm(
                    gradient,
                    dim=-1,
                )

        sigma_term = sigma if self.add_posterior_stddev else 0.0
        acquisition_value = sigma_term + self.phi * gradients_all_orders[0]
        if len(gradients_all_orders) > 1:
            for gradient in gradients_all_orders[1:]:
                acquisition_value = acquisition_value + self.phi2 * gradient
        acquisition_value = self.apply_weight_func(x, acquisition_value)
        self.intermediate_data = {
            "sigma": sigma_term,
            "gradients_all_orders": gradients_all_orders,
        }
        return acquisition_value

    def calculate_gradients_analytical(self, x: Tensor) -> Tensor:
        """Calculate analytical posterior-mean gradients using autodiff.

        Parameters
        ----------
        x : Tensor
            Query locations with shape ``batch_shape x q x d``.

        Returns
        -------
        Tensor
            Gradient tensor with shape ``batch_shape x n_gradient_dims`` for
            the single-candidate case ``q == 1``.
        """
        def posterior_mean(values: Tensor) -> Tensor:
            return self.model.posterior(values).mean.squeeze()

        with torch.enable_grad():
            gradient = elementwise_derivative_analytical(posterior_mean, x)
        if self.gradient_dims is not None:
            gradient = torch.index_select(
                gradient,
                -1,
                torch.as_tensor(self.gradient_dims, device=gradient.device),
            )
        return gradient.squeeze(1)

    def calculate_gradients_numerical(self, x: Tensor, order: int = 1) -> Tensor:
        """Estimate gradients by finite differences in normalized x-space.

        Parameters
        ----------
        x : Tensor
            Query locations with shape ``batch_shape x q x d``.
        order : int, optional
            Derivative order.

        Returns
        -------
        Tensor
            Gradient tensor with shape ``batch_shape x n_gradient_dims`` for
            the single-candidate case ``q == 1``.
        """
        def posterior_mean(values: Tensor) -> Tensor:
            return self._mean_and_sigma(values)[0].reshape(-1, 1, 1)

        gradient = elementwise_derivative_finite_difference(
            posterior_mean,
            x,
            order=order,
            step_size=self.finite_difference_interval,
            dims=self.gradient_dims,
        )
        return gradient.squeeze(1)

    def set_background_gradient(self, background_gradient: float) -> None:
        """Set the background gradient used by numerical differentiation.

        Parameters
        ----------
        background_gradient : float
            Scalar gradient estimate to subtract from the first-derivative term
            when ``subtract_background_gradient`` is enabled.
        """
        self.background_gradient = background_gradient


class ComprehensiveAugmentedAcquisitionFunction(
    PosteriorStandardDeviationDerivedAcquisition
):
    """Combined XANES acquisition using uncertainty, gradient, and residue.

    This acquisition multiplies posterior uncertainty by a clipped sum of a
    gradient-based term and an optional reference-spectrum residue term.
    """

    def __init__(
        self,
        model: Model,
        bo_tool,
        maximize: bool = True,
        beta: float = 0.999,
        gamma: float = 0.95,
        acqf_weight_func_floor_value: float = 0.1,
        acqf_weight_func_edge_offset: float = -1.6,
        acqf_weight_func_post_edge_gain: float = 5.0,
        acqf_weight_func_post_edge_offset: float = 1.0,
        acqf_weight_func_post_edge_width: float = 0.5,
        acqf_weight_func_post_edge_decay_location: float = 50.0,
        gradient_dims: Optional[list[int]] = None,
        gradient_order: int = 1,
        differentiation_method: str = "analytical",
        reference_spectra_x: Optional[Tensor] = None,
        reference_spectra_y: Optional[Tensor] = None,
        phi_g: float = 0.1,
        phi_g2: float = 0.001,
        phi_r: float = 100.0,
        addon_term_lower_bound: float = 1e-2,
        add_posterior_stddev: bool = True,
        estimate_posterior_mean_by_interpolation: bool = False,
        subtract_background_gradient: bool = True,
    ) -> None:
        """Initialize the comprehensive XANES acquisition function.

        Parameters
        ----------
        model : Model
            Underlying GP model.
        bo_tool : BayesianOptimizationTool
            Optimization tool that owns the GP model, observed data, and data
            transforms.
        maximize : bool, optional
            Forwarded to the BoTorch parent class.
        beta : float, optional
            Decay factor for additive-term weights.
        gamma : float, optional
            Decay factor for the acquisition weighting mask amplitude.
        acqf_weight_func_floor_value : float, optional
            Minimum value applied to the acquisition weighting function.
        acqf_weight_func_edge_offset : float, optional
            Offset, in units of estimated edge width, applied to the edge
            location when constructing the weighting function.
        acqf_weight_func_post_edge_gain : float, optional
            Gain controlling the post-edge Gaussian boost.
        acqf_weight_func_post_edge_offset : float, optional
            Offset applied to the center of the post-edge Gaussian term.
        acqf_weight_func_post_edge_width : float, optional
            Width multiplier for the post-edge Gaussian term.
        acqf_weight_func_post_edge_decay_location : float, optional
            Decay location, in units of edge width, for the trailing sigmoid.
        gradient_dims : list[int], optional
            Feature dimensions used by the gradient term.
        gradient_order : int, optional
            Highest derivative order used by the gradient term.
        differentiation_method : str, optional
            Derivative method used by the gradient term.
        reference_spectra_x : Tensor, optional
            Reference-energy grid with shape ``(n_reference_points,)``.
        reference_spectra_y : Tensor, optional
            Reference spectra with shape
            ``(n_reference_spectra, n_reference_points)``.
        phi_g : float, optional
            Weight of the first-derivative term.
        phi_g2 : float, optional
            Weight of higher-order derivative terms.
        phi_r : float, optional
            Weight of the reference-spectrum residue term.
        addon_term_lower_bound : float, optional
            Lower clipping bound applied to the sum of additive terms before
            multiplying by posterior uncertainty.
        add_posterior_stddev : bool, optional
            Whether posterior standard deviation contributes directly.
        estimate_posterior_mean_by_interpolation : bool, optional
            Whether to replace the GP posterior mean with spline interpolation
            of observed data.
        subtract_background_gradient : bool, optional
            Whether to subtract a background gradient estimate from the first
            derivative term when using numerical gradients.
        """
        super().__init__(
            model=model,
            bo_tool=bo_tool,
            maximize=maximize,
            beta=beta,
            gamma=gamma,
            acqf_weight_func_floor_value=acqf_weight_func_floor_value,
            acqf_weight_func_edge_offset=acqf_weight_func_edge_offset,
            acqf_weight_func_post_edge_gain=acqf_weight_func_post_edge_gain,
            acqf_weight_func_post_edge_offset=acqf_weight_func_post_edge_offset,
            acqf_weight_func_post_edge_width=acqf_weight_func_post_edge_width,
            acqf_weight_func_post_edge_decay_location=(
                acqf_weight_func_post_edge_decay_location
            ),
            add_posterior_stddev=add_posterior_stddev,
            estimate_posterior_mean_by_interpolation=estimate_posterior_mean_by_interpolation,
        )
        self.acqf_g = GradientAwarePosteriorStandardDeviation(
            model=model,
            bo_tool=bo_tool,
            maximize=maximize,
            beta=beta,
            gamma=gamma,
            acqf_weight_func_floor_value=acqf_weight_func_floor_value,
            acqf_weight_func_edge_offset=acqf_weight_func_edge_offset,
            acqf_weight_func_post_edge_gain=acqf_weight_func_post_edge_gain,
            acqf_weight_func_post_edge_offset=acqf_weight_func_post_edge_offset,
            acqf_weight_func_post_edge_width=acqf_weight_func_post_edge_width,
            acqf_weight_func_post_edge_decay_location=(
                acqf_weight_func_post_edge_decay_location
            ),
            gradient_dims=gradient_dims,
            method=differentiation_method,
            order=gradient_order,
            phi=phi_g,
            phi2=phi_g2,
            add_posterior_stddev=False,
            estimate_posterior_mean_by_interpolation=estimate_posterior_mean_by_interpolation,
            subtract_background_gradient=subtract_background_gradient,
        )
        if reference_spectra_x is not None and reference_spectra_y is not None:
            self.acqf_r = FittingResiduePosteriorStandardDeviation(
                model=model,
                bo_tool=bo_tool,
                maximize=maximize,
                beta=beta,
                gamma=gamma,
                acqf_weight_func_floor_value=acqf_weight_func_floor_value,
                acqf_weight_func_edge_offset=acqf_weight_func_edge_offset,
                acqf_weight_func_post_edge_gain=acqf_weight_func_post_edge_gain,
                acqf_weight_func_post_edge_offset=acqf_weight_func_post_edge_offset,
                acqf_weight_func_post_edge_width=acqf_weight_func_post_edge_width,
                acqf_weight_func_post_edge_decay_location=(
                    acqf_weight_func_post_edge_decay_location
                ),
                reference_spectra_x=reference_spectra_x,
                reference_spectra_y=reference_spectra_y,
                phi=phi_r,
                add_posterior_stddev=False,
                estimate_posterior_mean_by_interpolation=estimate_posterior_mean_by_interpolation,
            )
            self.phi_r = self.acqf_r.phi
        else:
            logger.warning("No reference spectra provided; residue term disabled.")
            self.acqf_r = DummyAcquisition()
            self.phi_r = 0.0
        self.gradient_order = gradient_order
        self.phi_g = self.acqf_g.phi
        self.phi_g2 = self.acqf_g.phi2
        self.add_posterior_stddev = add_posterior_stddev
        self.addon_term_lower_bound = addon_term_lower_bound

    def set_weight_func(self, function: Callable) -> None:
        """Attach the same weighting mask to all internal acquisition terms.

        Parameters
        ----------
        function : Callable
            Weighting function in normalized x-space that is broadcast-compatible
            with all internal acquisition terms.
        """
        super().set_weight_func(function)
        self.acqf_g.set_weight_func(function)
        if hasattr(self.acqf_r, "set_weight_func"):
            self.acqf_r.set_weight_func(function)

    def set_background_gradient(self, background_gradient: float) -> None:
        """Propagate background gradient to the gradient-based term.

        Parameters
        ----------
        background_gradient : float
            Scalar background-gradient estimate in normalized x-space.
        """
        self.acqf_g.set_background_gradient(background_gradient)

    def update_hyperparams_following_schedule(self) -> None:
        """Decay all internal additive weights after an update.

        The schedule is forwarded to both the gradient and residue subterms,
        and the top-level weighting-mask mixing factor ``alpha`` is also
        decayed.
        """
        self.acqf_r.update_hyperparams_following_schedule()
        self.acqf_g.update_hyperparams_following_schedule()
        self.alpha = self.alpha * self.gamma

    @t_batch_mode_transform()
    def forward(self, x: Tensor) -> Tensor:
        """Return the combined XANES acquisition value.

        Parameters
        ----------
        x : Tensor
            Query locations with shape ``batch_shape x q x d``.

        Returns
        -------
        Tensor
            Acquisition values with shape ``batch_shape`` for ``q == 1``.
        """
        mu, sigma = self._mean_and_sigma(x)
        acquisition_value = sigma - sigma.min() if self.add_posterior_stddev else 1.0

        gradient_term = (
            self.acqf_g(x, mu_x=mu, sigma_x=sigma) if self.phi_g > 0 else torch.tensor(0.0, device=x.device)
        )
        residue_term = (
            self.acqf_r(x, mu_x=mu, sigma_x=sigma) if self.phi_r > 0 else torch.tensor(0.0, device=x.device)
        )
        acquisition_value = acquisition_value * torch.clip(
            gradient_term + residue_term,
            self.addon_term_lower_bound,
            None,
        )
        return self.apply_weight_func(x, acquisition_value)
