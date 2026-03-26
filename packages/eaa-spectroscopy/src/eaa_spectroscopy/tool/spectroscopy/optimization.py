import logging
import warnings
from typing import Optional

import botorch
import numpy as np
import torch

from eaa_core.tool.base import tool
from eaa_core.tool.optimization import BayesianOptimizationTool
from eaa_spectroscopy.util import estimate_edge_location_and_width, sort_unique_xy

logger = logging.getLogger(__name__)


class AdaptiveXANESBayesianOptimization(BayesianOptimizationTool):
    """Bayesian optimization tool with built-in XANES acquisition weighting.

    The tool owns the schedule for constructing the XANES acquisition
    weighting function so it can be used directly from custom control loops
    without requiring a task manager.
    """

    name: str = "adaptive_xanes_bayesian_optimization"

    def __init__(
        self,
        n_updates_create_acqf_weight_func: Optional[int] = None,
        n_discrete_choices: int = 1000,
        duplicate_distance_threshold: Optional[float] = None,
        stopping_uncertainty_threshold: Optional[float] = None,
        stopping_n_updates_to_begin: int = 10,
        stopping_check_interval: int = 5,
        n_max_measurements: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the adaptive XANES Bayesian-optimization tool.

        Parameters
        ----------
        n_updates_create_acqf_weight_func : int, optional
            Number of post-build adaptive updates to complete before enabling
            XANES acquisition weighting. If ``None``, the weighting function is
            never created automatically.
        n_discrete_choices : int, optional
            Number of normalized candidate points used by the discrete
            acquisition optimizer.
        duplicate_distance_threshold : float, optional
            Threshold in normalized x-space below which a candidate is treated
            as a duplicate of an already sampled point. If omitted, a default
            threshold of ``1e-4`` times the normalized span is used.
        stopping_uncertainty_threshold : float, optional
            Threshold on the maximum weighted posterior standard deviation used
            by :meth:`should_stop`. If ``None``, the uncertainty stopping rule
            is disabled.
        stopping_n_updates_to_begin : int, optional
            Minimum number of adaptive updates before evaluating the
            uncertainty-based stopping rule.
        stopping_check_interval : int, optional
            Evaluate the uncertainty-based stopping rule every
            ``stopping_check_interval`` adaptive updates after
            ``stopping_n_updates_to_begin``.
        n_max_measurements : int, optional
            Hard cap on the total number of measurements considered by
            :meth:`should_stop`.
        *args
            Positional arguments forwarded to
            :class:`BayesianOptimizationTool`.
        **kwargs
            Keyword arguments forwarded to
            :class:`BayesianOptimizationTool`.
        """
        self.n_updates_create_acqf_weight_func = n_updates_create_acqf_weight_func
        self.n_discrete_choices = n_discrete_choices
        self.duplicate_distance_threshold = duplicate_distance_threshold
        self.stopping_uncertainty_threshold = stopping_uncertainty_threshold
        self.stopping_n_updates_to_begin = stopping_n_updates_to_begin
        self.stopping_check_interval = stopping_check_interval
        self.n_max_measurements = n_max_measurements
        self.n_adaptive_update_calls = 0
        self.stop_reason: str | None = None
        super().__init__(*args, **kwargs)

    def build(self, acquisition_function_kwargs: dict = None) -> None:
        """Build the BO model and initialize XANES acquisition state."""
        self.n_adaptive_update_calls = 0
        self.stop_reason = None
        super().build(acquisition_function_kwargs=acquisition_function_kwargs)
        self.configure_xanes_acquisition()

    def get_num_adaptive_updates_completed(self) -> int:
        """Return the number of adaptive updates completed after build."""
        return self.n_adaptive_update_calls

    def get_duplicate_distance_threshold(self) -> torch.Tensor:
        """Return the normalized duplicate-distance threshold.

        Returns
        -------
        torch.Tensor
            Per-dimension duplicate threshold in normalized x-space.
        """
        if self.xs_transformed.numel() == 0:
            dtype = self.bounds.dtype
            device = self.bounds.device
        else:
            dtype = self.xs_transformed.dtype
            device = self.xs_transformed.device
        if self.duplicate_distance_threshold is not None:
            return torch.full(
                (self.n_dims_in,),
                float(self.duplicate_distance_threshold),
                dtype=dtype,
                device=device,
            )
        return torch.full((self.n_dims_in,), 1e-4, dtype=dtype, device=device)

    def filter_duplicate_choices(self, choices: torch.Tensor) -> torch.Tensor:
        """Filter discrete candidates that are too close to sampled points.

        Parameters
        ----------
        choices : torch.Tensor
            Candidate points in normalized x-space with shape
            ``(n_choices, n_dims)``.

        Returns
        -------
        torch.Tensor
            Filtered candidate tensor in normalized x-space.
        """
        if self.xs_transformed is None or self.xs_transformed.numel() == 0:
            return choices

        measured_points = self.xs_transformed.to(
            dtype=choices.dtype,
            device=choices.device,
        )
        duplicate_distance_threshold = self.get_duplicate_distance_threshold().to(
            dtype=choices.dtype,
            device=choices.device,
        )
        diff_mat = torch.abs(choices[:, None, :] - measured_points[None, :, :])
        duplicate_mask = (diff_mat < duplicate_distance_threshold).all(dim=-1).any(dim=-1)
        filtered_choices = choices[~duplicate_mask]
        if filtered_choices.shape[0] == 0:
            warnings.warn(
                "All discrete XANES candidates were filtered as duplicates; "
                "falling back to the full candidate grid."
            )
            return choices
        return filtered_choices

    def estimate_edge_location_and_width(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        input_is_transformed: bool = True,
        run_in_transformed_space: bool = True,
        return_normalized_values: bool = True,
    ) -> tuple[float, float]:
        """Estimate XANES edge location and width from observed data.

        Parameters
        ----------
        x : torch.Tensor
            Observed energies with shape ``(n_samples, 1)``.
        y : torch.Tensor
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
        acquisition_function = self.acquisition_function
        if hasattr(acquisition_function, "estimate_edge_location_and_width"):
            return acquisition_function.estimate_edge_location_and_width(
                x=x,
                y=y,
                input_is_transformed=input_is_transformed,
                run_in_transformed_space=run_in_transformed_space,
                return_normalized_values=return_normalized_values,
            )

        if not input_is_transformed and run_in_transformed_space:
            x, y = self.transform_data(
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

    def estimate_background_gradient(self) -> float | None:
        """Estimate the pre-edge background gradient in normalized x-space.

        Returns
        -------
        float | None
            Estimated scalar background gradient, or ``None`` when a valid
            pre-edge interval cannot be formed.
        """
        edge_location, edge_width = self.estimate_edge_location_and_width(
            self.xs_transformed,
            self.ys_transformed,
            input_is_transformed=True,
            run_in_transformed_space=True,
            return_normalized_values=True,
        )
        x0 = 0.0
        x1 = edge_location - edge_width * 2.0
        if x1 <= x0:
            return None
        query = torch.tensor(
            [[x0], [x1]],
            dtype=self.xs_transformed.dtype,
            device=self.xs_transformed.device,
        )
        mu, _ = self.get_posterior_mean_and_std(
            query,
            transform=False,
            untransform=True,
            compute_sigma=False,
        )
        y0, y1 = mu.squeeze()
        gradient = ((y1 - y0) / (x1 - x0)).detach()
        return float(gradient)

    def configure_xanes_acquisition(self) -> None:
        """Refresh XANES-specific acquisition state from the current model."""
        acquisition_function = self.acquisition_function
        if acquisition_function is None:
            return

        if hasattr(acquisition_function, "set_background_gradient"):
            background_gradient = self.estimate_background_gradient()
            if background_gradient is not None:
                acquisition_function.set_background_gradient(background_gradient)

        if (
            self.n_updates_create_acqf_weight_func is None
            or self.n_adaptive_update_calls < self.n_updates_create_acqf_weight_func
        ):
            return
        if not hasattr(acquisition_function, "build_acquisition_weight_function"):
            return
        if getattr(acquisition_function, "weight_func", None) is not None:
            return

        logger.info(
            "Building XANES acquisition weight function with floor value %s.",
            acquisition_function.acqf_weight_func_floor_value,
        )
        acquisition_function.build_acquisition_weight_function()

    def should_stop(self) -> bool:
        """Return whether the tool-level XANES stopping criterion has triggered.

        Returns
        -------
        bool
            ``True`` when either the configured maximum-measurement cap or the
            weighted-uncertainty threshold has been reached.
        """
        self.stop_reason = None
        n_measurements = int(self.xs_untransformed.shape[0])
        if self.n_max_measurements is not None and n_measurements >= self.n_max_measurements:
            self.stop_reason = "max_measurements_reached"
            return True

        if self.stopping_uncertainty_threshold is None:
            return False
        if self.model is None or self.acquisition_function is None:
            return False

        n_updates = self.get_num_adaptive_updates_completed()
        if n_updates < self.stopping_n_updates_to_begin:
            return False
        if (
            n_updates - self.stopping_n_updates_to_begin
        ) % self.stopping_check_interval != 0:
            return False

        n_query = max(n_measurements * 5, 32)
        x_query = torch.linspace(
            0.0,
            1.0,
            n_query,
            dtype=self.xs_transformed.dtype,
            device=self.xs_transformed.device,
        ).view(-1, 1)
        _, sigma = self.get_posterior_mean_and_std(
            x_query,
            transform=False,
            untransform=True,
        )
        sigma = sigma.squeeze()
        weight_func = getattr(self.acquisition_function, "weight_func", None)
        if weight_func is not None:
            weights = torch.clip(weight_func(x_query).squeeze(), 0, 1)
        else:
            weights = 1.0

        if torch.max(sigma * weights) < self.stopping_uncertainty_threshold:
            self.stop_reason = "weighted_max_uncertainty_below_threshold"
            return True
        return False

    @tool(name="update")
    def update(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
    ) -> None:
        """Update the BO tool and refresh XANES acquisition weighting when due.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            New untransformed input locations with shape
            ``(n_samples, n_features)``.
        y : torch.Tensor | np.ndarray
            New untransformed observations with shape
            ``(n_samples, n_observations)``.
        """
        had_built_model = (
            self.model is not None
            and self.acquisition_function is not None
            and self.input_transform is not None
            and self.outcome_transform is not None
        )
        super().update(x, y)

        if not had_built_model:
            return

        self.n_adaptive_update_calls += 1
        self.configure_xanes_acquisition()

    @tool(name="suggest")
    def suggest(
        self,
        n_suggestions: int = 1,
    ) -> torch.Tensor:
        """Suggest new XANES points using discrete acquisition maximization.

        Parameters
        ----------
        n_suggestions : int, optional
            Number of candidates to suggest.

        Returns
        -------
        torch.Tensor
            Suggested points in untransformed energy space.
        """
        if self.acquisition_function is None:
            raise RuntimeError("Acquisition function is not built. Call `build()` first.")
        if self.n_dims_in != 1:
            raise NotImplementedError(
                "AdaptiveXANESBayesianOptimization currently supports discrete "
                "optimization only for one-dimensional XANES search spaces."
            )

        if self.xs_transformed is None or self.xs_transformed.numel() == 0:
            dtype = self.bounds.dtype
            device = self.bounds.device
        else:
            dtype = self.xs_transformed.dtype
            device = self.xs_transformed.device
        choices = torch.linspace(
            0.0,
            1.0,
            self.n_discrete_choices,
            dtype=dtype,
            device=device,
        ).view(-1, 1)
        choices = self.filter_duplicate_choices(choices)
        candidates, _ = botorch.optim.optimize_acqf_discrete(
            acq_function=self.acquisition_function,
            q=n_suggestions,
            choices=choices,
            unique=True,
        )
        candidates = self.untransform_data(x=candidates)[0]
        lower = self.bounds[0].to(dtype=candidates.dtype, device=candidates.device)
        upper = self.bounds[1].to(dtype=candidates.dtype, device=candidates.device)
        candidates = torch.max(torch.min(candidates, upper), lower)
        return candidates.detach()
