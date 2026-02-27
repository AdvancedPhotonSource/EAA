from typing import Annotated, Callable, Tuple, List, Type
import logging

import botorch.generation
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.acquisition import AcquisitionFunction
import gpytorch
import torch
from sciagent.tool.base import BaseTool, ToolReturnType, tool

from eaa.util import to_tensor

logger = logging.getLogger(__name__)


class BaseSequentialOptimizationTool(BaseTool):
    name: str = "base_sequential_optimization"

    def __init__(
        self,
        require_approval: bool = False,
        *args,
        **kwargs,
    ):
        """A base class for sequential optimization tools.
        
        Subclasses of this class should find the **maximizer** of the objective function.

        Parameters
        ----------
        require_approval : bool, optional
            Whether to require approval for the tool to be used.
        """
        self.xs_untransformed: torch.Tensor | None = None
        self.ys_untransformed: torch.Tensor | None = None
        if not isinstance(getattr(type(self), "xs_transformed", None), property):
            self.xs_transformed: torch.Tensor | None = None
        if not isinstance(getattr(type(self), "ys_transformed", None), property):
            self.ys_transformed: torch.Tensor | None = None
        super().__init__(*args, require_approval=require_approval, **kwargs)

    def build(self, *args, **kwargs) -> None:
        return None

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def suggest(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def visualize_status(self) -> matplotlib.figure.Figure:
        """Visualize observed optimization data.

        Returns
        -------
        matplotlib.figure.Figure
            Figure showing xs_untransformed versus ys_untransformed.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        if self.xs_untransformed is None or self.ys_untransformed is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig
        if self.xs_untransformed.numel() == 0 or self.ys_untransformed.numel() == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        xs = self.xs_untransformed.detach().cpu().numpy()
        ys = self.ys_untransformed.detach().cpu().numpy()
        if xs.ndim == 1:
            xs = xs[:, None]
        if ys.ndim == 1:
            ys = ys[:, None]

        if xs.shape[1] == 1:
            order = np.argsort(xs[:, 0])
            ax.plot(xs[order, 0], ys[order, 0], "o-", label="observations")
            ax.plot(
                xs[-1, 0],
                ys[-1, 0],
                marker="*",
                color="red",
                markersize=14,
                linestyle="None",
                label="latest",
            )
            ax.set_xlabel("x (untransformed)")
        else:
            idx = np.arange(xs.shape[0])
            ax.plot(idx, ys[:, 0], "o-", label="observations")
            ax.plot(
                idx[-1],
                ys[-1, 0],
                marker="*",
                color="red",
                markersize=14,
                linestyle="None",
                label="latest",
            )
            ax.set_xlabel("sample index")
            ax.set_title("Observed y vs sample index (x is multi-dimensional)")
        ax.set_ylabel("y (untransformed)")
        ax.grid(True)
        ax.legend()
        return fig


class BayesianOptimizationTool(BaseSequentialOptimizationTool):
    name: str = "bayesian_optimization"

    def __init__(
        self,
        bounds: Tuple[List[float], List[float]],
        acquisition_function_class: Type[
            AcquisitionFunction
        ] = botorch.acquisition.ExpectedImprovement,
        acquisition_function_kwargs: dict = None,
        model_class: Type[gpytorch.models.GP] = botorch.models.SingleTaskGP,
        model_kwargs: dict = None,
        optimization_function: Callable = botorch.optim.optimize_acqf,
        optimization_function_kwargs: dict = None,
        n_observations: int = 1,
        kernel_lengthscales: torch.Tensor = None,
        require_approval: bool = False,
        *args,
        **kwargs,
    ):
        """The Bayesian optimization tool that finds the maximizer of
        the objective function.

        Parameters
        ----------
        bounds : Tuple[List[float], List[float]]
            The bounds of each dimension of the search space. It should be
            a list of lists; the first list contains the lower bounds and the
            second list contains the upper bounds. The length of each sublist
            should be equal to the number of dimensions of the search space,
            or the number of parameters to be tuned.
        acquisition_function_class : Type[AcquisitionFunction]
            The class of the acquisition function.
        acquisition_function_kwargs : dict
            The keyword arguments for the acquisition function class.
        model_class : Type[gpytorch.models.GP]
            The class of the Gaussian process model. The simplest choice
            for single-task optimization problems is
            `botorch.models.SingleTaskGP`.
        model_kwargs : dict
            The keyword arguments for the model class.
        optimization_function : Callable
            The function to optimize the acquisition function. This function should be one
            of, or derived from, functions in `botorch.optim`. The following
            arguments are mandatory:

            - acquisition_function
            - bounds
            - q
            - num_restarts
            - raw_samples

            The rest of the arguments, if any, should be supplied using
            `optimization_function_kwargs`.
        optimization_function_kwargs : dict
            The keyword arguments for the optimization function.
        n_observations : int, optional
            The number of observations coming from the objective function.
            For single-task optimization problems, this should be 1.
        kernel_lengthscales : torch.Tensor, optional
            Kernel lengthscale in the un-transformed space. If given,
            the lengthscales of the kernel will be overriden with thse
            values.
        """
        self.bounds = to_tensor(bounds).float()
        self.acquisition_function = None
        self.acquisition_function_class = acquisition_function_class
        self.acquisition_function_kwargs = (
            acquisition_function_kwargs
            if acquisition_function_kwargs is not None
            else {}
        )
        self.model = None
        self.model_class = model_class
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.optimization_function = optimization_function
        self.optimization_function_kwargs = (
            optimization_function_kwargs
            if optimization_function_kwargs is not None
            else {}
        )
        self.n_dims_in = len(bounds[0])
        self.n_dims_out = n_observations
        self.kernel_lengthscales = (
            to_tensor(kernel_lengthscales) if kernel_lengthscales is not None else None
        )

        self.input_transform = None
        self.outcome_transform = None

        super().__init__(*args, build=False, require_approval=require_approval, **kwargs)

        # Untransformed data
        self.xs_untransformed = torch.tensor([])
        self.ys_untransformed = torch.tensor([])

        # Transformed data
        self.xs_transformed = torch.tensor([])
        self.ys_transformed = torch.tensor([])

    def check_x_data(self, data: torch.Tensor):
        if not (data.ndim == 2 and data.shape[1] == self.n_dims_in):
            raise ValueError(
                f"Expected input data of shape (n_samples, {self.n_dims_in}), "
                f"but got {data.shape}."
            )

    def check_y_data(self, data: torch.Tensor):
        if not (data.ndim == 2 and data.shape[1] == self.n_dims_out):
            raise ValueError(
                f"Expected output data of shape (n_samples, {self.n_dims_out}), "
                f"but got {data.shape}."
            )

    def get_random_initial_points(
        self, n_points: int
    ) -> Annotated[torch.Tensor, "initial_points"]:
        """Get a random set of initial points within the bounds. The observed data
        can be used to form the initial training data for the Gaussian process
        model.

        Parameters
        ----------
        n_points : int
            The number of initial points to observe.

        Returns
        -------
        torch.Tensor
            A (n_points, n_features) tensor giving the initial points to observe.
        """
        return (
            torch.rand(n_points, self.n_dims_in) * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )

    def build(self, acquisition_function_kwargs: dict = None) -> None:
        """Build the Gaussian process model and data transform modules. This function
        should be called after the initial data are collected and updated to the tool
        using the `update` method.
        """
        self.initialize_transforms()
        self.train_transforms_and_transform_data()
        self.initialize_model(self.xs_transformed, self.ys_transformed)
        self.fit_kernel_hyperparameters()
        self.build_acquisition_function(acquisition_function_kwargs)

    def initialize_model(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """Initialize the Gaussian process model with recorded data.

        Parameters
        ----------
        x_train : torch.Tensor
            A (n_samples, n_features) tensor giving the *transformed* input parameters.
        y_train : torch.Tensor
            A (n_samples, n_observations) tensor giving the *transformed* observations
            of the objective function.
        """
        self.model = self.model_class(x_train, y_train)

    def fit_kernel_hyperparameters(self, *args, **kwargs):
        """Fit the kernel hyperparameters of the Gaussian process model."""
        fitting_func = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        botorch.fit.fit_gpytorch_mll(fitting_func)

        if hasattr(self.model.covar_module, "lengthscale"):
            logger.info(
                "Fitted kernel hyperparameters (untransformed): {}".format(
                    self.unscale_by_normalizer_bounds(
                        self.model.covar_module.lengthscale
                    ).detach()
                )
            )
            if self.kernel_lengthscales is not None:
                self.model.covar_module.lengthscale = self.scale_by_normalizer_bounds(
                    self.kernel_lengthscales
                )
                logger.info(
                    "Overriden kernel hyperparameters (untransformed): {}".format(
                        self.unscale_by_normalizer_bounds(
                            self.model.covar_module.lengthscale
                        ).detach()
                    )
                )

    def build_acquisition_function(self, acquisition_function_kwargs: dict = None):
        """Build the acquisition function."""
        if acquisition_function_kwargs is not None:
            self.acquisition_function_kwargs.update(acquisition_function_kwargs)
        self.acquisition_function = self.acquisition_function_class(
            model=self.model,
            **self.acquisition_function_kwargs,
        )

    def initialize_transforms(self):
        """Build the input and outcome transforms."""
        self.input_transform = Normalize(d=self.n_dims_in, bounds=self.bounds)
        self.outcome_transform = Standardize(m=self.n_dims_out)

    def train_transforms_and_transform_data(self):
        """Train the outcome transforms and transform the recorded raw data.

        The bounds of the input transformed are set when the transform is instantiated
        and are not trained here.
        """
        self.xs_transformed, self.ys_transformed = self.transform_data(
            self.xs_untransformed, self.ys_untransformed, train_x=False, train_y=True
        )

    def transform_data(self, x=None, y=None, train_x=False, train_y=False):
        """Transform data into the normalized and/or standardized space.
        This function can also be used to find the transforms using given data
        if such transforms are not yet learned.

        Parameters
        ----------
        x : torch.Tensor, optional
            A (n_samples, n_features) tensor giving the input parameters.
        y : torch.Tensor, optional
            A (n_samples, n_observations) tensor giving the observations of the
            objective function.
        train_x : bool, optional
            If True, the input data are used to train the input transform.
        train_y : bool, optional
            If True, the output data are used to train the output transform.

        Returns
        -------
        x : torch.Tensor
            A (n_samples, n_features) tensor giving the transformed input parameters.
        y : torch.Tensor
            A (n_samples, n_observations) tensor giving the transformed observations
            of the objective function.
        """
        if x is not None:
            x = x.double()
        if y is not None:
            y = y.double()
        if x is not None and self.input_transform is not None:
            do_squeeze = False
            if x.ndim == 1:
                x = x[:, None]
                do_squeeze = True
            if train_x:
                self.input_transform.train()
            else:
                self.input_transform.eval()
            x = self.input_transform(x)
            if do_squeeze:
                x = x[:, 0]
        if y is not None and self.outcome_transform is not None:
            if train_y:
                self.outcome_transform.train()
            else:
                self.outcome_transform.eval()
            y, _ = self.outcome_transform(y)
        return x, y

    def untransform_data(self, x=None, y=None):
        """Un-transform data from the normalized and/or standardized space.

        Parameters
        ----------
        x : torch.Tensor, optional
            A (n_samples, n_features) tensor giving the transformed input parameters.
        y : torch.Tensor, optional
            A (n_samples, n_observations) tensor giving the transformed observations
            of the objective function.

        Returns
        -------
        x : torch.Tensor
            A (n_samples, n_features) tensor giving the un-transformed input parameters.
        y : torch.Tensor
            A (n_samples, n_observations) tensor giving the un-transformed observations
            of the objective function.
        """
        if x is not None and self.input_transform is not None:
            self.input_transform.training = False
            x = self.input_transform.untransform(x)
        if y is not None and self.outcome_transform is not None:
            self.outcome_transform.training = False
            y, _ = self.outcome_transform.untransform(y)
        return x, y

    def scale_by_normalizer_bounds(self, x, dim=0):
        """
        Scale data in x-space by 1 / span_of_normalizer_bounds.

        Parameters
        ----------
        x : Any
            The input data.
        dim : int, optional
            Use the `dim`-th dimension of the normalizer bounds to calculate the scaling factor.
            If x has a shape of [n, ..., d] where d equals to the number of dimensions of the bounds,
            the scaling factors are calculated separately for each dimension and the `dim` argument is
            disregarded.

        Returns
        -------
        Any
            The scaled data.
        """
        if isinstance(x, torch.Tensor) and x.ndim >= 2:
            return x / (self.input_transform.bounds[1] - self.input_transform.bounds[0])
        else:
            s = (
                self.input_transform.bounds[1][dim]
                - self.input_transform.bounds[0][dim]
            )
            return x / s

    def unscale_by_normalizer_bounds(self, x, dim=0):
        """
        Scale data in x-space by span_of_normalizer_bounds.

        Parameters
        ----------
        x : Any
            The input data.
        dim : int, optional
            Use the `dim`-th dimension of the normalizer bounds to calculate the scaling factor.
            If x has a shape of [n, ..., d] where d equals to the number of dimensions of the bounds,
            the scaling factors are calculated separately for each dimension and the `dim` argument is
            disregarded.

        Returns
        -------
        Any
            The scaled data.
        """
        if isinstance(x, torch.Tensor) and x.ndim >= 2:
            return x * (self.input_transform.bounds[1] - self.input_transform.bounds[0])
        else:
            s = (
                self.input_transform.bounds[1][dim]
                - self.input_transform.bounds[0][dim]
            )
            return x * s

    def scale_by_standardizer_scale(self, y, dim=0):
        """
        Scale data in y-space by the standardizer scale.

        Parameters
        ----------
        y : torch.Tensor
            The input data.
        dim : int, optional
            Use the `dim`-th dimension of the standardizer scale to calculate the scaling factor.
            If y has a shape of [n, ..., d] where d equals to the number of dimensions of the standardizer scale,
            the scaling factors are calculated separately for each dimension and the `dim` argument is
            disregarded.

        Returns
        -------
        torch.Tensor
            The scaled data.
        """
        return y / self.outcome_transform.stdvs[0][dim]

    def unscale_by_standardizer_scale(self, y, dim=0):
        """
        Un-scale data in y-space by the standardizer scale.

        Parameters
        ----------
        y : torch.Tensor
            The input data.
        dim : int, optional
            Use the `dim`-th dimension of the standardizer scale to calculate the scaling factor.
            If y has a shape of [n, ..., d] where d equals to the number of dimensions of the standardizer scale,
            the scaling factors are calculated separately for each dimension and the `dim` argument is
            disregarded.

        Returns
        -------
        torch.Tensor
            The un-scaled data.
        """
        return y * self.outcome_transform.stdvs[dim]

    @tool(name="update", return_type=ToolReturnType.NUMBER)
    def update(
        self,
        x: Annotated[torch.Tensor | np.ndarray, "The input parameters."],
        y: Annotated[torch.Tensor | np.ndarray, "The observations of the objective function."],
    ) -> None:
        """Update the Bayesian optimization tool with new observation.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            A tensor or numpy array of shape (n_samples, n_features) giving
            the *un-transformed* input parameters (raw values before normalization).
        y : torch.Tensor | np.ndarray
            A tensor or numpy array of shape (n_samples, n_observations) giving the
            *un-transformed* observations (raw values before standardization) 
            of the objective function. For multi-task optimization problems, 
            `n_observations` should be equal to the number of tasks.
        """
        x = to_tensor(x)
        y = to_tensor(y)

        self.check_x_data(x)
        self.check_y_data(y)

        self.xs_untransformed = torch.cat([self.xs_untransformed, x])
        self.ys_untransformed = torch.cat([self.ys_untransformed, y])

        if self.input_transform is not None and self.outcome_transform is not None:
            x, y = self.transform_data(x, y)
            self.xs_transformed = torch.cat([self.xs_transformed, x])
            self.ys_transformed = torch.cat([self.ys_transformed, y])
            if self.model is not None:
                self.model.condition_on_observations(x, y)
            else:
                logger.info(
                    "GP model is not updated because it is not built yet by calling "
                    "`build`."
                )
        else:
            logger.info(
                "GP model and variable buffers are not updated because normalization "
                "and standardization transforms are not built yet by calling `build`."
            )
            
        if (
            len(self.xs_untransformed) != len(self.xs_transformed) 
            or len(self.ys_untransformed) != len(self.ys_transformed)
        ):
            logger.debug(
                "The number of untransformed and transformed data are not equal. "
                "This is expected if normalization and standardization transforms "
                "are not built yet by calling `build`. However, if you have "
                "already done so, this is unexpected."
            )

    @tool(name="suggest", return_type=ToolReturnType.NUMBER)
    def suggest(
        self, 
        n_suggestions: Annotated[int, "The number of suggestions to make."] = 1
    ) -> Annotated[torch.Tensor, "suggested_points"]:
        """Suggest a new point to observe.

        Parameters
        ----------
        n_suggestions : int, optional
            The number of suggestions to make.

        Returns
        -------
        torch.Tensor
            A (n_samples, n_features) tensor giving the suggested points to observe.
            The values are in the untransformed space (raw observation before
            normalization and standardization).
        """
        if isinstance(
            self.acquisition_function, botorch.acquisition.AnalyticAcquisitionFunction
        ):
            if n_suggestions > 1:
                raise ValueError(
                    "Analytic acquisition functions only support a single suggestion."
                )

        candidates, _ = self.optimization_function(
            acq_function=self.acquisition_function,
            bounds=torch.stack(
                [
                    torch.zeros(self.n_dims_in, dtype=self.xs_transformed.dtype),
                    torch.ones(self.n_dims_in, dtype=self.xs_transformed.dtype),
                ]
            ),
            q=n_suggestions,
            num_restarts=20,
            raw_samples=50,
            **self.optimization_function_kwargs,
        )

        candidates = self.untransform_data(x=candidates)[0]
        # Safety net: numerical transforms/optimizer tolerances can produce tiny
        # out-of-bound values after untransform. Clamp explicitly to raw bounds.
        lower = self.bounds[0].to(dtype=candidates.dtype, device=candidates.device)
        upper = self.bounds[1].to(dtype=candidates.dtype, device=candidates.device)
        candidates = torch.max(torch.min(candidates, upper), lower)
        return candidates.detach()

    def visualize_status(self) -> matplotlib.figure.Figure:
        """Visualize observed data and GP posterior status.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with observations and posterior mean ±1 std.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        if self.xs_untransformed is None or self.ys_untransformed is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig
        if self.xs_untransformed.numel() == 0 or self.ys_untransformed.numel() == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        xs = self.xs_untransformed.detach().cpu().numpy()
        ys = self.ys_untransformed.detach().cpu().numpy()
        if xs.ndim == 1:
            xs = xs[:, None]
        if ys.ndim == 1:
            ys = ys[:, None]

        if self.model is None:
            if xs.shape[1] == 1:
                order = np.argsort(xs[:, 0])
                ax.plot(xs[order, 0], ys[order, 0], "o-", label="observations")
                ax.plot(
                    xs[-1, 0],
                    ys[-1, 0],
                    marker="*",
                    color="red",
                    markersize=14,
                    linestyle="None",
                    label="latest",
                )
                ax.set_xlabel("x (untransformed)")
            else:
                idx = np.arange(xs.shape[0])
                ax.plot(idx, ys[:, 0], "o-", label="observations")
                ax.plot(
                    idx[-1],
                    ys[-1, 0],
                    marker="*",
                    color="red",
                    markersize=14,
                    linestyle="None",
                    label="latest",
                )
                ax.set_xlabel("sample index")
                ax.set_title("Observed y vs sample index (x is multi-dimensional)")
            ax.set_ylabel("y (untransformed)")
            ax.grid(True)
            ax.legend()
            return fig

        with torch.no_grad():
            x_query_untransformed = self.xs_untransformed
            if self.input_transform is not None:
                x_query_transformed = self.transform_data(
                    x=x_query_untransformed, y=None, train_x=False, train_y=False
                )[0]
            else:
                x_query_transformed = x_query_untransformed
            posterior = self.model.posterior(x_query_transformed)
            mean_t = posterior.mean.squeeze(-1)
            std_t = posterior.variance.clamp_min(0).sqrt().squeeze(-1)
            _, mean_u = self.untransform_data(x=None, y=mean_t[:, None])
            if self.outcome_transform is not None:
                std_u = self.unscale_by_standardizer_scale(std_t)
            else:
                std_u = std_t

        mean_np = mean_u.detach().cpu().numpy().reshape(-1)
        std_np = std_u.detach().cpu().numpy().reshape(-1)

        if xs.shape[1] == 1:
            x_plot = xs[:, 0]
            order = np.argsort(x_plot)
            x_sorted = x_plot[order]
            y_sorted = ys[:, 0][order]
            mean_sorted = mean_np[order]
            std_sorted = std_np[order]
            ax.plot(x_sorted, y_sorted, "o", label="observations")
            ax.plot(x_sorted, mean_sorted, "-", label="posterior mean")
            ax.fill_between(
                x_sorted,
                mean_sorted - std_sorted,
                mean_sorted + std_sorted,
                alpha=0.25,
                label="posterior ±1σ",
            )
            ax.plot(
                xs[-1, 0],
                ys[-1, 0],
                marker="*",
                color="red",
                markersize=14,
                linestyle="None",
                label="latest",
            )
            ax.set_xlabel("x (untransformed)")
        else:
            idx = np.arange(xs.shape[0])
            ax.plot(idx, ys[:, 0], "o", label="observations")
            ax.plot(idx, mean_np, "-", label="posterior mean")
            ax.fill_between(
                idx,
                mean_np - std_np,
                mean_np + std_np,
                alpha=0.25,
                label="posterior ±1σ",
            )
            ax.plot(
                idx[-1],
                ys[-1, 0],
                marker="*",
                color="red",
                markersize=14,
                linestyle="None",
                label="latest",
            )
            ax.set_xlabel("sample index")
            ax.set_title("Posterior shown at observed points (x is multi-dimensional)")
        ax.set_ylabel("y (untransformed)")
        ax.grid(True)
        ax.legend()
        return fig


class QuadraticOptimizationTool(BaseSequentialOptimizationTool):
    name: str = "quadratic_optimization"

    def __init__(
        self,
        negative_definite_tolerance: float = 1e-10,
        non_concave_step_limit: float = 1.0,
        require_approval: bool = False,
        *args,
        **kwargs,
    ):
        """A lightweight optimizer that fits a quadratic surface to observations.

        Unlike the Bayesian optimizer, this tool works purely with closed-form
        quadratic regression and returns the maximizer of the fitted surface.

        Parameters
        ----------
        negative_definite_tolerance : float
            Tolerance for declaring the quadratic concave.
        non_concave_step_limit : float
            Step size to take along the steepest ascent direction when the
            quadratic fit is not concave.
        """
        self.n_dims_in: int | None = None
        self.n_observations: int = 1
        self.negative_definite_tolerance = negative_definite_tolerance
        self.non_concave_step_limit = non_concave_step_limit
        super().__init__(*args, build=False, require_approval=require_approval, **kwargs)
        
    @property
    def xs_transformed(self) -> torch.Tensor:
        return self.xs_untransformed

    @property
    def ys_transformed(self) -> torch.Tensor:
        return self.ys_untransformed

    def validate_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(
                f"Expected input data of shape (n_samples, n_features), but got {tuple(x.shape)}."
            )
        if self.n_dims_in is None:
            self.n_dims_in = x.shape[1]
        elif x.shape[1] != self.n_dims_in:
            raise ValueError(
                f"Expected input dimensionality {self.n_dims_in}, but received {x.shape[1]}."
            )
        return x

    def validate_y(self, y: torch.Tensor, n_samples: int) -> torch.Tensor:
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim != 2:
            raise ValueError(
                f"Expected output data of shape (n_samples, 1), but got {tuple(y.shape)}."
            )
        if y.shape[1] != self.n_observations:
            raise ValueError(
                f"Quadratic optimization only supports a single objective value but received {y.shape[1]}."
            )
        if y.shape[0] != n_samples:
            raise ValueError(
                f"Mismatched number of samples between x ({n_samples}) and y ({y.shape[0]})."
            )
        return y

    def build_design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Construct the design matrix for quadratic regression."""
        n_samples, n_features = x.shape
        features = [torch.ones((n_samples, 1), dtype=x.dtype, device=x.device), x]
        quadratic_terms = []
        for i in range(n_features):
            for j in range(i, n_features):
                term = (x[:, i] * x[:, j]).unsqueeze(-1)
                quadratic_terms.append(term)
        if quadratic_terms:
            features.append(torch.cat(quadratic_terms, dim=1))
        return torch.cat(features, dim=1)

    def required_num_points(self) -> int:
        n_quad_terms = self.n_dims_in * (self.n_dims_in + 1) // 2
        return 1 + self.n_dims_in + n_quad_terms

    def fit_quadratic(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit the quadratic surface y = x^T Q x + r^T x + c."""
        design = self.build_design_matrix(self.xs_untransformed)
        y = self.ys_untransformed
        try:
            solution = torch.linalg.lstsq(design, y).solution
        except RuntimeError:
            solution = torch.linalg.pinv(design) @ y
        solution = solution.squeeze(-1)

        idx = 0
        constant = solution[idx]
        idx += 1
        linear = solution[idx : idx + self.n_dims_in]
        idx += self.n_dims_in
        quad_coeffs = solution[idx:]

        Q = torch.zeros(
            (self.n_dims_in, self.n_dims_in),
            dtype=solution.dtype,
            device=solution.device,
        )
        q_idx = 0
        for i in range(self.n_dims_in):
            for j in range(i, self.n_dims_in):
                coeff = quad_coeffs[q_idx]
                q_idx += 1
                if i == j:
                    Q[i, j] = coeff
                else:
                    half_coeff = coeff / 2
                    Q[i, j] = half_coeff
                    Q[j, i] = half_coeff
        return Q, linear, constant

    def evaluate_quadratic(
        self,
        x: torch.Tensor | np.ndarray,
        Q: torch.Tensor | np.ndarray,
        linear: torch.Tensor | np.ndarray,
        constant: float | torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Evaluate y = x^T Q x + linear^T x + constant for each row in x.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            A tensor or numpy array of shape (n_samples, n_features).
        Q : torch.Tensor | np.ndarray
            A square tensor or numpy array of shape (n_features, n_features).
        linear : torch.Tensor | np.ndarray
            A tensor or numpy array of shape (n_features,).
        constant : float | torch.Tensor | np.ndarray
            A scalar constant term.

        Returns
        -------
        torch.Tensor
            A tensor of shape (n_samples, 1) containing quadratic values.
        """
        x_tensor = to_tensor(x).float()
        Q_tensor = to_tensor(Q).float()
        linear_tensor = to_tensor(linear).float()
        constant_tensor = to_tensor(constant).float()

        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)
        if x_tensor.ndim != 2:
            raise ValueError(
                f"Expected x of shape (n_samples, n_features), but got {tuple(x_tensor.shape)}."
            )
        if Q_tensor.ndim != 2 or Q_tensor.shape[0] != Q_tensor.shape[1]:
            raise ValueError(
                f"Expected Q to be square, but got shape {tuple(Q_tensor.shape)}."
            )
        if x_tensor.shape[1] != Q_tensor.shape[0]:
            raise ValueError(
                f"Expected x features to match Q dimensions, but got {x_tensor.shape[1]} and {Q_tensor.shape[0]}."
            )
        if linear_tensor.ndim != 1 or linear_tensor.shape[0] != x_tensor.shape[1]:
            raise ValueError(
                f"Expected linear to have shape ({x_tensor.shape[1]},), but got {tuple(linear_tensor.shape)}."
            )

        quadratic = torch.einsum("bi,ij,bj->b", x_tensor, Q_tensor, x_tensor)
        return (quadratic + x_tensor @ linear_tensor + constant_tensor)[:, None]

    @tool(name="update", return_type=ToolReturnType.NUMBER)
    def update(
        self,
        x: Annotated[torch.Tensor | np.ndarray, "The input parameters."],
        y: Annotated[
            torch.Tensor | np.ndarray, "The observations of the objective function."
        ],
    ) -> None:
        """Update the quadratic optimizer with new observations.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            A tensor or numpy array of shape (n_samples, n_features) giving the
            *un-transformed* input parameters.
        y : torch.Tensor | np.ndarray
            A tensor or numpy array of shape (n_samples, 1) giving the
            *un-transformed* objective values.
        """
        x_tensor = to_tensor(x).float()
        y_tensor = to_tensor(y).float()
        x_tensor = self.validate_x(x_tensor)
        y_tensor = self.validate_y(y_tensor, x_tensor.shape[0])

        if self.xs_untransformed is None:
            self.xs_untransformed = x_tensor
            self.ys_untransformed = y_tensor
        else:
            self.xs_untransformed = torch.cat([self.xs_untransformed, x_tensor], dim=0)
            self.ys_untransformed = torch.cat([self.ys_untransformed, y_tensor], dim=0)

    @tool(name="suggest", return_type=ToolReturnType.NUMBER)
    def suggest(
        self,
        n_suggestions: Annotated[int, "The number of suggestions to return."] = 1,
    ) -> Annotated[torch.Tensor, "suggested_points"]:
        """Return the maximizer of the fitted quadratic surface.

        Returns
        -------
        torch.Tensor
            A (1, n_features) tensor giving the suggested point.
        """
        if n_suggestions != 1:
            raise ValueError("Quadratic optimization supports only one suggestion.")

        if self.n_dims_in is None or self.xs_untransformed is None:
            raise ValueError("No observations available. Please call `update` first.")

        if self.xs_untransformed.shape[0] < self.required_num_points():
            raise ValueError(
                f"Quadratic fitting requires at least {self.required_num_points()} "
                f"observations, but only {self.xs_untransformed.shape[0]} were given."
            )

        Q, linear, _ = self.fit_quadratic()
        symmetric_Q = 0.5 * (Q + Q.transpose(0, 1))
        eigenvalues = torch.linalg.eigvalsh(symmetric_Q)
        
        # When the quadratic is not concave, take a step along the steepest ascent direction.
        if torch.max(eigenvalues) >= -self.negative_definite_tolerance:
            x_current = self.xs_untransformed[-1]
            gradient = (symmetric_Q + symmetric_Q.transpose(0, 1)) @ x_current + linear
            grad_norm = torch.linalg.norm(gradient)
            if grad_norm == 0:
                return x_current.unsqueeze(0).detach()
            step = self.non_concave_step_limit * gradient / grad_norm
            return (x_current + step).unsqueeze(0).detach()

        try:
            maximizer = -0.5 * torch.linalg.solve(symmetric_Q, linear)
        except RuntimeError as exc:
            raise ValueError(
                "Unable to solve for the maximizer because the quadratic is singular."
            ) from exc

        maximizer = maximizer.unsqueeze(0)
        return maximizer.detach()
