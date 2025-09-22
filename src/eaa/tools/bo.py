from typing import Annotated, Callable, Tuple, List, Type, Dict, Any
import logging

import botorch.generation
import numpy as np
import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.acquisition import AcquisitionFunction
import gpytorch
import torch

from eaa.tools.base import BaseTool, ToolReturnType
from eaa.util import to_tensor

logger = logging.getLogger(__name__)


class BayesianOptimizationTool(BaseTool):
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
        """The Bayesian optimization tool.

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

        # Untransformed data
        self.xs_untransformed = torch.tensor([])
        self.ys_untransformed = torch.tensor([])

        # Transformed data
        self.xs_transformed = torch.tensor([])
        self.ys_transformed = torch.tensor([])

        self.input_transform = None
        self.outcome_transform = None

        super().__init__(*args, build=False, require_approval=require_approval, **kwargs)
        
        self.exposed_tools: List[Dict[str, Any]] = [
            {
                "name": "update",
                "function": self.update,
                "return_type": ToolReturnType.NUMBER
            },
            {
                "name": "suggest",
                "function": self.suggest,
                "return_type": ToolReturnType.NUMBER
            }
        ]

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

    def build(self) -> None:
        """Build the Gaussian process model and data transform modules. This function
        should be called after the initial data are collected and updated to the tool
        using the `update` method.
        """
        self.initialize_transforms()
        self.train_transforms_and_transform_data()
        self.initialize_model(self.xs_transformed, self.ys_transformed)
        self.fit_kernel_hyperparameters()
        self.build_acquisition_function()

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

    def build_acquisition_function(self):
        """Build the acquisition function."""
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
        return candidates.detach()
