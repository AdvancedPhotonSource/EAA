from typing import Annotated

import numpy as np
from sciagent.tool.base import BaseTool, ToolReturnType, tool


class MultivariateLinearRegression(BaseTool):
    """A linear regression tool for multivariate input/output prediction."""

    name: str = "multivariate_linear_regression"

    def __init__(
        self,
        require_approval: bool = False,
        *args,
        **kwargs,
    ):
        self.xs: np.ndarray | None = None
        self.ys: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.n_feat_in: int | None = None
        self.n_feat_out: int | None = None
        super().__init__(*args, require_approval=require_approval, **kwargs)

    @staticmethod
    def to_2d_array(data, name: str) -> np.ndarray:
        array = np.asarray(data, dtype=float)
        if array.ndim != 2:
            raise ValueError(
                f"`{name}` must be a 2D array with shape (n_samples, n_features), "
                f"but got shape {array.shape}."
            )
        return array

    @tool(name="update", return_type=ToolReturnType.TEXT)
    def update(
        self,
        x: Annotated[list[list[float]], "Input data with shape (n_samples, n_feat_in)."],
        y: Annotated[list[list[float]], "Target data with shape (n_samples, n_feat_out)."],
    ) -> str:
        """Update the model with new training data and refit coefficients.

        Parameters
        ----------
        x : list[list[float]]
            Input data with shape (n_samples, n_feat_in).
        y : list[list[float]]
            Target data with shape (n_samples, n_feat_out).
        """
        x_arr = self.to_2d_array(x, name="x")
        y_arr = self.to_2d_array(y, name="y")

        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                "The number of samples in `x` and `y` must match, "
                f"but got {x_arr.shape[0]} and {y_arr.shape[0]}."
            )

        if self.n_feat_in is None:
            self.n_feat_in = x_arr.shape[1]
        elif x_arr.shape[1] != self.n_feat_in:
            raise ValueError(
                f"`x` must have {self.n_feat_in} features, but got {x_arr.shape[1]}."
            )

        if self.n_feat_out is None:
            self.n_feat_out = y_arr.shape[1]
        elif y_arr.shape[1] != self.n_feat_out:
            raise ValueError(
                f"`y` must have {self.n_feat_out} features, but got {y_arr.shape[1]}."
            )

        if self.xs is None:
            self.xs = x_arr
            self.ys = y_arr
        else:
            self.xs = np.vstack([self.xs, x_arr])
            self.ys = np.vstack([self.ys, y_arr])

        x_aug = np.hstack(
            [self.xs, np.ones((self.xs.shape[0], 1), dtype=self.xs.dtype)]
        )
        beta, *_ = np.linalg.lstsq(x_aug, self.ys, rcond=None)
        self.w = beta[:-1, :]
        self.b = beta[-1, :]

        return (
            f"Model updated with {self.xs.shape[0]} total samples. "
            f"Input features: {self.n_feat_in}, output features: {self.n_feat_out}."
        )

    @tool(name="predict", return_type=ToolReturnType.LIST)
    def predict(
        self,
        x: Annotated[list[list[float]], "Input data with shape (n_samples, n_feat_in)."],
    ) -> list[list[float]]:
        """Predict outputs for new inputs using y = xW + b.

        Parameters
        ----------
        x : list[list[float]]
            Input data with shape (n_samples, n_feat_in).

        Returns
        -------
        list[list[float]]
            Predicted outputs with shape (n_samples, n_feat_out).
        """
        if self.w is None or self.b is None:
            raise ValueError("Model is not fitted. Call `update` with training data first.")

        x_arr = self.to_2d_array(x, name="x")
        if x_arr.shape[1] != self.n_feat_in:
            raise ValueError(
                f"`x` must have {self.n_feat_in} features, but got {x_arr.shape[1]}."
            )

        y_pred = x_arr @ self.w + self.b
        return y_pred.tolist()
