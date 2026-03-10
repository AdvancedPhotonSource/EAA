from typing import Annotated

import numpy as np
import matplotlib.pyplot as plt
from eaa.core.tooling.base import BaseTool, ToolReturnType, tool


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

    def get_n_parameter_drift_points_collected(self) -> int:
        if self.xs is None:
            return 0
        return int(self.xs.shape[0])

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

    def visualize_status(self):
        """Visualize observed data and fitted linear model for supported shapes.

        Supported cases:
        - 1D input -> 1D output: scatter + fitted line
        - 2D input -> 1D output: 3D scatter + fitted plane

        Returns
        -------
        matplotlib.figure.Figure | None
            Figure for supported shapes, otherwise None.
        """
        if (
            self.xs is None
            or self.ys is None
            or self.w is None
            or self.b is None
            or self.n_feat_in is None
            or self.n_feat_out is None
        ):
            return None

        if self.n_feat_in == 1 and self.n_feat_out == 1:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            x_values = self.xs[:, 0]
            y_values = self.ys[:, 0]
            ax.scatter(x_values, y_values, label="data")

            x_line = np.linspace(x_values.min(), x_values.max(), 200)[:, None]
            y_line = (x_line @ self.w + self.b)[:, 0]
            ax.plot(x_line[:, 0], y_line, color="tab:orange", label="fitted line")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
            ax.legend()
            return fig

        if self.n_feat_in == 2 and self.n_feat_out == 1:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")

            x1 = self.xs[:, 0]
            x2 = self.xs[:, 1]
            y_values = self.ys[:, 0]
            ax.scatter(x1, x2, y_values, color="tab:blue", label="data")

            x1_grid = np.linspace(x1.min(), x1.max(), 30)
            x2_grid = np.linspace(x2.min(), x2.max(), 30)
            x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
            x_mesh = np.column_stack([x1_mesh.ravel(), x2_mesh.ravel()])
            y_mesh = (x_mesh @ self.w + self.b)[:, 0].reshape(x1_mesh.shape)
            ax.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.35, cmap="viridis")

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("y")
            ax.legend()
            return fig

        return None
