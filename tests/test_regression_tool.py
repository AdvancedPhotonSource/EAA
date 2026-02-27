import numpy as np
import pytest
from matplotlib.figure import Figure

from eaa.tool.regression import MultivariateLinearRegression


class TestMultivariateLinearRegression:
    def test_update_and_predict_exact_multivariate_mapping(self):
        tool = MultivariateLinearRegression()

        x = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, -1.0],
            ]
        )
        w_true = np.array([[2.0, -1.0], [3.0, 4.0]])
        b_true = np.array([5.0, -2.0])
        y = x @ w_true + b_true

        message = tool.update(x.tolist(), y.tolist())
        assert "Model updated with 5 total samples" in message

        x_test = np.array([[3.0, 2.0], [-1.0, 4.0]])
        y_expected = x_test @ w_true + b_true
        y_pred = np.array(tool.predict(x_test.tolist()))

        assert y_pred.shape == (2, 2)
        assert np.allclose(y_pred, y_expected, atol=1e-10)

    def test_incremental_update_appends_data_and_refits(self):
        tool = MultivariateLinearRegression()

        x1 = [[0.0], [1.0]]
        y1 = [[1.0], [3.0]]
        tool.update(x1, y1)

        x2 = [[2.0], [3.0]]
        y2 = [[5.0], [7.0]]
        message = tool.update(x2, y2)

        assert "Model updated with 4 total samples" in message

        y_pred = np.array(tool.predict([[4.0]]))
        assert y_pred.shape == (1, 1)
        assert np.allclose(y_pred, np.array([[9.0]]), atol=1e-10)

    def test_predict_before_fit_raises(self):
        tool = MultivariateLinearRegression()
        with pytest.raises(ValueError, match="Model is not fitted"):
            tool.predict([[1.0, 2.0]])

    def test_update_rejects_non_2d_data(self):
        tool = MultivariateLinearRegression()
        with pytest.raises(ValueError, match="`x` must be a 2D array"):
            tool.update([1.0, 2.0], [[1.0], [2.0]])

        with pytest.raises(ValueError, match="`y` must be a 2D array"):
            tool.update([[1.0], [2.0]], [1.0, 2.0])

    def test_update_rejects_mismatched_sample_count(self):
        tool = MultivariateLinearRegression()
        with pytest.raises(ValueError, match="number of samples in `x` and `y` must match"):
            tool.update([[1.0], [2.0]], [[1.0]])

    def test_rejects_feature_mismatch_across_calls(self):
        tool = MultivariateLinearRegression()
        tool.update([[0.0, 1.0], [1.0, 0.0]], [[1.0], [2.0]])

        with pytest.raises(ValueError, match="`x` must have 2 features"):
            tool.update([[1.0], [2.0]], [[3.0], [4.0]])

        with pytest.raises(ValueError, match="`y` must have 1 features"):
            tool.update([[1.0, 2.0]], [[3.0, 4.0]])

        with pytest.raises(ValueError, match="`x` must have 2 features"):
            tool.predict([[1.0]])

    def test_visualize_status_returns_figure_for_1d_to_1d(self):
        tool = MultivariateLinearRegression()
        tool.update([[0.0], [1.0], [2.0]], [[1.0], [3.0], [5.0]])

        fig = tool.visualize_status()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        assert fig.axes[0].name == "rectilinear"

    def test_visualize_status_returns_figure_for_2d_to_1d(self):
        tool = MultivariateLinearRegression()
        x = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        y = [[1.0], [3.0], [2.0], [4.0]]
        tool.update(x, y)

        fig = tool.visualize_status()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        assert fig.axes[0].name == "3d"

    def test_visualize_status_returns_none_for_unsupported_shapes(self):
        tool = MultivariateLinearRegression()
        tool.update([[0.0], [1.0], [2.0]], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        assert tool.visualize_status() is None
