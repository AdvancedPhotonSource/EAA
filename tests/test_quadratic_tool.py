import argparse
import logging

import pytest
import torch

from eaa.tool.optimization import QuadraticOptimizationTool
import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestQuadraticOptimizationTool(tutils.BaseTester):
    def test_quadratic_tool_finds_known_maximum(self):
        tutils.set_seed(123)
        tool = QuadraticOptimizationTool()

        Q = torch.tensor([[-4.0, -1.0], [-1.0, -2.0]])
        linear = torch.tensor([8.0, 3.0])
        constant = 1.5

        def objective(x: torch.Tensor) -> torch.Tensor:
            # y = x^T Q x + linear^T x + constant
            quadratic = torch.einsum("bi,ij,bj->b", x, Q, x)
            return (quadratic + x @ linear + constant)[:, None]

        xs = torch.randn(30, 2)
        ys = objective(xs)
        tool.update(xs, ys)

        best = tool.suggest()
        optimum = (-0.5 * torch.linalg.solve(Q, linear)).detach()
        assert torch.allclose(best[0], optimum, atol=1e-4)

        repeated = tool.suggest()
        assert repeated.shape == (1, 2)
        assert torch.allclose(repeated[0], best[0])

    def test_quadratic_tool_requires_enough_data(self):
        tool = QuadraticOptimizationTool()
        xs = torch.tensor([[0.0], [1.0]])
        ys = -(xs - 0.5) ** 2
        tool.update(xs, ys)
        with pytest.raises(ValueError):
            tool.suggest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tester = TestQuadraticOptimizationTool()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_quadratic_tool_finds_known_maximum()
    tester.test_quadratic_tool_requires_enough_data()
