
import argparse
import logging

import gpytorch.constraints
import botorch
import gpytorch
import torch

from eaa.tools.bo import BayesianOptimizationTool

import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestBayesianOptimization(tutils.BaseTester):
    
    def visualize_gp(self, bo_tool: BayesianOptimizationTool):
        x_ticks = torch.linspace(-10, 10, 100)
        y_ticks = torch.linspace(-10, 10, 100)
        x, y = torch.meshgrid(x_ticks, y_ticks, indexing='xy')
        x = torch.stack([x, y], dim=-1).reshape(-1, 2)
        x = bo_tool.transform_data(x, train_x=False)[0]
        posterior = bo_tool.model.posterior(x)
        
        mu = posterior.mean.detach().cpu().numpy()
        sigma = posterior.variance.sqrt().detach().cpu().numpy()
        
        mu = mu.reshape(100, 100)
        sigma = sigma.reshape(100, 100)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(mu, extent=[x_ticks.min(), x_ticks.max(), y_ticks.min(), y_ticks.max()])
        ax[0].scatter(bo_tool.xs_raw[:, 0], bo_tool.xs_raw[:, 1], color='gray')
        ax[0].scatter(bo_tool.xs_raw[-1:, 0], bo_tool.xs_raw[-1:, 1], color='red', marker='x')
        ax[0].set_title('mean')
        ax[1].imshow(sigma, extent=[x_ticks.min(), x_ticks.max(), y_ticks.min(), y_ticks.max()])
        ax[1].scatter(bo_tool.xs_raw[:, 0], bo_tool.xs_raw[:, 1], color='gray')
        ax[1].scatter(bo_tool.xs_raw[-1:, 0], bo_tool.xs_raw[-1:, 1], color='red', marker='x')
        ax[1].set_title('std')
        plt.show()
    
    def test_bayesian_optimization(self):
        def objective_function(x: torch.Tensor) -> torch.Tensor:
            # Expected input shape: (n_samples, n_features)
            # Maximum: x = [1, 2]
            y = torch.exp(-((x[:, 0] - 1) ** 2 + (x[:, 1] - 2) ** 2) / (2 * 10 ** 2))
            return y[:, None]
        
        tutils.set_seed(42)
        
        tool = BayesianOptimizationTool(
            bounds=([-10, -10], [10, 10]),
            objective_function=objective_function,
            acquisition_function_class=botorch.acquisition.LogExpectedImprovement,
            acquisition_function_kwargs={
                "best_f": -100
            },
            model_class=botorch.models.SingleTaskGP,
            model_kwargs={
                "covar_module": gpytorch.kernels.MaternKernel(
                    nu=2.5,
                )
            },
            optimization_function=botorch.optim.optimize_acqf,
        )
        
        xs_init = tool.get_random_initial_points(n_points=20)
        for x in xs_init:
            x = x[None, :]
            y = objective_function(x)
            tool.update(x, y)
        tool.build()
        
        for i in range(50):
            candidates = tool.suggest(n_suggestions=1)
            logging.info(f"Candidate suggested: {candidates[0]}")
            y = objective_function(candidates)
            tool.update(candidates, y)
            # if self.debug:
            #     self.visualize_gp(tool)
        
        final_suggestion = candidates[0]
        assert torch.allclose(final_suggestion.float(), torch.tensor([1.0, 2.0]), rtol=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tester = TestBayesianOptimization()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_bayesian_optimization()
