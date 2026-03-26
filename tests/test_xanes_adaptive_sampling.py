import argparse

import botorch
import gpytorch
import numpy as np
import pytest
import torch

from eaa_core.task_manager.tuning.bo import BayesianOptimizationTaskManager
from eaa_core.tool.optimization import BayesianOptimizationTool
from eaa_spectroscopy.acquisition_function.xanes import (
    ComprehensiveAugmentedAcquisitionFunction,
)
from eaa_spectroscopy.task_manager.spectroscopy.xanes import (
    XANESAdaptiveSamplingTaskManager,
)
from eaa_spectroscopy.tool.spectroscopy import (
    AdaptiveXANESBayesianOptimization,
    SimulatedSpectrumMeasurementTool,
)

import test_utils as tutils


@pytest.fixture(autouse=True)
def force_cpu_execution(monkeypatch: pytest.MonkeyPatch):
    """Run all XANES adaptive-sampling tests on CPU only."""
    previous_device = torch.get_default_device()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(previous_device)


class TestXANESAdaptiveSampling(tutils.BaseTester):
    @staticmethod
    def load_ybco_xanes_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the YBCO XANES CI dataset.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Energy grid, test spectrum, and stacked reference spectra.
        """
        data_path = (
            f"{tutils.BaseTester.get_ci_data_dir()}/data/xanes_spectrum/data_YBCO.csv"
        )
        data = np.genfromtxt(data_path, delimiter=",", names=True)
        energy = np.asarray(data["energy_ev"], dtype=np.float64)
        spectrum = np.asarray(data["test"], dtype=np.float64)
        reference_spectra = np.stack(
            [
                np.asarray(data["ref1"], dtype=np.float64),
                np.asarray(data["ref2"], dtype=np.float64),
            ]
        )
        return energy, spectrum, reference_spectra

    def test_bayesian_optimization_task_manager_with_objective_tool(self):
        energy, spectrum, reference_spectra = self.load_ybco_xanes_data()
        measurement_tool = SimulatedSpectrumMeasurementTool(
            data=(energy, spectrum),
            noise_std=0.0,
        )
        bo_tool = BayesianOptimizationTool(
            bounds=([float(energy.min())], [float(energy.max())]),
            acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
            acquisition_function_kwargs={
                "differentiation_method": "numerical",
                "gradient_order": 1,
                "phi_g": 0.2,
                "phi_g2": 0.0,
                "reference_spectra_x": torch.as_tensor(energy, dtype=torch.double),
                "reference_spectra_y": torch.as_tensor(
                    reference_spectra,
                    dtype=torch.double,
                ),
                "add_posterior_stddev": True,
                "acqf_weight_func_floor_value": 0.01,
                "acqf_weight_func_post_edge_gain": 3.0,
                "acqf_weight_func_post_edge_offset": 2.0,
                "acqf_weight_func_post_edge_width": 1.0,
            },
            model_class=botorch.models.SingleTaskGP,
            model_kwargs={"covar_module": gpytorch.kernels.MaternKernel(nu=2.5)},
            optimization_function=botorch.optim.optimize_acqf,
            optimization_function_kwargs={"num_restarts": 8, "raw_samples": 32},
        )

        task_manager = BayesianOptimizationTaskManager(
            llm_config=None,
            bayesian_optimization_tool=bo_tool,
            objective_function=measurement_tool,
            objective_function_method="measure",
            n_initial_points=10,
            session_db_path=None,
        )
        task_manager.run(n_iterations=10)

        xs = task_manager.bayesian_optimization_tool.xs_untransformed
        ys = task_manager.bayesian_optimization_tool.ys_untransformed
        assert xs.shape == (20, 1)
        assert ys.shape == (20, 1)
        assert torch.isfinite(xs).all()
        assert torch.isfinite(ys).all()
        assert (xs >= float(energy.min())).all()
        assert (xs <= float(energy.max())).all()

    def test_simulated_spectrum_measurement_tool_shapes(self):
        energy, spectrum, _ = self.load_ybco_xanes_data()
        tool = SimulatedSpectrumMeasurementTool(data=(energy, spectrum))

        x_query = torch.tensor(
            [
                [float(energy[0])],
                [float(energy[len(energy) // 2])],
                [float(energy[-1])],
            ],
            dtype=torch.double,
        )
        y_query = tool.measure(x_query, add_noise=False)

        assert y_query.shape == (3, 1)
        assert torch.isfinite(y_query).all()

    def test_xanes_adaptive_sampling_task_manager(self):
        tutils.set_seed(123)

        energy, spectrum, reference_spectra = self.load_ybco_xanes_data()
        measurement_tool = SimulatedSpectrumMeasurementTool(
            data=(energy, spectrum),
            noise_std=0.0,
        )

        bo_tool = AdaptiveXANESBayesianOptimization(
            bounds=([float(energy.min())], [float(energy.max())]),
            acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
            acquisition_function_kwargs={
                "differentiation_method": "numerical",
                "gradient_order": 1,
                "phi_g": 1e3,
                "phi_g2": 1e2,
                "reference_spectra_x": torch.as_tensor(energy, dtype=torch.double),
                "reference_spectra_y": torch.as_tensor(
                    reference_spectra,
                    dtype=torch.double,
                ),
                "add_posterior_stddev": True,
                "acqf_weight_func_floor_value": 0.01,
                "acqf_weight_func_post_edge_gain": 3.0,
                "acqf_weight_func_post_edge_offset": 2.0,
                "acqf_weight_func_post_edge_width": 1.0,
            },
            model_class=botorch.models.SingleTaskGP,
            model_kwargs={"covar_module": gpytorch.kernels.MaternKernel(nu=2.5)},
            noise_std=1e-4,
            n_discrete_choices=1000,
            n_updates_create_acqf_weight_func=2,
            n_max_measurements=40,
        )
        initial_points = torch.linspace(
            float(energy.min()),
            float(energy.max()),
            15,
            dtype=torch.double,
        ).view(-1, 1)

        task_manager = XANESAdaptiveSamplingTaskManager(
            llm_config=None,
            measurement_tool=measurement_tool,
            bayesian_optimization_tool=bo_tool,
            initial_points=initial_points,
            session_db_path=None,
        )
        task_manager.run(n_iterations=20)

        if self.debug:
            import matplotlib.pyplot as plt
            bo_tool.visualize_status()
            plt.show()

        assert bo_tool.acquisition_function.weight_func is not None
        assert task_manager.stop_reason in {None, "max_measurements_reached"}
        assert bo_tool.xs_untransformed.shape[1] == 1
        assert bo_tool.ys_untransformed.shape[1] == 1
        assert initial_points.shape[0] <= bo_tool.xs_untransformed.shape[0] <= bo_tool.n_max_measurements
        assert bo_tool.ys_untransformed.shape[0] == bo_tool.xs_untransformed.shape[0]
        assert torch.isfinite(bo_tool.xs_untransformed).all()
        assert torch.isfinite(bo_tool.ys_untransformed).all()

    def test_adaptive_xanes_bo_builds_weight_without_task_manager(self):
        tutils.set_seed(123)

        energy, spectrum, reference_spectra = self.load_ybco_xanes_data()
        measurement_tool = SimulatedSpectrumMeasurementTool(
            data=(energy, spectrum),
            noise_std=0.0,
        )

        bo_tool = AdaptiveXANESBayesianOptimization(
            bounds=([float(energy.min())], [float(energy.max())]),
            acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
            acquisition_function_kwargs={
                "differentiation_method": "numerical",
                "gradient_order": 1,
                "phi_g": 0.2,
                "phi_g2": 0.0,
                "reference_spectra_x": torch.as_tensor(energy, dtype=torch.double),
                "reference_spectra_y": torch.as_tensor(
                    reference_spectra,
                    dtype=torch.double,
                ),
                "add_posterior_stddev": True,
                "acqf_weight_func_floor_value": 0.01,
                "acqf_weight_func_post_edge_gain": 3.0,
                "acqf_weight_func_post_edge_offset": 2.0,
                "acqf_weight_func_post_edge_width": 1.0,
            },
            model_class=botorch.models.SingleTaskGP,
            model_kwargs={"covar_module": gpytorch.kernels.MaternKernel(nu=2.5)},
            noise_std=1e-4,
            n_discrete_choices=256,
            n_updates_create_acqf_weight_func=2,
            n_max_measurements=8,
        )

        initial_points = torch.linspace(
            float(energy.min()),
            float(energy.max()),
            6,
            dtype=torch.double,
        ).view(-1, 1)
        for x in initial_points:
            x = x.view(1, 1)
            y = measurement_tool.measure(x, add_noise=False)
            bo_tool.update(x, y)
        bo_tool.build()

        assert getattr(bo_tool.acquisition_function, "weight_func", None) is None
        assert np.isfinite(bo_tool.acquisition_function.acqf_g.background_gradient)

        candidate = bo_tool.suggest(n_suggestions=1)
        bo_tool.update(candidate, measurement_tool.measure(candidate, add_noise=False))
        assert getattr(bo_tool.acquisition_function, "weight_func", None) is None
        assert np.isfinite(bo_tool.acquisition_function.acqf_g.background_gradient)
        assert bo_tool.should_stop() is False

        candidate = bo_tool.suggest(n_suggestions=1)
        bo_tool.update(candidate, measurement_tool.measure(candidate, add_noise=False))
        assert bo_tool.acquisition_function.weight_func is not None
        assert np.isfinite(bo_tool.acquisition_function.acqf_g.background_gradient)
        assert bo_tool.n_adaptive_update_calls == 2
        assert bo_tool.should_stop() is True
        assert bo_tool.stop_reason == "max_measurements_reached"

    def test_adaptive_xanes_bo_filters_duplicate_discrete_candidates(self):
        tutils.set_seed(123)

        energy, spectrum, reference_spectra = self.load_ybco_xanes_data()
        measurement_tool = SimulatedSpectrumMeasurementTool(
            data=(energy, spectrum),
            noise_std=0.0,
        )

        bo_tool = AdaptiveXANESBayesianOptimization(
            bounds=([float(energy.min())], [float(energy.max())]),
            acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
            acquisition_function_kwargs={
                "differentiation_method": "numerical",
                "gradient_order": 1,
                "phi_g": 0.2,
                "phi_g2": 0.0,
                "reference_spectra_x": torch.as_tensor(energy, dtype=torch.double),
                "reference_spectra_y": torch.as_tensor(
                    reference_spectra,
                    dtype=torch.double,
                ),
                "add_posterior_stddev": True,
                "acqf_weight_func_floor_value": 0.01,
                "acqf_weight_func_post_edge_gain": 3.0,
                "acqf_weight_func_post_edge_offset": 2.0,
                "acqf_weight_func_post_edge_width": 1.0,
            },
            model_class=botorch.models.SingleTaskGP,
            model_kwargs={"covar_module": gpytorch.kernels.MaternKernel(nu=2.5)},
            noise_std=1e-4,
            n_discrete_choices=16,
        )

        discrete_grid = torch.linspace(
            float(energy.min()),
            float(energy.max()),
            16,
            dtype=torch.double,
        ).view(-1, 1)
        initial_points = discrete_grid[[0, 5, 10, 15]]
        initial_values = measurement_tool.measure(initial_points, add_noise=False)
        bo_tool.update(initial_points, initial_values)
        bo_tool.build()

        candidate = bo_tool.suggest(n_suggestions=1)
        assert torch.all(torch.abs(initial_points - candidate) > 1e-9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    tester = TestXANESAdaptiveSampling()
    tester.setup_method(
        name="",
        generate_data=False,
        generate_gold=False,
        debug=True,
    )
    tester.test_bayesian_optimization_task_manager_with_objective_tool()
    tester.test_simulated_spectrum_measurement_tool_shapes()
    tester.test_xanes_adaptive_sampling_task_manager()
    tester.test_adaptive_xanes_bo_builds_weight_without_task_manager()
