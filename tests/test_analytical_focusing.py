import argparse
import os

import numpy as np
import tifffile
from matplotlib.figure import Figure

from eaa.task_manager.tuning.analytical_focusing import (
    AnalyticalScanningMicroscopeFocusingTaskManager,
)
from eaa.tool.imaging.acquisition import SimulatedAcquireImage
from eaa.tool.imaging.param_tuning import SimulatedSetParameters
from sciagent.tool.base import BaseTool

import test_utils as tutils


class TestAnalyticalFocusing(tutils.BaseTester):
    def _build_task_manager(self):
        image_path = os.path.join(
            self.get_ci_input_data_dir(),
            "simulated_images",
            "grid_test_pattern_roi.tiff",
        )
        image = tifffile.imread(image_path)
        if image.ndim == 3:
            image = image[..., 0]

        acquisition_tool = SimulatedAcquireImage(
            whole_image=image,
            add_axis_ticks=True,
            add_grid_lines=False,
            invert_yaxis=False,
            add_line_scan_candidates_to_image=False,
            plot_image_in_log_scale=False,
        )

        param_setting_tool = SimulatedSetParameters(
            acquisition_tool=acquisition_tool,
            parameter_names=["z"],
            true_parameters=[3.0],
            parameter_ranges=[(0.0,), (10.0,)],
            drift_factor=10,
        )

        task_manager = AnalyticalScanningMicroscopeFocusingTaskManager(
            param_setting_tool=param_setting_tool,
            acquisition_tool=acquisition_tool,
            initial_parameters={"z": 10.0},
            parameter_ranges=[(0.0,), (10.0,)],
            line_scan_tool_x_coordinate_args=("x_center",),
            line_scan_tool_y_coordinate_args=("y_center",),
            image_acquisition_tool_x_coordinate_args=("x_center",),
            image_acquisition_tool_y_coordinate_args=("y_center",),
        )
        return task_manager, acquisition_tool

    def test_task_manager_runs(self, monkeypatch):
        task_manager, acquisition_tool = self._build_task_manager()
        monkeypatch.setattr(task_manager, "run_conversation", lambda: None)
        n_initial_points = 2
        n_bo_iterations = 1
        task_manager.run(
            initial_2d_scan_kwargs={"y_center": 175, "x_center": 175, "size_y": 350, "size_x": 350},
            initial_line_scan_kwargs={
                "x_center": 160,
                "y_center": 170,
                "length": 60,
                "scan_step": 1.0,
            },
            n_initial_points=n_initial_points,
            initial_sampling_window_size=(0.5,),
            n_max_iterations=n_bo_iterations,
            parameter_change_step_limit=0.5,
        )
        assert (
            task_manager.param_setting_tool.len_parameter_history
            == n_initial_points + n_bo_iterations + 1
        )
        assert acquisition_tool.counter_acquire_image >= n_initial_points + n_bo_iterations

    def test_task_manager_runs_without_offset_calibration(self, monkeypatch):
        task_manager, acquisition_tool = self._build_task_manager()
        task_manager.run_offset_calibration = False
        monkeypatch.setattr(task_manager, "run_conversation", lambda: None)
        n_initial_points = 2
        n_bo_iterations = 1
        task_manager.run(
            initial_2d_scan_kwargs=None,
            initial_line_scan_kwargs={
                "x_center": 160,
                "y_center": 170,
                "length": 60,
                "scan_step": 1.0,
            },
            n_initial_points=n_initial_points,
            initial_sampling_window_size=(0.5,),
            n_max_iterations=n_bo_iterations,
            parameter_change_step_limit=0.5,
        )
        assert (
            task_manager.param_setting_tool.len_parameter_history
            == n_initial_points + n_bo_iterations + 1
        )
        assert acquisition_tool.counter_acquire_image == 0

    def test_linear_drift_prediction_fit_and_apply(self):
        task_manager, _ = self._build_task_manager()
        task_manager.use_linear_drift_prediction = True
        task_manager.n_parameter_drift_points_before_prediction = 3
        task_manager.initial_image_acquisition_position = np.array([10.0, 20.0], dtype=float)
        task_manager.image_acquisition_kwargs = {
            "y_center": 0.0,
            "x_center": 0.0,
            "size_y": 128,
            "size_x": 128,
        }
        task_manager.line_scan_kwargs = {
            "x_center": 1.0,
            "y_center": 2.0,
            "length": 10.0,
            "scan_step": 1.0,
        }

        # Drift model: delta_y = 2 * z + 1, delta_x = -z + 3
        for z in [0.0, 1.0, 2.0]:
            drift = np.array([2.0 * z + 1.0, -z + 3.0], dtype=float)
            current_position = task_manager.initial_image_acquisition_position + drift
            task_manager.update_linear_drift_models(
                parameters=np.array([z], dtype=float),
                current_position_yx=current_position,
            )

        assert task_manager.should_apply_linear_drift_prediction()
        task_manager.apply_predicted_image_acquisition_position(np.array([4.0], dtype=float))

        assert np.isclose(task_manager.image_acquisition_kwargs["y_center"], 19.0)
        assert np.isclose(task_manager.image_acquisition_kwargs["x_center"], 19.0)
        assert np.isclose(task_manager.line_scan_kwargs["y_center"], 21.0)
        assert np.isclose(task_manager.line_scan_kwargs["x_center"], 20.0)

    def test_run_iteration_applies_registration_offset_and_updates_model(self, monkeypatch):
        task_manager, acquisition_tool = self._build_task_manager()
        task_manager.use_linear_drift_prediction = True
        task_manager.n_parameter_drift_points_before_prediction = 1
        task_manager.initial_image_acquisition_position = np.array([0.0, 0.0], dtype=float)
        task_manager.initialize_kwargs_buffers(
            initial_line_scan_kwargs={
                "x_center": 160.0,
                "y_center": 170.0,
                "length": 60.0,
                "scan_step": 1.0,
            },
            initial_2d_scan_kwargs={"y_center": 0.0, "x_center": 0.0, "size_y": 200, "size_x": 200},
        )
        task_manager.update_linear_drift_models(
            parameters=np.array([0.0], dtype=float),
            current_position_yx=np.array([0.0, 0.0], dtype=float),
        )

        def fake_run_2d_scan():
            kwargs = task_manager.image_acquisition_kwargs
            acquisition_tool.update_image_acquisition_call_history(
                x_center=float(kwargs["x_center"]),
                y_center=float(kwargs["y_center"]),
                size_x=float(kwargs["size_x"]),
                size_y=float(kwargs["size_y"]),
                psize_x=1.0,
                psize_y=1.0,
            )

        monkeypatch.setattr(task_manager, "run_2d_scan", fake_run_2d_scan)
        monkeypatch.setattr(
            task_manager,
            "find_offset",
            lambda: (np.array([0.0, 0.0]), np.array([100.0, -50.0])),
        )
        monkeypatch.setattr(task_manager, "run_line_scan", lambda: 1.0)
        monkeypatch.setattr(task_manager, "update_optimization_model", lambda fwhm: None)

        captured_offsets = []

        def capture_offset(offset):
            captured_offsets.append(np.array(offset, dtype=float))

        monkeypatch.setattr(task_manager, "apply_offset_to_image_acquisition_kwargs", capture_offset)
        task_manager.run_tuning_iteration(np.array([1.0], dtype=float))
        assert any(np.allclose(offset, np.array([100.0, -50.0])) for offset in captured_offsets)

    def test_run_tuning_iteration_calls_drift_visualization_after_update(self, monkeypatch):
        task_manager, _ = self._build_task_manager()
        task_manager.run_offset_calibration = True

        task_manager.initialize_kwargs_buffers(
            initial_line_scan_kwargs={
                "x_center": 160.0,
                "y_center": 170.0,
                "length": 60.0,
                "scan_step": 1.0,
            },
            initial_2d_scan_kwargs={"y_center": 0.0, "x_center": 0.0, "size_y": 200, "size_x": 200},
        )

        monkeypatch.setattr(task_manager, "run_2d_scan", lambda: None)
        monkeypatch.setattr(
            task_manager,
            "find_offset",
            lambda: (np.array([0.0, 0.0]), np.array([0.0, 0.0])),
        )
        monkeypatch.setattr(task_manager, "run_line_scan", lambda: 1.0)
        monkeypatch.setattr(task_manager, "update_optimization_model", lambda fwhm: None)

        call_order = []
        monkeypatch.setattr(
            task_manager,
            "update_linear_drift_models",
            lambda parameters: call_order.append("update"),
        )
        monkeypatch.setattr(
            task_manager,
            "record_linear_drift_model_visualizations",
            lambda: call_order.append("visualize"),
        )

        task_manager.run_tuning_iteration(np.array([1.0], dtype=float))

        assert call_order == ["update", "visualize"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-gold", action="store_true")
    args = parser.parse_args()

    tester = TestAnalyticalFocusing()
    tester.setup_method(
        name="",
        generate_data=False,
        generate_gold=args.generate_gold,
        debug=True,
    )
    tester.test_task_manager_runs()
    tester.test_task_manager_runs_without_offset_calibration()
