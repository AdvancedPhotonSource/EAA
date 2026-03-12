import argparse
import os

import numpy as np
import tifffile

from eaa.task_manager.tuning.analytical_focusing import (
    AnalyticalScanningMicroscopeFocusingTaskManager,
)
from eaa.tool.imaging.acquisition import SimulatedAcquireImage
from eaa.tool.imaging.param_tuning import SimulatedSetParameters
from eaa.tool.imaging.registration import ImageRegistration

import test_utils as tutils


class DummyRegistrationTool:
    def __init__(self, name: str, offset=(0.0, 0.0)):
        self.name = name
        self.offset = np.array(offset, dtype=float)

    def get_offset(self, target="previous"):
        return self.offset.tolist()


class TestAnalyticalFocusing(tutils.BaseTester):
    def _build_task_manager(self, registration_tools=None):
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
        if registration_tools is None:
            registration_tools = [
                ImageRegistration(image_acquisition_tool=acquisition_tool)
            ]

        task_manager = AnalyticalScanningMicroscopeFocusingTaskManager(
            param_setting_tool=param_setting_tool,
            acquisition_tool=acquisition_tool,
            initial_parameters={"z": 10.0},
            parameter_ranges=[(0.0,), (10.0,)],
            registration_tools=registration_tools,
            line_scan_tool_x_coordinate_args=("x_center",),
            line_scan_tool_y_coordinate_args=("y_center",),
            image_acquisition_tool_x_coordinate_args=("x_center",),
            image_acquisition_tool_y_coordinate_args=("y_center",),
            session_db_path=None,
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

    def test_select_drift_uses_linear_model_after_priming(self):
        registration_tools = [
            DummyRegistrationTool("primary"),
            DummyRegistrationTool("secondary"),
        ]
        task_manager, _ = self._build_task_manager(
            registration_tools=registration_tools
        )
        task_manager.registration_selection_priming_iterations = 3

        for z in [0.0, 1.0, 2.0]:
            drift = np.array([2.0 * z + 1.0, -z + 3.0], dtype=float)
            task_manager.update_linear_drift_models(
                parameters=np.array([z], dtype=float),
                drift_wrt_initial_yx=drift,
            )

        chosen_drift, chosen_source = task_manager._select_drift(
            candidate_drifts={
                "primary": np.array([0.0, 0.0], dtype=float),
                "secondary": np.array([9.0, -1.0], dtype=float),
            },
            x_current=np.array([4.0], dtype=float),
        )

        assert chosen_source == "secondary"
        assert np.allclose(chosen_drift, np.array([9.0, -1.0], dtype=float))

    def test_predict_linear_drift_model_returns_zero_without_samples(self):
        task_manager, _ = self._build_task_manager()

        predicted_drift = task_manager.predict_linear_drift_model(
            np.array([4.0], dtype=float)
        )

        assert np.allclose(predicted_drift, np.array([0.0, 0.0], dtype=float))

    def test_predict_linear_drift_model_uses_fitted_models(self):
        registration_tools = [
            DummyRegistrationTool("primary"),
            DummyRegistrationTool("secondary"),
        ]
        task_manager, _ = self._build_task_manager(
            registration_tools=registration_tools
        )

        for z in [0.0, 1.0, 2.0]:
            drift = np.array([2.0 * z + 1.0, -z + 3.0], dtype=float)
            task_manager.update_linear_drift_models(
                parameters=np.array([z], dtype=float),
                drift_wrt_initial_yx=drift,
            )

        predicted_drift = task_manager.predict_linear_drift_model(
            np.array([4.0], dtype=float)
        )

        assert np.allclose(predicted_drift, np.array([9.0, -1.0], dtype=float))

    def test_run_iteration_applies_registration_offset_and_updates_model(self, monkeypatch):
        task_manager, acquisition_tool = self._build_task_manager()
        task_manager.initialize_kwargs_buffers(
            initial_line_scan_kwargs={
                "x_center": 160.0,
                "y_center": 170.0,
                "length": 60.0,
                "scan_step": 1.0,
            },
            initial_2d_scan_kwargs={"y_center": 0.0, "x_center": 0.0, "size_y": 200, "size_x": 200},
        )
        task_manager.initial_image_acquisition_position = np.array([0.0, 0.0], dtype=float)

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
            "find_position_correction",
            lambda registration_tool, target: (
                np.array([0.0, 0.0]),
                np.array([100.0, -50.0]),
            ),
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
        task_manager.initial_image_acquisition_position = np.array([0.0, 0.0], dtype=float)

        monkeypatch.setattr(task_manager, "run_2d_scan", lambda: None)
        monkeypatch.setattr(
            task_manager,
            "apply_drift_correction",
            lambda x_current: np.array([0.0, 0.0], dtype=float),
        )
        monkeypatch.setattr(task_manager, "run_line_scan", lambda: 1.0)
        monkeypatch.setattr(task_manager, "update_optimization_model", lambda fwhm: None)

        call_order = []
        monkeypatch.setattr(
            task_manager,
            "update_linear_drift_models",
            lambda parameters, drift_wrt_initial_yx: call_order.append("update"),
        )
        monkeypatch.setattr(
            task_manager,
            "record_linear_drift_model_visualizations",
            lambda: call_order.append("visualize"),
        )

        task_manager.run_tuning_iteration(np.array([1.0], dtype=float))

        assert call_order == ["update", "visualize"]

    def test_registration_and_scan_position_corrections_across_two_iterations(self):
        registration_tool = DummyRegistrationTool("dummy", offset=(2.5, -4.0))
        task_manager, acquisition_tool = self._build_task_manager(
            registration_tools=[registration_tool]
        )
        task_manager.initialize_kwargs_buffers(
            initial_line_scan_kwargs={
                "x_center": 160.0,
                "y_center": 170.0,
                "length": 60.0,
                "scan_step": 1.0,
            },
            initial_2d_scan_kwargs={
                "y_center": 175.0,
                "x_center": 175.0,
                "size_y": 350,
                "size_x": 350,
            },
        )

        task_manager.run_2d_scan()
        task_manager.run_line_scan()

        task_manager.param_setting_tool.set_parameters(np.array([9.0], dtype=float))
        task_manager.image_acquisition_kwargs["y_center"] += 7.0
        task_manager.image_acquisition_kwargs["x_center"] -= 6.0
        task_manager.run_2d_scan()

        line_scan_correction, image_acquisition_correction = (
            task_manager.find_position_correction(registration_tool, target="previous")
        )
        assert np.allclose(line_scan_correction, np.array([4.5, -2.0], dtype=float))
        assert np.allclose(
            image_acquisition_correction,
            np.array([-2.5, 4.0], dtype=float),
        )

        chosen_line_scan_correction = task_manager.apply_drift_correction(
            np.array([9.0], dtype=float)
        )
        assert np.allclose(
            chosen_line_scan_correction,
            np.array([4.5, -2.0], dtype=float),
        )
        assert np.allclose(
            task_manager.extract_line_scan_position(task_manager.line_scan_kwargs),
            np.array([174.5, 158.0], dtype=float),
        )
        assert np.allclose(
            task_manager.extract_image_acquisition_position(
                task_manager.image_acquisition_kwargs
            ),
            np.array([179.5, 173.0], dtype=float),
        )

        task_manager.run_line_scan()
        assert len(acquisition_tool.line_scan_call_history) == 2
        assert np.allclose(
            [
                acquisition_tool.line_scan_call_history[-1]["y_center"],
                acquisition_tool.line_scan_call_history[-1]["x_center"],
            ],
            np.array([174.5, 158.0], dtype=float),
        )


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
    tester.test_registration_and_scan_position_corrections_across_two_iterations()
