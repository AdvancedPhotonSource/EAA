import argparse
import os

import tifffile

from eaa.task_manager.tuning.analytical_focusing import (
    AnalyticalScanningMicroscopeFocusingTaskManager,
)
from eaa.tool.imaging.acquisition import SimulatedAcquireImage
from eaa.tool.imaging.param_tuning import SimulatedSetParameters

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
            line_scan_tool_x_coordinate_args=("start_x", "end_x"),
            line_scan_tool_y_coordinate_args=("start_y", "end_y"),
            image_acquisition_tool_x_coordinate_args=("loc_x",),
            image_acquisition_tool_y_coordinate_args=("loc_y",),
        )
        return task_manager, acquisition_tool

    def test_task_manager_runs(self, monkeypatch):
        task_manager, acquisition_tool = self._build_task_manager()
        monkeypatch.setattr(task_manager, "run_conversation", lambda: None)
        n_initial_points = 2
        n_bo_iterations = 1
        task_manager.run(
            initial_2d_scan_kwargs={"loc_y": 0, "loc_x": 0, "size_y": 350, "size_x": 350},
            initial_line_scan_kwargs={
                "start_x": 130,
                "start_y": 170,
                "end_x": 190,
                "end_y": 170,
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
