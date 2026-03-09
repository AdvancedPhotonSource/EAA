import os
import argparse

import numpy as np
import tifffile

from sciagent.api.llm_config import OpenAIConfig

from eaa.task_manager.imaging import analytical_feature_tracking
from eaa.task_manager.imaging.analytical_feature_tracking import (
    AnalyticalFeatureTrackingTaskManager,
)
from eaa.tool.imaging.acquisition import SimulatedAcquireImage

import test_utils as tutils

import logging

logging.basicConfig(level=logging.INFO)

class TestAnalyticalFeatureTracking(tutils.BaseTester):
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
        task_manager = AnalyticalFeatureTrackingTaskManager(
            llm_config=OpenAIConfig(
                model="gpt-4",
                api_key="fake-key",
                base_url="https://api.openai.com/v1",
            ),
            image_acquisition_tool=acquisition_tool,
            image_acquisition_tool_x_coordinate_args=("x_center",),
            image_acquisition_tool_y_coordinate_args=("y_center",),
        )
        return task_manager, acquisition_tool, image

    def test_get_position_deltas_matches_spiral_pattern(self):
        task_manager, _, _ = self._build_task_manager()
        expected_positions = [
            (0, 0),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-2, -1),
            (-2, 0),
            (-2, 1),
            (-2, 2),
            (-1, 2),
        ]
        for idx, expected in enumerate(expected_positions):
            assert (
                task_manager.get_position_deltas(idx, (1, 1)) == expected
            ), f"Index {idx} mismatch"

    def test_feature_tracking_run_returns_expected_offset(self, monkeypatch):
        call_state = {"count": 0}

        def mock_check_feature_presence_llm(**kwargs):
            call_state["count"] += 1
            return call_state["count"] > 1

        monkeypatch.setattr(
            analytical_feature_tracking,
            "check_feature_presence_llm",
            mock_check_feature_presence_llm,
        )
        task_manager, _, image = self._build_task_manager()
        reference_loc = (60, 270)
        size = (100, 100)
        reference_image = image[
            reference_loc[0] : reference_loc[0] + size[0],
            reference_loc[1] : reference_loc[1] + size[1],
        ]
        
        drift = (100, 0)
        current_kwargs = {
            "y_center": reference_loc[0] + drift[0] + size[0] / 2,
            "x_center": reference_loc[1] + drift[1] + size[1] / 2,
            "size_y": size[0],
            "size_x": size[1],
        }
        step_size = (80.0, 80.0)
        offset = task_manager.run(
            current_acquisition_kwargs=current_kwargs,
            reference_image=reference_image,
            step_size=step_size,
            n_max_rounds=2,
        )
        expected_offset = np.array([-drift[0], -drift[1]])
        np.testing.assert_allclose(offset, expected_offset, atol=1.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-gold", action="store_true")
    args = parser.parse_args()

    tester = TestAnalyticalFeatureTracking()
    tester.setup_method(
        name="",
        generate_data=False,
        generate_gold=args.generate_gold,
        debug=True,
    )
    tester.test_feature_tracking_run_returns_expected_offset()
