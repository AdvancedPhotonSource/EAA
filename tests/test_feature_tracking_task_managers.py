from eaa.core.tooling.base import BaseTool, ToolReturnType, tool
from eaa.task_manager.imaging.feature_tracking import FeatureTrackingTaskManager
from eaa.task_manager.imaging.roi_search import ROISearchTaskManager


class DummyAcquireImageTool(BaseTool):
    def __init__(self):
        self.counter_acquire_image = 0
        super().__init__()

    @tool(name="acquire_image", return_type=ToolReturnType.IMAGE_PATH)
    def acquire_image(self) -> str:
        return "dummy.png"


def test_roi_search_task_manager_run_uses_feedback_loop(monkeypatch):
    task_manager = ROISearchTaskManager(
        build=False,
        use_coding_tools=False,
        image_acquisition_tool=DummyAcquireImageTool(),
        session_db_path=None,
    )
    captured = {}

    def fake_run_feedback_loop(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(task_manager, "run_feedback_loop", fake_run_feedback_loop)

    task_manager.run(
        feature_description="bright spot",
        y_range=(0.0, 10.0),
        x_range=(5.0, 15.0),
        fov_size=(2.0, 3.0),
        step_size=(1.0, 1.5),
    )

    assert "bright spot" in captured["initial_prompt"]
    assert captured["message_with_yielded_image"].startswith("Here is the image")


def test_feature_tracking_task_manager_run_uses_feedback_loop(monkeypatch):
    task_manager = FeatureTrackingTaskManager(
        build=False,
        use_coding_tools=False,
        image_acquisition_tool=DummyAcquireImageTool(),
        session_db_path=None,
    )
    captured = {}

    def fake_run_feedback_loop(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(task_manager, "run_feedback_loop", fake_run_feedback_loop)

    task_manager.run(
        reference_image_path="reference.png",
        initial_position=(1.0, 2.0),
        initial_fov_size=(3.0, 4.0),
        y_range=(0.0, 10.0),
        x_range=(5.0, 15.0),
    )

    assert captured["initial_image_path"] == "reference.png"
    assert "reference image" in captured["initial_prompt"]
    assert "image_path_tool_response" in captured["hook_functions"]


def test_feature_tracking_task_manager_no_longer_owns_fov_search():
    assert not hasattr(FeatureTrackingTaskManager, "run_fov_search")
