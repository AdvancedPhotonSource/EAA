from eaa_core.tool.base import BaseTool, tool
from eaa_imaging.task_manager.imaging.feature_tracking import FeatureTrackingTaskManager
from eaa_imaging.task_manager.imaging.roi_search import (
    MultiAgentROISearchTaskManager,
    MultiAgentROISearchState,
    ROISearchTaskManager,
)


class DummyAcquireImageTool(BaseTool):
    def __init__(self):
        self.counter_acquire_image = 0
        super().__init__()

    @tool(name="acquire_image")
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


def test_multi_agent_roi_search_uses_dedicated_graph_then_feedback_loop(monkeypatch):
    task_manager = MultiAgentROISearchTaskManager(
        build=True,
        use_coding_tools=False,
        image_acquisition_tool=DummyAcquireImageTool(),
        session_db_path=None,
    )
    task_manager.model = object()

    responses = [
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "acquire_image",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "user", "content": "position prompt"},
        ),
        (
            {
                "role": "assistant",
                "content": '{"foi_present": true, "fov_description": "bright spot is visible"}',
            },
            {"role": "user", "content": "checker prompt"},
        ),
    ]

    def fake_invoke_model_raw(*args, **kwargs):
        return responses.pop(0)

    feedback_loop_kwargs = {}

    def fake_run_feedback_loop(**kwargs):
        feedback_loop_kwargs.update(kwargs)

    monkeypatch.setattr(task_manager, "invoke_model_raw", fake_invoke_model_raw)
    monkeypatch.setattr(task_manager, "run_feedback_loop", fake_run_feedback_loop)

    task_manager.run(
        feature_description="bright spot",
        y_range=(0.0, 10.0),
        x_range=(5.0, 15.0),
        fov_size=(2.0, 3.0),
        step_size=(1.0, 1.5),
    )

    assert isinstance(task_manager.task_state, MultiAgentROISearchState)
    assert task_manager.task_state.foi_present is True
    assert task_manager.task_state.last_image_path == "dummy.png"
    assert len(task_manager.task_state.position_tool_calls) == 1
    assert feedback_loop_kwargs["initial_image_path"] == "dummy.png"
    assert "bright spot is visible" in feedback_loop_kwargs["initial_prompt"]
    assert "centered in the FOV" in feedback_loop_kwargs["initial_prompt"]


def test_multi_agent_roi_search_repeats_until_feature_present(monkeypatch):
    task_manager = MultiAgentROISearchTaskManager(
        build=True,
        use_coding_tools=False,
        image_acquisition_tool=DummyAcquireImageTool(),
        session_db_path=None,
    )
    task_manager.model = object()

    responses = [
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "acquire_image",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "user", "content": "first position prompt"},
        ),
        (
            {
                "role": "assistant",
                "content": '{"foi_present": false, "fov_description": "only background"}',
            },
            {"role": "user", "content": "first checker prompt"},
        ),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "acquire_image",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "user", "content": "second position prompt"},
        ),
        (
            {
                "role": "assistant",
                "content": '{"foi_present": true, "fov_description": "feature appears"}',
            },
            {"role": "user", "content": "second checker prompt"},
        ),
    ]

    def fake_invoke_model_raw(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(task_manager, "invoke_model_raw", fake_invoke_model_raw)
    monkeypatch.setattr(task_manager, "run_feedback_loop", lambda **kwargs: None)

    task_manager.run(
        initial_prompt="Find a feature.",
        max_search_rounds=3,
    )

    assert task_manager.task_state.search_round_index == 2
    assert len(task_manager.task_state.position_tool_calls) == 2
    assert task_manager.task_state.foi_present is True


def test_multi_agent_roi_search_reprompts_position_proposer_on_bad_response(monkeypatch):
    task_manager = MultiAgentROISearchTaskManager(
        build=True,
        use_coding_tools=False,
        image_acquisition_tool=DummyAcquireImageTool(),
        session_db_path=None,
    )
    task_manager.model = object()

    responses = [
        (
            {"role": "assistant", "content": "I would scan the next point."},
            {"role": "user", "content": "position prompt"},
        ),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "acquire_image",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "user", "content": "retry position prompt"},
        ),
        (
            {
                "role": "assistant",
                "content": '{"foi_present": true, "fov_description": "feature appears"}',
            },
            {"role": "user", "content": "checker prompt"},
        ),
    ]
    prompts = []

    def fake_invoke_model_raw(message=None, **kwargs):
        prompts.append(message)
        return responses.pop(0)

    monkeypatch.setattr(task_manager, "invoke_model_raw", fake_invoke_model_raw)
    monkeypatch.setattr(task_manager, "run_feedback_loop", lambda **kwargs: None)

    task_manager.run(initial_prompt="Find a feature.")

    assert "could not be used" in prompts[1]
    assert "acquire_image" in prompts[1]
    assert task_manager.task_state.foi_present is True


def test_multi_agent_roi_search_reprompts_image_checker_on_invalid_json(monkeypatch):
    task_manager = MultiAgentROISearchTaskManager(
        build=True,
        use_coding_tools=False,
        image_acquisition_tool=DummyAcquireImageTool(),
        session_db_path=None,
    )
    task_manager.model = object()

    responses = [
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "acquire_image",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "user", "content": "position prompt"},
        ),
        (
            {
                "role": "assistant",
                "content": "```json\n{\"foi_present\": true}\n```",
            },
            {"role": "user", "content": "checker prompt"},
        ),
        (
            {
                "role": "assistant",
                "content": '{"foi_present": true, "fov_description": "feature appears"}',
            },
            {"role": "user", "content": "retry checker prompt"},
        ),
    ]
    prompts = []

    def fake_invoke_model_raw(message=None, **kwargs):
        prompts.append(message)
        return responses.pop(0)

    monkeypatch.setattr(task_manager, "invoke_model_raw", fake_invoke_model_raw)
    monkeypatch.setattr(task_manager, "run_feedback_loop", lambda **kwargs: None)

    task_manager.run(initial_prompt="Find a feature.")

    assert "could not be parsed or used" in prompts[2]
    assert "triple backticks" in prompts[2]
    assert task_manager.task_state.fov_description == "feature appears"


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


def test_feature_tracking_task_manager_no_longer_owns_fov_search():
    assert not hasattr(FeatureTrackingTaskManager, "run_fov_search")
