import json
import sqlite3

from eaa_core.task_manager.loops.monitoring import (
    MonitoringTaskManager,
    MonitoringTaskState,
)
from eaa_core.task_manager.state import ChatGraphState


def make_chat_state(content: str) -> ChatGraphState:
    """Build a completed chat state with one assistant response."""
    message = {"role": "assistant", "content": content}
    return ChatGraphState(messages=[message], full_history=[message])


def fetch_monitoring_rows(log_path):
    """Return all rows from the monitoring log."""
    with sqlite3.connect(log_path) as connection:
        return connection.execute(
            """
            SELECT observations_json, observations_summary, anomaly_detected,
                   actions_taken, suggested_next_action
            FROM monitoring_log
            ORDER BY time
            """
        ).fetchall()


def test_monitoring_task_manager_runs_one_iteration_and_logs(tmp_path, monkeypatch):
    log_path = tmp_path / "monitoring.sqlite"
    task_manager = MonitoringTaskManager(
        build=False,
        checkpoint_db_path=None,
        monitoring_log_path=str(log_path),
    )
    task_manager.task_graph = task_manager.build_task_graph()
    responses = iter(
        [
            make_chat_state(
                json.dumps(
                    {
                        "ready": True,
                        "monitoring_action": "Read the beam current.",
                        "anomaly_criteria": "Normal when current is above 10 mA.",
                        "anomaly_response_action": "Notify the operator.",
                        "normal_response_action": "do nothing",
                        "monitoring_interval_sec": 0,
                        "max_monitoring_actions": 1,
                    }
                )
            ),
            make_chat_state(
                json.dumps(
                    {
                        "observations": {"beam_current": 12.5},
                        "observations_summary": "Beam current is stable.",
                        "anomaly_detected": False,
                        "actions_taken": "No action taken.",
                        "suggested_next_action": "Continue monitoring.",
                    }
                )
            ),
        ]
    )

    monkeypatch.setattr(
        task_manager,
        "invoke_fresh_chat_graph",
        lambda **kwargs: next(responses),
    )

    task_manager.run("Monitor beam current.")

    assert isinstance(task_manager.task_state, MonitoringTaskState)
    assert task_manager.task_state.count_monitoring_action == 1
    assert task_manager.task_state.anomaly_detected is False
    rows = fetch_monitoring_rows(log_path)
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == {"beam_current": 12.5}
    assert rows[0][1:] == (
        "Beam current is stable.",
        0,
        "No action taken.",
        "Continue monitoring.",
    )


def test_monitoring_intake_reasks_for_missing_information(tmp_path, monkeypatch):
    log_path = tmp_path / "monitoring.sqlite"
    task_manager = MonitoringTaskManager(
        build=False,
        checkpoint_db_path=None,
        monitoring_log_path=str(log_path),
    )
    task_manager.task_graph = task_manager.build_task_graph()
    responses = iter(
        [
            make_chat_state("What interval should I use?"),
            make_chat_state(
                json.dumps(
                    {
                        "ready": True,
                        "monitoring_action": "Read detector counts.",
                        "anomaly_criteria": "Normal when counts are below 1000.",
                        "anomaly_response_action": "Pause acquisition.",
                        "normal_response_action": "do nothing",
                        "monitoring_interval_sec": 0,
                        "max_monitoring_actions": 1,
                    }
                )
            ),
            make_chat_state(
                json.dumps(
                    {
                        "observations": {"counts": 1200},
                        "observations_summary": "Counts are high.",
                        "anomaly_detected": True,
                        "actions_taken": "Paused acquisition.",
                        "suggested_next_action": "Inspect detector gain.",
                    }
                )
            ),
        ]
    )
    prompts = []
    published = []
    monkeypatch.setattr(task_manager, "publish_webui_message", published.append)

    monkeypatch.setattr(
        task_manager,
        "invoke_fresh_chat_graph",
        lambda **kwargs: next(responses),
    )

    def fake_get_user_input(prompt, *args, **kwargs):
        prompts.append(prompt)
        assert kwargs["display_prompt_in_webui"] is False
        return "Use a 0 second interval and run once."

    monkeypatch.setattr(task_manager, "get_user_input", fake_get_user_input)

    task_manager.run("Monitor detector counts.")

    assert len(prompts) == 1
    assert prompts[0] == "What interval should I use?\n"
    assert published == [
        {"role": "user", "content": "Use a 0 second interval and run once."}
    ]
    assert task_manager.task_state.count_monitoring_action == 1
    assert task_manager.task_state.anomaly_detected is True
    assert fetch_monitoring_rows(log_path)[0][2] == 1


def test_monitoring_task_manager_stops_after_finite_action_count(tmp_path, monkeypatch):
    log_path = tmp_path / "monitoring.sqlite"
    task_manager = MonitoringTaskManager(
        build=False,
        checkpoint_db_path=None,
        monitoring_log_path=str(log_path),
    )
    task_manager.task_graph = task_manager.build_task_graph()
    responses = iter(
        [
            make_chat_state(
                json.dumps(
                    {
                        "ready": True,
                        "monitoring_action": "Read motor position.",
                        "anomaly_criteria": "Normal when position is near 1.0.",
                        "anomaly_response_action": "Stop the motor.",
                        "normal_response_action": "do nothing",
                        "monitoring_interval_sec": 0,
                        "max_monitoring_actions": 2,
                    }
                )
            ),
            make_chat_state(
                json.dumps(
                    {
                        "observations": {"position": 1.0},
                        "observations_summary": "Position is normal.",
                        "anomaly_detected": False,
                        "actions_taken": "No action taken.",
                        "suggested_next_action": "",
                    }
                )
            ),
            make_chat_state(
                json.dumps(
                    {
                        "observations": {"position": 1.1},
                        "observations_summary": "Position is normal.",
                        "anomaly_detected": False,
                        "actions_taken": "No action taken.",
                        "suggested_next_action": "",
                    }
                )
            ),
        ]
    )

    monkeypatch.setattr(
        task_manager,
        "invoke_fresh_chat_graph",
        lambda **kwargs: next(responses),
    )

    task_manager.run("Monitor motor position.")

    assert task_manager.task_state.count_monitoring_action == 2
    assert len(fetch_monitoring_rows(log_path)) == 2


def test_monitoring_intake_accepts_null_for_indefinite_count():
    parsed = MonitoringTaskManager.parse_intake_response(
        json.dumps(
            {
                "ready": True,
                "monitoring_action": "Read beam current.",
                "anomaly_criteria": "Normal when current is above 10 mA.",
                "anomaly_response_action": "Notify the operator.",
                "normal_response_action": "do nothing",
                "monitoring_interval_sec": 5,
                "max_monitoring_actions": None,
            }
        )
    )

    assert parsed["max_monitoring_actions"] is None
