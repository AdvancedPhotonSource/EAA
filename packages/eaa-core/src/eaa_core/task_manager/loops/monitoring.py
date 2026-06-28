from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import json
import sqlite3
import time

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.message_proc import generate_openai_message, get_message_elements_as_text
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.task_manager.persistence import configure_sqlite_connection
from eaa_core.task_manager.state import ChatGraphState, TaskManagerState
from eaa_core.tool.base import BaseTool


class MonitoringTaskState(TaskManagerState):
    """State for the monitoring task graph."""

    monitoring_action: str = ""
    anomaly_criteria: str = ""
    anomaly_response_action: str = ""
    normal_response_action: str = "do nothing"
    monitoring_interval_sec: float = 0.0
    max_monitoring_actions: Optional[int] = None
    count_monitoring_action: int = 0
    anomaly_detected: bool = False
    log_path: str = ""
    intake_messages: list[dict[str, Any]] = Field(default_factory=list)
    intake_complete: bool = False
    log_initialized: bool = False


class MonitoringTaskManager(BaseTaskManager):
    """Task manager that monitors an experiment status on a timed loop."""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        tools: list[BaseTool] = (),
        monitoring_log_path: str = "monitoring_log.sqlite",
        checkpoint_db_path: Optional[str] = "checkpoint.sqlite",
        build: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a monitoring task manager.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            Configuration used to build the chat model.
        memory_config : MemoryManagerConfig, optional
            Optional long-term memory configuration.
        tools : list[BaseTool], optional
            Tools exposed to the monitoring chat graph.
        monitoring_log_path : str, default="monitoring_log.sqlite"
            SQLite database path used for monitoring logs. Relative paths are
            resolved from the current working directory.
        checkpoint_db_path : Optional[str], default="checkpoint.sqlite"
            SQLite database path used for LangGraph checkpoints.
        build : bool, default=True
            Whether to build the task manager during initialization.
        """
        self.monitoring_log_path = self.resolve_monitoring_log_path(
            monitoring_log_path,
        )
        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            tools=tools,
            checkpoint_db_path=checkpoint_db_path,
            build=build,
            *args,
            **kwargs,
        )
        self.task_state = MonitoringTaskState(log_path=self.monitoring_log_path)
        self.active_state = self.task_state

    @staticmethod
    def resolve_monitoring_log_path(monitoring_log_path: str) -> str:
        """Resolve the monitoring log path relative to the current directory."""
        path = Path(monitoring_log_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return str(path.resolve())

    def get_task_state_model(self) -> type[TaskManagerState]:
        """Return the state model used by the monitoring task graph."""
        return MonitoringTaskState

    def build_task_graph(self, checkpointer: Any = None) -> CompiledStateGraph:
        """Build the monitoring workflow graph."""
        builder = StateGraph(MonitoringTaskState)
        builder.add_node("intake_chat", self.intake_chat)
        builder.add_node("initialize_log", self.initialize_log)
        builder.add_node("monitor_once", self.monitor_once)
        builder.add_node("sleep_until_next_monitoring_action", self.sleep_until_next_monitoring_action)
        builder.add_edge(START, "intake_chat")
        builder.add_conditional_edges(
            "intake_chat",
            self.route_after_intake_chat,
        )
        builder.add_edge("initialize_log", "monitor_once")
        builder.add_conditional_edges(
            "monitor_once",
            self.route_after_monitor_once,
        )
        builder.add_edge("sleep_until_next_monitoring_action", "monitor_once")
        return builder.compile(checkpointer=checkpointer)

    def run(
        self,
        monitoring_request: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """Run the monitoring task graph.

        Parameters
        ----------
        monitoring_request : str, optional
            Initial user description of the monitoring task.
        """
        if monitoring_request is not None or not isinstance(
            self.task_state,
            MonitoringTaskState,
        ):
            self.task_state = MonitoringTaskState(
                initial_prompt=monitoring_request or "",
                log_path=self.monitoring_log_path,
            )
        elif not self.task_state.log_path:
            self.task_state.log_path = self.monitoring_log_path
        return super().run(*args, **kwargs)

    def intake_chat(self, state: MonitoringTaskState) -> dict[str, Any]:
        """Collect and parse monitoring setup details through the chat graph."""
        if not state.intake_messages and state.initial_prompt:
            state.intake_messages.append(
                generate_openai_message(
                    content=state.initial_prompt,
                    role="user",
                )
            )
        chat_state = self.invoke_fresh_chat_graph(
            system_prompt=self.build_intake_system_prompt(),
            messages=state.intake_messages,
        )
        assistant_message = chat_state.latest_response or {}
        assistant_content = self.message_content_as_text(assistant_message)
        intake_payload = self.try_parse_json_object(assistant_content)
        if intake_payload is None:
            state.intake_messages.append(assistant_message)
            user_response = self.get_user_input(
                prompt=f"{assistant_content}\n",
                display_prompt_in_webui=False,
            )
            self.append_intake_user_response(state, user_response)
            return state.model_dump()
        try:
            if not isinstance(intake_payload, dict):
                raise ValueError("The response JSON must be an object.")
            parsed = self.validate_intake_payload(intake_payload)
        except ValueError as exc:
            state.intake_messages.append(assistant_message)
            user_response = self.get_user_input(
                prompt=f"{assistant_content}\n{exc}\n",
                display_prompt_in_webui=False,
            )
            self.append_intake_user_response(state, user_response)
            return state.model_dump()

        state.monitoring_action = parsed["monitoring_action"]
        state.anomaly_criteria = parsed["anomaly_criteria"]
        state.anomaly_response_action = parsed["anomaly_response_action"]
        state.normal_response_action = parsed["normal_response_action"]
        state.monitoring_interval_sec = parsed["monitoring_interval_sec"]
        state.max_monitoring_actions = parsed["max_monitoring_actions"]
        state.log_path = self.monitoring_log_path
        state.intake_complete = True
        state.intake_messages.append(assistant_message)
        return state.model_dump()

    def route_after_intake_chat(self, state: MonitoringTaskState) -> str:
        """Route after intake either to logging setup or another intake turn."""
        if state.intake_complete:
            return "initialize_log"
        return "intake_chat"

    def append_intake_user_response(
        self,
        state: MonitoringTaskState,
        user_response: str,
    ) -> None:
        """Append and display a user reply collected during monitoring intake."""
        user_message = generate_openai_message(content=user_response, role="user")
        state.intake_messages.append(user_message)
        self.node_factory.update_message_history_for_state(
            state,
            user_message,
            update_context=False,
            update_full_history=True,
            write_to_webui=True,
        )

    def initialize_log(self, state: MonitoringTaskState) -> dict[str, Any]:
        """Create the monitoring SQLite log table if needed."""
        self.create_monitoring_log(state.log_path)
        state.log_initialized = True
        return state.model_dump()

    def monitor_once(self, state: MonitoringTaskState) -> dict[str, Any]:
        """Run one monitoring action through a fresh chat context and log it."""
        chat_state = self.invoke_fresh_chat_graph(
            system_prompt=self.build_monitoring_system_prompt(),
            messages=[
                generate_openai_message(
                    content=self.build_monitoring_prompt(state),
                    role="user",
                )
            ],
        )
        assistant_message = chat_state.latest_response or {}
        assistant_content = self.message_content_as_text(assistant_message)
        parsed = self.parse_monitoring_response(assistant_content)
        self.insert_monitoring_log(
            log_path=state.log_path,
            observations=parsed["observations"],
            observations_summary=parsed["observations_summary"],
            anomaly_detected=parsed["anomaly_detected"],
            actions_taken=parsed["actions_taken"],
            suggested_next_action=parsed["suggested_next_action"],
        )
        state.anomaly_detected = parsed["anomaly_detected"]
        state.count_monitoring_action += 1
        return state.model_dump()

    def route_after_monitor_once(self, state: MonitoringTaskState) -> str:
        """Route after a monitoring iteration."""
        if (
            state.max_monitoring_actions is not None
            and state.count_monitoring_action >= state.max_monitoring_actions
        ):
            return END
        return "sleep_until_next_monitoring_action"

    def sleep_until_next_monitoring_action(
        self,
        state: MonitoringTaskState,
    ) -> dict[str, Any]:
        """Wait for the configured interval before the next monitoring action."""
        time.sleep(state.monitoring_interval_sec)
        return state.model_dump()

    def invoke_fresh_chat_graph(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> ChatGraphState:
        """Run the chat graph once with a fresh state and custom system prompt."""
        chat_graph = self.chat_graph or self.build_chat_graph()
        previous_system_message = self.assistant_system_message
        self.assistant_system_message = system_prompt
        try:
            initial_state = ChatGraphState(
                messages=list(messages),
                full_history=list(messages),
                termination_behavior="return",
                await_user_input=False,
            )
            result = chat_graph.invoke(
                initial_state,
                context=self.memory_manager.get_runtime_context(),
            )
        finally:
            self.assistant_system_message = previous_system_message
        return ChatGraphState.model_validate(result)

    @staticmethod
    def build_intake_system_prompt() -> str:
        """Return the system prompt used for monitoring intake."""
        return (
            "The user wants you to monitor the status of an experiment "
            "instrument, data stream, state, process, or object for anomalous "
            "situations. Collect the required setup details from the user.\n\n"
            "Mandatory details:\n"
            "- The monitoring action: what should be done to obtain observables.\n"
            "- The anomaly criteria: conditions under which the state is normal.\n"
            "- The response action when an anomaly is detected.\n"
            "- The interval between monitoring actions, in seconds.\n\n"
            "Optional detail:\n"
            "- The response action when the situation is normal. If omitted, use "
            "\"do nothing\".\n"
            "- The number of times to run monitoring actions. If omitted, use null.\n\n"
            "If any mandatory detail is missing, ask the user for only the missing "
            "information. Once all details are available, return only a strict JSON "
            "object with no Markdown fences and these fields: ready, "
            "monitoring_action, anomaly_criteria, anomaly_response_action, "
            "normal_response_action, monitoring_interval_sec, "
            "max_monitoring_actions. Use null for an indefinite run."
        )

    @staticmethod
    def build_monitoring_system_prompt() -> str:
        """Return the system prompt used for each monitoring iteration."""
        return (
            "You are running one iteration of an experiment monitoring workflow. "
            "Use the available tools as needed to perform the requested monitoring "
            "and response actions. Return only a strict JSON object after all "
            "required actions are complete."
        )

    def build_monitoring_prompt(self, state: MonitoringTaskState) -> str:
        """Build the prompt for one monitoring iteration."""
        normal_action = state.normal_response_action or "do nothing"
        previous_suggestion = self.get_latest_suggested_next_action(state.log_path)
        suggestion_text = (
            f"\nPrevious suggested next action: {previous_suggestion}"
            if previous_suggestion
            else ""
        )
        return (
            "Perform one monitoring iteration with the following setup:\n"
            f"- Monitoring action: {state.monitoring_action}\n"
            f"- Normal/anomaly criteria: {state.anomaly_criteria}\n"
            f"- Action if anomaly is detected: {state.anomaly_response_action}\n"
            f"- Action if the situation is normal: {normal_action}\n"
            f"{suggestion_text}\n\n"
            "After completing the monitoring action and the appropriate response "
            "action, return only a strict JSON object with these fields: "
            "observations, observations_summary, anomaly_detected, actions_taken, "
            "suggested_next_action. `observations` must be a JSON object. "
            "`anomaly_detected` must be a boolean."
        )

    @staticmethod
    def message_content_as_text(message: dict[str, Any]) -> str:
        """Return text content from an OpenAI-style message payload."""
        if not message:
            return ""
        return get_message_elements_as_text(message)["content"].strip()

    @classmethod
    def parse_intake_response(cls, content: str) -> dict[str, Any]:
        """Parse and validate the intake JSON object."""
        data = cls.parse_json_object(content)
        return cls.validate_intake_payload(data)

    @classmethod
    def validate_intake_payload(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize a parsed intake JSON object."""
        if data.get("ready") is not True:
            raise ValueError("Monitoring setup is not ready yet.")
        required = [
            "monitoring_action",
            "anomaly_criteria",
            "anomaly_response_action",
            "monitoring_interval_sec",
        ]
        missing = [
            field
            for field in required
            if data.get(field) is None or data.get(field) == ""
        ]
        if missing:
            raise ValueError(
                "Missing mandatory monitoring setup fields: "
                f"{', '.join(missing)}."
            )
        interval = float(data["monitoring_interval_sec"])
        if interval < 0:
            raise ValueError("monitoring_interval_sec must be non-negative.")
        return {
            "monitoring_action": str(data["monitoring_action"]),
            "anomaly_criteria": str(data["anomaly_criteria"]),
            "anomaly_response_action": str(data["anomaly_response_action"]),
            "normal_response_action": str(
                data.get("normal_response_action") or "do nothing"
            ),
            "monitoring_interval_sec": interval,
            "max_monitoring_actions": cls.normalize_max_monitoring_actions(
                data.get("max_monitoring_actions")
            ),
        }

    @classmethod
    def parse_monitoring_response(cls, content: str) -> dict[str, Any]:
        """Parse and validate one monitoring iteration JSON object."""
        data = cls.parse_json_object(content)
        observations = data.get("observations")
        if observations is None:
            observations = {}
        if not isinstance(observations, dict):
            observations = {"value": observations}
        return {
            "observations": observations,
            "observations_summary": str(data.get("observations_summary") or ""),
            "anomaly_detected": bool(data.get("anomaly_detected", False)),
            "actions_taken": str(data.get("actions_taken") or ""),
            "suggested_next_action": str(data.get("suggested_next_action") or ""),
        }

    @staticmethod
    def parse_json_object(content: str) -> dict[str, Any]:
        """Parse a strict JSON object from model text."""
        try:
            data = json.loads(content.strip())
        except json.JSONDecodeError as exc:
            raise ValueError("The response was not valid JSON.") from exc
        if not isinstance(data, dict):
            raise ValueError("The response JSON must be an object.")
        return data

    @staticmethod
    def try_parse_json_object(content: str) -> Optional[Any]:
        """Return parsed JSON, or ``None`` when text is not JSON."""
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def normalize_max_monitoring_actions(value: Any) -> Optional[int]:
        """Normalize the configured monitoring action limit."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "inf",
            "infinite",
            "indefinite",
            "none",
            "null",
        }:
            return None
        count = int(value)
        if count <= 0:
            raise ValueError("max_monitoring_actions must be positive or null.")
        return count

    @staticmethod
    def create_monitoring_log(log_path: str) -> None:
        """Create the monitoring log table."""
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with configure_sqlite_connection(sqlite3.connect(path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_log (
                    time TEXT NOT NULL,
                    observations_json TEXT NOT NULL,
                    observations_summary TEXT NOT NULL,
                    anomaly_detected INTEGER NOT NULL,
                    actions_taken TEXT NOT NULL,
                    suggested_next_action TEXT NOT NULL
                )
                """
            )
            connection.commit()

    @staticmethod
    def insert_monitoring_log(
        *,
        log_path: str,
        observations: dict[str, Any],
        observations_summary: str,
        anomaly_detected: bool,
        actions_taken: str,
        suggested_next_action: str,
    ) -> None:
        """Insert one monitoring result into the SQLite log."""
        with configure_sqlite_connection(sqlite3.connect(log_path)) as connection:
            connection.execute(
                """
                INSERT INTO monitoring_log (
                    time,
                    observations_json,
                    observations_summary,
                    anomaly_detected,
                    actions_taken,
                    suggested_next_action
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(observations, default=str),
                    observations_summary,
                    int(anomaly_detected),
                    actions_taken,
                    suggested_next_action,
                ),
            )
            connection.commit()

    @staticmethod
    def get_latest_suggested_next_action(log_path: str) -> str:
        """Return the most recent suggested next action from the log."""
        if not log_path or not Path(log_path).exists():
            return ""
        with configure_sqlite_connection(sqlite3.connect(log_path)) as connection:
            row = connection.execute(
                """
                SELECT suggested_next_action
                FROM monitoring_log
                WHERE suggested_next_action != ''
                ORDER BY time DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return ""
        return str(row[0] or "")
