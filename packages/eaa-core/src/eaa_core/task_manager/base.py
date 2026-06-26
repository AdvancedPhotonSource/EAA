from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence
import json
import logging
import re
import shlex
import sqlite3
import time

from langgraph.graph import START, StateGraph

from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.llm.model import build_chat_model, invoke_chat_model
from eaa_core.gui.runtime import WebUIRuntimeController, WebUIRuntimeServer
from eaa_core.message_proc import (
    complete_unresponded_tool_calls,
    convert_tagged_text_to_openai_message,
    generate_openai_message,
    get_message_elements_as_text,
    get_message_preview,
    print_message,
)
from eaa_core.task_manager.commands import UserInputCommand, parse_user_input_command
from eaa_core.task_manager.memory_manager import MemoryManager
from eaa_core.task_manager.nodes import NodeFactory
from eaa_core.task_manager.persistence import (
    PrunableSqliteSaver,
    SQLiteTranscriptStore,
    configure_sqlite_connection,
)
from eaa_core.task_manager.prompts import render_prompt_template
from eaa_core.task_manager.skills import (
    SkillMetadata,
    build_skill_context_message,
    discover_skills,
    resolve_skill,
)
from eaa_core.task_manager.state import (
    ChatGraphState,
    ChatRuntimeContext,
    TaskManagerState,
    build_compatible_checkpoint_state,
    get_state_checkpoint_config,
    load_latest_checkpoint_state_from_connection,
)
from eaa_core.task_manager.tool_executor import SerialToolExecutor
from eaa_core.tool.base import BaseTool
from eaa_core.tool.coding import SandboxType
from eaa_core.tool.tool_manager import ToolManager

logger = logging.getLogger(__name__)
SKILL_SELECTION_PATTERN = re.compile(r"(?P<prefix>^|\s)/skill\s+(?P<name>\S+)")
BUILTIN_SKILL_DIRS = (str(Path(__file__).resolve().parents[1] / "skills"),)

GraphName = Literal["chat_graph", "task_graph"]
GraphInvokeCommand = Literal["completed", "chat", "return", "exit"]


@dataclass
class GraphInvokeResult:
    """Result of a graph invocation with interruption command handling."""

    final_state: Any = None
    command: GraphInvokeCommand = "completed"


def get_checkpoint_state_model(
    graph_name: GraphName,
) -> type[TaskManagerState]:
    """Return the state model for a checkpointed graph."""
    if graph_name == "chat_graph":
        return ChatGraphState
    if graph_name == "task_graph":
        return TaskManagerState
    raise ValueError(f"Unsupported graph name for checkpoint loading: {graph_name}.")


class TaskManagerAgentAdapter:
    """Compatibility adapter for code paths that still expect `task_manager.agent`."""

    def __init__(self, task_manager: "BaseTaskManager"):
        self.task_manager = task_manager
        self.tool_manager = task_manager.tool_manager

    def receive(
        self,
        message: Optional[str | dict[str, Any] | list[dict[str, Any]]] = None,
        *,
        image_path: Optional[str | list[str]] = None,
        context: Optional[list[dict[str, Any]]] = None,
        return_outgoing_message: bool = False,
    ):
        """Invoke the task manager model helper."""
        return self.task_manager.invoke_model_raw(
            message=message,
            image_path=image_path,
            context=context,
            return_outgoing_message=return_outgoing_message,
        )

    def handle_tool_call(self, message: dict[str, Any]):
        """Execute tool calls found in an assistant response."""
        return self.task_manager.tool_executor.execute_tool_calls_from_message(message)


class BaseTaskManager:
    """LangGraph-backed base task manager for EAA.

    Parameters
    ----------
    llm_config : LLMConfig, optional
        Configuration used to build the chat model.
    name : str, optional
        Human-readable task manager name used when registering or launching
        task managers from other agents.
    memory_config : Optional[MemoryManagerConfig], optional
        Configuration for the long-term memory store.
    tools : list[BaseTool], optional
        Base tools exposed to the task manager.
    skill_dirs : Optional[Sequence[str]], optional
        Directories searched for EAA skills.
    checkpoint_db_path : Optional[str], default="checkpoint.sqlite"
        Path to the SQLite database used for LangGraph checkpoints.
    session_id : str, default="default"
        Logical main-agent session id used to namespace checkpoint threads.
    transcript_db_path : Optional[str], default="transcript.sqlite"
        Path to the SQLite database used for durable WebUI transcript display.
    use_webui : bool, default=False
        Whether to enable WebUI-driven user input and WebUI display writes.
    prune_checkpoints : bool, default=True
        Whether to keep only the latest checkpoint per graph thread in the
        checkpoint database.
    build : bool, default=True
        Whether to initialize persistence, model, tools, memory, and graphs
        during construction.
    is_subagent : bool, default=False
        Whether this task manager is a subordinate agent. Subordinate agents do
        not receive the subagent-launching tool.
    """

    def __init__(
        self,
        llm_config: LLMConfig = None,
        name: Optional[str] = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        tools: list[BaseTool] = (),
        skill_dirs: Optional[Sequence[str]] = None,
        checkpoint_db_path: Optional[str] = "checkpoint.sqlite",
        session_id: str = "default",
        transcript_db_path: Optional[str] = "transcript.sqlite",
        transcript_table_name: str = "transcript_messages",
        use_webui: bool = False,
        webui_runtime_host: str = "127.0.0.1",
        webui_runtime_port: int = 8010,
        webui_upload_dir: str = ".tmp",
        runtime_controller: WebUIRuntimeController | None = None,
        runtime_conversation_id: str = "primary",
        prune_checkpoints: bool = True,
        build: bool = True,
        is_subagent: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the task manager."""
        if "session_db_path" in kwargs:
            raise ValueError(
                "`session_db_path` is no longer supported. Use "
                "`transcript_db_path` for WebUI transcript persistence and "
                "`checkpoint_db_path` for LangGraph checkpoints."
            )
        self.name = self.resolve_name(name)
        self.chat_state = ChatGraphState()
        self.task_state = TaskManagerState()
        self.active_state: TaskManagerState = self.task_state
        self.llm_config = llm_config
        if isinstance(memory_config, dict):
            memory_config = MemoryManagerConfig.from_dict(memory_config)
        self.memory_config = memory_config
        self.skill_dirs = (
            list(BUILTIN_SKILL_DIRS) if skill_dirs is None else list(skill_dirs)
        )
        self.skill_catalog: list[SkillMetadata] = discover_skills(self.skill_dirs)
        self.use_webui = use_webui
        self.coding_tool_request_approval = True
        self.prune_checkpoints = prune_checkpoints
        self.is_subagent = is_subagent
        self.checkpoint_db_path = checkpoint_db_path
        self.session_id = str(session_id).strip()
        if not self.session_id:
            raise ValueError("`session_id` must be a non-empty string.")
        self.transcript_db_path = transcript_db_path
        self.transcript_table_name = transcript_table_name
        self.transcript_store = SQLiteTranscriptStore(
            transcript_db_path,
            table_name=transcript_table_name,
        )
        self.webui_runtime_host = webui_runtime_host
        self.webui_runtime_port = webui_runtime_port
        self.webui_upload_dir = webui_upload_dir
        self.runtime_conversation_id = runtime_conversation_id
        self.runtime_controller: WebUIRuntimeController | None = runtime_controller
        self.runtime_server: WebUIRuntimeServer | None = None
        if use_webui and runtime_controller is None:
            self.runtime_controller = WebUIRuntimeController(
                self,
                upload_dir=webui_upload_dir,
            )
            self.runtime_server = WebUIRuntimeServer(
                self.runtime_controller,
                host=webui_runtime_host,
                port=webui_runtime_port,
            )
        self.memory_manager = MemoryManager(self)
        self.tool_manager = ToolManager(
            self,
            tools=tools,
            skill_dirs=self.skill_dirs,
            coding_tool_request_approval=self.coding_tool_request_approval,
            include_subagent_tool=not self.is_subagent,
        )
        self.tool_executor = SerialToolExecutor(
            approval_handler=self._request_tool_approval_via_task_manager,
            tools=self.tool_manager,
        )
        self.tool_manager.bind_executor(self.tool_executor)
        self.model = None
        self.agent = TaskManagerAgentAdapter(self)
        self.node_factory = NodeFactory(self)
        if not getattr(self, "assistant_system_message", None):
            self.assistant_system_message = self.get_default_system_prompt()
        elif self.is_subagent:
            self.assistant_system_message = "\n\n".join(
                [self.assistant_system_message, self.get_subagent_system_prompt()]
            )
        self.chat_graph = None
        self.task_graph = None
        self.checkpoint_connections: dict[tuple[str, str], sqlite3.Connection] = {}
        self.checkpoint_graphs: dict[tuple[str, str], Any] = {}

        if build:
            self.build()

    @classmethod
    def get_default_name(cls) -> str:
        """Return the default task manager name."""
        return BaseTool.camel_to_snake(cls.__name__)

    @classmethod
    def resolve_name(cls, name: Optional[str]) -> str:
        """Return a validated task manager name."""
        resolved_name = cls.get_default_name() if name is None else str(name).strip()
        if not resolved_name:
            raise ValueError("`name` must be a non-empty string when provided.")
        return resolved_name

    def get_default_system_prompt(self) -> str:
        """Return the default system prompt for the task manager."""
        prompt = render_prompt_template(
            "eaa_core.task_manager.prompts",
            "system_base.md",
            {
                "available_skills_text": self.format_available_skills_for_prompt(),
            },
        )
        if self.is_subagent:
            prompt = "\n\n".join(
                [
                    prompt,
                    self.get_subagent_system_prompt(),
                ]
            )
        return prompt

    def get_subagent_system_prompt(self) -> str:
        """Return the prompt addendum for subordinate task managers."""
        return render_prompt_template(
            "eaa_core.task_manager.prompts",
            "subagent.md",
            {},
        )

    def format_available_skills_for_prompt(self) -> str:
        """Return the available-skill summary injected into the system prompt."""
        if not self.skill_catalog:
            return "No skills are currently available."
        lines = ["Available skills:"]
        lines.extend(
            f"- {skill.name}: {skill.description} [{skill.path}]"
            for skill in self.skill_catalog
        )
        return "\n".join(lines)

    def build_selected_skill_messages(self, skill_name: str) -> list[dict[str, object]]:
        """Return context messages for an explicitly selected skill.

        Parameters
        ----------
        skill_name : str
            Skill name or path selected with ``/skill``.

        Returns
        -------
        list[dict[str, object]]
            Messages containing only the selected skill's ``SKILL.md``.
        """
        skill = resolve_skill(self.skill_catalog, skill_name)
        if skill is None:
            names = ", ".join(item.name for item in self.skill_catalog) or "none"
            raise ValueError(f"Unknown skill requested: {skill_name}. Available skills: {names}")
        return [build_skill_context_message(skill)]

    def expand_skill_command_in_text(self, text: str) -> list[dict[str, object]]:
        """Build context messages for ``/skill <name>`` user text.

        Parameters
        ----------
        text : str
            User input text that may contain a skill selection command.

        Returns
        -------
        list[dict[str, object]]
            Skill context message followed by any remaining user message.
        """
        command = parse_user_input_command(text)
        if command.kind != "skill" or not command.argument:
            match = SKILL_SELECTION_PATTERN.search(text)
            if match is None:
                return [convert_tagged_text_to_openai_message(text, role="user")]
            skill_name = match.group("name")
            remaining_text = (
                text[: match.start()]
                + match.group("prefix")
                + text[match.end() :]
            ).strip()
            remaining_text = re.sub(r"\s{2,}", " ", remaining_text)
            messages = self.build_selected_skill_messages(skill_name)
            if remaining_text:
                messages.append(convert_tagged_text_to_openai_message(remaining_text, role="user"))
            return messages
        messages = self.build_selected_skill_messages(command.argument)
        if command.text:
            messages.append(convert_tagged_text_to_openai_message(command.text, role="user"))
        return messages

    def set_active_state(
        self,
        state: TaskManagerState,
        graph_name: Optional[GraphName] = None,
    ) -> None:
        """Set the active state and update the matching state holder.

        Parameters
        ----------
        state : TaskManagerState
            State instance to make active.
        graph_name : Optional[GraphName], optional
            Graph slot that owns the state. When omitted, the slot is inferred
            from the concrete state model for backwards compatibility.
        """
        if not isinstance(state, TaskManagerState):
            state = TaskManagerState.model_validate(state)
        if graph_name is None:
            graph_name = self.infer_graph_name_for_state(state)
        if graph_name == "chat_graph":
            self.chat_state = ChatGraphState.model_validate(state.model_dump())
            self.active_state = self.chat_state
        elif graph_name == "task_graph":
            self.task_state = state
            self.active_state = self.task_state
        else:
            raise ValueError(f"Unsupported active graph name: {graph_name}.")

    def infer_graph_name_for_state(self, state: TaskManagerState) -> GraphName:
        """Infer a graph slot from a state model."""
        if isinstance(state, ChatGraphState):
            return "chat_graph"
        return "task_graph"

    @property
    def context(self) -> list[dict[str, Any]]:
        """Return the canonical active conversation context."""
        return self.active_state.messages

    @context.setter
    def context(self, value: list[dict[str, Any]]) -> None:
        """Replace the canonical active conversation context."""
        self.active_state.messages = list(value)

    @property
    def full_history(self) -> list[dict[str, Any]]:
        """Return the canonical full transcript."""
        return self.active_state.full_history

    @full_history.setter
    def full_history(self, value: list[dict[str, Any]]) -> None:
        """Replace the canonical full transcript."""
        self.active_state.full_history = list(value)

    def recover_active_state_from_checkpoint(
        self,
        graph: Any,
        checkpoint_config: Optional[dict[str, Any]],
        graph_name: GraphName,
        state_model: type[TaskManagerState],
    ) -> bool:
        """Restore active state from the latest checkpoint snapshot.

        Parameters
        ----------
        graph : Any
            Compiled LangGraph instance used for the interrupted run.
        checkpoint_config : Optional[dict[str, Any]]
            Checkpoint configuration containing the graph thread id.
        graph_name : {"chat_graph", "task_graph"}
            Graph slot that owns the recovered state.
        state_model : type[TaskManagerState]
            State model used to validate the checkpoint payload.

        Returns
        -------
        bool
            ``True`` when a checkpoint snapshot was recovered, otherwise
            ``False``.
        """
        if graph is None or checkpoint_config is None:
            return False
        snapshot = graph.get_state(checkpoint_config)
        if (
            snapshot.created_at is None
            or snapshot.values is None
            or len(snapshot.values) == 0
        ):
            return False
        self.set_active_state(
            state_model.model_validate(snapshot.values),
            graph_name,
        )
        return True

    def format_interruption_message_content(
        self,
        base_message: str,
        checkpoint_recovered: bool,
    ) -> str:
        """Build a user-facing interruption message with state recovery details.

        Parameters
        ----------
        base_message : str
            First sentence describing the interrupted graph.
        checkpoint_recovered : bool
            Whether active state was restored from a checkpoint before
            composing the message.

        Returns
        -------
        str
            Message content including the latest recovered message preview.
        """
        lines = [base_message]
        if not checkpoint_recovered:
            lines.append(
                "Warning: checkpoint recovery was unavailable, so the restored "
                "conversation may miss messages produced during the interrupted graph run."
            )
        latest_message = self.active_state.get_latest_message()
        if latest_message is not None:
            preview = get_message_preview(latest_message, max_characters=100)
            lines.append(f"Last recovered message: {preview}")
        lines.append("You can now provide new instructions.")
        return "\n".join(lines)

    def append_interruption_resume_input(
        self,
        base_message: str,
        checkpoint_recovered: bool,
    ) -> GraphInvokeCommand:
        """Record an interruption notice and handle user resume input.

        Parameters
        ----------
        base_message : str
            First sentence describing the interrupted graph.
        checkpoint_recovered : bool
            Whether active state was restored from a checkpoint before
            composing the notice.
        Returns
        -------
        GraphInvokeCommand
            ``"completed"`` when user text was appended and the graph should
            resume. Other values are slash commands for the caller to handle.
        """
        self.resolve_unmatched_tool_calls_for_interruption()
        interrupt_message = generate_openai_message(
            content=self.format_interruption_message_content(
                base_message,
                checkpoint_recovered=checkpoint_recovered,
            ),
            role="system",
        )
        if not self.use_webui:
            print_message(interrupt_message)
        self.update_message_history(
            interrupt_message,
            update_context=True,
            update_full_history=True,
        )
        while True:
            user_input = self.get_user_input(
                prompt="Enter instructions to resume the interrupted graph: ",
                display_prompt_in_webui=self.use_webui,
            )
            command = parse_user_input_command(user_input)
            if self.handle_runtime_command(command):
                continue
            if command.kind == "skill" and not command.argument:
                self.display_available_skills()
                continue
            if command.kind == "skill":
                try:
                    skill_messages = self.build_selected_skill_messages(command.argument)
                except ValueError as exc:
                    self.record_system_message(
                        str(exc),
                        update_context=True,
                        write_to_webui=True,
                    )
                    continue
                for skill_message in skill_messages:
                    self.update_message_history(
                        skill_message,
                        update_context=True,
                        update_full_history=True,
                    )
                if command.text:
                    user_message = generate_openai_message(content=command.text, role="user")
                    if not self.use_webui:
                        print_message(user_message)
                    self.update_message_history(
                        user_message,
                        update_context=True,
                        update_full_history=True,
                    )
                self.active_state.await_user_input = False
                return "completed"
            if command.kind in {"exit", "return", "chat"}:
                return command.kind
            if command.kind == "monitor":
                self.record_system_message(
                    "The `/monitor` command is only supported from chat mode.",
                    update_context=True,
                    write_to_webui=True,
                )
                continue
            for user_message in self.expand_skill_command_in_text(command.text):
                if not self.use_webui:
                    print_message(user_message)
                self.update_message_history(
                    user_message,
                    update_context=True,
                    update_full_history=True,
                )
            self.active_state.await_user_input = False
            return "completed"

    def resolve_unmatched_tool_calls_for_interruption(self) -> None:
        """Append synthetic tool responses for interrupted tool calls."""
        original_context_length = len(self.active_state.messages)
        complete_unresponded_tool_calls(
            self.active_state.messages,
            placeholder_content=(
                "The tool call was interrupted before it completed. "
                "No tool result is available. Please call the tool again if it is still needed."
            ),
        )
        self.active_state.full_history.extend(
            self.active_state.messages[original_context_length:]
        )

    def get_interrupted_checkpoint_resume_node(
        self,
        graph: Any,
        checkpoint_config: Optional[dict[str, Any]],
    ) -> Optional[str]:
        """Return the graph node to mark as completed after interruption recovery."""
        if graph is None or checkpoint_config is None:
            return None
        snapshot = graph.get_state(checkpoint_config)
        next_nodes = getattr(snapshot, "next", ()) or ()
        if "execute_tools" in next_nodes:
            return "execute_tools"
        return None

    def invoke_graph_with_interruption_recovery(
        self,
        graph: Any,
        graph_input: Optional[TaskManagerState],
        graph_kwargs: dict[str, Any],
        graph_name: GraphName,
        state_model: type[TaskManagerState],
        interruption_message: str,
    ) -> GraphInvokeResult:
        """Invoke a graph and resume it in place after Ctrl-C interruptions.

        Parameters
        ----------
        graph : Any
            Compiled LangGraph instance to invoke.
        graph_input : Optional[TaskManagerState]
            Initial graph input. Use ``None`` when resuming from checkpoint
            state.
        graph_kwargs : dict[str, Any]
            Keyword arguments passed to ``graph.invoke``.
        graph_name : {"chat_graph", "task_graph"}
            Graph slot that owns the recovered state.
        state_model : type[TaskManagerState]
            State model used to validate checkpoint payloads.
        interruption_message : str
            First sentence of the interruption notice.

        Returns
        -------
        Any
            Graph invocation result, including any slash command that stopped
            the resume loop.
        """
        active_input = graph_input
        active_kwargs = dict(graph_kwargs)
        while True:
            try:
                if self.runtime_controller is not None:
                    self.runtime_controller.check_interrupt()
                    self.runtime_controller.set_status(
                        "running",
                        input_requested=False,
                        conversation_id=self.runtime_conversation_id,
                    )
                result = graph.invoke(active_input, **active_kwargs)
                if self.runtime_controller is not None:
                    self.runtime_controller.set_status(
                        "idle",
                        input_requested=False,
                        conversation_id=self.runtime_conversation_id,
                    )
                return GraphInvokeResult(final_state=result, command="completed")
            except KeyboardInterrupt:
                if self.runtime_controller is not None:
                    self.runtime_controller.set_status(
                        "interrupted",
                        input_requested=False,
                        conversation_id=self.runtime_conversation_id,
                    )
                checkpoint_config = active_kwargs.get("config")
                checkpoint_recovered = self.recover_active_state_from_checkpoint(
                    graph=graph,
                    checkpoint_config=checkpoint_config,
                    graph_name=graph_name,
                    state_model=state_model,
                )
                resume_as_node = self.get_interrupted_checkpoint_resume_node(
                    graph=graph,
                    checkpoint_config=checkpoint_config,
                )
                command = self.append_interruption_resume_input(
                    interruption_message,
                    checkpoint_recovered=checkpoint_recovered,
                )
                if command != "completed":
                    return GraphInvokeResult(command=command)
                if checkpoint_config is None:
                    active_input = self.active_state
                    continue
                update_kwargs = {}
                if resume_as_node is not None:
                    update_kwargs["as_node"] = resume_as_node
                graph.update_state(
                    checkpoint_config,
                    self.active_state.model_dump(),
                    **update_kwargs,
                )
                active_input = None

    def build(self, *args, **kwargs):
        """Build persistence, model, tools, and graphs."""
        self.build_db()
        self.build_model()
        self.build_tools()
        self.build_memory_store()
        self.chat_graph = self.build_chat_graph()
        self.task_graph = self.build_task_graph()

    def build_db(self, *args, **kwargs):
        """Initialize message persistence and optionally hydrate prior messages."""
        self.transcript_store.connect()
        if self.runtime_controller is not None:
            self.runtime_controller.build()

    def start_webui_runtime(self) -> None:
        """Start the agent-side WebUI runtime server."""
        if not self.use_webui:
            raise RuntimeError(
                "Cannot start WebUI runtime when `use_webui` is False. "
                "Initialize the task manager with `use_webui=True`."
            )
        if self.runtime_server is None:
            return
        self.transcript_store.connect()
        self.runtime_controller.build()
        self.runtime_server.start()

    def stop_webui_runtime(self) -> None:
        """Stop the agent-side WebUI runtime server."""
        if self.runtime_server is not None:
            self.runtime_server.stop()

    def build_model(self, *args, **kwargs):
        """Build the chat model if an LLM config is provided."""
        if self.llm_config is None:
            logger.info("Skipping model build because `llm_config` is not provided.")
            return
        self.model = build_chat_model(self.llm_config)

    def build_tools(self, *args, **kwargs):
        """Register local tools and built-in helper tools."""
        self.tool_manager.build()

    def build_memory_store(self) -> None:
        """Build the long-term memory store used by the chat graph."""
        self.memory_manager.build_store()

    def build_task_graph(self, checkpointer: Any = None):
        """Build the task-manager-specific graph if needed."""
        return None

    def resolve_checkpoint_storage(
        self,
        graph_name: str,
        checkpoint_db_path: Optional[str] = None,
    ) -> tuple[str, str]:
        """Resolve the shared checkpoint database path and graph thread id.

        Parameters
        ----------
        graph_name : str
            Graph identifier being checkpointed.
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and persistence instead
            of ``self.checkpoint_db_path``.

        Returns
        -------
        tuple[str, str]
            Resolved SQLite database path and checkpoint thread id.
        """
        checkpoint_path = checkpoint_db_path or self.checkpoint_db_path
        if checkpoint_path is None:
            raise ValueError(
                "Checkpointing requires `checkpoint_db_path`."
            )
        shared_path = str(Path(checkpoint_path).expanduser().resolve())
        return shared_path, f"{self.session_id}:{graph_name}"

    def get_checkpointed_graph(
        self,
        graph_name: GraphName,
        checkpoint_db_path: Optional[str] = None,
        load_state: bool = True,
    ) -> tuple[Any, Optional[dict[str, Any]], Optional[TaskManagerState]]:
        """Return a checkpointed graph together with config and optional state.

        Parameters
        ----------
        graph_name : {"chat_graph", "task_graph"}
            Graph identifier whose persistent graph should be returned.
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and persistence instead
            of ``self.checkpoint_db_path``.
        load_state : bool, default=True
            Whether to also load and validate the latest checkpointed state.

        Returns
        -------
        tuple[Any, Optional[dict[str, Any]], Optional[TaskManagerState]]
            Compiled graph, checkpoint config, and the latest checkpointed
            state if ``load_state`` is True and a checkpoint exists.
        """
        resolved_checkpoint_path, checkpoint_thread_id = self.resolve_checkpoint_storage(
            graph_name,
            checkpoint_db_path=checkpoint_db_path,
        )
        cache_key = (graph_name, resolved_checkpoint_path)
        if cache_key not in self.checkpoint_graphs:
            connection = configure_sqlite_connection(
                sqlite3.connect(resolved_checkpoint_path, check_same_thread=False)
            )
            saver = PrunableSqliteSaver(
                connection,
                prune_checkpoints=self.prune_checkpoints,
            )
            saver.setup()
            if graph_name == "chat_graph":
                graph = self.build_chat_graph(checkpointer=saver)
            elif graph_name == "task_graph":
                graph = self.build_task_graph(checkpointer=saver)
            else:
                raise ValueError(f"Unsupported graph name for checkpointing: {graph_name}.")
            self.checkpoint_connections[cache_key] = connection
            self.checkpoint_graphs[cache_key] = graph
        graph = self.checkpoint_graphs[cache_key]
        config = get_state_checkpoint_config(
            checkpoint_thread_id=checkpoint_thread_id,
        )
        loaded_state: Optional[TaskManagerState] = None
        if load_state:
            state_model = get_checkpoint_state_model(graph_name)
            snapshot = graph.get_state(config)
            if (
                snapshot.created_at is not None
                and snapshot.values is not None
                and len(snapshot.values) > 0
            ):
                loaded_state = state_model.model_validate(snapshot.values)
            else:
                latest_state = load_latest_checkpoint_state_from_connection(
                    connection=self.checkpoint_connections[cache_key],
                    prune_checkpoints=self.prune_checkpoints,
                )
                if latest_state is not None:
                    loaded_state = build_compatible_checkpoint_state(
                        state_model.__name__,
                        latest_state,
                    )
        return graph, config, loaded_state

    def _collect_tool_names(self, tools: list[BaseTool]) -> set[str]:
        """Collect model-visible exposed tool names from tool objects."""
        return self.tool_manager._collect_tool_names(tools)

    def handle_runtime_command(self, command: UserInputCommand) -> bool:
        """Apply runtime configuration commands.

        Parameters
        ----------
        command : UserInputCommand
            Parsed user command.

        Returns
        -------
        bool
            ``True`` when the command was recognized and consumed.
        """
        if command.kind == "set_coding_tool_approval":
            try:
                request_approval = self._parse_bool_command_argument(
                    command.argument,
                    command_name="/setcodingtoolapproval",
                )
            except ValueError as exc:
                self.record_system_message(str(exc), update_context=True, write_to_webui=True)
                return True
            self.tool_manager.set_coding_tool_request_approval(request_approval)
            state = "enabled" if request_approval else "disabled"
            self.record_system_message(
                f"Coding tool approval is now {state}.",
                update_context=True,
                write_to_webui=True,
            )
            return True
        if command.kind == "set_coding_tool_sandbox_type":
            try:
                sandbox_type, visible_dirs = self._parse_sandbox_type_command_argument(
                    command.argument
                )
            except ValueError as exc:
                self.record_system_message(str(exc), update_context=True, write_to_webui=True)
                return True
            self.tool_manager.set_coding_tool_sandbox_type(
                sandbox_type,
                visible_dirs=visible_dirs,
            )
            sandbox_label = "None" if sandbox_type is None else sandbox_type
            visible_label = ", ".join(visible_dirs) if visible_dirs else "current working directory only"
            self.record_system_message(
                (
                    f"Coding tool sandbox type is now {sandbox_label}. "
                    f"Bubblewrap visible dirs: {visible_label}."
                ),
                update_context=True,
                write_to_webui=True,
            )
            return True
        return False

    @staticmethod
    def _parse_bool_command_argument(argument: str, *, command_name: str) -> bool:
        """Parse a boolean slash-command argument."""
        tokens = shlex.split(argument)
        if len(tokens) != 1:
            raise ValueError(f"Usage: `{command_name} true|false`.")
        value = tokens[0].lower()
        if value in {"true", "yes", "on", "1"}:
            return True
        if value in {"false", "no", "off", "0"}:
            return False
        raise ValueError(f"Usage: `{command_name} true|false`.")

    @staticmethod
    def _parse_sandbox_type_command_argument(
        argument: str,
    ) -> tuple[SandboxType, Optional[list[str]]]:
        """Parse coding-tool sandbox type and optional visible dirs."""
        tokens = shlex.split(argument)
        if not tokens:
            raise ValueError(
                "Usage: `/setcodingtoolsandboxtype none|bubblewrap|container "
                "[visible_dir ...]`."
            )
        sandbox_value = tokens[0].lower()
        if sandbox_value == "none":
            sandbox_type: SandboxType = None
        elif sandbox_value in {"bubblewrap", "container"}:
            sandbox_type = sandbox_value
        else:
            raise ValueError(
                "Sandbox type must be one of: none, bubblewrap, container."
            )
        visible_dirs = tokens[1:] or None
        return sandbox_type, visible_dirs

    def register_tools(self, tools: BaseTool | list[BaseTool]) -> None:
        """Register one or more tools with the serial executor."""
        self.tool_manager.register_tools(tools)

    def record_system_message(
        self,
        content: str,
        image_path: Optional[str | list[str]] = None,
        update_context: bool = True,
        write_to_webui: bool = True,
    ) -> None:
        """Append a system message to history.

        Parameters
        ----------
        content : str
            Message text to append.
        image_path : Optional[str | list[str]], optional
            Optional image path or paths to attach to the system message.
        update_context : bool, default=False
            Whether to append the system message to the active model context.
        write_to_webui : bool, default=True
            Whether to publish the message to live WebUI clients.
        """
        self.update_message_history(
            generate_openai_message(content=content, role="system", image_path=image_path),
            update_context=update_context,
            update_full_history=True,
            write_to_webui=write_to_webui,
        )

    def update_message_history(
        self,
        message: Dict[str, Any],
        update_context: bool = True,
        update_full_history: bool = True,
        write_to_webui: bool = True,
    ) -> None:
        """Append a message to in-memory history and durable transcript storage.

        Parameters
        ----------
        message : Dict[str, Any]
            Message payload to append.
        update_context : bool, default=True
            Whether to append the message to the active context.
        update_full_history : bool, default=True
            Whether to append the message to the transcript history.
        write_to_webui : bool, default=True
            Whether to publish the message to live WebUI clients.
        """
        if update_context:
            self.active_state.messages.append(message)
        if write_to_webui:
            self.publish_webui_message(message)
        if update_full_history:
            self.active_state.full_history.append(message)
            self.record_transcript_message(message)

    def publish_webui_message(self, message: Dict[str, Any]) -> None:
        """Publish a display-only message to live WebUI clients.

        Parameters
        ----------
        message : Dict[str, Any]
            Message payload to expose through the WebUI.
        """
        if self.runtime_controller is not None:
            if isinstance(message.get("content"), list) or isinstance(message.get("tool_calls"), list):
                display_message = get_message_elements_as_text(message)
                display_message["content"] = display_message["content"].strip()
                image_urls = display_message.get("image") or []
                if isinstance(image_urls, list):
                    display_message["images"] = image_urls
                    display_message["image"] = image_urls[0] if image_urls else None
                if self.runtime_conversation_id == "primary":
                    self.runtime_controller.publish_message(display_message)
                else:
                    self.runtime_controller.publish_message(
                        display_message,
                        conversation_id=self.runtime_conversation_id,
                    )
            else:
                if self.runtime_conversation_id == "primary":
                    self.runtime_controller.publish_message(message)
                else:
                    self.runtime_controller.publish_message(
                        message,
                        conversation_id=self.runtime_conversation_id,
                    )

    def record_transcript_message(self, message: Dict[str, Any]) -> None:
        """Persist one transcript message."""
        self.transcript_store.append_message(message)

    def get_user_input(
        self,
        prompt: str = "Enter a message: ",
        display_prompt_in_webui: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """Get user input from the terminal or the WebUI runtime."""
        if self.runtime_controller is not None:
            logger.info("Waiting for user input through the WebUI runtime.")
            if display_prompt_in_webui:
                self.update_message_history(
                    {"role": "system", "content": prompt},
                    update_context=False,
                    update_full_history=False,
                    write_to_webui=True,
                )
            return self.runtime_controller.request_input(
                prompt,
                conversation_id=self.runtime_conversation_id,
            )
        self.set_user_input_requested(True)
        message = input(prompt)
        self.set_user_input_requested(False)
        return message

    def set_user_input_requested(self, requested: bool) -> None:
        """Update the WebUI pending-input status flag."""
        if self.runtime_controller is not None:
            self.runtime_controller.set_status(
                "waiting_for_input" if requested else "running",
                input_requested=requested,
                conversation_id=self.runtime_conversation_id,
            )

    def _request_tool_approval_via_task_manager(self, tool_name: str, tool_kwargs: Dict[str, Any]) -> bool:
        """Relay tool approval requests through the task manager input path."""
        prompt = (
            f"Tool '{tool_name}' requires approval before execution.\n"
            f"Arguments: {json.dumps(tool_kwargs, default=str)}\n"
            "Approve? [y/N]: "
        )
        if self.runtime_controller is not None:
            return self.runtime_controller.request_approval(
                tool_name,
                tool_kwargs,
                conversation_id=self.runtime_conversation_id,
            )
        response = self.get_user_input(prompt, display_prompt_in_webui=False)
        return response.strip().lower() in {"y", "yes"}

    def get_model_messages(self, context: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
        """Return the message list sent to the model, including the system prompt."""
        messages = []
        if self.assistant_system_message:
            messages.append(generate_openai_message(content=self.assistant_system_message, role="system"))
        if context is not None:
            messages.extend(context)
        return messages

    def invoke_model_raw(
        self,
        message: Optional[str | dict[str, Any] | list[dict[str, Any]]] = None,
        *,
        image_path: Optional[str | list[str]] = None,
        context: Optional[list[dict[str, Any]]] = None,
        return_outgoing_message: bool = False,
    ):
        """Invoke the model without mutating task-manager state.

        Parameters
        ----------
        message : str or dict or list of dict, optional
            Message payload to append before invoking the model.
        image_path : str or list of str, optional
            Image path payload to append before invoking the model.
        context : list of dict, optional
            Explicit conversation context to send to the model. If omitted,
            the active task-manager context is used.
        return_outgoing_message : bool, default=False
            Whether to return the synthesized outgoing message alongside the
            assistant response.

        Returns
        -------
        dict or tuple[dict, dict | list[dict] | None]
            Assistant response, or `(response, outgoing_message)` when
            `return_outgoing_message` is `True`.
        """
        if self.model is None:
            raise RuntimeError("No model is configured for this task manager.")
        effective_context = list(context) if context is not None else list(self.context)
        outgoing_message = None
        if message is not None:
            if isinstance(message, str):
                if image_path is None:
                    outgoing_message = convert_tagged_text_to_openai_message(message, role="user")
                else:
                    outgoing_message = generate_openai_message(
                        content=message,
                        role="user",
                        image_path=image_path,
                    )
                effective_context.append(outgoing_message)
            elif isinstance(message, dict):
                outgoing_message = message
                effective_context.append(outgoing_message)
            elif isinstance(message, list):
                outgoing_message = list(message)
                effective_context.extend(message)
            else:
                raise ValueError("Unsupported message payload type.")
        elif image_path is not None:
            outgoing_message = generate_openai_message(content="", role="user", image_path=image_path)
            effective_context.append(outgoing_message)

        response = invoke_chat_model(
            self.model,
            messages=self.get_model_messages(effective_context),
            tool_schemas=self.tool_executor.list_tool_schemas(),
        )
        if return_outgoing_message:
            return response, outgoing_message
        return response

    def display_available_skills(self) -> str:
        """Display skills available to the task manager."""
        if not self.skill_catalog:
            text = "No skills are available."
        else:
            text = "\n".join(
                ["Skills available to the agent:"]
                + [
                    f"{index}. {skill.name} - {skill.description} [{skill.path}]"
                    for index, skill in enumerate(self.skill_catalog, start=1)
                ]
            )
        if self.use_webui:
            self.publish_webui_message({"role": "system", "content": text})
        else:
            print(text)
        return text

    def enter_monitoring_mode(self, task_description: str):
        """Parse and launch a monitoring workflow."""
        parsing_prompt = (
            "Parse the following task description and return a JSON object with:\n"
            "- task_description: str\n"
            "- time_interval: float\n"
            "Return only the JSON object.\n"
            f"Task description: {task_description}\n"
        )
        local_context: list[dict[str, Any]] = []
        while True:
            response, outgoing = self.invoke_model_raw(
                parsing_prompt,
                context=local_context,
                return_outgoing_message=True,
            )
            self.update_message_history(outgoing, update_context=False, update_full_history=True)
            self.update_message_history(response, update_context=False, update_full_history=True)
            local_context.extend([outgoing, response])
            try:
                parsed = json.loads(response["content"])
            except json.JSONDecodeError:
                parsing_prompt = self.get_user_input(
                    prompt=f"Failed to parse the task description. Please try again. {response['content']}",
                    display_prompt_in_webui=self.use_webui,
                )
                continue
            break
        self.run_monitoring(
            task_description=parsed["task_description"],
            time_interval=parsed["time_interval"],
        )

    def run_monitoring(
        self,
        task_description: str,
        time_interval: float,
        initial_prompt: Optional[str] = None,
    ):
        """Run a periodic monitoring loop."""
        if initial_prompt is None:
            initial_prompt = (
                "You are given the following monitoring task: "
                f"{task_description}\n"
                "Add TERMINATE if everything is fine or fixed. Add NEED HUMAN if immediate input is required."
            )
        while True:
            try:
                self.run_conversation(message=initial_prompt, termination_behavior="return")
                time.sleep(time_interval)
            except KeyboardInterrupt:
                self.update_message_history(
                    generate_openai_message(
                        content="Keyboard interrupt detected. Terminating monitoring task.",
                        role="system",
                    )
                )
                return

    def prerun_check(self, *args, **kwargs) -> bool:
        """Run preflight validation before execution."""
        return True

    def run(self, *args, **kwargs) -> None:
        """Run the task graph using the current state as input.

        Raises
        ------
        ValueError
            If the task manager does not define a runnable task graph.
        """
        graph = self.task_graph
        graph_kwargs: dict[str, Any] = {}
        if self.checkpoint_db_path is not None:
            graph, checkpoint_config, _ = self.get_checkpointed_graph(
                "task_graph",
                load_state=False,
            )
            self.task_graph = graph
            graph_kwargs["config"] = checkpoint_config
        if graph is None:
            raise ValueError("The task manager does not define a runnable task graph.")
        initial_state = self.task_state
        initial_state.copy_messages_and_history_from_state(self.active_state)
        self.set_active_state(initial_state, "task_graph")
        invoke_result = self.invoke_graph_with_interruption_recovery(
            graph=graph,
            graph_input=initial_state,
            graph_kwargs=graph_kwargs,
            graph_name="task_graph",
            state_model=type(initial_state),
            interruption_message="Keyboard interrupt detected. The current task was interrupted.",
        )
        if invoke_result.command == "chat":
            self.run_conversation(
                termination_behavior="user",
            )
            return
        if invoke_result.command != "completed":
            return
        final_state = invoke_result.final_state
        state_model = type(initial_state)
        if not issubclass(state_model, TaskManagerState):
            state_model = TaskManagerState
        self.set_active_state(state_model.model_validate(final_state), "task_graph")
        if bool(getattr(self.task_state, "chat_requested", False)):
            self.run_conversation(
                termination_behavior="user",
            )

    def run_from_checkpoint(self, checkpoint_db_path: Optional[str] = None) -> None:
        """Resume a task manager from a task-graph checkpoint.

        Parameters
        ----------
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and updates instead of
            ``self.checkpoint_db_path``.
        """
        self.prerun_check()
        graph, checkpoint_config, loaded_state = self.get_checkpointed_graph(
            "task_graph",
            checkpoint_db_path=checkpoint_db_path,
        )
        snapshot = graph.get_state(checkpoint_config)
        fallback_loaded = (
            snapshot.created_at is None
            or snapshot.values is None
            or len(snapshot.values) == 0
        )
        if loaded_state is None:
            resolved_checkpoint_path, _ = self.resolve_checkpoint_storage(
                "task_graph",
                checkpoint_db_path=checkpoint_db_path,
            )
            raise ValueError(
                f"No task-graph checkpoint found in shared checkpoint DB "
                f"{resolved_checkpoint_path}."
            )
        self.set_active_state(loaded_state, "task_graph")
        self.task_graph = graph
        if graph is None or checkpoint_config is None:
            raise ValueError("The task manager does not define a checkpointable task graph.")
        graph_input = loaded_state if fallback_loaded else None
        state_model = type(loaded_state)
        if not issubclass(state_model, TaskManagerState):
            state_model = TaskManagerState
        invoke_result = self.invoke_graph_with_interruption_recovery(
            graph=graph,
            graph_input=graph_input,
            graph_kwargs={"config": checkpoint_config},
            graph_name="task_graph",
            state_model=state_model,
            interruption_message="Keyboard interrupt detected. The current task was interrupted.",
        )
        if invoke_result.command == "chat":
            self.run_conversation(
                termination_behavior="user",
            )
            return
        if invoke_result.command != "completed":
            return
        final_state = invoke_result.final_state
        self.set_active_state(state_model.model_validate(final_state), "task_graph")
        if bool(getattr(self.task_state, "chat_requested", False)):
            self.run_conversation(
                termination_behavior="user",
            )

    def build_chat_graph(self, checkpointer: Any = None):
        """Build the base chat graph."""
        node_factory = self.node_factory
        builder = StateGraph(ChatGraphState, context_schema=ChatRuntimeContext)
        builder.add_node(
            "await_or_ingest_user_input",
            node_factory.await_or_ingest_user_input,
        )
        builder.add_node(
            "call_model",
            node_factory.call_model,
            input_schema=ChatGraphState,
        )
        builder.add_node(
            "execute_tools",
            node_factory.execute_tools,
            input_schema=ChatGraphState,
        )
        builder.add_node(
            "image_followup",
            node_factory.image_followup,
            input_schema=ChatGraphState,
        )
        builder.add_node(
            "finalize_round",
            node_factory.finalize_chat_round,
            input_schema=ChatGraphState,
        )
        builder.add_edge(START, "await_or_ingest_user_input")
        builder.add_conditional_edges(
            "await_or_ingest_user_input",
            node_factory.route_after_chat_input,
        )
        builder.add_conditional_edges(
            "call_model",
            node_factory.route_after_chat_response,
        )
        builder.add_conditional_edges(
            "execute_tools",
            node_factory.route_after_tool_execution,
        )
        builder.add_edge("image_followup", "finalize_round")
        builder.add_conditional_edges(
            "finalize_round",
            node_factory.route_after_chat_round,
        )
        return builder.compile(checkpointer=checkpointer)

    def run_conversation(
        self,
        message: Optional[str | Dict[str, Any] | list[Dict[str, Any]]] = None,
        max_agent_iterations: Optional[int] = None,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_last_images_to_keep_in_context: Optional[int] = None,
        message_with_yielded_image: str = "Here is the image the tool returned.",
        termination_behavior: Optional[Literal["return", "user"]] = "user",
        inherit_activate_state_messages: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Start a free-style conversation using the chat graph.

        Parameters
        ----------
        message : str or dict or list of dict, optional
            Optional bootstrap message payload for the next chat turn.
        max_agent_iterations : int, optional
            Maximum number of assistant tool-execution cycles before returning
            control to the caller or user.
        n_first_images_to_keep_in_context, n_last_images_to_keep_in_context : int, optional
            Number of earliest and latest image-bearing messages to keep in
            context. When both are ``None``, all images remain in context. If
            either value is not ``None``, pruning runs and the ``None`` side is
            interpreted as ``0``.

            For ``n`` image-bearing messages, image ordinals are kept as
            follows, with counts capped by the available images::

                n_first_images_to_keep_in_context = 1
                n_last_images_to_keep_in_context = 2
                image_ordinals_kept = {0, n - 2, n - 1}

                n_first_images_to_keep_in_context = None
                n_last_images_to_keep_in_context = 2
                image_ordinals_kept = {n - 2, n - 1}

                n_first_images_to_keep_in_context = 0
                n_last_images_to_keep_in_context = 2
                image_ordinals_kept = {n - 2, n - 1}

        message_with_yielded_image : str, default="Here is the image the tool returned."
            Follow-up message used when a tool returns image paths.
        termination_behavior : {"return", "user"}, default="user"
            Behavior after a non-tool assistant response or an interrupt.
            `"user"` returns control to the user for another instruction,
            while `"return"` exits back to the caller.
        inherit_activate_state_messages : bool, default=True
            Whether to copy ``messages`` and ``full_history`` from the active
            state into the new chat state before running the graph.

        Returns
        -------
        None
        """
        initial_state = ChatGraphState(
            round_index=self.active_state.round_index,
            termination_behavior=termination_behavior or "user",
            bootstrap_message=message,
            max_agent_iterations=max_agent_iterations,
            n_first_images_to_keep_in_context=n_first_images_to_keep_in_context,
            n_last_images_to_keep_in_context=n_last_images_to_keep_in_context,
            message_with_yielded_image=message_with_yielded_image,
            await_user_input=message is None,
        )
        if inherit_activate_state_messages:
            initial_state.copy_messages_and_history_from_state(self.active_state)
        self.set_active_state(initial_state, "chat_graph")
        graph = self.chat_graph
        graph_kwargs: dict[str, Any] = {
            "context": self.memory_manager.get_runtime_context(),
        }
        if self.checkpoint_db_path is not None:
            graph, checkpoint_config, _ = self.get_checkpointed_graph(
                "chat_graph",
                load_state=False,
            )
            self.chat_graph = graph
            graph_kwargs["config"] = checkpoint_config
        invoke_result = self.invoke_graph_with_interruption_recovery(
            graph=graph,
            graph_input=initial_state,
            graph_kwargs=graph_kwargs,
            graph_name="chat_graph",
            state_model=ChatGraphState,
            interruption_message="Keyboard interrupt detected. The current chat run was interrupted.",
        )
        if invoke_result.command != "completed":
            return
        final_state = invoke_result.final_state
        self.set_active_state(ChatGraphState.model_validate(final_state), "chat_graph")
        if self.chat_state.monitor_requested:
            self.enter_monitoring_mode(self.chat_state.monitor_task_description)

    def run_conversation_from_checkpoint(
        self,
        checkpoint_db_path: Optional[str] = None,
    ) -> None:
        """Resume the chat graph directly from a saved checkpoint.

        Parameters
        ----------
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and updates instead of
            ``self.checkpoint_db_path``.

        Notes
        -----
        By default checkpoints are loaded from ``checkpoint_db_path``. Pass
        ``checkpoint_db_path`` to resume from a different SQLite file.
        """
        graph, checkpoint_config, loaded_state = self.get_checkpointed_graph(
            "chat_graph",
            checkpoint_db_path=checkpoint_db_path,
        )
        snapshot = graph.get_state(checkpoint_config)
        fallback_loaded = (
            snapshot.created_at is None
            or snapshot.values is None
            or len(snapshot.values) == 0
        )
        if loaded_state is None:
            resolved_checkpoint_path, _ = self.resolve_checkpoint_storage(
                "chat_graph",
                checkpoint_db_path=checkpoint_db_path,
            )
            raise ValueError(
                f"No chat-graph checkpoint found in shared checkpoint DB "
                f"{resolved_checkpoint_path}."
            )
        resumed_state = ChatGraphState.model_validate(loaded_state.model_dump())
        restart_from_state = (
            fallback_loaded
            or resumed_state.exit_requested
            or resumed_state.return_requested
        )
        if restart_from_state:
            resumed_state.exit_requested = False
            resumed_state.return_requested = False
            resumed_state.bootstrap_message = None
            resumed_state.await_user_input = True
        self.set_active_state(resumed_state, "chat_graph")
        self.chat_graph = graph
        graph_input: Optional[ChatGraphState] = None
        if restart_from_state:
            graph_input = resumed_state
        invoke_result = self.invoke_graph_with_interruption_recovery(
            graph=graph,
            graph_input=graph_input,
            graph_kwargs={
                "config": checkpoint_config,
                "context": self.memory_manager.get_runtime_context(),
            },
            graph_name="chat_graph",
            state_model=ChatGraphState,
            interruption_message="Keyboard interrupt detected. The current chat run was interrupted.",
        )
        if invoke_result.command != "completed":
            return
        final_state = invoke_result.final_state
        self.set_active_state(ChatGraphState.model_validate(final_state), "chat_graph")
        if self.chat_state.monitor_requested:
            self.enter_monitoring_mode(self.chat_state.monitor_task_description)
