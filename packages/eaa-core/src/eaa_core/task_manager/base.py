from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, TypeVar
import json
import logging
import sqlite3
import time

from langgraph.graph import START, StateGraph

from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.llm.model import build_chat_model, invoke_chat_model
from eaa_core.message_proc import (
    generate_openai_message,
    get_message_elements_as_text,
    print_message,
)
from eaa_core.skill import load_skills
from eaa_core.task_manager.memory_manager import MemoryManager
from eaa_core.task_manager.nodes import NodeFactory
from eaa_core.task_manager.persistence import PrunableSqliteSaver, SQLiteMessageStore
from eaa_core.task_manager.prompts import render_prompt_template
from eaa_core.task_manager.state import (
    ChatGraphState,
    ChatRuntimeContext,
    FeedbackLoopState,
    TaskManagerState,
)
from eaa_core.task_manager.tool_executor import SerialToolExecutor
from eaa_core.tool.base import BaseTool
from eaa_core.tool.coding import BashCodingTool, PythonCodingTool
from eaa_core.tool.skill import SkillLibraryTool

logger = logging.getLogger(__name__)

GraphName = Literal["chat_graph", "feedback_loop_graph", "task_graph"]
StateT = TypeVar("StateT", bound=TaskManagerState)


def get_state_checkpoint_config(checkpoint_thread_id: str) -> dict[str, Any]:
    """Return the LangGraph config used for a checkpoint file."""
    return {
        "configurable": {
            "thread_id": checkpoint_thread_id,
        }
    }


def get_checkpoint_state_model(
    graph_name: GraphName,
) -> type[TaskManagerState]:
    """Return the state model for a checkpointed graph."""
    if graph_name == "chat_graph":
        return ChatGraphState
    if graph_name == "feedback_loop_graph":
        return FeedbackLoopState
    if graph_name == "task_graph":
        return TaskManagerState
    raise ValueError(f"Unsupported graph name for checkpoint loading: {graph_name}.")


def load_latest_checkpoint_state_from_connection(
    connection: sqlite3.Connection,
    prune_checkpoints: bool,
) -> Optional[dict[str, Any]]:
    """Load the newest transcript-bearing checkpoint state from a connection."""
    saver = PrunableSqliteSaver(
        connection,
        prune_checkpoints=prune_checkpoints,
    )
    saver.setup()
    latest_state, _ = saver.load_latest_checkpoint_state()
    return latest_state


def build_compatible_checkpoint_state(
    graph_name: GraphName,
    incoming_state: TaskManagerState | dict[str, Any],
) -> Optional[TaskManagerState]:
    """Translate a checkpoint state into the target graph's compatible state.

    Parameters
    ----------
    graph_name : {"chat_graph", "feedback_loop_graph", "task_graph"}
        Target graph identifier whose state model should be produced.
    incoming_state : TaskManagerState or dict[str, Any]
        Source checkpoint state from any graph.

    Returns
    -------
    Optional[TaskManagerState]
        Compatible state for the target graph, or ``None`` when no transcript
        data is available to seed the new graph.
    """
    state_data = (
        incoming_state.model_dump()
        if isinstance(incoming_state, TaskManagerState)
        else dict(incoming_state)
    )
    messages = state_data.get("messages")
    if not isinstance(messages, list):
        messages = []
    full_history = state_data.get("full_history")
    if not isinstance(full_history, list):
        full_history = list(messages)
    if len(messages) == 0 and len(full_history) == 0:
        return None
    shared_fields = {
        "messages": list(messages),
        "full_history": list(full_history),
        "await_user_input": True,
        "round_index": int(state_data.get("round_index", 0) or 0),
        "store_all_images_in_context": bool(
            state_data.get("store_all_images_in_context", True)
        ),
    }
    if graph_name == "chat_graph":
        return ChatGraphState(
            **shared_fields,
            bootstrap_message=None,
            termination_behavior="user",
            monitor_requested=bool(state_data.get("monitor_requested", False)),
            monitor_task_description=str(state_data.get("monitor_task_description", "") or ""),
            exit_requested=False,
            return_requested=False,
        )
    if graph_name == "feedback_loop_graph":
        return FeedbackLoopState(
            **shared_fields,
            initial_prompt="",
            initial_image_path=None,
            initial_prompt_pending=False,
            message_with_yielded_image="Here is the image the tool returned.",
            max_rounds=99,
            n_first_images_to_keep_in_context=None,
            n_last_images_to_keep_in_context=None,
            allow_non_image_tool_responses=True,
            allow_multiple_tool_calls=False,
            expected_tool_call_sequence=None,
            expected_tool_call_sequence_tolerance=0,
            termination_behavior="ask",
            max_arounds_reached_behavior="return",
            chat_requested=bool(state_data.get("chat_requested", False)),
            exit_requested=False,
            return_requested=False,
        )
    if graph_name == "task_graph":
        return TaskManagerState(**shared_fields)
    raise ValueError(f"Unsupported graph name for checkpoint loading: {graph_name}.")


class TaskManagerAgentAdapter:
    """Compatibility adapter for code paths that still expect `task_manager.agent`."""

    def __init__(self, task_manager: "BaseTaskManager"):
        self.task_manager = task_manager
        self.tool_manager = task_manager.tool_executor

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
    memory_config : Optional[MemoryManagerConfig], optional
        Configuration for the long-term memory store.
    tools : list[BaseTool], optional
        Base tools exposed to the task manager.
    skill_dirs : Optional[Sequence[str]], optional
        Directories searched for EAA skills.
    session_db_path : Optional[str], default="session.sqlite"
        Path to the shared SQLite session database. This database stores
        LangGraph checkpoints, explicit WebUI display messages, WebUI input
        messages, and WebUI status flags.
    use_webui : bool, default=False
        Whether to enable WebUI-driven user input and WebUI display writes.
    use_coding_tools : bool, default=True
        Whether to register built-in coding tools.
    run_codes_in_sandbox : bool, default=False
        Whether built-in coding tools should execute code in sandbox mode.
    prune_checkpoints : bool, default=True
        Whether to keep only the latest checkpoint per graph thread in the
        shared SQLite session database.
    build : bool, default=True
        Whether to initialize persistence, model, tools, memory, and graphs
        during construction.
    """

    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        tools: list[BaseTool] = (),
        skill_dirs: Optional[Sequence[str]] = None,
        session_db_path: Optional[str] = "session.sqlite",
        use_webui: bool = False,
        use_coding_tools: bool = True,
        run_codes_in_sandbox: bool = False,
        prune_checkpoints: bool = True,
        build: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the task manager."""
        self.common_state = TaskManagerState()
        self.chat_state = ChatGraphState()
        self.feedback_state = FeedbackLoopState()
        self.task_state = TaskManagerState()
        self.active_state: TaskManagerState = self.task_state
        self.llm_config = llm_config
        if isinstance(memory_config, dict):
            memory_config = MemoryManagerConfig.from_dict(memory_config)
        self.memory_config = memory_config
        self.tools = list(tools)
        self.skill_dirs = list(skill_dirs) if skill_dirs else []
        self.skill_tool: Optional[SkillLibraryTool] = None
        self.use_webui = use_webui
        self.use_coding_tools = use_coding_tools
        self.run_codes_in_sandbox = run_codes_in_sandbox
        self.prune_checkpoints = prune_checkpoints
        self.session_db_path = session_db_path
        self.memory_manager = MemoryManager(self)
        self.persistence = SQLiteMessageStore(self.session_db_path)
        self.tool_executor = SerialToolExecutor(
            approval_handler=self._request_tool_approval_via_task_manager,
        )
        self.model = None
        self.agent = TaskManagerAgentAdapter(self)
        self.node_factory = NodeFactory(self)
        if not getattr(self, "assistant_system_message", None):
            self.assistant_system_message = self.get_default_system_prompt()
        self.chat_graph = None
        self.feedback_loop_graph = None
        self.task_graph = None
        self.checkpoint_connections: dict[tuple[str, str], sqlite3.Connection] = {}
        self.checkpoint_graphs: dict[tuple[str, str], Any] = {}

        if use_webui and not session_db_path:
            raise ValueError("`use_webui` requires `session_db_path` to be set.")
        if build:
            self.build()

    def get_default_system_prompt(self) -> str:
        """Return the default system prompt for the task manager."""
        return render_prompt_template(
            "eaa_core.task_manager.prompts",
            "system_base.md",
            {
                "available_skills_text": self.format_available_skills_for_prompt(),
            },
        )

    def format_available_skills_for_prompt(self) -> str:
        """Return the available-skill summary injected into the system prompt."""
        skill_catalog = load_skills(self.skill_dirs)
        if not skill_catalog:
            return "No skills are currently available."
        lines = ["Available skills:"]
        lines.extend(
            f"- {skill.name}: {skill.description}"
            for skill in skill_catalog
        )
        return "\n".join(lines)

    def set_active_state(
        self,
        state: TaskManagerState,
        graph_name: Optional[GraphName] = None,
    ) -> None:
        """Set the active state and update the matching state holder.
        Also update common state with the state.

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
        elif graph_name == "feedback_loop_graph":
            self.feedback_state = FeedbackLoopState.model_validate(state.model_dump())
            self.active_state = self.feedback_state
        elif graph_name == "task_graph":
            self.task_state = state
            self.active_state = self.task_state
        else:
            raise ValueError(f"Unsupported active graph name: {graph_name}.")
        self.sync_common_state_from_state(self.active_state)

    def infer_graph_name_for_state(self, state: TaskManagerState) -> GraphName:
        """Infer a graph slot from a state model."""
        if isinstance(state, ChatGraphState):
            return "chat_graph"
        if isinstance(state, FeedbackLoopState):
            return "feedback_loop_graph"
        return "task_graph"

    def sync_common_state_from_state(self, state: TaskManagerState) -> None:
        """Copy common fields from a graph state into ``common_state``."""
        self.common_state = TaskManagerState(
            messages=list(state.messages),
            full_history=list(state.full_history),
            await_user_input=state.await_user_input,
            round_index=state.round_index,
            store_all_images_in_context=state.store_all_images_in_context,
        )

    def common_state_payload(self) -> dict[str, Any]:
        """Return common fields copied into graph-specific state models."""
        return {
            "messages": list(self.common_state.messages),
            "full_history": list(self.common_state.full_history),
            "await_user_input": self.common_state.await_user_input,
            "round_index": self.common_state.round_index,
            "store_all_images_in_context": self.common_state.store_all_images_in_context,
        }

    def sync_state_from_common(self, state: StateT) -> StateT:
        """Return ``state`` with common fields copied from ``common_state``."""
        return state.model_copy(update=self.common_state_payload())

    @property
    def context(self) -> list[dict[str, Any]]:
        """Return the canonical active conversation context."""
        return self.common_state.messages

    @context.setter
    def context(self, value: list[dict[str, Any]]) -> None:
        """Replace the canonical active conversation context."""
        self.common_state.messages = list(value)
        self.active_state.messages = list(value)

    @property
    def full_history(self) -> list[dict[str, Any]]:
        """Return the canonical full transcript."""
        return self.common_state.full_history

    @full_history.setter
    def full_history(self, value: list[dict[str, Any]]) -> None:
        """Replace the canonical full transcript."""
        self.common_state.full_history = list(value)
        self.active_state.full_history = list(value)

    def build(self, *args, **kwargs):
        """Build persistence, model, tools, and graphs."""
        self.build_db()
        self.build_model()
        self.build_tools()
        self.build_memory_store()
        self.chat_graph = self.build_chat_graph()
        self.feedback_loop_graph = self.build_feedback_loop_graph()
        self.task_graph = self.build_task_graph()

    def build_db(self, *args, **kwargs):
        """Initialize message persistence and optionally hydrate prior messages."""
        self.persistence.connect()

    def build_model(self, *args, **kwargs):
        """Build the chat model if an LLM config is provided."""
        if self.llm_config is None:
            logger.info("Skipping model build because `llm_config` is not provided.")
            return
        self.model = build_chat_model(self.llm_config)

    def build_tools(self, *args, **kwargs):
        """Register local tools and built-in helper tools."""
        self.tool_executor.register_tools(self._collect_base_tools())

    def build_memory_store(self) -> None:
        """Build the long-term memory store used by the chat graph."""
        self.memory_manager.build_store()

    def build_task_graph(self, checkpointer: Any = None):
        """Build the task-manager-specific graph if needed."""
        return None

    def handoff_to_chat(
        self,
        store_all_images_in_context: bool = True,
    ) -> None:
        """Exit the active workflow into chat mode.

        Parameters
        ----------
        store_all_images_in_context : bool, default=True
            Whether image follow-up messages should remain in active chat
            context after the handoff.
        """
        source_state = self.active_state
        self.set_active_state(
            ChatGraphState(
                messages=list(source_state.messages),
                full_history=list(source_state.full_history),
                round_index=source_state.round_index,
                termination_behavior="user",
                store_all_images_in_context=store_all_images_in_context,
                bootstrap_message=None,
                await_user_input=True,
            ),
            "chat_graph",
        )
        self.run_conversation(
            store_all_images_in_context=store_all_images_in_context,
            termination_behavior="user",
        )

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
            of ``self.session_db_path``.

        Returns
        -------
        tuple[str, str]
            Resolved SQLite database path and checkpoint thread id.
        """
        checkpoint_path = checkpoint_db_path or self.session_db_path
        if checkpoint_path is None:
            raise ValueError(
                "Checkpointing requires `session_db_path` or `checkpoint_db_path` "
                "because the WebUI relay and LangGraph checkpoints can share one "
                "SQLite file."
            )
        shared_path = str(Path(checkpoint_path).expanduser().resolve())
        return shared_path, graph_name

    def get_checkpointed_graph(
        self,
        graph_name: Literal["chat_graph", "feedback_loop_graph", "task_graph"],
        checkpoint_db_path: Optional[str] = None,
        load_state: bool = True,
    ) -> tuple[Any, Optional[dict[str, Any]], Optional[TaskManagerState]]:
        """Return a checkpointed graph together with config and optional state.

        Parameters
        ----------
        graph_name : {"chat_graph", "feedback_loop_graph", "task_graph"}
            Graph identifier whose persistent graph should be returned.
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and persistence instead
            of ``self.session_db_path``.
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
            connection = sqlite3.connect(resolved_checkpoint_path, check_same_thread=False)
            saver = PrunableSqliteSaver(
                connection,
                prune_checkpoints=self.prune_checkpoints,
            )
            saver.setup()
            if graph_name == "chat_graph":
                graph = self.build_chat_graph(checkpointer=saver)
            elif graph_name == "feedback_loop_graph":
                graph = self.build_feedback_loop_graph(checkpointer=saver)
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
            snapshot = graph.get_state(config)
            if (
                snapshot.created_at is not None
                and snapshot.values is not None
                and len(snapshot.values) > 0
            ):
                state_model = get_checkpoint_state_model(graph_name)
                loaded_state = state_model.model_validate(snapshot.values)
            else:
                latest_state = load_latest_checkpoint_state_from_connection(
                    connection=self.checkpoint_connections[cache_key],
                    prune_checkpoints=self.prune_checkpoints,
                )
                if latest_state is not None:
                    loaded_state = build_compatible_checkpoint_state(graph_name, latest_state)
        return graph, config, loaded_state

    def _collect_base_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = list(self.tools)
        self._merge_tools(tools, self._build_default_tools())
        return tools

    def _merge_tools(self, tools: list[BaseTool], new_tools: list[BaseTool]) -> None:
        """Merge tool lists while avoiding duplicated tool names."""
        seen_names = self._collect_tool_names(tools)
        for tool in new_tools:
            tool_names = self._collect_tool_names([tool])
            if tool_names and tool_names & seen_names:
                continue
            tools.append(tool)
            seen_names.update(tool_names)

    def _collect_tool_names(self, tools: list[BaseTool]) -> set[str]:
        """Collect exposed tool names from tool objects."""
        names: set[str] = set()
        for tool in tools:
            for exposed in tool.exposed_tools:
                names.add(exposed.name)
        return names

    def _build_default_tools(self) -> list[BaseTool]:
        """Return default built-in tools."""
        tools: list[BaseTool] = []
        self.skill_tool = SkillLibraryTool(skill_dirs=self.skill_dirs)
        tools.append(self.skill_tool)
        if not self.use_coding_tools:
            return tools
        tools.extend(
            [
                PythonCodingTool(run_in_sandbox=self.run_codes_in_sandbox),
                BashCodingTool(run_in_sandbox=self.run_codes_in_sandbox),
            ]
        )
        return tools

    def register_tools(self, tools: BaseTool | list[BaseTool]) -> None:
        """Register one or more tools with the serial executor."""
        self.tool_executor.register_tools(tools)

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
            Whether to append the message to the explicit WebUI display table.
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
        """Append a message to in-memory history for checkpoint persistence.

        Parameters
        ----------
        message : Dict[str, Any]
            Message payload to append.
        update_context : bool, default=True
            Whether to append the message to the active context.
        update_full_history : bool, default=True
            Whether to append the message to the transcript history.
        write_to_webui : bool, default=True
            Whether to append the message to the explicit WebUI display table.
        """
        if update_context:
            self.common_state.messages.append(message)
            self.active_state.messages.append(message)
        if update_full_history:
            self.common_state.full_history.append(message)
            self.active_state.full_history.append(message)
        if write_to_webui and self.session_db_path is not None:
            self.persistence.append_message(message)

    def add_webui_message_to_db(self, message: Dict[str, Any]) -> None:
        """Append a display-only message to the WebUI.

        Parameters
        ----------
        message : Dict[str, Any]
            Message payload to expose through the WebUI.
        """
        self.persistence.append_message(message)

    def get_user_input(
        self,
        prompt: str = "Enter a message: ",
        display_prompt_in_webui: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """Get user input from the terminal or the WebUI relay DB."""
        if self.use_webui:
            logger.info("Waiting for user input through the WebUI relay database.")
            self.set_user_input_requested(True)
            if display_prompt_in_webui:
                self.update_message_history(
                    {"role": "system", "content": prompt},
                    update_context=False,
                    update_full_history=False,
                    write_to_webui=True,
                )
            while True:
                queued_input = self.persistence.dequeue_webui_input()
                if queued_input is not None:
                    self.set_user_input_requested(False)
                    return queued_input
                time.sleep(1)
        self.set_user_input_requested(True)
        message = input(prompt)
        self.set_user_input_requested(False)
        return message

    def set_user_input_requested(self, requested: bool) -> None:
        """Persist the WebUI pending-input status flag."""
        self.persistence.set_user_input_requested(requested)

    def _request_tool_approval_via_task_manager(self, tool_name: str, tool_kwargs: Dict[str, Any]) -> bool:
        """Relay tool approval requests through the task manager input path."""
        prompt = (
            f"Tool '{tool_name}' requires approval before execution.\n"
            f"Arguments: {json.dumps(tool_kwargs, default=str)}\n"
            "Approve? [y/N]: "
        )
        response = self.get_user_input(prompt, display_prompt_in_webui=self.use_webui)
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
        dict or tuple[dict, dict | None]
            Assistant response, or `(response, outgoing_message)` when
            `return_outgoing_message` is `True`.
        """
        if self.model is None:
            raise RuntimeError("No model is configured for this task manager.")
        effective_context = list(context) if context is not None else list(self.context)
        outgoing_message = None
        if message is not None:
            if isinstance(message, str):
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

    def display_command_help(self) -> str:
        """Display the available interactive commands."""
        text = (
            "Below are supported commands.\n"
            "* `/exit`: exit the current loop\n"
            "* `/chat`: enter chat mode\n"
            "* `/monitor <task description>`: enter monitoring mode\n"
            "* `/skill`: display skills available to the agent\n"
            "* `/return`: return to upper level task\n"
        )
        if self.use_webui:
            self.add_webui_message_to_db({"role": "system", "content": text})
        else:
            print(text)
            self.add_webui_message_to_db({"role": "system", "content": text})
        return text

    def display_available_skills(self) -> str:
        """Display skills available through the skill library tool."""
        skill_catalog = self.skill_tool.skill_catalog if self.skill_tool is not None else []
        if not skill_catalog:
            text = "No skills are available."
        else:
            text = "\n".join(
                ["Skills available to the agent:"]
                + [
                    f"{index}. {skill.name} - {skill.description} [{skill.path}]"
                    for index, skill in enumerate(skill_catalog, start=1)
                ]
            )
        if self.use_webui:
            self.add_webui_message_to_db({"role": "system", "content": text})
        else:
            print(text)
            self.add_webui_message_to_db({"role": "system", "content": text})
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
                self.run_feedback_loop(initial_prompt=initial_prompt, termination_behavior="return")
                time.sleep(time_interval)
            except KeyboardInterrupt:
                self.add_webui_message_to_db(
                    generate_openai_message(
                        content="Keyboard interrupt detected. Terminating monitoring task.",
                        role="system",
                    )
                )
                return

    def enforce_tool_call_sequence(
        self,
        expected_tool_call_sequence: list[str],
        tolerance: int = 0,
    ) -> None:
        """Warn the model if the recent tool-call order differs from the expected sequence."""
        if len(self.tool_executor.tool_execution_history) <= 1:
            return
        n_actual = min(
            len(self.tool_executor.tool_execution_history),
            len(expected_tool_call_sequence),
        ) - tolerance
        if n_actual <= 0:
            return
        actual_sequence = [
            entry["tool_name"] for entry in self.tool_executor.tool_execution_history[-n_actual:]
        ]
        expanded_expected = list(expected_tool_call_sequence) * 2
        for index in range(len(expanded_expected) - len(actual_sequence) + 1):
            if expanded_expected[index : index + len(actual_sequence)] == actual_sequence:
                return
        self.update_message_history(
            generate_openai_message(
                content=(
                    f"The tool call sequence {actual_sequence} is not as expected. "
                    "Are you making the right tool calls in the right order? "
                    "If this is intended to address an exception, ignore this message."
                ),
                role="user",
            ),
            update_context=True,
            update_full_history=False,
        )

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
        if self.session_db_path is not None:
            graph, checkpoint_config, _ = self.get_checkpointed_graph(
                "task_graph",
                load_state=False,
            )
            self.task_graph = graph
            graph_kwargs["config"] = checkpoint_config
        if graph is None:
            raise ValueError("The task manager does not define a runnable task graph.")
        initial_state = self.sync_state_from_common(self.task_state)
        self.set_active_state(initial_state, "task_graph")
        try:
            final_state = graph.invoke(initial_state, **graph_kwargs)
        except KeyboardInterrupt:
            interrupt_message = generate_openai_message(
                content=(
                    "Keyboard interrupt detected. The current task was interrupted. "
                    "You can now provide new instructions."
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
            if (
                isinstance(initial_state, FeedbackLoopState)
                and initial_state.termination_behavior == "ask"
            ):
                self.run_conversation(
                    store_all_images_in_context=True,
                    termination_behavior="user",
                )
            return
        state_model = type(initial_state)
        if not issubclass(state_model, TaskManagerState):
            state_model = TaskManagerState
        self.set_active_state(state_model.model_validate(final_state), "task_graph")
        if bool(getattr(self.task_state, "chat_requested", False)):
            self.handoff_to_chat()

    def run_from_checkpoint(self, checkpoint_db_path: Optional[str] = None) -> None:
        """Resume a task manager from a task-graph checkpoint.

        Parameters
        ----------
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and updates instead of
            ``self.session_db_path``.
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
        final_state = graph.invoke(graph_input, config=checkpoint_config)
        state_model = type(loaded_state)
        if not issubclass(state_model, TaskManagerState):
            state_model = TaskManagerState
        self.set_active_state(state_model.model_validate(final_state), "task_graph")
        if bool(getattr(self.task_state, "chat_requested", False)):
            self.handoff_to_chat()

    def _message_contains_image(self, message: dict[str, Any]) -> bool:
        """Return whether a message contains image payloads."""
        return get_message_elements_as_text(message)["image"] is not None

    def execute_tools_for_state(
        self,
        state: TaskManagerState,
        *,
        message_with_yielded_image: str,
        allow_non_image_tool_responses: bool,
        store_all_images_in_context: bool = True,
    ) -> dict[str, Any]:
        """Execute tool calls for a graph state.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state whose latest assistant response will be
            executed.
        message_with_yielded_image : str
            Compatibility argument retained for callers that still route tool
            execution through the task manager boundary.
        allow_non_image_tool_responses : bool
            Compatibility argument retained for callers that still route tool
            execution through the task manager boundary.
        store_all_images_in_context : bool, default=True
            Compatibility argument retained for callers that still route tool
            execution through the task manager boundary.

        Returns
        -------
        dict[str, Any]
            Updated graph state payload.
        """
        return self.node_factory.execute_tools_for_state(state)

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
        builder.add_edge("image_followup", "call_model")
        return builder.compile(checkpointer=checkpointer)

    def build_feedback_loop_graph(self, checkpointer: Any = None):
        """Build the base feedback-loop graph."""
        node_factory = self.node_factory
        builder = StateGraph(FeedbackLoopState)
        builder.add_node(
            "handle_human_gate",
            node_factory.handle_human_gate,
        )
        builder.add_node(
            "reprompt_model",
            node_factory.reprompt_model,
        )
        builder.add_node(
            "execute_tools",
            node_factory.execute_tools,
            input_schema=FeedbackLoopState,
        )
        builder.add_node(
            "image_followup",
            node_factory.image_followup,
            input_schema=FeedbackLoopState,
        )
        builder.add_node(
            "finalize_round",
            node_factory.finalize_round,
        )
        builder.add_node(
            "call_model",
            node_factory.call_model,
            input_schema=FeedbackLoopState,
        )
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            node_factory.route_after_feedback_response,
        )
        builder.add_conditional_edges(
            "handle_human_gate",
            node_factory.route_after_feedback_response,
        )
        builder.add_conditional_edges(
            "reprompt_model",
            node_factory.route_after_feedback_response,
        )
        builder.add_conditional_edges(
            "execute_tools",
            node_factory.route_after_tool_execution,
        )
        builder.add_edge("image_followup", "finalize_round")
        builder.add_conditional_edges(
            "finalize_round",
            node_factory.route_after_feedback_round,
        )
        return builder.compile(checkpointer=checkpointer)

    def run_conversation(
        self,
        message: Optional[str | Dict[str, Any] | list[Dict[str, Any]]] = None,
        store_all_images_in_context: bool = True,
        termination_behavior: Optional[Literal["return", "user"]] = "user",
        *args,
        **kwargs,
    ) -> None:
        """Start a free-style conversation using the chat graph.

        Parameters
        ----------
        message : str or dict or list of dict, optional
            Optional bootstrap message payload for the next chat turn.
        store_all_images_in_context : bool, default=True
            Whether all images should remain in the active chat context.
        termination_behavior : {"return", "user"}, default="user"
            Behavior after a non-tool assistant response or an interrupt.
            `"user"` returns control to the user for another instruction,
            while `"return"` exits back to the caller.

        Returns
        -------
        None
        """
        initial_state = ChatGraphState(
            messages=list(self.common_state.messages),
            full_history=list(self.common_state.full_history),
            round_index=self.common_state.round_index,
            termination_behavior=termination_behavior or "user",
            store_all_images_in_context=store_all_images_in_context,
            bootstrap_message=message,
            await_user_input=message is None,
        )
        self.set_active_state(initial_state, "chat_graph")
        graph = self.chat_graph
        graph_kwargs: dict[str, Any] = {
            "context": self.memory_manager.get_runtime_context(),
        }
        if self.session_db_path is not None:
            graph, checkpoint_config, _ = self.get_checkpointed_graph(
                "chat_graph",
                load_state=False,
            )
            self.chat_graph = graph
            graph_kwargs["config"] = checkpoint_config
        try:
            final_state = graph.invoke(
                initial_state,
                **graph_kwargs,
            )
        except KeyboardInterrupt:
            interrupt_message = generate_openai_message(
                content=(
                    "Keyboard interrupt detected. The current chat run was interrupted. "
                    "You can now provide new instructions."
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
            if (termination_behavior or "user") == "user":
                self.run_conversation(
                    store_all_images_in_context=store_all_images_in_context,
                    termination_behavior=termination_behavior,
                )
            return
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
            ``self.session_db_path``.

        Notes
        -----
        By default checkpoints are loaded from ``session_db_path``. Pass
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
        final_state = graph.invoke(
            graph_input,
            config=checkpoint_config,
            context=self.memory_manager.get_runtime_context(),
        )
        self.set_active_state(ChatGraphState.model_validate(final_state), "chat_graph")
        if self.chat_state.monitor_requested:
            self.enter_monitoring_mode(self.chat_state.monitor_task_description)

    def run_feedback_loop(
        self,
        initial_prompt: str,
        initial_image_path: Optional[str | list[str]] = None,
        message_with_yielded_image: str = "Here is the image the tool returned.",
        max_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_last_images_to_keep_in_context: Optional[int] = None,
        allow_non_image_tool_responses: bool = True,
        allow_multiple_tool_calls: bool = False,
        expected_tool_call_sequence: Optional[list[str]] = None,
        expected_tool_call_sequence_tolerance: int = 0,
        termination_behavior: Literal["ask", "return"] = "ask",
        max_arounds_reached_behavior: Literal["return", "raise"] = "return",
        *args,
        **kwargs,
    ) -> None:
        """Run the feedback-loop graph with the configured workflow settings.

        Parameters
        ----------
        initial_prompt : str
            Initial prompt sent to the model on the first feedback-loop turn.
        initial_image_path : str or list of str, optional
            Optional image path payload sent with the initial prompt.
        message_with_yielded_image : str, default="Here is the image the tool returned."
            Follow-up message used when a tool returns image paths.
        max_rounds : int, default=99
            Maximum number of feedback-loop rounds.
        n_first_images_to_keep_in_context : int, optional
            Number of earliest images to keep in the active context.
        n_last_images_to_keep_in_context : int, optional
            Number of most recent images to keep in the active context.
        allow_non_image_tool_responses : bool, default=True
            Whether non-image tool outputs are accepted.
        allow_multiple_tool_calls : bool, default=False
            Whether the assistant may issue multiple tool calls in one response.
        expected_tool_call_sequence : list of str, optional
            Expected tool-call order used for validation messaging.
        expected_tool_call_sequence_tolerance : int, default=0
            Allowed mismatch tolerance for the expected tool-call order.
        termination_behavior : {"ask", "return"}, default="ask"
            Behavior after a terminal feedback response or an interrupt.
            `"ask"` enters chat mode for new user instructions, while
            `"return"` exits back to the caller.
        max_arounds_reached_behavior : {"return", "raise"}, default="return"
            Behavior when `max_rounds` is reached.

        Returns
        -------
        None
        """
        if termination_behavior not in ["ask", "return"]:
            raise ValueError("`termination_behavior` must be either 'ask' or 'return'.")
        initial_state = FeedbackLoopState(
            messages=list(self.common_state.messages),
            full_history=list(self.common_state.full_history),
            round_index=0,
            await_user_input=False,
            initial_prompt=initial_prompt,
            initial_image_path=initial_image_path,
            message_with_yielded_image=message_with_yielded_image,
            max_rounds=max_rounds,
            n_first_images_to_keep_in_context=n_first_images_to_keep_in_context,
            n_last_images_to_keep_in_context=n_last_images_to_keep_in_context,
            allow_non_image_tool_responses=allow_non_image_tool_responses,
            allow_multiple_tool_calls=allow_multiple_tool_calls,
            expected_tool_call_sequence=expected_tool_call_sequence,
            expected_tool_call_sequence_tolerance=expected_tool_call_sequence_tolerance,
            termination_behavior=termination_behavior,
            max_arounds_reached_behavior=max_arounds_reached_behavior,
        )
        self.set_active_state(initial_state, "feedback_loop_graph")
        graph = self.feedback_loop_graph
        graph_kwargs: dict[str, Any] = {}
        if self.session_db_path is not None:
            graph, checkpoint_config, _ = self.get_checkpointed_graph(
                "feedback_loop_graph",
                load_state=False,
            )
            self.feedback_loop_graph = graph
            graph_kwargs["config"] = checkpoint_config
        try:
            final_state = graph.invoke(initial_state, **graph_kwargs)
        except KeyboardInterrupt:
            interrupt_message = generate_openai_message(
                content=(
                    "Keyboard interrupt detected. The current feedback loop was interrupted. "
                    "You can now provide new instructions."
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
            if termination_behavior == "ask":
                self.run_conversation(
                    store_all_images_in_context=True,
                    termination_behavior="user",
                )
            return
        self.set_active_state(
            FeedbackLoopState.model_validate(final_state),
            "feedback_loop_graph",
        )
        if self.feedback_state.chat_requested:
            self.handoff_to_chat()

    def run_feedback_loop_from_checkpoint(
        self,
        checkpoint_db_path: Optional[str] = None,
    ) -> None:
        """Resume the feedback-loop graph directly from a saved checkpoint.

        Parameters
        ----------
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and updates instead of
            ``self.session_db_path``.

        Notes
        -----
        By default checkpoints are loaded from ``session_db_path``. Pass
        ``checkpoint_db_path`` to resume from a different SQLite file.
        """
        graph, checkpoint_config, loaded_state = self.get_checkpointed_graph(
            "feedback_loop_graph",
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
                "feedback_loop_graph",
                checkpoint_db_path=checkpoint_db_path,
            )
            raise ValueError(
                f"No feedback-loop checkpoint found in shared checkpoint DB "
                f"{resolved_checkpoint_path}."
            )
        resumed_state = FeedbackLoopState.model_validate(loaded_state.model_dump())
        restart_from_human_gate = (
            fallback_loaded
            or resumed_state.exit_requested
            or resumed_state.return_requested
        )
        if restart_from_human_gate:
            resumed_state.exit_requested = False
            resumed_state.return_requested = False
            resumed_state.await_user_input = True
        self.set_active_state(resumed_state, "feedback_loop_graph")
        self.feedback_loop_graph = graph
        if restart_from_human_gate:
            checkpoint_config = graph.update_state(
                checkpoint_config,
                resumed_state.model_dump(),
                as_node="handle_human_gate",
            )
        final_state = graph.invoke(None, config=checkpoint_config)
        self.set_active_state(
            FeedbackLoopState.model_validate(final_state),
            "feedback_loop_graph",
        )
        if self.feedback_state.chat_requested:
            self.handoff_to_chat()
