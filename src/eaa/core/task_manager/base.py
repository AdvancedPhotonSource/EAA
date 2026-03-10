from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Sequence
import json
import logging
import time

from langgraph.graph import START, StateGraph

from eaa.api.llm_config import LLMConfig
from eaa.api.memory import MemoryManagerConfig
from eaa.core.llm.model import build_chat_model, invoke_chat_model
from eaa.core.message_proc import (
    generate_openai_message,
    get_message_elements_as_text,
    get_tool_call_info,
    has_tool_call,
    print_message,
)
from eaa.core.skill import SkillMetadata, SkillTool, load_skills, split_markdown_into_message_sections
from eaa.core.task_manager.nodes import NodeFactory
from eaa.core.task_manager.persistence import SQLiteMessageStore
from eaa.core.task_manager.state import ChatGraphState, FeedbackLoopState, TaskManagerState
from eaa.core.task_manager.tool_executor import SerialToolExecutor
from eaa.core.tooling.base import BaseTool, ToolReturnType
from eaa.tool.coding import BashCodingTool, PythonCodingTool

logger = logging.getLogger(__name__)


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

    def handle_tool_call(self, message: dict[str, Any], return_tool_return_types: bool = False):
        """Execute tool calls found in an assistant response."""
        return self.task_manager.execute_tool_calls_from_message(
            message,
            return_tool_return_types=return_tool_return_types,
        )


class BaseTaskManager:
    """LangGraph-backed base task manager for EAA."""

    assistant_system_message = ""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        tools: list[BaseTool] = (),
        skill_dirs: Optional[Sequence[str]] = None,
        message_db_path: Optional[str] = None,
        fill_context_with_message_db: bool = False,
        use_webui: bool = False,
        use_coding_tools: bool = True,
        run_codes_in_sandbox: bool = False,
        allow_parallel_tool_execution: bool = False,
        build: bool = True,
        *args,
        memory_vector_store: Optional[Any] = None,
        memory_notability_filter: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        memory_formatter: Optional[Callable[[list[Any]], str]] = None,
        memory_embedder: Optional[Callable[[Sequence[str]], list[list[float]]]] = None,
        **kwargs,
    ):
        """Initialize the task manager."""
        self.state = TaskManagerState()
        self.llm_config = llm_config
        if isinstance(memory_config, dict):
            memory_config = MemoryManagerConfig.from_dict(memory_config)
        self.memory_config = memory_config
        self.tools = list(tools)
        self.skill_dirs = list(skill_dirs) if skill_dirs else []
        self.skill_catalog: list[SkillMetadata] = []
        self.use_webui = use_webui
        self.use_coding_tools = use_coding_tools
        self.run_codes_in_sandbox = run_codes_in_sandbox
        self.allow_parallel_tool_execution = allow_parallel_tool_execution
        self.message_db_path = message_db_path
        self.fill_context_with_message_db = fill_context_with_message_db
        self.webui_user_input_last_timestamp = 0
        self._memory_vector_store = memory_vector_store
        self._memory_notability_filter = memory_notability_filter
        self._memory_formatter = memory_formatter
        self._memory_embedder = memory_embedder
        self.persistence = SQLiteMessageStore(message_db_path)
        self.tool_executor = SerialToolExecutor(
            approval_handler=self._request_tool_approval_via_task_manager,
            allow_parallel_tool_execution=allow_parallel_tool_execution,
        )
        self.model = None
        self.agent = TaskManagerAgentAdapter(self)
        self.node_factory = NodeFactory(self)
        self.chat_graph = None
        self.feedback_loop_graph = None
        self.task_graph = None
        if use_webui and not message_db_path:
            raise ValueError("`use_webui` requires `message_db_path` to be set.")
        if build:
            self.build()

    @property
    def context(self) -> list[dict[str, Any]]:
        """Return the active conversation context."""
        return self.state.messages

    @context.setter
    def context(self, value: list[dict[str, Any]]) -> None:
        """Replace the active conversation context."""
        self.state.messages = value

    @property
    def full_history(self) -> list[dict[str, Any]]:
        """Return the full transcript."""
        return self.state.full_history

    @full_history.setter
    def full_history(self, value: list[dict[str, Any]]) -> None:
        """Replace the full transcript."""
        self.state.full_history = value

    def build(self, *args, **kwargs):
        """Build persistence, model, tools, and graphs."""
        self.build_db()
        self.build_model()
        self.build_tools()
        self.chat_graph = self.build_chat_graph()
        self.feedback_loop_graph = self.build_feedback_loop_graph()
        self.task_graph = self.build_task_graph()

    def build_db(self, *args, **kwargs):
        """Initialize message persistence and optionally hydrate prior messages."""
        self.persistence.connect()
        if self.use_webui:
            self.webui_user_input_last_timestamp = self.persistence.get_latest_webui_input_timestamp()
        if self.fill_context_with_message_db:
            loaded_messages = self.persistence.load_messages()
            self.context.extend(loaded_messages)
            self.full_history.extend(loaded_messages)

    def build_model(self, *args, **kwargs):
        """Build the chat model if an LLM config is provided."""
        if self.llm_config is None:
            logger.info("Skipping model build because `llm_config` is not provided.")
            return
        self.model = build_chat_model(self.llm_config)

    def build_tools(self, *args, **kwargs):
        """Register local and skill-provided tools."""
        self.tool_executor.register_tools(self._collect_base_tools())

    def build_task_graph(self):
        """Build the task-manager-specific graph if needed."""
        return None

    def _collect_base_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = list(self.tools)
        self._merge_tools(tools, self._build_default_tools())
        self._merge_tools(tools, self._build_skill_tools())
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
        if not self.use_coding_tools:
            return []
        return [
            PythonCodingTool(run_in_sandbox=self.run_codes_in_sandbox),
            BashCodingTool(run_in_sandbox=self.run_codes_in_sandbox),
        ]

    def _build_skill_tools(self) -> list[BaseTool]:
        """Discover skills and expose them as tools."""
        if not self.skill_dirs:
            self.skill_catalog = []
            return []
        self.skill_catalog = load_skills(self.skill_dirs)
        return [SkillTool(skill) for skill in self.skill_catalog]

    def register_tools(self, tools: BaseTool | list[BaseTool]) -> None:
        """Register one or more tools with the serial executor."""
        self.tool_executor.register_tools(tools)

    def record_system_message(
        self,
        content: str,
        image_path: Optional[str | list[str]] = None,
        update_context: bool = False,
    ) -> None:
        """Append a system message to history."""
        self.update_message_history(
            generate_openai_message(content=content, role="system", image_path=image_path),
            update_context=update_context,
            update_full_history=True,
        )

    def update_message_history(
        self,
        message: Dict[str, Any],
        update_context: bool = True,
        update_full_history: bool = True,
        update_db: bool = True,
    ) -> None:
        """Append a message to in-memory and persisted history."""
        if update_context:
            self.context.append(message)
        if update_full_history:
            self.full_history.append(message)
        if update_db:
            self.persistence.append_message(message)

    def add_message_to_db(self, message: Dict[str, Any]) -> None:
        """Persist a message to the SQLite transcript store."""
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
                self.add_message_to_db({"role": "system", "content": prompt})
            while True:
                message = self.persistence.poll_new_webui_input(self.webui_user_input_last_timestamp)
                if message is not None:
                    self.webui_user_input_last_timestamp = self.persistence.get_latest_webui_input_timestamp()
                    self.set_user_input_requested(False)
                    return message
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

    def execute_tool_calls_from_message(
        self,
        message: dict[str, Any],
        *,
        return_tool_return_types: bool = False,
    ):
        """Execute tool calls found in an assistant message."""
        if not has_tool_call(message):
            empty = ([], []) if return_tool_return_types else []
            return empty
        results = self.tool_executor.execute_tool_calls(message["tool_calls"])
        tool_messages = [result.message for result in results]
        tool_return_types = [result.return_type for result in results]
        return (tool_messages, tool_return_types) if return_tool_return_types else tool_messages

    def _parse_tool_response_payload(self, content: Any) -> Optional[Dict[str, Any]]:
        """Parse dict-like tool payloads from tool message content."""
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _extract_image_paths_from_tool_response(self, content: Any) -> list[str]:
        """Extract one or multiple image paths from a tool response payload."""
        payload = self._parse_tool_response_payload(content)
        if payload is not None:
            image_paths = payload.get("image_paths")
            if isinstance(image_paths, list):
                return [value for value in image_paths if isinstance(value, str)]
            image_path = payload.get("image_path")
            if isinstance(image_path, str):
                return [image_path]
            return []
        if isinstance(content, str):
            return [content]
        return []

    def _build_skill_doc_messages(
        self,
        tool_response: Dict[str, Any],
        tool_call_info: Optional[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """Expand skill documentation payloads into message sections."""
        if tool_call_info is None:
            return []
        tool_name = tool_call_info.get("function", {}).get("name")
        skill_tool_names = {skill.tool_name for skill in self.skill_catalog}
        if tool_name not in skill_tool_names:
            return []
        payload = self._parse_tool_response_payload(tool_response.get("content"))
        if payload is None or not isinstance(payload.get("files"), dict):
            return []
        skill_root = payload.get("path")
        skill_root_path = Path(skill_root) if isinstance(skill_root, str) else None
        messages = []
        for relative_path, file_content in payload["files"].items():
            if not isinstance(relative_path, str) or not isinstance(file_content, str):
                continue
            markdown_path = skill_root_path / relative_path if skill_root_path is not None else None
            for section in split_markdown_into_message_sections(file_content, markdown_path=markdown_path):
                if len(section["image_paths"]) == 0:
                    messages.append(generate_openai_message(content=section["text"], role="user"))
                    continue
                try:
                    messages.append(
                        generate_openai_message(
                            content=section["text"],
                            role="user",
                            image_path=section["image_paths"][0],
                        )
                    )
                except Exception as exc:
                    logger.warning("Failed to load skill image '%s': %s", section["image_paths"][0], exc)
                    messages.append(generate_openai_message(content=section["text"], role="user"))
                for image_path in section["image_paths"][1:]:
                    try:
                        messages.append(generate_openai_message(content="", role="user", image_path=image_path))
                    except Exception as exc:
                        logger.warning("Failed to load skill image '%s': %s", image_path, exc)
        return messages

    def _inject_skill_doc_messages_to_context(
        self,
        tool_response: Dict[str, Any],
        tool_call_info: Optional[Dict[str, Any]],
    ) -> None:
        """Append expanded skill-doc messages to context and full history."""
        for message in self._build_skill_doc_messages(tool_response, tool_call_info):
            self.update_message_history(message, update_context=True, update_full_history=True)

    def get_manager_metadata_summary(self) -> str:
        """Return a JSON summary of the task manager configuration."""
        llm_summary = repr(self.llm_config)
        llm_import_path = None
        if self.llm_config is not None:
            llm_class = self.llm_config.__class__
            llm_import_path = f"{llm_class.__module__}.{llm_class.__name__}"
        tool_info = []
        for tool in self._collect_base_tools():
            tool_class = tool.__class__
            tool_info.append(
                {
                    "class_name": tool_class.__name__,
                    "import_path": f"{tool_class.__module__}.{tool_class.__name__}",
                }
            )
        return json.dumps(
            {
                "llm_config": llm_summary,
                "llm_config_import_path": llm_import_path,
                "tools": tool_info,
            },
            indent=2,
            default=str,
        )

    def launch_task_manager(self, task_request: str) -> None:
        """Run a skill-driven subtask using the discovered skill tools."""
        if not self.skill_catalog:
            system_message = generate_openai_message(
                content="No skills are available to run the subtask.",
                role="system",
            )
            self.update_message_history(system_message, update_context=True, update_full_history=True)
            print_message(system_message)
            return
        skill_catalog_json = json.dumps(
            [
                {
                    "name": skill.name,
                    "tool_name": skill.tool_name,
                    "description": skill.description,
                    "path": skill.path,
                }
                for skill in self.skill_catalog
            ],
            indent=2,
        )
        self.record_system_message(
            "The user requested to launch a sub-task manager for a specified task. "
            "Select the correct skill, fetch its docs, create a Python script, and execute it. "
            "If critical setup information is missing, ask a single clarification question and include NEED HUMAN. "
            f"Available skill tools:\n{skill_catalog_json}",
            update_context=True,
        )
        self.record_system_message(
            "Current task manager metadata (llm config + tools):\n"
            f"{self.get_manager_metadata_summary()}",
            update_context=True,
        )
        self.run_feedback_loop(
            initial_prompt=task_request.strip() or "(no additional description provided)",
            termination_behavior="return",
            allow_multiple_tool_calls=True,
        )

    def display_command_help(self) -> str:
        """Display the available interactive commands."""
        text = (
            "Below are supported commands.\n"
            "* `/exit`: exit the current loop\n"
            "* `/chat`: enter chat mode\n"
            "* `/monitor <task description>`: enter monitoring mode\n"
            "* `/subtask <task description>`: run a skill-driven subtask\n"
            "* `/skill`: display skills available to the agent\n"
            "* `/return`: return to upper level task\n"
        )
        if self.use_webui:
            self.add_message_to_db({"role": "system", "content": text})
        else:
            print(text)
            self.add_message_to_db({"role": "system", "content": text})
        return text

    def display_available_skills(self) -> str:
        """Display discovered skills."""
        if not self.skill_catalog:
            text = "No skills are available."
        else:
            text = "\n".join(
                ["Skills available to the agent:"]
                + [
                    f"{index}. {skill.name} ({skill.tool_name}) - {skill.description} [{skill.path}]"
                    for index, skill in enumerate(self.skill_catalog, start=1)
                ]
            )
        if self.use_webui:
            self.add_message_to_db({"role": "system", "content": text})
        else:
            print(text)
            self.add_message_to_db({"role": "system", "content": text})
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
                self.add_message_to_db(
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
        self.context.append(
            generate_openai_message(
                content=(
                    f"The tool call sequence {actual_sequence} is not as expected. "
                    "Are you making the right tool calls in the right order? "
                    "If this is intended to address an exception, ignore this message."
                ),
                role="user",
            )
        )

    def postprocess_tool_result(
        self,
        tool_response: Dict[str, Any],
        tool_response_type: ToolReturnType,
        *,
        message_with_yielded_image: str,
        allow_non_image_tool_responses: bool,
        hook_functions: Optional[dict[str, Callable]] = None,
        tool_call_info: Optional[Dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Generate follow-up messages after a tool finishes."""
        hook_functions = hook_functions or {}
        followup_messages: list[dict[str, Any]] = []
        for skill_doc_message in self._build_skill_doc_messages(tool_response, tool_call_info):
            followup_messages.append(skill_doc_message)
        if tool_response_type in (ToolReturnType.IMAGE_PATH, ToolReturnType.DICT):
            image_paths = self._extract_image_paths_from_tool_response(tool_response.get("content"))
            if len(image_paths) > 0:
                hook = hook_functions.get("image_path_tool_response")
                if hook is not None:
                    for image_path in image_paths:
                        hook_messages = hook(image_path) or []
                        followup_messages.extend(list(hook_messages))
                else:
                    followup_messages.append(
                        generate_openai_message(
                            content=message_with_yielded_image,
                            image_path=image_paths,
                            role="user",
                        )
                    )
            elif tool_response_type == ToolReturnType.IMAGE_PATH:
                logger.warning(
                    "Tool returned IMAGE_PATH but no valid image path was found in %s",
                    tool_response.get("content"),
                )
            elif not allow_non_image_tool_responses:
                followup_messages.append(
                    generate_openai_message(
                        content=(
                            f"The tool should return an image path, but got {str(tool_response_type)}. "
                            "Make sure you call the right tool correctly."
                        ),
                        role="user",
                    )
                )
        elif not allow_non_image_tool_responses:
            followup_messages.append(
                generate_openai_message(
                    content=(
                        f"The tool should return an image path, but got {str(tool_response_type)}. "
                        "Make sure you call the right tool correctly."
                    ),
                    role="user",
                )
            )
        return followup_messages

    def prerun_check(self, *args, **kwargs) -> bool:
        """Run preflight validation before execution."""
        return True

    def run(self, *args, **kwargs) -> None:
        """Run the task manager."""
        self.prerun_check()

    def _message_contains_image(self, message: dict[str, Any]) -> bool:
        """Return whether a message contains image payloads."""
        return get_message_elements_as_text(message)["image"] is not None

    def _apply_followup_messages(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        store_all_images_in_context: bool = True,
    ) -> None:
        """Append post-tool follow-up messages while optionally pruning image context."""
        for message in messages:
            update_context = True
            if self._message_contains_image(message) and not store_all_images_in_context:
                update_context = False
            if not self.use_webui:
                print_message(message)
            self.update_message_history(
                message,
                update_context=update_context,
                update_full_history=True,
            )

    def invoke_model_for_state(
        self,
        state: TaskManagerState,
        *,
        message: Optional[str | dict[str, Any] | list[dict[str, Any]]] = None,
        image_path: Optional[str | list[str]] = None,
        context: Optional[list[dict[str, Any]]] = None,
        update_context: bool = True,
        update_full_history: bool = True,
        await_user_input_resolver: Optional[Callable[[TaskManagerState], bool]] = None,
    ) -> dict[str, Any]:
        """Invoke the model for a graph state and persist the exchange.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state to mutate in place.
        message : str or dict or list of dict, optional
            Message payload to append before invoking the model.
        image_path : str or list of str, optional
            Image path payload to append before invoking the model.
        context : list of dict, optional
            Explicit conversation context to send to the model. If omitted,
            the active task-manager context is used.
        update_context : bool, default=True
            Whether the outgoing and assistant messages update the active
            context.
        update_full_history : bool, default=True
            Whether the outgoing and assistant messages update the full
            transcript.
        await_user_input_resolver : callable, optional
            Callback used to derive `state.await_user_input` after the model
            response is recorded.

        Returns
        -------
        dict
            Updated graph state payload for LangGraph.
        """
        self.state = state
        response, outgoing = self.invoke_model_raw(
            message=message,
            image_path=image_path,
            context=context,
            return_outgoing_message=True,
        )
        if outgoing is not None:
            if not self.use_webui:
                print_message(outgoing)
            self.update_message_history(
                outgoing,
                update_context=update_context,
                update_full_history=update_full_history,
            )
        if not self.use_webui:
            print_message(response)
        self.update_message_history(
            response,
            update_context=update_context,
            update_full_history=update_full_history,
        )
        if await_user_input_resolver is not None:
            state.await_user_input = await_user_input_resolver(state)
        return state.model_dump()

    def execute_tools_for_state(
        self,
        state: TaskManagerState,
        *,
        message_with_yielded_image: str,
        allow_non_image_tool_responses: bool,
        hook_functions: Optional[dict[str, Callable]] = None,
        store_all_images_in_context: bool = True,
    ) -> dict[str, Any]:
        """Execute tool calls from the latest response and persist follow-up messages.

        Args:
            state: The active graph state whose latest assistant response will be executed.
            message_with_yielded_image: Text used for image follow-up user messages.
            allow_non_image_tool_responses: Whether non-image tool results are acceptable.
            hook_functions: Optional post-tool hook mapping.
            store_all_images_in_context: Whether follow-up images remain in active context.

        Returns:
            The updated graph state payload for LangGraph.
        """
        self.state = state
        response = state.latest_response
        tool_messages, tool_return_types = self.execute_tool_calls_from_message(
            response,
            return_tool_return_types=True,
        )
        tool_call_info_list = get_tool_call_info(response, index=None) if has_tool_call(response) else []
        for tool_message in tool_messages:
            if not self.use_webui:
                print_message(tool_message)
            self.update_message_history(tool_message, update_context=True, update_full_history=True)

        followup_messages: list[dict[str, Any]] = []
        for index, (tool_message, tool_return_type) in enumerate(zip(tool_messages, tool_return_types)):
            tool_call_info = tool_call_info_list[index] if index < len(tool_call_info_list) else None
            followup_messages.extend(
                self.postprocess_tool_result(
                    tool_message,
                    tool_return_type,
                    message_with_yielded_image=message_with_yielded_image,
                    allow_non_image_tool_responses=allow_non_image_tool_responses,
                    hook_functions=hook_functions,
                    tool_call_info=tool_call_info,
                )
            )
        self._apply_followup_messages(
            followup_messages,
            store_all_images_in_context=store_all_images_in_context,
        )
        return state.model_dump()

    def build_chat_graph(self):
        """Build the base chat graph."""
        node_factory = self.node_factory
        builder = StateGraph(ChatGraphState)
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
        builder.add_edge(START, "await_or_ingest_user_input")
        builder.add_conditional_edges(
            "await_or_ingest_user_input",
            node_factory.route_after_chat_input,
        )
        builder.add_conditional_edges(
            "call_model",
            node_factory.route_after_chat_response,
        )
        builder.add_edge("execute_tools", "call_model")
        return builder.compile()

    def build_feedback_loop_graph(self):
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
        builder.add_edge("execute_tools", "finalize_round")
        builder.add_conditional_edges(
            "finalize_round",
            node_factory.route_after_feedback_round,
        )
        return builder.compile()

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
            messages=list(self.context),
            full_history=list(self.full_history),
            round_index=self.state.round_index,
            termination_behavior=termination_behavior or "user",
            store_all_images_in_context=store_all_images_in_context,
            bootstrap_message=message,
            await_user_input=message is None,
        )
        self.state = initial_state
        try:
            final_state = self.chat_graph.invoke(initial_state)
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
        self.state = ChatGraphState.model_validate(final_state)

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
        hook_functions: Optional[dict[str, Callable]] = None,
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
        hook_functions : dict, optional
            Optional post-tool hook callbacks.
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
            messages=list(self.context),
            full_history=list(self.full_history),
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
            hook_functions=hook_functions or {},
            expected_tool_call_sequence=expected_tool_call_sequence,
            expected_tool_call_sequence_tolerance=expected_tool_call_sequence_tolerance,
            termination_behavior=termination_behavior,
            max_arounds_reached_behavior=max_arounds_reached_behavior,
        )
        self.state = initial_state
        try:
            final_state = self.feedback_loop_graph.invoke(initial_state)
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
                self.run_conversation(store_all_images_in_context=True, termination_behavior="user")
            return
        self.state = FeedbackLoopState.model_validate(final_state)
