from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END
from langgraph.runtime import Runtime

from eaa.core.exceptions import MaxRoundsReached
from eaa.core.message_proc import (
    generate_openai_message,
    get_tool_call_info,
    has_tool_call,
    print_message,
    purge_context_images,
)
from eaa.core.task_manager.state import (
    ChatGraphState,
    ChatRuntimeContext,
    FeedbackLoopState,
    TaskManagerState,
)
from eaa.core.tooling.base import ToolReturnType

if TYPE_CHECKING:
    from eaa.core.task_manager.base import BaseTaskManager


class NodeFactory:
    """Factory for LangGraph nodes and routers used by task managers."""

    def __init__(self, task_manager: "BaseTaskManager"):
        """Initialize the node factory.

        Args:
            task_manager: Task manager that owns graph state and execution helpers.
        """
        self.task_manager = task_manager

    def chat_response_requires_user_input(self, state: TaskManagerState) -> bool:
        """Return whether a chat response should hand control back to the user.

        Args:
            state: Active graph state.

        Returns:
            Whether the latest assistant response requires user input.
        """
        return not has_tool_call(state.latest_response)

    def feedback_response_requires_user_input(self, state: FeedbackLoopState) -> bool:
        """Return whether the latest feedback response should prompt the user.

        Args:
            state: Active feedback-loop graph state.

        Returns:
            Whether the latest assistant response requires user input.
        """
        response = state.latest_response or {}
        content = response.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
        content = content or ""
        if "NEED HUMAN" in content:
            return True
        return "TERMINATE" in content and state.termination_behavior == "ask"

    def update_message_history_for_state(
        self,
        state: TaskManagerState,
        message: dict[str, object],
        *,
        update_context: bool = True,
        update_full_history: bool = True,
        write_to_webui: bool = True,
    ) -> None:
        """Append a message directly to a graph state.

        Args:
            state: Active graph state.
            message: Message payload to append.
            update_context: Whether to append to active context.
            update_full_history: Whether to append to full history.
            write_to_webui: Whether to append to explicit WebUI display storage.
        """
        if update_context:
            state.messages.append(message)
        if update_full_history:
            state.full_history.append(message)
        if write_to_webui and self.task_manager.session_db_path is not None:
            self.task_manager.persistence.append_message(message)

    def apply_followup_messages_for_state(
        self,
        state: TaskManagerState,
        messages: list[dict[str, object]],
        *,
        store_all_images_in_context: bool = True,
    ) -> None:
        """Append follow-up messages directly to a graph state.

        Args:
            state: Active graph state.
            messages: Follow-up messages to append.
            store_all_images_in_context: Whether image messages remain in active context.
        """
        for message in messages:
            update_context = True
            if self.task_manager._message_contains_image(message) and not store_all_images_in_context:
                update_context = False
            if not self.task_manager.use_webui:
                print_message(message)
            self.update_message_history_for_state(
                state,
                message,
                update_context=update_context,
                update_full_history=True,
            )

    def invoke_model_for_state(
        self,
        state: TaskManagerState,
        *,
        message: str | dict[str, object] | list[dict[str, object]] | None = None,
        image_path: str | list[str] | None = None,
        context: list[dict[str, object]] | None = None,
        update_context: bool = True,
        update_full_history: bool = True,
        await_user_input_resolver: object | None = None,
    ) -> dict[str, object]:
        """Invoke the model and append the exchange to the provided state.

        Args:
            state: Active graph state.
            message: Optional outgoing message payload.
            image_path: Optional image payload.
            context: Explicit model context.
            update_context: Whether outgoing and assistant messages update context.
            update_full_history: Whether outgoing and assistant messages update transcript.
            await_user_input_resolver: Optional callback to derive await-user-input.

        Returns:
            Updated graph state payload.
        """
        response, outgoing = self.task_manager.invoke_model_raw(
            message=message,
            image_path=image_path,
            context=context,
            return_outgoing_message=True,
        )
        if outgoing is not None:
            if not self.task_manager.use_webui:
                print_message(outgoing)
            self.update_message_history_for_state(
                state,
                outgoing,
                update_context=update_context,
                update_full_history=update_full_history,
            )
        if not self.task_manager.use_webui:
            print_message(response)
        self.update_message_history_for_state(
            state,
            response,
            update_context=update_context,
            update_full_history=update_full_history,
        )
        if await_user_input_resolver is not None:
            state.await_user_input = await_user_input_resolver(state)
        return state.model_dump()

    def execute_tools_for_state(self, state: TaskManagerState) -> dict[str, object]:
        """Execute tool calls and append tool messages to the provided state.

        Args:
            state: Active graph state.

        Returns:
            Updated graph state payload.
        """
        response = state.latest_response
        tool_messages, tool_return_types = self.task_manager.tool_executor.execute_tool_calls_from_message(
            response,
            return_tool_return_types=True,
        )
        for tool_message in tool_messages:
            if not self.task_manager.use_webui:
                print_message(tool_message)
            self.update_message_history_for_state(
                state,
                tool_message,
                update_context=True,
                update_full_history=True,
            )
        state.latest_tool_return_types = list(tool_return_types)
        return state.model_dump()

    def enforce_tool_call_sequence_for_state(
        self,
        state: FeedbackLoopState,
        expected_tool_call_sequence: list[str],
        tolerance: int = 0,
    ) -> None:
        """Append a warning if recent tool calls violate the expected sequence.

        Args:
            state: Active feedback-loop state.
            expected_tool_call_sequence: Expected tool-call order.
            tolerance: Allowed mismatch tolerance.
        """
        if len(self.task_manager.tool_executor.tool_execution_history) <= 1:
            return
        n_actual = min(
            len(self.task_manager.tool_executor.tool_execution_history),
            len(expected_tool_call_sequence),
        ) - tolerance
        if n_actual <= 0:
            return
        actual_sequence = [
            entry["tool_name"]
            for entry in self.task_manager.tool_executor.tool_execution_history[-n_actual:]
        ]
        expanded_expected = list(expected_tool_call_sequence) * 2
        for index in range(len(expanded_expected) - len(actual_sequence) + 1):
            if expanded_expected[index : index + len(actual_sequence)] == actual_sequence:
                return
        self.update_message_history_for_state(
            state,
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
            write_to_webui=False,
        )

    def await_or_ingest_user_input(self, state: ChatGraphState) -> dict[str, object]:
        """Ingest bootstrap messages or prompt the user for a new chat turn.

        Args:
            state: Active chat graph state.

        Returns:
            Updated graph state payload.
        """
        if state.bootstrap_message is not None:
            bootstrap = state.bootstrap_message
            if isinstance(bootstrap, str):
                message = generate_openai_message(content=bootstrap, role="user")
                if not self.task_manager.use_webui:
                    print_message(message)
                self.update_message_history_for_state(
                    state,
                    message,
                    update_context=True,
                    update_full_history=True,
                )
            elif isinstance(bootstrap, dict):
                if not self.task_manager.use_webui:
                    print_message(bootstrap)
                self.update_message_history_for_state(
                    state,
                    bootstrap,
                    update_context=True,
                    update_full_history=True,
                )
            elif isinstance(bootstrap, list):
                for message in bootstrap:
                    if not self.task_manager.use_webui:
                        print_message(message)
                    self.update_message_history_for_state(
                        state,
                        message,
                        update_context=True,
                        update_full_history=True,
                    )
            else:
                raise ValueError("`message` must be one of: str, dict, list[dict], or None.")
            state.bootstrap_message = None
            state.await_user_input = False
            return state.model_dump()
        if not state.await_user_input:
            return state.model_dump()

        while True:
            user_message = self.task_manager.get_user_input(
                prompt=(
                    "Enter a message (/exit: exit; /return: return to upper level task; "
                    "/help: show command help; /skill: list skills): "
                )
            )
            stripped = user_message.strip()
            command, _, remainder = stripped.partition(" ")
            command_lower = command.lower()
            if command_lower == "/exit" and remainder == "":
                state.exit_requested = True
                return state.model_dump()
            if command_lower == "/return" and remainder == "":
                state.return_requested = True
                return state.model_dump()
            if command_lower == "/monitor":
                if remainder.strip():
                    state.monitor_requested = True
                    state.monitor_task_description = remainder.strip()
                    state.await_user_input = False
                    return state.model_dump()
                continue
            if command_lower == "/subtask":
                state.subtask_requested = True
                state.subtask_task_description = remainder.strip()
                state.await_user_input = False
                return state.model_dump()
            if command_lower == "/skill" and remainder == "":
                self.task_manager.display_available_skills()
                continue
            if command_lower == "/help" and remainder == "":
                self.task_manager.display_command_help()
                continue
            message = generate_openai_message(content=user_message, role="user")
            if not self.task_manager.use_webui:
                print_message(message)
            self.update_message_history_for_state(
                state,
                message,
                update_context=True,
                update_full_history=True,
            )
            state.await_user_input = False
            return state.model_dump()

    def route_after_chat_input(self, state: ChatGraphState) -> str:
        """Route the chat graph after user-input ingestion.

        Args:
            state: Active chat graph state.

        Returns:
            Next node name.
        """
        if (
            state.monitor_requested
            or state.subtask_requested
            or state.exit_requested
            or state.return_requested
        ):
            return END
        return "call_model"

    def call_model(
        self,
        state: TaskManagerState,
        runtime: Runtime[ChatRuntimeContext] | None = None,
    ) -> dict[str, object]:
        """Call the model for the current graph turn.

        Args:
            state: Active graph state.

        Returns:
            Updated graph state payload.
        """
        await_user_input_resolver = (
            self.feedback_response_requires_user_input
            if isinstance(state, FeedbackLoopState)
            else self.chat_response_requires_user_input
        )
        message = None
        image_path = None
        context = list(state.messages)
        if isinstance(state, FeedbackLoopState) and state.initial_prompt_pending:
            message = state.initial_prompt
            image_path = state.initial_image_path
        memory_message = None
        latest_message = state.messages[-1] if state.messages else None
        is_chat_user_turn = isinstance(state, ChatGraphState) and state.last_message_is_from_user()
        if (
            is_chat_user_turn
            and runtime is not None
            and runtime.context is not None
            and runtime.context.memory_store is not None
        ):
            memory_results = self.task_manager.memory_manager.retrieve_user_memories(
                latest_message,
                runtime.context.memory_namespace,
            )
            memory_message = self.task_manager.memory_manager.build_memory_context_message(memory_results)
            context = self.task_manager.memory_manager.inject_memory_context(
                list(state.messages),
                memory_message,
            )
        result = self.invoke_model_for_state(
            state,
            message=message,
            image_path=image_path,
            context=context,
            await_user_input_resolver=await_user_input_resolver,
        )
        if (
            is_chat_user_turn
            and runtime is not None
            and runtime.context is not None
            and runtime.context.memory_store is not None
            and self.task_manager.memory_manager.user_message_triggers_memory(latest_message)
        ):
            self.task_manager.memory_manager.save_user_memory(
                latest_message,
                runtime.context.memory_namespace,
            )
        if isinstance(state, FeedbackLoopState) and state.initial_prompt_pending:
            state.initial_prompt_pending = False
            result["initial_prompt_pending"] = False
        return result

    def route_after_chat_response(self, state: ChatGraphState) -> str:
        """Route the chat graph after the model responds.

        Args:
            state: Active chat graph state.

        Returns:
            Next node name.
        """
        response = state.latest_response or {}
        if has_tool_call(response):
            return "execute_tools"
        if state.termination_behavior == "return":
            return END
        return "await_or_ingest_user_input"

    def execute_tools(self, state: TaskManagerState) -> dict[str, object]:
        """Execute tool calls for the active graph state.

        Args:
            state: Active graph state.

        Returns:
            Updated graph state payload.
        """
        if isinstance(state, FeedbackLoopState):
            return self.task_manager.execute_tools_for_state(
                state,
                message_with_yielded_image=state.message_with_yielded_image,
                allow_non_image_tool_responses=state.allow_non_image_tool_responses,
                hook_functions=self.task_manager.active_feedback_hook_functions,
                store_all_images_in_context=state.store_all_images_in_context,
            )
        return self.task_manager.execute_tools_for_state(
            state,
            message_with_yielded_image="Here is the image the tool returned.",
            allow_non_image_tool_responses=True,
            store_all_images_in_context=state.store_all_images_in_context,
        )

    def tool_followup_required(
        self,
        state: TaskManagerState,
        *,
        allow_non_image_tool_responses: bool,
    ) -> bool:
        """Return whether the latest tool batch requires follow-up handling.

        Args:
            state: Active graph state.
            allow_non_image_tool_responses: Whether non-image tool results are acceptable.

        Returns:
            Whether the follow-up node should run.
        """
        response = state.latest_response or {}
        tool_call_info_list = get_tool_call_info(response, index=None) or []
        tool_messages = state.latest_tool_messages
        tool_return_types = state.latest_tool_return_types
        for index, tool_message in enumerate(tool_messages):
            tool_call_info = tool_call_info_list[index] if index < len(tool_call_info_list) else None
            if len(
                self.task_manager.tool_executor.build_skill_doc_messages(
                    tool_message,
                    tool_call_info,
                    self.task_manager.skill_catalog,
                )
            ) > 0:
                return True
            tool_return_type = (
                tool_return_types[index]
                if index < len(tool_return_types)
                else ToolReturnType.TEXT
            )
            if tool_return_type in (ToolReturnType.IMAGE_PATH, ToolReturnType.DICT):
                image_paths = self.task_manager.tool_executor.extract_image_paths_from_tool_response(
                    tool_message.get("content")
                )
                if len(image_paths) > 0:
                    return True
                if (
                    tool_return_type == ToolReturnType.DICT
                    and not allow_non_image_tool_responses
                ):
                    return True
            elif not allow_non_image_tool_responses:
                return True
        return False

    def build_tool_followup_messages(
        self,
        state: TaskManagerState,
        *,
        message_with_yielded_image: str,
        allow_non_image_tool_responses: bool,
        hook_functions: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Build follow-up messages for the latest tool batch.

        Args:
            state: Active graph state.
            message_with_yielded_image: Text used when a tool returns images.
            allow_non_image_tool_responses: Whether non-image tool results are acceptable.
            hook_functions: Optional post-tool hook mapping.

        Returns:
            Follow-up messages to append after tool execution.
        """
        response = state.latest_response or {}
        tool_call_info_list = get_tool_call_info(response, index=None) or []
        tool_messages = state.latest_tool_messages
        tool_return_types = state.latest_tool_return_types
        followup_messages: list[dict[str, object]] = []
        for index, tool_message in enumerate(tool_messages):
            tool_call_info = tool_call_info_list[index] if index < len(tool_call_info_list) else None
            tool_return_type = (
                tool_return_types[index]
                if index < len(tool_return_types)
                else ToolReturnType.TEXT
            )
            followup_messages.extend(
                self.task_manager.tool_executor.build_tool_followup_messages(
                    tool_message,
                    tool_return_type,
                    skill_catalog=self.task_manager.skill_catalog,
                    message_with_yielded_image=message_with_yielded_image,
                    allow_non_image_tool_responses=allow_non_image_tool_responses,
                    hook_functions=hook_functions,
                    tool_call_info=tool_call_info,
                )
            )
        return followup_messages

    def image_followup(self, state: TaskManagerState) -> dict[str, object]:
        """Append follow-up messages after a tool batch completes.

        Args:
            state: Active graph state.

        Returns:
            Updated graph state payload.
        """
        if isinstance(state, FeedbackLoopState):
            followup_messages = self.build_tool_followup_messages(
                state,
                message_with_yielded_image=state.message_with_yielded_image,
                allow_non_image_tool_responses=state.allow_non_image_tool_responses,
                hook_functions=self.task_manager.active_feedback_hook_functions,
            )
            self.apply_followup_messages_for_state(state, followup_messages)
            return state.model_dump()
        followup_messages = self.build_tool_followup_messages(
            state,
            message_with_yielded_image="Here is the image the tool returned.",
            allow_non_image_tool_responses=True,
        )
        self.apply_followup_messages_for_state(
            state,
            followup_messages,
            store_all_images_in_context=state.store_all_images_in_context,
        )
        return state.model_dump()

    def route_after_tool_execution(self, state: TaskManagerState) -> str:
        """Route execution after tools have completed.

        Args:
            state: Active graph state.

        Returns:
            Next node name.
        """
        allow_non_image_tool_responses = (
            state.allow_non_image_tool_responses
            if isinstance(state, FeedbackLoopState)
            else True
        )
        if self.tool_followup_required(
            state,
            allow_non_image_tool_responses=allow_non_image_tool_responses,
        ):
            return "image_followup"
        if isinstance(state, FeedbackLoopState):
            return "finalize_round"
        return "call_model"

    def route_after_feedback_response(self, state: FeedbackLoopState) -> str:
        """Route feedback-loop execution after each assistant response.

        Args:
            state: Active feedback-loop graph state.

        Returns:
            Next node name.
        """
        if state.chat_requested or state.exit_requested or state.return_requested:
            return END
        response = state.latest_response or {}
        content = response.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
        content = content or ""
        if "TERMINATE" in content and state.termination_behavior == "return":
            return END
        if state.await_user_input:
            return "handle_human_gate"
        tool_calls = response.get("tool_calls") or []
        if len(tool_calls) == 0 and "NEED HUMAN" not in content:
            return "reprompt_model"
        if len(tool_calls) > 1 and not state.allow_multiple_tool_calls:
            return "reprompt_model"
        return "execute_tools"

    def handle_human_gate(self, state: FeedbackLoopState) -> dict[str, object]:
        """Handle TERMINATE and NEED HUMAN responses.

        Args:
            state: Active feedback-loop graph state.

        Returns:
            Updated graph state payload.
        """
        if not state.await_user_input:
            return state.model_dump()
        message = self.task_manager.get_user_input(
            prompt=(
                "Termination condition triggered. What to do next? "
                "(`/exit`: exit; `/chat`: chat mode; `/help`: show command help): "
            ),
            display_prompt_in_webui=self.task_manager.use_webui,
        )
        if message.lower() == "/exit":
            state.exit_requested = True
            state.await_user_input = False
            return state.model_dump()
        if message.lower() == "/chat":
            state.await_user_input = False
            state.chat_requested = True
            return state.model_dump()
        if message.lower() == "/help":
            state.await_user_input = True
            self.task_manager.display_command_help()
            return state.model_dump()
        return self.invoke_model_for_state(
            state,
            message=message,
            context=list(state.messages),
            await_user_input_resolver=self.feedback_response_requires_user_input,
        )

    def reprompt_model(self, state: FeedbackLoopState) -> dict[str, object]:
        """Reprompt the model when its previous response did not satisfy loop constraints.

        Args:
            state: Active feedback-loop graph state.

        Returns:
            Updated graph state payload.
        """
        tool_calls = (state.latest_response or {}).get("tool_calls") or []
        if len(tool_calls) > 1 and not state.allow_multiple_tool_calls:
            corrective_message = (
                "There are more than one tool calls in your response. "
                "Make sure you only make one call at a time. Please redo your tool calls."
            )
        else:
            corrective_message = (
                "There is no tool call in the response. Make sure you call the tool correctly. "
                'If you need human intervention, say "NEED HUMAN".'
            )
        return self.invoke_model_for_state(
            state,
            message=corrective_message,
            context=list(state.messages),
            await_user_input_resolver=self.feedback_response_requires_user_input,
        )

    def finalize_round(self, state: FeedbackLoopState) -> dict[str, object]:
        """Finalize one feedback-loop round and prune image context if configured.

        Args:
            state: Active feedback-loop graph state.

        Returns:
            Updated graph state payload.
        """
        if state.expected_tool_call_sequence is not None:
            self.enforce_tool_call_sequence_for_state(
                state,
                state.expected_tool_call_sequence,
                state.expected_tool_call_sequence_tolerance,
            )
        if (
            state.n_last_images_to_keep_in_context is not None
            or state.n_first_images_to_keep_in_context is not None
        ):
            keep_first = state.n_first_images_to_keep_in_context or 0
            keep_last = state.n_last_images_to_keep_in_context or 0
            state.messages = purge_context_images(
                context=state.messages,
                keep_first_n=keep_first,
                keep_last_n=keep_last - 1,
                keep_text=True,
            )
        state.round_index += 1
        if state.round_index >= state.max_rounds and state.max_arounds_reached_behavior == "raise":
            raise MaxRoundsReached()
        return state.model_dump()

    def route_after_feedback_round(self, state: FeedbackLoopState) -> str:
        """Route after a feedback-loop round completes.

        Args:
            state: Active feedback-loop graph state.

        Returns:
            Next node name.
        """
        if state.round_index >= state.max_rounds:
            return END
        return "call_model"
