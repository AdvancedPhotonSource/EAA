from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END
from langgraph.runtime import Runtime

from eaa.core.exceptions import MaxRoundsReached
from eaa.core.message_proc import generate_openai_message, has_tool_call, print_message, purge_context_images
from eaa.core.task_manager.state import (
    ChatGraphState,
    ChatRuntimeContext,
    FeedbackLoopState,
    TaskManagerState,
)

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

    def await_or_ingest_user_input(self, state: ChatGraphState) -> dict[str, object]:
        """Ingest bootstrap messages or prompt the user for a new chat turn.

        Args:
            state: Active chat graph state.

        Returns:
            Updated graph state payload.
        """
        self.task_manager.state = state
        if state.bootstrap_message is not None:
            bootstrap = state.bootstrap_message
            if isinstance(bootstrap, str):
                message = generate_openai_message(content=bootstrap, role="user")
                if not self.task_manager.use_webui:
                    print_message(message)
                self.task_manager.update_message_history(
                    message,
                    update_context=True,
                    update_full_history=True,
                )
            elif isinstance(bootstrap, dict):
                if not self.task_manager.use_webui:
                    print_message(bootstrap)
                self.task_manager.update_message_history(
                    bootstrap,
                    update_context=True,
                    update_full_history=True,
                )
            elif isinstance(bootstrap, list):
                for message in bootstrap:
                    if not self.task_manager.use_webui:
                        print_message(message)
                    self.task_manager.update_message_history(
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
                    self.task_manager.enter_monitoring_mode(remainder.strip())
                continue
            if command_lower == "/subtask":
                self.task_manager.launch_task_manager(remainder.strip())
                continue
            if command_lower == "/skill" and remainder == "":
                self.task_manager.display_available_skills()
                continue
            if command_lower == "/help" and remainder == "":
                self.task_manager.display_command_help()
                continue
            message = generate_openai_message(content=user_message, role="user")
            if not self.task_manager.use_webui:
                print_message(message)
            self.task_manager.update_message_history(
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
        if state.exit_requested or state.return_requested:
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
        context = self.task_manager.context
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
                list(self.task_manager.context),
                memory_message,
            )
        result = self.task_manager.invoke_model_for_state(
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
            )
        return self.task_manager.execute_tools_for_state(
            state,
            message_with_yielded_image="Here is the image the tool returned.",
            allow_non_image_tool_responses=True,
            store_all_images_in_context=state.store_all_images_in_context,
        )

    def route_after_feedback_response(self, state: FeedbackLoopState) -> str:
        """Route feedback-loop execution after each assistant response.

        Args:
            state: Active feedback-loop graph state.

        Returns:
            Next node name.
        """
        if state.exit_requested or state.return_requested:
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
        self.task_manager.state = state
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
            self.task_manager.run_conversation(store_all_images_in_context=True)
            chat_messages = list(self.task_manager.context)
            chat_full_history = list(self.task_manager.full_history)
            self.task_manager.state = state
            state.messages = chat_messages
            state.full_history = chat_full_history
            return state.model_dump()
        if message.lower() == "/help":
            state.await_user_input = True
            self.task_manager.display_command_help()
            return state.model_dump()
        return self.task_manager.invoke_model_for_state(
            state,
            message=message,
            context=self.task_manager.context,
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
        return self.task_manager.invoke_model_for_state(
            state,
            message=corrective_message,
            context=self.task_manager.context,
            await_user_input_resolver=self.feedback_response_requires_user_input,
        )

    def finalize_round(self, state: FeedbackLoopState) -> dict[str, object]:
        """Finalize one feedback-loop round and prune image context if configured.

        Args:
            state: Active feedback-loop graph state.

        Returns:
            Updated graph state payload.
        """
        self.task_manager.state = state
        if state.expected_tool_call_sequence is not None:
            self.task_manager.enforce_tool_call_sequence(
                state.expected_tool_call_sequence,
                state.expected_tool_call_sequence_tolerance,
            )
        if (
            state.n_last_images_to_keep_in_context is not None
            or state.n_first_images_to_keep_in_context is not None
        ):
            keep_first = state.n_first_images_to_keep_in_context or 0
            keep_last = state.n_last_images_to_keep_in_context or 0
            self.task_manager.context = purge_context_images(
                context=self.task_manager.context,
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
