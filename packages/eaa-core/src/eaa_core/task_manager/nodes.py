from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END
from langgraph.runtime import Runtime

from eaa_core.exceptions import MaxRoundsReached
from eaa_core.message_proc import (
    generate_openai_message,
    has_tool_call,
    print_message,
    purge_context_images,
)
from eaa_core.task_manager.state import (
    ChatGraphState,
    ChatRuntimeContext,
    FeedbackLoopState,
    TaskManagerState,
)

if TYPE_CHECKING:
    from eaa_core.task_manager.base import BaseTaskManager


class NodeFactory:
    """Factory for LangGraph nodes and routers used by task managers.

    Notes
    -----
    The factory keeps node logic state-centric: nodes read flags from the
    passed state object, mutate that state in place, and return a dumped state
    payload for LangGraph. Router methods interpret the same fields to decide
    the next edge in the graph.
    """

    def __init__(self, task_manager: "BaseTaskManager"):
        """Initialize the node factory.

        Parameters
        ----------
        task_manager : BaseTaskManager
            Task manager that owns model invocation, tool execution, persistence,
            and interactive helper methods used by the nodes.
        """
        self.task_manager = task_manager

    def chat_response_requires_user_input(self, state: TaskManagerState) -> bool:
        """Return whether a chat response should hand control back to the user.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state whose latest assistant response is inspected.

        Returns
        -------
        bool
            ``True`` when the latest assistant response has no tool call and
            the chat graph should wait for another user turn.
        """
        return not has_tool_call(state.latest_response)

    def feedback_response_requires_user_input(self, state: FeedbackLoopState) -> bool:
        """Return whether the latest feedback response should prompt the user.

        Parameters
        ----------
        state : FeedbackLoopState
            Active feedback-loop graph state.

        Returns
        -------
        bool
            ``True`` when the latest assistant response contains ``NEED HUMAN``
            or contains ``TERMINATE`` while ``state.termination_behavior`` is
            ``"ask"``.
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

        Parameters
        ----------
        state : TaskManagerState
            Active graph state to mutate.
        message : dict[str, object]
            Message payload to append.
        update_context : bool, default=True
            Whether to append the message to ``state.messages``.
        update_full_history : bool, default=True
            Whether to append the message to ``state.full_history``.
        write_to_webui : bool, default=True
            Whether to mirror the message to the explicit WebUI display store
            when a session database is configured.
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

        Parameters
        ----------
        state : TaskManagerState
            Active graph state to mutate.
        messages : list[dict[str, object]]
            Follow-up messages to append in order.
        store_all_images_in_context : bool, default=True
            Whether image-bearing follow-up messages should remain in
            ``state.messages``. Messages are always recorded in
            ``state.full_history``.

        Notes
        -----
        When ``store_all_images_in_context`` is ``False``, image-bearing
        follow-up messages are treated as transcript-only messages so they do
        not bloat the active model context.
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
        """Invoke the model and record the resulting exchange on the state.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state to mutate.
        message : str or dict or list of dict, optional
            Optional outgoing payload to send before model invocation.
        image_path : str or list of str, optional
            Optional image payload paired with ``message`` when ``message`` is a
            plain string.
        context : list[dict[str, object]], optional
            Explicit message context to send to the model. When omitted, the
            caller is expected to have already prepared ``state.messages``.
        update_context : bool, default=True
            Whether the outgoing and assistant messages should be appended to
            ``state.messages``.
        update_full_history : bool, default=True
            Whether the outgoing and assistant messages should be appended to
            ``state.full_history``.
        await_user_input_resolver : callable, optional
            Resolver called after the assistant response is recorded. When
            provided, its return value becomes ``state.await_user_input``.

        Returns
        -------
        dict[str, object]
            Dumped state payload after the exchange is recorded.
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

        Parameters
        ----------
        state : TaskManagerState
            Active graph state whose latest assistant response contains the
            tool calls to execute.

        Returns
        -------
        dict[str, object]
            Dumped state payload after tool messages have been appended and
            the tool transcript has been refreshed.
        """
        response = state.latest_response
        tool_messages = self.task_manager.tool_executor.execute_tool_calls_from_message(response)
        for tool_message in tool_messages:
            if not self.task_manager.use_webui:
                print_message(tool_message)
            self.update_message_history_for_state(
                state,
                tool_message,
                update_context=True,
                update_full_history=True,
            )
        return state.model_dump()

    def enforce_tool_call_sequence_for_state(
        self,
        state: FeedbackLoopState,
        expected_tool_call_sequence: list[str],
        tolerance: int = 0,
    ) -> None:
        """Append a warning if recent tool calls violate the expected sequence.

        Parameters
        ----------
        state : FeedbackLoopState
            Active feedback-loop state.
        expected_tool_call_sequence : list[str]
            Expected repeating sequence of tool names.
        tolerance : int, default=0
            Number of recent mismatches to ignore when comparing the observed
            suffix against the expected sequence.

        Notes
        -----
        When the recent tool-call suffix cannot be aligned with the expected
        sequence, a synthetic user message is appended to ``state.messages`` to
        steer the model back onto the expected workflow.
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
        """Ingest bootstrap input or wait for the next interactive chat turn.

        Parameters
        ----------
        state : ChatGraphState
            Active chat graph state.

        Returns
        -------
        dict[str, object]
            Dumped chat state after bootstrap handling, slash-command handling,
            or user-message ingestion.

        Notes
        -----
        Behavior depends on the current state fields.

        - If ``state.bootstrap_message`` is not ``None``, the bootstrap payload
          is normalized into one or more user messages, appended to the state,
          ``state.bootstrap_message`` is cleared, and
          ``state.await_user_input`` is set to ``False``.
        - If ``state.await_user_input`` is ``False`` and no bootstrap payload
          is pending, the node is a no-op.
        - Otherwise, the node prompts for terminal or WebUI input and handles
          slash commands as control signals:

          ``/exit``
              Sets ``state.exit_requested``.
          ``/return``
              Sets ``state.return_requested``.
          ``/monitor <task>``
              Sets ``state.monitor_requested`` and stores the task
              description in ``state.monitor_task_description``.
          ``/skill`` and ``/help``
              Perform side effects only and keep waiting for input.

        Regular user text is appended to the state and flips
        ``state.await_user_input`` to ``False`` so the graph can continue to
        the model call.
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

        Parameters
        ----------
        state : ChatGraphState
            Active chat graph state after ``await_or_ingest_user_input``.

        Returns
        -------
        str
            ``END`` when the input node requested exit, return, or monitoring;
            otherwise ``"call_model"``.
        """
        if (
            state.monitor_requested
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

        Parameters
        ----------
        state : TaskManagerState
            Active graph state.
        runtime : Runtime[ChatRuntimeContext], optional
            LangGraph runtime object. Chat runs may provide memory-store
            context through ``runtime.context``.

        Returns
        -------
        dict[str, object]
            Dumped state payload after the model response is recorded.

        Notes
        -----
        Behavior depends on the state subtype and state flags.

        - For feedback-loop states, if ``state.initial_prompt_pending`` is
          ``True``, the node sends ``state.initial_prompt`` together with
          ``state.initial_image_path`` on top of the current context. After the
          assistant response is recorded, ``state.initial_prompt_pending`` is
          set to ``False``.
        - Otherwise, the node sends only the prepared context.
        - For chat states whose last message is from the user, long-term memory
          snippets are injected when ``runtime.context`` provides a memory
          store, and qualifying user messages are saved back to memory after
          the response is recorded.
        - ``state.await_user_input`` is derived from
          ``chat_response_requires_user_input`` or
          ``feedback_response_requires_user_input`` depending on the state type.
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

        Parameters
        ----------
        state : ChatGraphState
            Active chat graph state after ``call_model``.

        Returns
        -------
        str
            ``"execute_tools"`` when the latest assistant response contains
            tool calls, ``END`` when ``state.termination_behavior`` is
            ``"return"``, and ``"await_or_ingest_user_input"`` otherwise.
        """
        response = state.latest_response or {}
        if has_tool_call(response):
            return "execute_tools"
        if state.termination_behavior == "return":
            return END
        return "await_or_ingest_user_input"

    def execute_tools(self, state: TaskManagerState) -> dict[str, object]:
        """Execute tool calls for the active graph state.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state whose latest assistant response contains tool
            calls.

        Returns
        -------
        dict[str, object]
            Dumped state payload after tool execution.

        Notes
        -----
        Feedback-loop states use the follow-up text and validation settings
        stored on the state. Plain chat states use a default image follow-up
        message and allow non-image tool outputs.
        """
        if isinstance(state, FeedbackLoopState):
            return self.task_manager.execute_tools_for_state(
                state,
                message_with_yielded_image=state.message_with_yielded_image,
                allow_non_image_tool_responses=state.allow_non_image_tool_responses,
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

        Parameters
        ----------
        state : TaskManagerState
            Active graph state after tool execution.
        allow_non_image_tool_responses : bool
            Whether non-image tool outputs are acceptable for the current
            workflow.

        Returns
        -------
        bool
            ``True`` when the graph should visit ``image_followup`` before the
            next model turn.

        Notes
        -----
        Follow-up is required when any tool result emits follow-up messages,
        returns image paths, or returns a non-image payload in a flow that
        requires image output.
        """
        tool_messages = state.latest_tool_messages
        for tool_message in tool_messages:
            if len(
                self.task_manager.tool_executor.extract_followup_messages_from_tool_response(
                    tool_message.get("content")
                )
            ) > 0:
                return True
            image_paths = self.task_manager.tool_executor.extract_image_paths_from_tool_response(
                tool_message.get("content")
            )
            if len(image_paths) > 0:
                return True
            if not allow_non_image_tool_responses:
                return True
        return False

    def build_tool_followup_messages(
        self,
        state: TaskManagerState,
        *,
        message_with_yielded_image: str,
        allow_non_image_tool_responses: bool,
    ) -> list[dict[str, object]]:
        """Build follow-up messages for the latest tool batch.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state after tool execution.
        message_with_yielded_image : str
            User-facing text used when a tool returns image paths.
        allow_non_image_tool_responses : bool
            Whether non-image tool outputs are acceptable for the current
            workflow.

        Returns
        -------
        list[dict[str, object]]
            Follow-up messages derived from the most recent tool responses,
            including emitted follow-up messages, image-yield messages, and
            non-image warnings when required.
        """
        tool_messages = state.latest_tool_messages
        followup_messages: list[dict[str, object]] = []
        for tool_message in tool_messages:
            followup_messages.extend(
                self.task_manager.tool_executor.build_tool_followup_messages(
                    tool_message,
                    message_with_yielded_image=message_with_yielded_image,
                    allow_non_image_tool_responses=allow_non_image_tool_responses,
                )
            )
        return followup_messages

    def image_followup(self, state: TaskManagerState) -> dict[str, object]:
        """Append follow-up messages after a tool batch completes.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state after ``execute_tools``.

        Returns
        -------
        dict[str, object]
            Dumped state payload after the generated follow-up messages are
            appended.

        Notes
        -----
        Feedback-loop states use per-workflow follow-up settings stored on the
        state. Chat states always use the default image follow-up wording and
        may optionally omit image messages from the active context when
        ``state.store_all_images_in_context`` is ``False``.
        """
        if isinstance(state, FeedbackLoopState):
            followup_messages = self.build_tool_followup_messages(
                state,
                message_with_yielded_image=state.message_with_yielded_image,
                allow_non_image_tool_responses=state.allow_non_image_tool_responses,
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

        Parameters
        ----------
        state : TaskManagerState
            Active graph state after ``execute_tools``.

        Returns
        -------
        str
            ``"image_followup"`` when tool outputs require follow-up,
            ``"finalize_round"`` for feedback-loop states that can advance
            directly, or ``"call_model"`` for plain chat states.
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

        Parameters
        ----------
        state : FeedbackLoopState
            Active feedback-loop graph state after ``call_model`` or another
            assistant-producing node.

        Returns
        -------
        str
            Next node name for the feedback workflow.

        Notes
        -----
        Routing depends on several state fields and response properties.

        - The graph ends immediately when ``state.chat_requested``,
          ``state.exit_requested``, or ``state.return_requested`` is set.
        - A response containing ``TERMINATE`` ends the graph when
          ``state.termination_behavior`` is ``"return"``.
        - When ``state.await_user_input`` is ``True``, the graph routes to
          ``handle_human_gate``.
        - Responses with no tool call, or with too many tool calls for the
          configured workflow, route to ``reprompt_model``.
        - Otherwise the response routes to ``execute_tools``.
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
        """Handle ``TERMINATE`` and ``NEED HUMAN`` responses.

        Parameters
        ----------
        state : FeedbackLoopState
            Active feedback-loop graph state.

        Returns
        -------
        dict[str, object]
            Dumped state payload after user input is handled.

        Notes
        -----
        When ``state.await_user_input`` is ``False``, this node is a no-op.
        Otherwise it prompts the user and interprets the response as follows.

        - ``/exit`` sets ``state.exit_requested``.
        - ``/chat`` sets ``state.chat_requested`` so the task-manager boundary
          can hand off to chat mode after the graph exits.
        - ``/help`` shows help and keeps waiting for input.
        - Any other text is appended as a user message and sent back through
          the model using the current ``state.messages`` context.
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
        """Reprompt the model when the previous response violated loop rules.

        Parameters
        ----------
        state : FeedbackLoopState
            Active feedback-loop graph state.

        Returns
        -------
        dict[str, object]
            Dumped state payload after the corrective prompt and assistant
            response are recorded.

        Notes
        -----
        The corrective prompt depends on the latest assistant response.

        - If too many tool calls were issued for the workflow, the prompt asks
          the model to retry with one tool call.
        - Otherwise, the prompt tells the model that no valid tool call was
          found and reminds it to emit ``NEED HUMAN`` when appropriate.
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
        """Finalize one feedback-loop round and prune image context if needed.

        Parameters
        ----------
        state : FeedbackLoopState
            Active feedback-loop graph state after tool execution and optional
            follow-up handling.

        Returns
        -------
        dict[str, object]
            Dumped state payload after round-finalization updates.

        Notes
        -----
        Finalization performs three state-dependent steps.

        - If ``state.expected_tool_call_sequence`` is set, the recent tool
          execution history is validated and a warning is appended on
          mismatch.
        - If ``state.n_first_images_to_keep_in_context`` or
          ``state.n_last_images_to_keep_in_context`` is set, image messages in
          ``state.messages`` are pruned while preserving the configured number
          of earliest and latest images.
        - ``state.round_index`` is incremented, and ``MaxRoundsReached`` is
          raised when the new round index reaches ``state.max_rounds`` and
          ``state.max_arounds_reached_behavior`` is ``"raise"``.
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

        Parameters
        ----------
        state : FeedbackLoopState
            Active feedback-loop graph state after ``finalize_round``.

        Returns
        -------
        str
            ``END`` when ``state.round_index`` has reached ``state.max_rounds``;
            otherwise ``"call_model"`` for the next feedback turn.
        """
        if state.round_index >= state.max_rounds:
            return END
        return "call_model"
