from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END
from langgraph.runtime import Runtime

from eaa_core.exceptions import MaxRoundsReached
from eaa_core.message_proc import (
    complete_unresponded_tool_calls,
    generate_openai_message,
    has_tool_call,
    print_message,
    purge_context_images,
)
from eaa_core.task_manager.commands import parse_user_input_command
from eaa_core.task_manager.state import (
    ChatGraphState,
    ChatRuntimeContext,
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

    def feedback_response_requires_user_input(self, state: TaskManagerState) -> bool:
        """Return whether the latest task response should prompt the user.

        Parameters
        ----------
        state : TaskManagerState
            Active task graph state.

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
            Whether to publish the message to live WebUI clients.
        """
        if update_context:
            state.messages.append(message)
        if write_to_webui:
            self.task_manager.publish_webui_message(message)
        if update_full_history:
            state.full_history.append(message)
            self.task_manager.record_transcript_message(message)

    def apply_followup_messages_for_state(
        self,
        state: TaskManagerState,
        messages: list[dict[str, object]],
        *,
        runtime_context: ChatRuntimeContext | None = None,
    ) -> None:
        """Append follow-up messages directly to a graph state.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state to mutate.
        messages : list[dict[str, object]]
            Follow-up messages to append in order.
        runtime_context : ChatRuntimeContext, optional
            Runtime context used to persist trigger-marked follow-up memories.
        """
        for message in messages:
            if not self.task_manager.use_webui:
                print_message(message)
            self.update_message_history_for_state(
                state,
                message,
                update_context=True,
                update_full_history=True,
            )
            self.task_manager.memory_manager.save_triggered_user_memory(
                message,
                runtime_context,
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
        if self.task_manager.runtime_controller is not None:
            self.task_manager.runtime_controller.check_interrupt()
        response, outgoing = self.task_manager.invoke_model_raw(
            message=message,
            image_path=image_path,
            context=context,
            return_outgoing_message=True,
        )
        if outgoing is None:
            outgoing_messages = []
        elif isinstance(outgoing, list):
            outgoing_messages = outgoing
        else:
            outgoing_messages = [outgoing]
        for outgoing_message in outgoing_messages:
            if not self.task_manager.use_webui:
                print_message(outgoing_message)
            self.update_message_history_for_state(
                state,
                outgoing_message,
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
        if self.task_manager.runtime_controller is not None:
            self.task_manager.runtime_controller.check_interrupt()
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
          ``/skill``
              Without an argument, displays available skills. With an
              argument, appends the selected ``SKILL.md`` to context.

        Regular user text is appended to the state and flips
        ``state.await_user_input`` to ``False`` so the graph can continue to
        the model call.
        """
        if state.bootstrap_message is not None:
            bootstrap = state.bootstrap_message
            if isinstance(bootstrap, str):
                for message in self.task_manager.expand_skill_command_in_text(bootstrap):
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
                    "/skill: list skills): "
                )
            )
            command = parse_user_input_command(user_message)
            if self.task_manager.handle_runtime_command(command):
                continue
            if command.kind == "exit":
                state.exit_requested = True
                return state.model_dump()
            if command.kind == "return":
                state.return_requested = True
                return state.model_dump()
            if command.kind == "skill" and not command.argument:
                self.task_manager.display_available_skills()
                continue
            if command.kind == "skill":
                try:
                    messages = self.task_manager.build_selected_skill_messages(command.argument)
                except ValueError as exc:
                    self.task_manager.record_system_message(
                        str(exc),
                        update_context=True,
                        write_to_webui=True,
                    )
                    continue
                for message in messages:
                    if not self.task_manager.use_webui:
                        print_message(message)
                    self.update_message_history_for_state(
                        state,
                        message,
                        update_context=True,
                        update_full_history=True,
                    )
                if command.text:
                    message = generate_openai_message(content=command.text, role="user")
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
            for message in self.task_manager.expand_skill_command_in_text(command.text):
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
            ``END`` when the input node requested exit or return;
            otherwise ``"call_model"``.
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

        - For task states with a configured initial prompt or image, if
          ``state.initial_prompt_pending`` is ``True``, the node sends
          ``state.initial_prompt`` together with ``state.initial_image_path``
          on top of the current context. After the assistant response is
          recorded, ``state.initial_prompt_pending`` is set to ``False``.
        - Otherwise, the node sends only the prepared context.
        - For chat states whose last message is from the user, long-term memory
          snippets are injected when ``runtime.context`` provides a memory
          store, and qualifying user messages are saved back to memory after
          the response is recorded.
        - ``state.await_user_input`` is derived from
          ``chat_response_requires_user_input`` or
          ``feedback_response_requires_user_input`` depending on the state type.
        """
        is_task_state = not isinstance(state, ChatGraphState)
        has_initial_prompt = bool(state.initial_prompt or state.initial_image_path)
        uses_task_workflow_response_rules = is_task_state and has_initial_prompt
        await_user_input_resolver = (
            self.feedback_response_requires_user_input
            if uses_task_workflow_response_rules
            else self.chat_response_requires_user_input
        )
        message = None
        image_path = None
        context = list(state.messages)
        if uses_task_workflow_response_rules and state.initial_prompt_pending:
            message = self.task_manager.expand_skill_command_in_text(state.initial_prompt)
            image_path = state.initial_image_path
            if image_path is not None and isinstance(message, list) and message:
                last_message = message[-1]
                content = last_message.get("content") if isinstance(last_message, dict) else None
                if isinstance(content, str):
                    message[-1] = generate_openai_message(
                        content=content,
                        role="user",
                        image_path=image_path,
                    )
                    image_path = None
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
        if is_chat_user_turn and runtime is not None:
            self.task_manager.memory_manager.save_triggered_user_memory(
                latest_message,
                runtime.context,
            )
        if uses_task_workflow_response_rules and state.initial_prompt_pending:
            state.initial_prompt_pending = False
            result["initial_prompt_pending"] = False
        if isinstance(state, ChatGraphState) and not has_tool_call(state.latest_response or {}):
            state.round_index = 0
            result["round_index"] = 0
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
            if (
                state.max_agent_iterations is not None
                and state.round_index >= state.max_agent_iterations
            ):
                return "finalize_round"
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
        Tool-result follow-up handling uses the settings stored on the state
        in later graph nodes.
        """
        return self.execute_tools_for_state(state)

    def tool_followup_required(
        self,
        state: TaskManagerState,
    ) -> bool:
        """Return whether the latest tool batch requires follow-up handling.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state after tool execution.

        Returns
        -------
        bool
            ``True`` when the graph should visit ``image_followup`` before the
            next model turn.

        Notes
        -----
        Follow-up is required when any tool result emits follow-up messages or
        returns image paths.
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
        return False

    def build_tool_followup_messages(
        self,
        state: TaskManagerState,
        *,
        message_with_yielded_image: str,
    ) -> list[dict[str, object]]:
        """Build follow-up messages for the latest tool batch.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state after tool execution.
        message_with_yielded_image : str
            User-facing text used when a tool returns image paths.

        Returns
        -------
        list[dict[str, object]]
            Follow-up messages derived from the most recent tool responses,
            including emitted follow-up messages and image-yield messages.
        """
        tool_messages = state.latest_tool_messages
        followup_messages: list[dict[str, object]] = []
        for tool_message in tool_messages:
            followup_messages.extend(
                self.task_manager.tool_executor.build_tool_followup_messages(
                    tool_message,
                    message_with_yielded_image=message_with_yielded_image,
                )
            )
        return followup_messages

    def image_followup(
        self,
        state: TaskManagerState,
        runtime: Runtime[ChatRuntimeContext] | None = None,
    ) -> dict[str, object]:
        """Append follow-up messages after a tool batch completes.

        Parameters
        ----------
        state : TaskManagerState
            Active graph state after ``execute_tools``.
        runtime : Runtime[ChatRuntimeContext], optional
            Runtime context that may carry long-term-memory storage.

        Returns
        -------
        dict[str, object]
            Dumped state payload after the generated follow-up messages are
            appended.

        Notes
        -----
        Task states use per-workflow follow-up settings stored on the state.
        Chat states always use the default image follow-up wording.

        Exceptions raised while materializing follow-up messages, such as
        failures to read an ``img_path`` returned by a tool, are converted into
        a plain user message and appended to the transcript so the workflow can
        continue and the model can react to the failure.
        """
        try:
            if not isinstance(state, ChatGraphState):
                followup_messages = self.build_tool_followup_messages(
                    state,
                    message_with_yielded_image=state.message_with_yielded_image,
                )
                self.apply_followup_messages_for_state(
                    state,
                    followup_messages,
                    runtime_context=None if runtime is None else runtime.context,
                )
                return state.model_dump()
            followup_messages = self.build_tool_followup_messages(
                state,
                message_with_yielded_image=getattr(
                    state,
                    "message_with_yielded_image",
                    "Here is the image the tool returned.",
                ),
            )
            self.apply_followup_messages_for_state(
                state,
                followup_messages,
                runtime_context=None if runtime is None else runtime.context,
            )
            return state.model_dump()
        except Exception as exc:
            error_message = generate_openai_message(
                content=(
                    "An error occurred while processing the tool follow-up output: "
                    f"{exc}. The tool may have returned an invalid image path or payload."
                ),
                role="user",
            )
            if not self.task_manager.use_webui:
                print_message(error_message)
            self.update_message_history_for_state(
                state,
                error_message,
                update_context=True,
                update_full_history=True,
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
            otherwise ``"finalize_round"``.
        """
        if self.tool_followup_required(state):
            return "image_followup"
        return "finalize_round"

    def finalize_chat_round(self, state: ChatGraphState) -> dict[str, object]:
        """Finalize one chat tool cycle and prune images before the next model call."""
        if (
            state.max_agent_iterations is not None
            and state.round_index >= state.max_agent_iterations
        ):
            original_context_length = len(state.messages)
            complete_unresponded_tool_calls(state.messages)
            for message in state.messages[original_context_length:]:
                state.full_history.append(message)
            state.await_user_input = True
            return state.model_dump()
        if (
            state.n_last_images_to_keep_in_context is not None
            or state.n_first_images_to_keep_in_context is not None
        ):
            state.messages = purge_context_images(
                context=state.messages,
                keep_first_n=state.n_first_images_to_keep_in_context or 0,
                keep_last_n=state.n_last_images_to_keep_in_context or 0,
                keep_text=True,
            )
        state.round_index += 1
        if (
            state.max_agent_iterations is not None
            and state.round_index >= state.max_agent_iterations
        ):
            state.await_user_input = True
        return state.model_dump()

    def route_after_chat_round(self, state: ChatGraphState) -> str:
        """Route after a chat tool cycle has been finalized."""
        if (
            state.max_agent_iterations is not None
            and state.round_index >= state.max_agent_iterations
        ):
            if state.termination_behavior == "return":
                return END
            return "await_or_ingest_user_input"
        return "call_model"

    def route_after_feedback_response(self, state: TaskManagerState) -> str:
        """Route task execution after each assistant response.

        Parameters
        ----------
        state : TaskManagerState
            Active task graph state after ``call_model`` or another
            assistant-producing node.

        Returns
        -------
        str
            Next node name for the task workflow.

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

    def handle_human_gate(self, state: TaskManagerState) -> dict[str, object]:
        """Handle ``TERMINATE`` and ``NEED HUMAN`` responses.

        Parameters
        ----------
        state : TaskManagerState
            Active task graph state.

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
        - ``/skill`` without an argument shows skills and keeps waiting for
          input.
        - ``/skill <name>`` appends the selected ``SKILL.md`` to context and
          sends the updated context back through the model.
        - Any other text is appended as a user message and sent back through
          the model using the current ``state.messages`` context.
        """
        if not state.await_user_input:
            return state.model_dump()
        message = self.task_manager.get_user_input(
            prompt=(
                "Termination condition triggered. What to do next? "
                "(`/exit`: exit; `/chat`: chat mode): "
            ),
            display_prompt_in_webui=self.task_manager.use_webui,
        )
        command = parse_user_input_command(message)
        if self.task_manager.handle_runtime_command(command):
            state.await_user_input = True
            return state.model_dump()
        if command.kind == "exit":
            state.exit_requested = True
            state.await_user_input = False
            return state.model_dump()
        if command.kind == "chat":
            state.await_user_input = False
            state.chat_requested = True
            return state.model_dump()
        if command.kind == "skill" and not command.argument:
            state.await_user_input = True
            self.task_manager.display_available_skills()
            return state.model_dump()
        if command.kind == "skill":
            try:
                messages = self.task_manager.build_selected_skill_messages(command.argument)
            except ValueError as exc:
                self.task_manager.record_system_message(
                    str(exc),
                    update_context=True,
                    write_to_webui=True,
                )
                state.await_user_input = True
                return state.model_dump()
            for skill_message in messages:
                if not self.task_manager.use_webui:
                    print_message(skill_message)
                self.update_message_history_for_state(
                    state,
                    skill_message,
                    update_context=True,
                    update_full_history=True,
                )
            if command.text:
                user_message = generate_openai_message(content=command.text, role="user")
                if not self.task_manager.use_webui:
                    print_message(user_message)
                self.update_message_history_for_state(
                    state,
                    user_message,
                    update_context=True,
                    update_full_history=True,
                )
            return self.invoke_model_for_state(
                state,
                context=list(state.messages),
                await_user_input_resolver=self.feedback_response_requires_user_input,
            )
        return self.invoke_model_for_state(
            state,
            message=self.task_manager.expand_skill_command_in_text(message),
            context=list(state.messages),
            await_user_input_resolver=self.feedback_response_requires_user_input,
        )

    def reprompt_model(self, state: TaskManagerState) -> dict[str, object]:
        """Reprompt the model when the previous response violated loop rules.

        Parameters
        ----------
        state : TaskManagerState
            Active task graph state.

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

    def finalize_round(self, state: TaskManagerState) -> dict[str, object]:
        """Finalize one task round and prune image context if needed.

        Parameters
        ----------
        state : TaskManagerState
            Active task graph state after tool execution and optional
            follow-up handling.

        Returns
        -------
        dict[str, object]
            Dumped state payload after round-finalization updates.

        Notes
        -----
        Finalization performs three state-dependent steps.

        - If ``state.n_first_images_to_keep_in_context`` or
          ``state.n_last_images_to_keep_in_context`` is set, image messages in
          ``state.messages`` are pruned while preserving the configured number
          of earliest and latest images.
        - ``state.round_index`` is incremented, and ``MaxRoundsReached`` is
          raised when the new round index reaches ``state.max_rounds`` and
          ``state.max_arounds_reached_behavior`` is ``"raise"``.
        """
        if (
            state.n_last_images_to_keep_in_context is not None
            or state.n_first_images_to_keep_in_context is not None
        ):
            keep_first = state.n_first_images_to_keep_in_context or 0
            keep_last = state.n_last_images_to_keep_in_context or 0
            state.messages = purge_context_images(
                context=state.messages,
                keep_first_n=keep_first,
                keep_last_n=keep_last,
                keep_text=True,
            )
        state.round_index += 1
        if state.round_index >= state.max_rounds and state.max_arounds_reached_behavior == "raise":
            raise MaxRoundsReached()
        return state.model_dump()

    def route_after_feedback_round(self, state: TaskManagerState) -> str:
        """Route after a task round completes.

        Parameters
        ----------
        state : TaskManagerState
            Active task graph state after ``finalize_round``.

        Returns
        -------
        str
            ``END`` when ``state.round_index`` has reached ``state.max_rounds``;
            otherwise ``"call_model"`` for the next task turn.
        """
        if state.round_index >= state.max_rounds:
            return END
        return "call_model"
