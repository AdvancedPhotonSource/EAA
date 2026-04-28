from typing import Any, Literal, Optional
import json

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from eaa_core.message_proc import (
    extract_message_text,
    has_tool_call,
)
from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.task_manager.base import load_latest_checkpoint_state_from_connection
from eaa_core.task_manager.state import TaskManagerState
from eaa_core.task_manager.prompts import render_prompt_template
from eaa_core.tool.base import BaseTool
from eaa_imaging.task_manager.imaging.base import ImagingBaseTaskManager
from eaa_imaging.task_manager.imaging.feature_tracking import initialize_feature_tracking_task_manager
from eaa_imaging.tool.imaging.acquisition import AcquireImage
from eaa_imaging.tool.imaging.registration import ImageRegistration


class MultiAgentROISearchState(TaskManagerState):
    """State for the multi-agent ROI-search task graph.

    Parameters
    ----------
    task_prompt : str
        Rendered ROI-search prompt shared by the position proposer and image
        checker agents.
    max_search_rounds : int
        Maximum number of position-proposal/acquisition attempts.
    search_round_index : int
        Number of completed position-proposal attempts.
    position_tool_calls : list[dict[str, Any]]
        Tool-call payloads proposed by the position proposer.
    acquisition_tool_messages : list[dict[str, Any]]
        Tool response messages returned after image acquisition.
    checker_results : list[dict[str, Any]]
        Parsed JSON decisions returned by the image checker.
    last_image_path : Optional[str]
        Image path from the latest acquisition response.
    fov_description : str
        Latest checker description of the field of view.
    foi_present : bool
        Whether the feature of interest was found by the checker.
    """

    task_prompt: str = ""
    max_search_rounds: int = 99
    search_round_index: int = 0
    position_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    acquisition_tool_messages: list[dict[str, Any]] = Field(default_factory=list)
    checker_results: list[dict[str, Any]] = Field(default_factory=list)
    last_image_path: Optional[str] = None
    fov_description: str = ""
    foi_present: bool = False


class ROISearchTaskManager(ImagingBaseTaskManager):
    """Search for a region of interest using the shared feedback-loop graph."""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        image_acquisition_tool: AcquireImage = None,
        image_registration_tool: ImageRegistration = None,
        additional_tools: list[BaseTool] = (),
        session_db_path: Optional[str] = "session.sqlite",
        build: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the ROI-search task manager.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            Configuration for the language model.
        memory_config : MemoryManagerConfig, optional
            Long-term memory configuration.
        image_acquisition_tool : AcquireImage
            Tool used to acquire microscope images.
        image_registration_tool : ImageRegistration, optional
            Optional registration tool available during ROI search.
        additional_tools : list[BaseTool], optional
            Additional tools to register alongside the imaging tools.
        session_db_path : str, optional
            SQLite path used for transcript persistence.
        build : bool, optional
            Whether to build the task manager immediately.
        *args
            Positional arguments forwarded to ``ImagingBaseTaskManager``.
        **kwargs
            Keyword arguments forwarded to ``ImagingBaseTaskManager``.
        """
        initialize_feature_tracking_task_manager(
            self,
            llm_config=llm_config,
            memory_config=memory_config,
            image_acquisition_tool=image_acquisition_tool,
            image_registration_tool=image_registration_tool,
            additional_tools=additional_tools,
            session_db_path=session_db_path,
            build=build,
            args=args,
            kwargs=kwargs,
        )

    def run(
        self,
        feature_description: str = None,
        y_range: tuple[float, float] = None,
        x_range: tuple[float, float] = None,
        fov_size: tuple[float, float] = None,
        step_size: tuple[float, float] = None,
        max_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_last_images_to_keep_in_context: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """Run a field-of-view search workflow.

        Parameters
        ----------
        feature_description : str, optional
            Text description of the feature to search for. The text can include
            an ``<img /path/to/image.png>`` reference image tag.
        y_range : tuple[float, float], optional
            Inclusive search range for the vertical stage coordinate.
        x_range : tuple[float, float], optional
            Inclusive search range for the horizontal stage coordinate.
        fov_size : tuple[float, float], optional
            Initial field-of-view size in ``(height, width)`` order.
        step_size : tuple[float, float], optional
            Initial grid-search step size in ``(dy, dx)`` order.
        max_rounds : int, optional
            Maximum number of feedback-loop rounds to allow.
        n_first_images_to_keep_in_context : int, optional
            Number of earliest images to keep in context when pruning.
        n_last_images_to_keep_in_context : int, optional
            Number of latest images to keep in context when pruning.
        initial_prompt : str, optional
            Explicit prompt override. When provided, the geometry arguments and
            feature description must be omitted.
        additional_prompt : str, optional
            Additional instructions appended to the generated prompt.
        *args
            Unused positional compatibility arguments.
        **kwargs
            Unused keyword compatibility arguments.
        """
        self.prerun_check()

        if initial_prompt is None:
            initial_prompt = render_prompt_template(
                "eaa_imaging.task_manager.prompts",
                "roi_search.md",
                {
                    "FEATURE_DESCRIPTION": feature_description,
                    "FOV_SIZE": fov_size,
                    "Y_MIN": y_range[0],
                    "Y_MAX": y_range[1],
                    "X_MIN": x_range[0],
                    "X_MAX": x_range[1],
                    "STEP_SIZE_Y": step_size[0],
                    "STEP_SIZE_X": step_size[1],
                },
            )
        elif any(
            value is not None
            for value in [feature_description, y_range, x_range, fov_size, step_size]
        ):
            raise ValueError(
                "`feature_description`, `y_range`, `x_range`, `fov_size`, and `step_size` "
                "should not be provided if `initial_prompt` is given."
            )

        if additional_prompt is not None:
            initial_prompt = initial_prompt + "\nAdditional instructions:\n" + additional_prompt

        self.run_feedback_loop(
            initial_prompt=initial_prompt,
            initial_image_path=None,
            message_with_yielded_image=(
                "Here is the image the tool returned. If the feature is there, "
                "report the coordinates of the FOV and include 'TERMINATE' in "
                "your response. Otherwise, continue to call tools to run the search. "
                "Include a brief description of what you see in the image in your response."
            ),
            max_rounds=max_rounds,
            n_first_images_to_keep_in_context=n_first_images_to_keep_in_context,
            n_last_images_to_keep_in_context=n_last_images_to_keep_in_context,
        )

    def run_from_checkpoint(self, checkpoint_db_path: Optional[str] = None) -> None:
        """Resume the ROI-search workflow from a checkpoint.

        Parameters
        ----------
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and updates instead of
            ``self.session_db_path``.
        """
        self.prerun_check()
        self.run_feedback_loop_from_checkpoint(checkpoint_db_path=checkpoint_db_path)


class MultiAgentROISearchTaskManager(ImagingBaseTaskManager):
    """Search for an ROI with a dedicated position-proposer/checker graph."""

    def __init__(
        self,
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        image_acquisition_tool: AcquireImage = None,
        image_registration_tool: ImageRegistration = None,
        additional_tools: list[BaseTool] = (),
        session_db_path: Optional[str] = "session.sqlite",
        build: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the multi-agent ROI-search task manager.

        Parameters
        ----------
        llm_config : LLMConfig, optional
            Configuration for the language model.
        memory_config : MemoryManagerConfig, optional
            Long-term memory configuration.
        image_acquisition_tool : AcquireImage
            Tool used to acquire microscope images.
        image_registration_tool : ImageRegistration, optional
            Optional registration tool available during final centering.
        additional_tools : list[BaseTool], optional
            Additional tools to register alongside the imaging tools.
        session_db_path : str, optional
            SQLite path used for transcript persistence.
        build : bool, optional
            Whether to build the task manager immediately.
        *args
            Positional arguments forwarded to ``ImagingBaseTaskManager``.
        **kwargs
            Keyword arguments forwarded to ``ImagingBaseTaskManager``.
        """
        initialize_feature_tracking_task_manager(
            self,
            llm_config=llm_config,
            memory_config=memory_config,
            image_acquisition_tool=image_acquisition_tool,
            image_registration_tool=image_registration_tool,
            additional_tools=additional_tools,
            session_db_path=session_db_path,
            build=build,
            args=args,
            kwargs=kwargs,
        )

    def build_task_graph(self, checkpointer: Any = None) -> CompiledStateGraph:
        """Build the position-proposer/image-checker ROI-search graph.

        Parameters
        ----------
        checkpointer : Any, optional
            LangGraph checkpointer.

        Returns
        -------
        CompiledStateGraph
            Compiled multi-agent ROI-search graph.
        """
        builder = StateGraph(MultiAgentROISearchState)
        builder.add_node("position_proposer", self.position_proposer)
        builder.add_node("image_checker", self.image_checker)
        builder.add_edge(START, "position_proposer")
        builder.add_edge("position_proposer", "image_checker")
        builder.add_conditional_edges(
            "image_checker",
            self.route_after_image_checker,
        )
        return builder.compile(checkpointer=checkpointer)

    def run(
        self,
        feature_description: str = None,
        y_range: tuple[float, float] = None,
        x_range: tuple[float, float] = None,
        fov_size: tuple[float, float] = None,
        step_size: tuple[float, float] = None,
        max_search_rounds: int = 99,
        max_centering_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_last_images_to_keep_in_context: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None,
        termination_behavior: Literal["ask", "return"] = "ask",
        max_arounds_reached_behavior: Literal["return", "raise"] = "return",
        *args,
        **kwargs,
    ) -> None:
        """Run multi-agent ROI search followed by feedback-loop centering.

        Parameters
        ----------
        feature_description : str, optional
            Text description of the feature to search for.
        y_range : tuple[float, float], optional
            Inclusive search range for the vertical stage coordinate.
        x_range : tuple[float, float], optional
            Inclusive search range for the horizontal stage coordinate.
        fov_size : tuple[float, float], optional
            Initial field-of-view size in ``(height, width)`` order.
        step_size : tuple[float, float], optional
            Initial grid-search step size in ``(dy, dx)`` order.
        max_search_rounds : int, optional
            Maximum number of proposer/checker attempts before failing.
        max_centering_rounds : int, optional
            Maximum number of final-centering feedback-loop rounds.
        n_first_images_to_keep_in_context : int, optional
            Number of earliest images to keep in feedback-loop context.
        n_last_images_to_keep_in_context : int, optional
            Number of latest images to keep in feedback-loop context.
        initial_prompt : str, optional
            Explicit prompt override. When provided, the geometry arguments and
            feature description must be omitted.
        additional_prompt : str, optional
            Additional instructions appended to the generated prompt.
        termination_behavior : {"ask", "return"}, optional
            Behavior when the centering agent emits ``TERMINATE``.
        max_arounds_reached_behavior : {"return", "raise"}, optional
            Behavior when final centering reaches ``max_centering_rounds``.
        *args
            Unused positional compatibility arguments.
        **kwargs
            Unused keyword compatibility arguments.
        """
        self.prerun_check()
        task_prompt = self.build_roi_search_prompt(
            feature_description=feature_description,
            y_range=y_range,
            x_range=x_range,
            fov_size=fov_size,
            step_size=step_size,
            initial_prompt=initial_prompt,
            additional_prompt=additional_prompt,
        )
        initial_state = MultiAgentROISearchState(
            task_prompt=task_prompt,
            max_search_rounds=max_search_rounds,
        )
        initial_state.copy_messages_and_history_from_state(self.active_state)
        self.set_active_state(initial_state, "task_graph")

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
        final_state = graph.invoke(initial_state, **graph_kwargs)
        self.set_active_state(
            MultiAgentROISearchState.model_validate(final_state),
            "task_graph",
        )
        search_state = self.task_state
        if not isinstance(search_state, MultiAgentROISearchState):
            raise TypeError("Multi-agent ROI search returned an unexpected state model.")
        if not search_state.foi_present or search_state.last_image_path is None:
            raise RuntimeError("ROI search ended before finding the feature of interest.")
        self.run_final_centering_feedback_loop(
            search_state,
            max_rounds=max_centering_rounds,
            n_first_images_to_keep_in_context=n_first_images_to_keep_in_context,
            n_last_images_to_keep_in_context=n_last_images_to_keep_in_context,
            termination_behavior=termination_behavior,
            max_arounds_reached_behavior=max_arounds_reached_behavior,
        )

    def run_from_checkpoint(self, checkpoint_db_path: Optional[str] = None) -> None:
        """Resume the multi-agent ROI-search graph from a checkpoint.

        Parameters
        ----------
        checkpoint_db_path : Optional[str], optional
            SQLite path to use for checkpoint loading and updates instead of
            ``self.session_db_path``.
        """
        self.prerun_check()
        graph, checkpoint_config, _ = self.get_checkpointed_graph(
            "task_graph",
            checkpoint_db_path=checkpoint_db_path,
            load_state=False,
        )
        snapshot = graph.get_state(checkpoint_config)
        fallback_loaded = (
            snapshot.created_at is None
            or snapshot.values is None
            or len(snapshot.values) == 0
        )
        if not fallback_loaded and snapshot.values is not None:
            resumed_state = MultiAgentROISearchState.model_validate(snapshot.values)
        else:
            resolved_checkpoint_path, _ = self.resolve_checkpoint_storage(
                "task_graph",
                checkpoint_db_path=checkpoint_db_path,
            )
            latest_state = load_latest_checkpoint_state_from_connection(
                connection=self.checkpoint_connections[("task_graph", resolved_checkpoint_path)],
                prune_checkpoints=self.prune_checkpoints,
            )
            if latest_state is None:
                raise ValueError(
                    f"No task-graph checkpoint found in shared checkpoint DB "
                    f"{resolved_checkpoint_path}."
                )
            resumed_state = MultiAgentROISearchState.model_validate(latest_state)
        self.set_active_state(resumed_state, "task_graph")
        self.task_graph = graph
        final_state = graph.invoke(None, config=checkpoint_config)
        self.set_active_state(
            MultiAgentROISearchState.model_validate(final_state),
            "task_graph",
        )
        search_state = self.task_state
        if isinstance(search_state, MultiAgentROISearchState) and search_state.foi_present:
            self.run_final_centering_feedback_loop(search_state)

    def build_roi_search_prompt(
        self,
        *,
        feature_description: Optional[str],
        y_range: Optional[tuple[float, float]],
        x_range: Optional[tuple[float, float]],
        fov_size: Optional[tuple[float, float]],
        step_size: Optional[tuple[float, float]],
        initial_prompt: Optional[str],
        additional_prompt: Optional[str],
    ) -> str:
        """Build the ROI-search prompt shared by graph nodes.

        Returns
        -------
        str
            Rendered prompt text.
        """
        if initial_prompt is None:
            if any(value is None for value in [y_range, x_range, fov_size, step_size]):
                raise ValueError(
                    "`y_range`, `x_range`, `fov_size`, and `step_size` must be "
                    "provided when `initial_prompt` is omitted."
                )
            initial_prompt = render_prompt_template(
                "eaa_imaging.task_manager.prompts",
                "roi_search.md",
                {
                    "FEATURE_DESCRIPTION": feature_description,
                    "FOV_SIZE": fov_size,
                    "Y_MIN": y_range[0],
                    "Y_MAX": y_range[1],
                    "X_MIN": x_range[0],
                    "X_MAX": x_range[1],
                    "STEP_SIZE_Y": step_size[0],
                    "STEP_SIZE_X": step_size[1],
                },
            )
        elif any(
            value is not None
            for value in [feature_description, y_range, x_range, fov_size, step_size]
        ):
            raise ValueError(
                "`feature_description`, `y_range`, `x_range`, `fov_size`, and `step_size` "
                "should not be provided if `initial_prompt` is given."
            )
        if additional_prompt is not None:
            initial_prompt = initial_prompt + "\nAdditional instructions:\n" + additional_prompt
        return initial_prompt

    def position_proposer(self, state: MultiAgentROISearchState) -> dict[str, Any]:
        """Ask the proposer agent to choose and acquire the next FOV.

        Parameters
        ----------
        state : MultiAgentROISearchState
            Active multi-agent ROI-search state.

        Returns
        -------
        dict[str, Any]
            Updated state payload.
        """
        prompt = self.build_position_proposer_prompt(state)
        while True:
            response, outgoing = self.invoke_model_raw(
                prompt,
                context=list(state.messages),
                return_outgoing_message=True,
            )
            self.record_model_exchange_for_state(state, outgoing, response)
            try:
                tool_calls = self.validate_position_proposer_response(response)
                tool_messages = self.tool_executor.execute_tool_calls(tool_calls)
                if len(tool_messages) != 1:
                    raise RuntimeError(
                        "Image acquisition did not return exactly one tool message."
                    )
                tool_message = tool_messages[0].message
                self.node_factory.update_message_history_for_state(state, tool_message)
                image_paths = self.tool_executor.extract_image_paths_from_tool_response(
                    tool_message.get("content")
                )
                state.position_tool_calls.append(tool_calls[0])
                state.acquisition_tool_messages.append(tool_message)
                if len(image_paths) == 0:
                    raise RuntimeError(
                        "Image acquisition tool response did not include `img_path`."
                    )
            except (RuntimeError, ValueError) as exc:
                prompt = self.build_position_proposer_retry_prompt(str(exc))
                continue
            state.last_image_path = image_paths[0]
            state.search_round_index += 1
            return state.model_dump()

    def image_checker(self, state: MultiAgentROISearchState) -> dict[str, Any]:
        """Ask the checker agent whether the latest image contains the FOI.

        Parameters
        ----------
        state : MultiAgentROISearchState
            Active multi-agent ROI-search state.

        Returns
        -------
        dict[str, Any]
            Updated state payload.
        """
        if state.last_image_path is None:
            raise RuntimeError("Image checker requires an acquired image.")
        prompt = self.build_image_checker_prompt(state)
        while True:
            response, outgoing = self.invoke_model_raw(
                prompt,
                image_path=state.last_image_path,
                context=list(state.messages),
                return_outgoing_message=True,
            )
            self.record_model_exchange_for_state(state, outgoing, response)
            try:
                if has_tool_call(response):
                    raise RuntimeError(
                        "Image checker must return JSON and must not call tools."
                    )
                result = self.parse_image_checker_response(response)
            except (RuntimeError, ValueError) as exc:
                prompt = self.build_image_checker_retry_prompt(str(exc))
                continue
            break
        state.checker_results.append(result)
        state.foi_present = bool(result["foi_present"])
        state.fov_description = result["fov_description"]
        return state.model_dump()

    def record_model_exchange_for_state(
        self,
        state: MultiAgentROISearchState,
        outgoing: Optional[dict[str, Any]],
        response: dict[str, Any],
    ) -> None:
        """Record a model exchange on the graph state and WebUI transcript.

        Parameters
        ----------
        state : MultiAgentROISearchState
            Active graph state.
        outgoing : Optional[dict[str, Any]]
            User message sent to the model, if one was generated.
        response : dict[str, Any]
            Assistant response from the model.
        """
        if outgoing is not None:
            self.node_factory.update_message_history_for_state(state, outgoing)
        self.node_factory.update_message_history_for_state(state, response)

    def validate_position_proposer_response(
        self,
        response: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Validate and return the proposer acquisition tool call.

        Parameters
        ----------
        response : dict[str, Any]
            Assistant response from the position proposer.

        Returns
        -------
        list[dict[str, Any]]
            Single ``acquire_image`` tool call.
        """
        if not has_tool_call(response):
            raise RuntimeError("Position proposer must call the `acquire_image` tool.")
        tool_calls = response.get("tool_calls") or []
        if len(tool_calls) != 1:
            raise RuntimeError("Position proposer must issue exactly one tool call.")
        tool_name = tool_calls[0].get("function", {}).get("name")
        if tool_name != "acquire_image":
            raise RuntimeError(
                f"Position proposer must call `acquire_image`, got `{tool_name}`."
            )
        return tool_calls

    def route_after_image_checker(self, state: MultiAgentROISearchState) -> str:
        """Route after image checking.

        Parameters
        ----------
        state : MultiAgentROISearchState
            State after image checking.

        Returns
        -------
        str
            ``END`` when the feature is present, otherwise
            ``"position_proposer"``.
        """
        if state.foi_present:
            return END
        if state.search_round_index >= state.max_search_rounds:
            raise RuntimeError(
                f"Feature of interest was not found after {state.max_search_rounds} "
                "search rounds."
            )
        return "position_proposer"

    def build_position_proposer_prompt(self, state: MultiAgentROISearchState) -> str:
        """Build the position-proposer prompt.

        Returns
        -------
        str
            Prompt text.
        """
        past_calls = [
            self.extract_tool_call_summary(tool_call)
            for tool_call in state.position_tool_calls
        ]
        return (
            "You are the position proposer working as a part of a workflow "
            "that searches for a user-specified ROI with the microscope. Here "
            "is the full task description:\n"
            "=== Beginning of task description ===\n"
            f"{state.task_prompt}\n"
            "=== End of task description ===\n\n"
            "As the position proposer, your job is to review the past "
            "acquisition tool calls below and choose the next field-of-view "
            "position to scan. Call exactly one tool, `acquire_image`, with "
            "the proposed position and FOV parameters. Do not answer in prose.\n\n"
            f"Past acquisition tool calls:\n{json.dumps(past_calls, indent=2)}"
        )

    def build_image_checker_prompt(self, state: MultiAgentROISearchState) -> str:
        """Build the image-checker prompt.

        Returns
        -------
        str
            Prompt text.
        """
        return (
            "You are the image checker working as a part of a workflow that "
            "searches for a user-specified ROI with the microscope. Here is "
            "the full task description:\n"
            "=== Beginning of task description ===\n"
            f"{state.task_prompt}\n"
            "=== End of task description ===\n\n"
            "As the image checker, your job is to review the attached acquired "
            "image and decide whether the feature of interest described in the "
            "task is present in this field of view. Return only valid JSON "
            "with exactly these keys. Do not wrap the JSON in Markdown or "
            "triple backticks. Output the raw JSON object directly:\n"
            "{\n"
            '  "foi_present": bool,\n'
            '  "fov_description": str\n'
            "}\n"
            "`fov_description` should briefly describe what is visible in the image."
        )

    def build_position_proposer_retry_prompt(self, error_message: str) -> str:
        """Build a correction prompt for invalid proposer behavior.

        Parameters
        ----------
        error_message : str
            Validation or tool-output error from the previous attempt.

        Returns
        -------
        str
            Prompt asking the model to retry correctly.
        """
        return (
            "Your previous position-proposer response could not be used:\n"
            f"{error_message}\n\n"
            "Try again. You must call exactly one tool, `acquire_image`, with "
            "the next proposed field-of-view position and scan parameters. Do "
            "not answer in prose."
        )

    def build_image_checker_retry_prompt(self, error_message: str) -> str:
        """Build a correction prompt for invalid image-checker behavior.

        Parameters
        ----------
        error_message : str
            Validation error from the previous checker attempt.

        Returns
        -------
        str
            Prompt asking the model to retry with raw JSON.
        """
        return (
            "Your previous image-checker response could not be parsed or used:\n"
            f"{error_message}\n\n"
            "Try again. Do not call tools. Do not use Markdown or triple "
            "backticks. Output only the raw JSON object with exactly these "
            'keys: "foi_present" as a boolean and "fov_description" as a string.'
        )

    def parse_image_checker_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate the image-checker JSON response.

        Parameters
        ----------
        response : dict[str, Any]
            Assistant response from the image checker.

        Returns
        -------
        dict[str, Any]
            Parsed checker result.
        """
        content = extract_message_text(response)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Image checker returned invalid JSON: {content}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Image checker response must be a JSON object.")
        if not isinstance(parsed.get("foi_present"), bool):
            raise ValueError("Image checker JSON must include boolean `foi_present`.")
        if not isinstance(parsed.get("fov_description"), str):
            raise ValueError("Image checker JSON must include string `fov_description`.")
        return {
            "foi_present": parsed["foi_present"],
            "fov_description": parsed["fov_description"],
        }

    def extract_tool_call_summary(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Return a compact tool-call summary for proposer prompts.

        Parameters
        ----------
        tool_call : dict[str, Any]
            OpenAI tool-call payload.

        Returns
        -------
        dict[str, Any]
            Tool name and parsed arguments.
        """
        function = tool_call.get("function", {})
        return {
            "name": function.get("name"),
            "arguments": self.tool_executor.parse_arguments(function.get("arguments")),
        }

    def run_final_centering_feedback_loop(
        self,
        search_state: MultiAgentROISearchState,
        *,
        max_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_last_images_to_keep_in_context: Optional[int] = None,
        termination_behavior: Literal["ask", "return"] = "ask",
        max_arounds_reached_behavior: Literal["return", "raise"] = "return",
    ) -> None:
        """Run the final feedback-loop centering workflow.

        Parameters
        ----------
        search_state : MultiAgentROISearchState
            Completed multi-agent search state.
        max_rounds : int, optional
            Maximum number of feedback-loop rounds.
        n_first_images_to_keep_in_context : int, optional
            Number of earliest images to keep in context.
        n_last_images_to_keep_in_context : int, optional
            Number of latest images to keep in context.
        termination_behavior : {"ask", "return"}, optional
            Behavior when the centering agent emits ``TERMINATE``.
        max_arounds_reached_behavior : {"return", "raise"}, optional
            Behavior when ``max_rounds`` is reached.
        """
        if search_state.last_image_path is None:
            raise RuntimeError("Final centering requires a located ROI image.")
        initial_prompt = (
            "You are the final centering agent working as a part of a workflow "
            "that searches for a user-specified ROI with the microscope. Here "
            "is the full task description:\n"
            "=== Beginning of task description ===\n"
            f"{search_state.task_prompt}\n"
            "=== End of task description ===\n\n"
            "As the final centering agent, your job is to adjust the "
            "field-of-view position so that the feature of interest found in "
            "the attached image is centered in the FOV. Use the `acquire_image` "
            "tool as needed to adjust and verify the FOV. When the feature is "
            "centered, include TERMINATE in your response.\n\n"
            f"Feature/FOV description from image checker:\n"
            f"{search_state.fov_description}"
        )
        self.run_feedback_loop(
            initial_prompt=initial_prompt,
            initial_image_path=search_state.last_image_path,
            message_with_yielded_image=(
                "Here is the image the tool returned. Continue adjusting the "
                "field-of-view position until the feature of interest is "
                "centered. When finished, include TERMINATE in your response."
            ),
            max_rounds=max_rounds,
            n_first_images_to_keep_in_context=n_first_images_to_keep_in_context,
            n_last_images_to_keep_in_context=n_last_images_to_keep_in_context,
            allow_non_image_tool_responses=True,
            termination_behavior=termination_behavior,
            max_arounds_reached_behavior=max_arounds_reached_behavior,
        )
