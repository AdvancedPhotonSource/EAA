from __future__ import annotations

import inspect
import re
from typing import Annotated, Any, ClassVar

from eaa_core.message_proc import extract_message_text
from eaa_core.task_manager.persistence import SQLiteTranscriptStore
from eaa_core.tool.base import BaseTool, tool


class SubagentTool(BaseTool):
    """Launch subordinate task managers for delegated agent work."""

    name: str = "subagent"
    registered_task_managers: ClassVar[dict[str, Any]] = {}

    def __init__(self, task_manager: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the subagent launcher.

        Parameters
        ----------
        task_manager : Any
            Parent task manager that owns this tool.
        *args
            Positional arguments forwarded to :class:`BaseTool`.
        **kwargs
            Keyword arguments forwarded to :class:`BaseTool`.
        """
        self.task_manager = task_manager
        super().__init__(*args, **kwargs)

    @classmethod
    def add_task_managers(cls, task_managers: Any | list[Any]) -> None:
        """Register one or more task manager objects for subtask launches.

        Parameters
        ----------
        task_managers : Any or list[Any]
            Task manager object or objects with a unique ``name`` and callable
            ``run`` method.
        """
        if isinstance(task_managers, (list, tuple)):
            normalized_task_managers = list(task_managers)
        else:
            normalized_task_managers = [task_managers]
        for task_manager in normalized_task_managers:
            name = cls._get_task_manager_name(task_manager)
            run_method = getattr(task_manager, "run", None)
            if not callable(run_method):
                raise ValueError(
                    f"Registered task manager {name!r} must define a callable "
                    "`run` method."
                )
            existing_task_manager = cls.registered_task_managers.get(name)
            if (
                existing_task_manager is not None
                and existing_task_manager is not task_manager
            ):
                raise ValueError(
                    f"A task manager named {name!r} is already registered."
                )
            cls.registered_task_managers[name] = task_manager

    @staticmethod
    def _get_task_manager_name(task_manager: Any) -> str:
        """Return a validated task manager registry name."""
        name = str(getattr(task_manager, "name", "")).strip()
        if not name:
            raise ValueError("Registered task managers must have a non-empty `name`.")
        return name

    @staticmethod
    def _transcript_table_name_for_conversation(conversation_id: str) -> str:
        """Return the transcript table name for a subordinate conversation."""
        table_suffix = re.sub(r"[^A-Za-z0-9_]", "_", conversation_id)
        return f"transcript_messages_{table_suffix}"

    @tool(name="subagent_tool.launch_subagent")
    def launch_subagent(
        self,
        message: Annotated[
            str,
            "Clear task instructions for the subagent, including expected output.",
        ],
    ) -> dict[str, str]:
        """Run a subagent conversation and return its final message.

        Parameters
        ----------
        message : str
            Instructions to send to the subagent.

        Returns
        -------
        dict[str, str]
            The subagent's final assistant response.
        """
        from eaa_core.task_manager.base import BaseTaskManager

        runtime_controller = getattr(self.task_manager, "runtime_controller", None)
        conversation_id: str | None = None
        if runtime_controller is not None:
            conversation = runtime_controller.create_conversation(kind="subagent")
            conversation_id = str(conversation["id"])
        transcript_table_name = "transcript_messages"
        if conversation_id is not None:
            transcript_table_name = self._transcript_table_name_for_conversation(
                conversation_id
            )
        inherited_tools = [
            tool_object
            for tool_object in self.task_manager.tool_manager
            if not isinstance(tool_object, SubagentTool)
        ]
        sub_task_manager = BaseTaskManager(
            llm_config=self.task_manager.llm_config,
            memory_config=self.task_manager.memory_config,
            tools=inherited_tools,
            skill_dirs=self.task_manager.skill_dirs,
            checkpoint_db_path=None,
            transcript_db_path=self.task_manager.transcript_db_path,
            transcript_table_name=transcript_table_name,
            use_webui=self.task_manager.use_webui,
            runtime_controller=runtime_controller,
            runtime_conversation_id=conversation_id or "primary",
            prune_checkpoints=self.task_manager.prune_checkpoints,
            is_subagent=True,
        )
        if sub_task_manager.model is None and self.task_manager.model is not None:
            sub_task_manager.model = self.task_manager.model
        try:
            sub_task_manager.run_conversation(
                message=message,
                termination_behavior="return",
                inherit_activate_state_messages=False,
            )
            response = sub_task_manager.chat_state.latest_response or {}
            return {"result": extract_message_text(response)}
        finally:
            if runtime_controller is not None and conversation_id is not None:
                runtime_controller.terminate_conversation(
                    conversation_id,
                    message="Subagent terminated",
                )

    @tool(name="subagent_tool.list_registered_task_managers")
    def list_registered_task_managers(self) -> list[dict[str, str]]:
        """Return task managers available for ``launch_subtask_manager``.

        Returns
        -------
        list[dict[str, str]]
            Registered task manager specs.
        """
        specs: list[dict[str, str]] = []
        for name, task_manager in self.registered_task_managers.items():
            run_method = getattr(task_manager, "run", None)
            specs.append(
                {
                    "name": name,
                    "task_class_name": type(task_manager).__name__,
                    "run_method_docstring": inspect.getdoc(run_method) or "",
                }
            )
        return specs

    @tool(name="subagent_tool.launch_subtask_manager")
    def launch_subtask_manager(
        self,
        task_manager_name: Annotated[
            str,
            "Name of the registered task manager to launch.",
        ],
        task_manager_kwargs: Annotated[
            dict,
            "Keyword arguments to pass to the selected task manager's run method.",
        ],
    ) -> dict[str, Any]:
        """Run a registered task manager and return its run result.

        Parameters
        ----------
        task_manager_name : str
            Name of the registered task manager to launch.
        task_manager_kwargs : dict
            Keyword arguments passed to ``matched_task_manager.run``.

        Returns
        -------
        dict[str, Any]
            The selected task manager's ``run`` return value.
        """
        if not isinstance(task_manager_kwargs, dict):
            raise ValueError("`task_manager_kwargs` must be a dictionary.")
        task_manager_kwargs["termination_behavior"] = "return"
        task_manager_name = str(task_manager_name).strip()
        matched_task_manager = self.registered_task_managers.get(task_manager_name)
        if matched_task_manager is None:
            available_names = ", ".join(self.registered_task_managers) or "none"
            raise ValueError(
                f"No registered task manager named {task_manager_name!r}. "
                f"Available task managers: {available_names}."
            )

        runtime_controller = getattr(self.task_manager, "runtime_controller", None)
        conversation_id = self._get_task_manager_name(matched_task_manager)
        if runtime_controller is not None:
            conversation = runtime_controller.create_conversation(
                label=conversation_id,
                kind="subagent",
            )
            conversation_id = str(conversation["id"])
        transcript_table_name = self._transcript_table_name_for_conversation(
            conversation_id
        )

        saved_state = {
            "checkpoint_db_path": getattr(
                matched_task_manager,
                "checkpoint_db_path",
                None,
            ),
            "transcript_db_path": getattr(
                matched_task_manager,
                "transcript_db_path",
                None,
            ),
            "transcript_table_name": getattr(
                matched_task_manager,
                "transcript_table_name",
                None,
            ),
            "transcript_store": getattr(matched_task_manager, "transcript_store", None),
            "use_webui": getattr(matched_task_manager, "use_webui", None),
            "runtime_controller": getattr(
                matched_task_manager,
                "runtime_controller",
                None,
            ),
            "runtime_conversation_id": getattr(
                matched_task_manager,
                "runtime_conversation_id",
                None,
            ),
        }
        matched_task_manager.checkpoint_db_path = None
        matched_task_manager.transcript_db_path = self.task_manager.transcript_db_path
        matched_task_manager.transcript_table_name = transcript_table_name
        matched_task_manager.transcript_store = SQLiteTranscriptStore(
            self.task_manager.transcript_db_path,
            table_name=transcript_table_name,
        )
        matched_task_manager.use_webui = self.task_manager.use_webui
        matched_task_manager.runtime_controller = runtime_controller
        matched_task_manager.runtime_conversation_id = conversation_id

        try:
            result = matched_task_manager.run(**task_manager_kwargs)
            return {"result": result}
        finally:
            for attribute, value in saved_state.items():
                setattr(matched_task_manager, attribute, value)
            if runtime_controller is not None:
                runtime_controller.terminate_conversation(
                    conversation_id,
                    message="Subtask manager terminated",
                )
