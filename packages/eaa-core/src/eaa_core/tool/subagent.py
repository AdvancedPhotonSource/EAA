from __future__ import annotations

import re
from typing import Annotated, Any

from eaa_core.message_proc import extract_message_text
from eaa_core.tool.base import BaseTool, tool


class SubagentTool(BaseTool):
    """Launch subordinate task managers for delegated agent work."""

    name: str = "subagent"

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
            table_suffix = re.sub(r"[^A-Za-z0-9_]", "_", conversation_id)
            transcript_table_name = f"transcript_messages_{table_suffix}"
        inherited_tools = [
            tool_object
            for tool_object in self.task_manager.tool_executor.tools
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
            use_coding_tools=self.task_manager.use_coding_tools,
            coding_tool_sandbox_type=self.task_manager.coding_tool_sandbox_type,
            bubblewrap_visible_dirs=self.task_manager.bubblewrap_visible_dirs,
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
