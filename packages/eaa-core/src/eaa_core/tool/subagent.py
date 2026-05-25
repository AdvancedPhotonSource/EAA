from __future__ import annotations

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

    @tool(name="launch_subagent")
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
            session_db_path=self.task_manager.session_db_path,
            use_webui=self.task_manager.use_webui,
            use_coding_tools=self.task_manager.use_coding_tools,
            run_codes_in_sandbox=self.task_manager.run_codes_in_sandbox,
            prune_checkpoints=self.task_manager.prune_checkpoints,
            is_subagent=True,
        )
        if sub_task_manager.model is None and self.task_manager.model is not None:
            sub_task_manager.model = self.task_manager.model
        sub_task_manager.run_conversation(
            message=message,
            termination_behavior="return",
            inherit_activate_state_messages=False,
        )
        response = sub_task_manager.chat_state.latest_response or {}
        return {"result": extract_message_text(response)}
