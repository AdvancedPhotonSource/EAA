"""Task-manager-owned tool collection and default tool handles."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from eaa_core.tool.base import BaseTool
from eaa_core.tool.coding import (
    BashCodingTool,
    PythonCodingTool,
    SandboxType,
    SimplePythonEvalTool,
)
from eaa_core.tool.subagent import SubagentTool
from eaa_core.tool.workspace import FileSystemTool, ImageRenderingTool, UvTool


class ToolManager(list[BaseTool]):
    """Manage task-manager tools and direct handles for built-in tools."""

    def __init__(
        self,
        task_manager: Any,
        tools: Sequence[BaseTool] = (),
        *,
        skill_dirs: Sequence[str] = (),
        coding_tool_request_approval: bool = True,
        include_subagent_tool: bool = True,
    ) -> None:
        """Initialize the tool manager.

        Parameters
        ----------
        task_manager : Any
            Parent task manager that owns this tool manager.
        tools : sequence of BaseTool, optional
            User-provided tools registered before default tools.
        skill_dirs : sequence of str, optional
            Skill directories readable by the filesystem tool without approval.
        coding_tool_request_approval : bool, default=True
            Initial approval requirement for coding tools.
        include_subagent_tool : bool, default=True
            Whether to create and register the subagent launcher.
        """
        super().__init__()
        self.task_manager = task_manager
        self.executor: Any = None
        self._initial_tools = list(tools)
        self._disabled_tool_names: set[str] = set()
        self.coding_tool_sandbox_type: SandboxType = None
        self.bubblewrap_visible_dirs: Optional[list[str]] = None
        self.coding_tool_request_approval = coding_tool_request_approval

        self.simple_python_eval_tool = SimplePythonEvalTool()
        self.file_system_tool = FileSystemTool(read_whitelist_paths=list(skill_dirs))
        self.image_rendering_tool = ImageRenderingTool()
        self.python_coding_tool = PythonCodingTool(
            sandbox_type=self.coding_tool_sandbox_type,
            bubblewrap_visible_dirs=self.bubblewrap_visible_dirs,
            require_approval=self.coding_tool_request_approval,
        )
        self.bash_coding_tool = BashCodingTool(
            sandbox_type=self.coding_tool_sandbox_type,
            bubblewrap_visible_dirs=self.bubblewrap_visible_dirs,
            require_approval=self.coding_tool_request_approval,
        )
        self.uv_tool = UvTool()
        self.subagent_tool = SubagentTool(task_manager) if include_subagent_tool else None

    def bind_executor(self, executor: Any) -> None:
        """Bind the executor that owns schemas for this tool list."""
        self.executor = executor

    def build(self) -> None:
        """Register user-provided tools followed by enabled default tools."""
        self.register_tools(self._initial_tools)
        self.register_tools(self.default_tools())

    def default_tools(self) -> list[BaseTool]:
        """Return enabled built-in tools in registration order."""
        tools: list[BaseTool] = []
        if "simple_python_eval_tool" not in self._disabled_tool_names:
            tools.append(self.simple_python_eval_tool)
        if "file_system_tool" not in self._disabled_tool_names:
            tools.append(self.file_system_tool)
        if "image_rendering_tool" not in self._disabled_tool_names:
            tools.append(self.image_rendering_tool)
        if "python_coding_tool" not in self._disabled_tool_names:
            tools.append(self.python_coding_tool)
        if "bash_coding_tool" not in self._disabled_tool_names:
            tools.append(self.bash_coding_tool)
        if "uv_tool" not in self._disabled_tool_names:
            tools.append(self.uv_tool)
        if (
            self.subagent_tool is not None
            and "subagent_tool" not in self._disabled_tool_names
        ):
            tools.append(self.subagent_tool)
        return tools

    def register_tools(self, tools: BaseTool | Sequence[BaseTool]) -> None:
        """Register one or more tools with the active executor."""
        tool_list = self._normalize_tools(tools)
        for tool in tool_list:
            if self._tool_has_registered_name_conflict(tool):
                continue
            self._set_known_tool_reference(tool)
            runtime_controller = getattr(self.task_manager, "runtime_controller", None)
            if runtime_controller is not None:
                runtime_controller.register_tool(tool)
            if self.executor is None:
                if tool not in self:
                    self.append(tool)
            else:
                self.executor.register_tools(tool)

    def unregister_tool(self, tool: BaseTool | None) -> None:
        """Remove a tool from the active list and executor schema registry."""
        if tool is None:
            return
        if self.executor is None:
            if tool in self:
                self.remove(tool)
            return
        self.executor.unregister_tool(tool)

    def disable_simple_python_eval_tool(self) -> None:
        """Disable the simple Python expression evaluation tool."""
        self._disabled_tool_names.add("simple_python_eval_tool")
        self.unregister_tool(self.simple_python_eval_tool)

    def disable_file_system_tool(self) -> None:
        """Disable the filesystem workspace tool."""
        self._disabled_tool_names.add("file_system_tool")
        self.unregister_tool(self.file_system_tool)

    def disable_image_rendering_tool(self) -> None:
        """Disable the image rendering workspace tool."""
        self._disabled_tool_names.add("image_rendering_tool")
        self.unregister_tool(self.image_rendering_tool)

    def disable_python_coding_tool(self) -> None:
        """Disable the Python coding tool."""
        self._disabled_tool_names.add("python_coding_tool")
        self.unregister_tool(self.python_coding_tool)

    def disable_bash_coding_tool(self) -> None:
        """Disable the Bash coding tool."""
        self._disabled_tool_names.add("bash_coding_tool")
        self.unregister_tool(self.bash_coding_tool)

    def disable_coding_tool(self) -> None:
        """Disable all built-in coding tools."""
        self.disable_python_coding_tool()
        self.disable_bash_coding_tool()

    def disable_uv_tool(self) -> None:
        """Disable the uv workspace tool."""
        self._disabled_tool_names.add("uv_tool")
        self.unregister_tool(self.uv_tool)

    def disable_subagent_tool(self) -> None:
        """Disable the subagent launcher tool."""
        self._disabled_tool_names.add("subagent_tool")
        self.unregister_tool(self.subagent_tool)

    def disable_workspace_tool(self) -> None:
        """Disable all built-in workspace tools."""
        self.disable_file_system_tool()
        self.disable_image_rendering_tool()
        self.disable_uv_tool()

    def set_coding_tool_sandbox_type(
        self,
        type: SandboxType,
        visible_dirs: Optional[Sequence[str]] = None,
    ) -> None:
        """Set the sandbox type for Python and Bash coding tools."""
        self.coding_tool_sandbox_type = type
        self.bubblewrap_visible_dirs = list(visible_dirs) if visible_dirs is not None else None
        self.set_python_coding_tool_sandbox_type(
            self.coding_tool_sandbox_type,
            visible_dirs=self.bubblewrap_visible_dirs,
        )
        self.set_bash_coding_tool_sandbox_type(
            self.coding_tool_sandbox_type,
            visible_dirs=self.bubblewrap_visible_dirs,
        )

    def set_python_coding_tool_sandbox_type(
        self,
        type: SandboxType,
        visible_dirs: Optional[Sequence[str]] = None,
    ) -> None:
        """Set the sandbox type for the Python coding tool."""
        self.python_coding_tool.set_sandbox_type(type, visible_dirs=visible_dirs)

    def set_bash_coding_tool_sandbox_type(
        self,
        type: SandboxType,
        visible_dirs: Optional[Sequence[str]] = None,
    ) -> None:
        """Set the sandbox type for the Bash coding tool."""
        self.bash_coding_tool.set_sandbox_type(type, visible_dirs=visible_dirs)

    def set_coding_tool_request_approval(self, request_approval: bool) -> None:
        """Set approval requirements for Python and Bash coding tools."""
        self.coding_tool_request_approval = request_approval
        self.task_manager.coding_tool_request_approval = request_approval
        self.set_python_coding_tool_request_approval(request_approval)
        self.set_bash_coding_tool_request_approval(request_approval)

    def set_python_coding_tool_request_approval(self, request_approval: bool) -> None:
        """Set approval requirements for the Python coding tool."""
        self._set_tool_request_approval(self.python_coding_tool, request_approval)

    def set_bash_coding_tool_request_approval(self, request_approval: bool) -> None:
        """Set approval requirements for the Bash coding tool."""
        self._set_tool_request_approval(self.bash_coding_tool, request_approval)

    def _set_tool_request_approval(
        self,
        tool: PythonCodingTool | BashCodingTool,
        request_approval: bool,
    ) -> None:
        tool.require_approval = request_approval
        if self.executor is None:
            return
        for exposed in tool.exposed_tools:
            if not exposed.model_visible or exposed.require_approval is not None:
                continue
            spec = self.executor.tool_specs.get(exposed.name)
            if spec is not None:
                spec.require_approval = request_approval

    @staticmethod
    def _normalize_tools(tools: BaseTool | Sequence[BaseTool]) -> list[BaseTool]:
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        normalized = list(tools)
        for tool in normalized:
            if not isinstance(tool, BaseTool):
                raise ValueError("Input should be a BaseTool or a list of BaseTool objects.")
        return normalized

    def _tool_has_registered_name_conflict(self, tool: BaseTool) -> bool:
        tool_names = self._collect_tool_names([tool])
        return bool(tool_names and tool_names & self._collect_tool_names(self))

    @staticmethod
    def _collect_tool_names(tools: Sequence[BaseTool]) -> set[str]:
        names: set[str] = set()
        for tool in tools:
            for exposed in tool.exposed_tools:
                if exposed.model_visible:
                    names.add(exposed.name)
        return names

    def _set_known_tool_reference(self, tool: BaseTool) -> None:
        if isinstance(tool, SimplePythonEvalTool):
            self.simple_python_eval_tool = tool
        if isinstance(tool, FileSystemTool):
            self.file_system_tool = tool
        if isinstance(tool, ImageRenderingTool):
            self.image_rendering_tool = tool
        if isinstance(tool, PythonCodingTool):
            self.python_coding_tool = tool
        if isinstance(tool, BashCodingTool):
            self.bash_coding_tool = tool
        if isinstance(tool, UvTool):
            self.uv_tool = tool
        if isinstance(tool, SubagentTool):
            self.subagent_tool = tool
