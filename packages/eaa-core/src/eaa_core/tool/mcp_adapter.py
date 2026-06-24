"""Compatibility adapters for MCP-backed tools."""

from __future__ import annotations

from typing import Any
import json

import numpy as np

from eaa_core.tool.mcp_client import MCPTool
from eaa_core.tool.param_tuning import SetParameters


def make_json_serializable(value: Any) -> Any:
    """Convert common NumPy values into JSON-serializable Python objects.

    Parameters
    ----------
    value : Any
        Value to normalize before sending it through MCP.

    Returns
    -------
    Any
        JSON-serializable value for arrays, NumPy scalars, mappings, and
        sequences.
    """
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list | tuple):
        return [make_json_serializable(item) for item in value]
    return value


def call_named_tool(tool: Any, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call a named exposed tool on a local or MCP-backed tool object.

    Parameters
    ----------
    tool : Any
        Tool object exposing ``exposed_tools``.
    tool_name : str
        Exposed tool name to call.
    arguments : dict[str, Any], optional
        Keyword arguments for the tool.

    Returns
    -------
    Any
        Tool result.
    """
    arguments = {} if arguments is None else arguments
    if hasattr(tool, tool_name):
        return getattr(tool, tool_name)(**arguments)
    for spec in getattr(tool, "exposed_tools", []):
        if spec.name == tool_name:
            return spec.function(**arguments)
    raise AttributeError(f"Tool {type(tool).__name__} does not expose {tool_name!r}.")


class MCPParameterSettingProxy(SetParameters):
    """SetParameters-compatible adapter for MCP-backed parameter tools.

    Parameters
    ----------
    mcp_tool : MCPTool
        MCP tool wrapper exposing a ``set_parameters`` tool.
    parameter_names : list[str]
        Ordered parameter names managed by the task manager.
    parameter_ranges : list[tuple[float, ...], tuple[float, ...]]
        Parameter bounds used by the task manager.
    """

    def __init__(
        self,
        mcp_tool: MCPTool,
        parameter_names: list[str],
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]],
        *args,
        **kwargs,
    ) -> None:
        self.mcp_tool = mcp_tool
        super().__init__(
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges,
            *args,
            **kwargs,
        )
        remote_schemas = {
            spec.name: spec.schema
            for spec in getattr(mcp_tool, "exposed_tools", [])
            if spec.schema is not None
        }
        for spec in self.exposed_tools:
            remote_name = self.resolve_remote_tool_name(spec.name.rsplit(".", maxsplit=1)[-1])
            if remote_name in remote_schemas:
                spec.schema = remote_schemas[remote_name]

    def resolve_remote_tool_name(self, short_name: str) -> str:
        """Resolve an exact or class-prefixed remote tool name."""
        remote_names = [
            spec.name
            for spec in getattr(self.mcp_tool, "exposed_tools", [])
        ]
        if short_name in remote_names:
            return short_name
        suffix = f".{short_name}"
        matches = [name for name in remote_names if name.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        return short_name

    def set_parameters(self, parameters: list[float]) -> str:
        """Set parameters through the remote tool and update local history."""
        normalized_parameters = make_json_serializable(parameters)
        result = call_named_tool(
            self.mcp_tool,
            self.resolve_remote_tool_name("set_parameters"),
            {"parameters": normalized_parameters},
        )
        self.update_parameter_history(normalized_parameters)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return json.dumps(result)
        return str(result)


def ensure_parameter_setting_tool_interface(
    tool: Any,
    parameter_names: list[str],
    parameter_ranges: list[tuple[float, ...], tuple[float, ...]],
) -> Any:
    """Return a parameter-setting object compatible with task managers."""
    if isinstance(tool, MCPTool):
        return MCPParameterSettingProxy(
            tool,
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges,
        )
    return tool
