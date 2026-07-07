"""Compatibility helpers for MCP-backed tools."""

from __future__ import annotations

from typing import Any

import numpy as np

from eaa_core.tool.base import BaseTool
from eaa_core.tool.mcp_client import MCPTool


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


class MCPRPCWrapper:
    """RPC-style wrapper around an MCP tool client.

    ``MCPRPCWrapper`` is intended for logic-driven task managers that call
    Python methods directly. It is not an agent-visible ``BaseTool``. The
    wrapper maps local method and argument names to facility-specific MCP tool
    names and argument names, then optionally syncs local attributes from the
    remote server through ``get_attribute_payload``.

    Parameters
    ----------
    mcp_tool_client : MCPTool
        MCP client exposing remote tools through ``exposed_tools``.
    mappings : dict[str, dict[str, Any]]
        Per-local-method mapping configuration. Each item must provide
        ``remote`` and may provide ``arguments`` and ``sync`` dictionaries.
    initial_attributes : dict[str, Any], optional
        Attributes to initialize on the wrapper for logic-driven task managers.
    local_methods : dict[str, Any], optional
        Local helper methods. Each callable receives the wrapper as its first
        argument.

    Examples
    --------
    Map a local task-manager method to a facility-specific remote tool and
    rename arguments before the MCP call:

    >>> wrapper = MCPRPCWrapper(
    ...     mcp_tool_client=mcp_tool_client,
    ...     mappings={
    ...         "local_measure": {
    ...             "remote": "facility_collect_frame",
    ...             "arguments": {
    ...                 "local_x": "sample_x",
    ...                 "local_y": "sample_y",
    ...             },
    ...         },
    ...     },
    ... )
    >>> wrapper.local_measure(local_x=1.0, local_y=2.0)
    # Calls facility_collect_frame(sample_x=1.0, sample_y=2.0)

    Sync local attributes after a successful remote call. The remote server
    must expose ``get_attribute_payload(name=...)`` for synced attributes:

    >>> wrapper = MCPRPCWrapper(
    ...     mcp_tool_client=mcp_tool_client,
    ...     mappings={
    ...         "local_measure": {
    ...             "remote": "facility_collect_frame",
    ...             "sync": {
    ...                 "latest_array": "detector.last_frame",
    ...                 "latest_metadata": "detector.last_metadata",
    ...             },
    ...         },
    ...     },
    ... )
    >>> wrapper.local_measure()
    >>> wrapper.latest_array
    array(...)

    Add a local helper method when the task manager needs local bookkeeping
    around an RPC call:

    >>> wrapper = MCPRPCWrapper(
    ...     mcp_tool_client=mcp_tool_client,
    ...     mappings={"set_values": {"remote": "facility_set_values"}},
    ...     initial_attributes={"parameter_names": ["a"], "parameter_history": {"a": []}},
    ...     local_methods={"set_values": set_parameters_with_local_history},
    ... )
    >>> wrapper.set_values([0.5])
    >>> wrapper.get_parameter_at_iteration(-1)
    [0.5]
    """

    def __init__(
        self,
        mcp_tool_client: MCPTool,
        mappings: dict[str, dict[str, Any]],
        initial_attributes: dict[str, Any] | None = None,
        local_methods: dict[str, Any] | None = None,
    ) -> None:
        self.mcp_tool_client = mcp_tool_client
        self.mappings = self._normalize_mappings(mappings)
        self.local_methods = dict(local_methods or {})
        for name, value in (initial_attributes or {}).items():
            setattr(self, name, value)

    @staticmethod
    def _normalize_mappings(
        mappings: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Validate and normalize local-to-remote mapping configuration."""
        if not isinstance(mappings, dict):
            raise TypeError("`mappings` must be a dictionary keyed by local method name.")

        normalized: dict[str, dict[str, Any]] = {}
        for local_name, config in mappings.items():
            if not isinstance(local_name, str) or not local_name:
                raise ValueError("Mapping local method names must be non-empty strings.")
            if isinstance(config, str):
                config = {"remote": config}
            if not isinstance(config, dict):
                raise TypeError(f"Mapping for {local_name!r} must be a dictionary.")
            remote_name = config.get("remote")
            if not isinstance(remote_name, str) or not remote_name:
                raise ValueError(f"Mapping for {local_name!r} must define a remote tool name.")
            argument_mapping = config.get("arguments", {})
            if not isinstance(argument_mapping, dict):
                raise TypeError(f"`arguments` mapping for {local_name!r} must be a dictionary.")
            sync_mapping = config.get("sync", {})
            if isinstance(sync_mapping, (set, list, tuple)):
                sync_mapping = {name: name for name in sync_mapping}
            if not isinstance(sync_mapping, dict):
                raise TypeError(f"`sync` mapping for {local_name!r} must be a dictionary.")
            normalized[local_name] = {
                "remote": remote_name,
                "arguments": dict(argument_mapping),
                "sync": dict(sync_mapping),
            }
        return normalized

    def __getattr__(self, name: str):
        """Return a configured local helper or RPC method."""
        if name in self.local_methods:
            local_method = self.local_methods[name]
            if not callable(local_method):
                raise TypeError(f"Local method {name!r} is not callable.")

            def local_function(*args, **kwargs):
                return local_method(self, *args, **kwargs)

            return local_function
        if name in self.mappings:
            def rpc_function(**kwargs):
                return self.call(name, **kwargs)

            return rpc_function
        raise AttributeError(name)

    def resolve_remote_tool_name(self, short_name: str) -> str:
        """Resolve an exact or uniquely class-prefixed remote tool name."""
        remote_names = [
            spec.name
            for spec in getattr(self.mcp_tool_client, "exposed_tools", [])
        ]
        if short_name in remote_names:
            return short_name
        suffix = f".{short_name}"
        matches = [name for name in remote_names if name.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AttributeError(
                f"Remote tool name {short_name!r} is ambiguous; matches: {matches}."
            )
        raise AttributeError(
            f"Tool {type(self.mcp_tool_client).__name__} does not expose {short_name!r}."
        )

    def call(self, local_name: str, **kwargs) -> Any:
        """Call a configured local RPC method by name."""
        if local_name not in self.mappings:
            raise AttributeError(f"No RPC mapping configured for {local_name!r}.")
        config = self.mappings[local_name]
        remote_kwargs = {
            config["arguments"].get(name, name): value
            for name, value in kwargs.items()
        }
        result = call_named_tool(
            self.mcp_tool_client,
            self.resolve_remote_tool_name(config["remote"]),
            make_json_serializable(remote_kwargs),
        )
        self.sync_attributes(config["sync"])
        return result

    def sync_attributes(self, sync_mapping: dict[str, str]) -> None:
        """Sync configured local attributes from remote ``get_attribute_payload``."""
        if not sync_mapping:
            return
        get_attribute_payload = self.resolve_remote_tool_name("get_attribute_payload")
        for local_attr, remote_attr in sync_mapping.items():
            payload = call_named_tool(
                self.mcp_tool_client,
                get_attribute_payload,
                {"name": remote_attr},
            )
            setattr(self, local_attr, self.decode_payload(payload))

    @staticmethod
    def decode_payload(payload: Any) -> Any:
        """Decode an EAA array payload or return a JSON-compatible value."""
        encoded_data = payload.get("encoded_data") if isinstance(payload, dict) else None
        if isinstance(encoded_data, dict) and encoded_data.get("type") == "array":
            return BaseTool.decode_array_payload(payload)
        return payload

    @property
    def len_parameter_history(self) -> int:
        """Return the length of a synced local ``parameter_history`` attribute."""
        parameter_history = getattr(self, "parameter_history")
        parameter_names = list(parameter_history.keys())
        if len(parameter_names) == 0:
            return 0
        return len(parameter_history[parameter_names[0]])

    def get_parameter_at_iteration(
        self,
        iteration: int,
        as_dict: bool = False,
    ) -> list[float] | dict[str, float]:
        """Return values from a local ``parameter_history`` attribute."""
        parameter_history = getattr(self, "parameter_history")
        if as_dict:
            return {
                key: parameter_history[key][iteration]
                for key in parameter_history
            }
        return [parameter_history[key][iteration] for key in parameter_history]

    def update_parameter_history(
        self,
        parameters: list[float] | dict[str, float],
    ) -> None:
        """Append values to a local ``parameter_history`` attribute."""
        parameter_history = getattr(self, "parameter_history")
        if isinstance(parameters, dict):
            for name, param in parameters.items():
                parameter_history[name].append(param)
            return
        for index, value in enumerate(parameters):
            parameter_history[self.parameter_names[index]].append(value)


def set_parameters_with_local_history(
    wrapper: MCPRPCWrapper,
    parameters: list[float],
) -> Any:
    """Call remote ``set_parameters`` and update wrapper-local history."""
    normalized_parameters = make_json_serializable(parameters)
    result = wrapper.call("set_parameters", parameters=normalized_parameters)
    wrapper.update_parameter_history(normalized_parameters)
    return result
