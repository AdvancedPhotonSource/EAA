"""AcquireImage-compatible adapter for MCP-backed acquisition tools."""

from __future__ import annotations

from typing import Any
import copy

import numpy as np

from eaa_core.tool.base import ExposedToolSpec, tool
from eaa_core.tool.mcp_adapter import call_named_tool
from eaa_core.tool.mcp_client import MCPTool
from eaa_imaging.tool.imaging.acquisition import AcquireImage


class MCPAcquireImageProxy(AcquireImage):
    """AcquireImage-compatible adapter around an MCP acquisition server.

    Parameters
    ----------
    mcp_tool : MCPTool
        MCP tool wrapper exposing ``acquire_image`` and optionally
        ``acquire_line_scan``.
    """

    name = "mcp_acquire_image_proxy"

    def __init__(
        self,
        mcp_tool: MCPTool,
        *args,
        require_approval: bool = False,
        **kwargs,
    ) -> None:
        self.mcp_tool = mcp_tool
        self._line_scan_return_gaussian_fit = False
        super().__init__(*args, require_approval=require_approval, **kwargs)
        self.exposed_tools = self.build_proxy_tool_specs()

    def __deepcopy__(self, memo):
        """Return a proxy copy that shares the MCP connection but copies local state."""
        copied = type(self)(
            self.mcp_tool,
            require_approval=self.require_approval,
        )
        memo[id(self)] = copied
        for attr in (
            "image_0",
            "image_km1",
            "image_k",
            "psize_0",
            "psize_km1",
            "psize_k",
            "image_0_path",
            "image_km1_path",
            "image_k_path",
            "image_acquisition_call_history",
            "line_scan_call_history",
            "_line_scan_return_gaussian_fit",
        ):
            setattr(copied, attr, copy.deepcopy(getattr(self, attr), memo))
        return copied

    def build_proxy_tool_specs(self) -> list[ExposedToolSpec]:
        """Build proxy tool specs using remote schemas when available."""
        def make_remote_function(tool_name: str):
            def remote_function(**kwargs):
                return call_named_tool(self.mcp_tool, tool_name, kwargs)

            return remote_function

        remote_specs = {
            spec.name: spec
            for spec in getattr(self.mcp_tool, "exposed_tools", [])
        }
        local_functions = {
            "acquire_image": self.acquire_image,
            "acquire_line_scan": self.acquire_line_scan,
            "get_current_image_info": self.get_current_image_info,
            "get_previous_image_info": self.get_previous_image_info,
            "get_initial_image_info": self.get_initial_image_info,
        }
        specs: list[ExposedToolSpec] = []
        for name, remote_spec in remote_specs.items():
            function = local_functions.get(name)
            if function is None:
                function = make_remote_function(name)
            specs.append(
                ExposedToolSpec(
                    name=name,
                    function=function,
                    require_approval=remote_spec.require_approval,
                    schema=remote_spec.schema,
                )
            )
        for spec in super().discover_tools():
            if spec.name not in remote_specs:
                specs.append(spec)
        return specs

    @property
    def line_scan_return_gaussian_fit(self) -> bool:
        """Return whether line scans should include Gaussian fit metadata."""
        return self._line_scan_return_gaussian_fit

    @line_scan_return_gaussian_fit.setter
    def line_scan_return_gaussian_fit(self, value: bool) -> None:
        self._line_scan_return_gaussian_fit = bool(value)
        for tool_name in ("set_attribute", "set_config"):
            try:
                call_named_tool(
                    self.mcp_tool,
                    tool_name,
                    {"name": "line_scan_return_gaussian_fit", "value": bool(value)},
                )
                return
            except AttributeError:
                continue

    def resolve_pixel_size(self, result: dict[str, Any], kwargs: dict[str, Any]) -> float:
        """Resolve image pixel size from a remote result or acquisition kwargs."""
        for key in ("psize", "pixel_size", "scan_step", "stepsize_x"):
            value = result.get(key, kwargs.get(key))
            if value is not None:
                return float(value)
        return 1.0

    def sync_image_buffers_from_result(
        self,
        result: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        """Load returned ``.npy`` image artifact and update local buffers."""
        array_path = result.get("array_path")
        if not isinstance(array_path, str):
            return
        image = np.load(array_path)
        self.update_image_buffers(
            image,
            psize=self.resolve_pixel_size(result, kwargs),
        )

    def record_image_acquisition_from_kwargs(
        self,
        result: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        """Record acquisition history from common remote acquisition schemas."""
        x_center = kwargs.get("x_center")
        y_center = kwargs.get("y_center")
        size_x = kwargs.get("size_x", kwargs.get("width"))
        size_y = kwargs.get("size_y", kwargs.get("height"))
        psize = self.resolve_pixel_size(result, kwargs)
        psize_x = kwargs.get("stepsize_x", psize)
        psize_y = kwargs.get("stepsize_y", psize)
        if None in {x_center, y_center, size_x, size_y}:
            return
        self.update_image_acquisition_call_history(
            x_center=x_center,
            y_center=y_center,
            size_x=size_x,
            size_y=size_y,
            psize_x=psize_x,
            psize_y=psize_y,
        )

    @tool(name="acquire_image")
    def acquire_image(self, **kwargs) -> dict[str, Any]:
        """Acquire an image through the remote MCP tool."""
        result = call_named_tool(self.mcp_tool, "acquire_image", kwargs)
        if isinstance(result, dict):
            self.record_image_acquisition_from_kwargs(result, kwargs)
            self.sync_image_buffers_from_result(result, kwargs)
        return result

    @tool(name="acquire_line_scan")
    def acquire_line_scan(self, **kwargs) -> dict[str, Any]:
        """Acquire a line scan through the remote MCP tool."""
        x_center = kwargs.get("x_center")
        y_center = kwargs.get("y_center")
        length = kwargs.get("length")
        step = kwargs.get("scan_step", kwargs.get("stepsize_x"))
        if None not in {x_center, y_center, length, step}:
            self.update_line_scan_call_history(
                step=step,
                x_center=x_center,
                y_center=y_center,
                length=length,
                angle=kwargs.get("angle", 0.0),
            )
        return call_named_tool(self.mcp_tool, "acquire_line_scan", kwargs)


def ensure_acquisition_tool_interface(tool: Any) -> Any:
    """Return an acquisition object compatible with imaging task managers."""
    if isinstance(tool, MCPTool):
        return MCPAcquireImageProxy(tool)
    return tool
