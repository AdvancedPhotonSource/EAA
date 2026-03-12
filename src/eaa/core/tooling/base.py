import base64
import inspect
import io
import os
import re
import typing
from dataclasses import dataclass
from enum import StrEnum, auto
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, List, Optional, get_args, get_type_hints

import matplotlib.pyplot as plt
import numpy as np

from eaa.core.util import get_timestamp


class ToolReturnType(StrEnum):
    """Supported tool return types."""

    TEXT = auto()
    IMAGE_PATH = auto()
    NUMBER = auto()
    BOOL = auto()
    LIST = auto()
    DICT = auto()
    EXCEPTION = auto()


@dataclass
class ExposedToolSpec:
    """Metadata for a single callable tool method."""

    name: str
    function: Callable[..., Any]
    return_type: ToolReturnType
    require_approval: Optional[bool] = None
    schema: Optional[Dict[str, Any]] = None


def tool(name: str, return_type: ToolReturnType):
    """Mark a method as an exposed tool."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(func):
            raise TypeError("@tool can only decorate callables.")
        setattr(func, "is_tool", True)
        setattr(func, "tool_name", name)
        setattr(func, "tool_return_type", return_type)
        return func

    return decorator


class BaseTool:
    """Base class for stateful tools."""

    name: str = "base_tool"

    def __init__(
        self,
        build: bool = True,
        *args,
        require_approval: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the tool.

        Parameters
        ----------
        build : bool, optional
            Whether to call :meth:`build` during initialization.
        *args
            Positional arguments forwarded to :meth:`build`.
        require_approval : bool, optional
            Whether execution of the tool requires approval.
        name : str, optional
            Instance-level tool name override.
        **kwargs
            Keyword arguments forwarded to :meth:`build`.
        """
        self.name = name if name is not None else self.get_default_name()
        self.require_approval = require_approval
        if build:
            self.build(*args, **kwargs)
        self.exposed_tools = self.discover_tools()

    def build(self, *args, **kwargs):
        """Build internal tool state."""

    @classmethod
    def get_default_name(cls) -> str:
        """Return the default instance name for the tool class."""
        declared_name = cls.__dict__.get("name")
        if isinstance(declared_name, str) and declared_name != BaseTool.name:
            return declared_name
        return cls.camel_to_snake(cls.__name__)

    @staticmethod
    def camel_to_snake(name: str) -> str:
        """Convert a CamelCase class name into snake_case."""
        step_one = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step_one).lower()

    def convert_image_to_base64(self, image: np.ndarray) -> str:
        """Convert an image array to a base64-encoded PNG string."""
        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def save_image_to_temp_dir(
        fig: plt.Figure,
        filename: Optional[str] = None,
        add_timestamp: bool = False,
    ) -> str:
        """Save a matplotlib figure into `.tmp` and return the file path."""
        os.makedirs(".tmp", exist_ok=True)
        filename = filename or "image.png"
        if not filename.endswith(".png"):
            filename = f"{filename}.png"
        if add_timestamp:
            stem, suffix = os.path.splitext(filename)
            filename = f"{stem}_{get_timestamp()}{suffix}"
        path = os.path.join(".tmp", filename)
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return path

    def discover_tools(self) -> List[ExposedToolSpec]:
        """Discover methods decorated with `@tool`."""
        discovered: List[ExposedToolSpec] = []
        seen: set[str] = set()
        for cls in self.__class__.mro():
            if cls is object:
                continue
            for attr_name, attr_value in cls.__dict__.items():
                if attr_name in seen:
                    continue
                target = self.unwrap_descriptor(attr_value)
                if target is None or not getattr(target, "is_tool", False):
                    continue
                seen.add(attr_name)
                bound = getattr(self, attr_name)
                overrides = getattr(self, "tool_name_overrides", None)
                name = overrides.get(attr_name) if overrides and attr_name in overrides else getattr(target, "tool_name", attr_name)
                discovered.append(
                    ExposedToolSpec(
                        name=name,
                        function=bound,
                        return_type=getattr(target, "tool_return_type"),
                    )
                )
        return discovered

    @staticmethod
    def unwrap_descriptor(attribute: Any) -> Optional[Callable[..., Any]]:
        """Resolve the underlying callable from class descriptors."""
        if isinstance(attribute, (staticmethod, classmethod)):
            return attribute.__func__
        if isinstance(attribute, (FunctionType, MethodType)):
            return attribute
        if callable(attribute):
            return attribute
        return None


def check(init_method: Callable):
    """Validate that a tool exposes at least one tool method."""

    def wrapper(self, *args, **kwargs):
        result = init_method(self, *args, **kwargs)
        if not hasattr(self, "exposed_tools") or len(getattr(self, "exposed_tools", [])) == 0:
            raise ValueError(
                "A subclass of BaseTool must define `exposed_tools` with at least one ExposedToolSpec."
            )
        for exposed in self.exposed_tools:
            if not isinstance(exposed, ExposedToolSpec):
                raise TypeError("Items in `exposed_tools` must be instances of ExposedToolSpec.")
        return result

    return wrapper


def generate_openai_tool_schema(tool_name: str, func: Callable) -> Dict[str, Any]:
    """Generate an OpenAI-compatible tool schema from a Python callable."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""
    python_type_to_json = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        tuple: "array",
        dict: "object",
    }

    def resolve_json_type(py_type):
        origin = typing.get_origin(py_type)
        args = typing.get_args(py_type)
        if origin in {list, typing.List}:
            item_type = python_type_to_json.get(args[0], "string") if args else "string"
            return {"type": "array", "items": {"type": item_type}}
        return {"type": python_type_to_json.get(py_type, "string")}

    properties = {}
    required = []
    for name, param in signature.parameters.items():
        if name not in type_hints:
            continue
        description = f"{name} parameter"
        if len(get_args(signature.parameters[name].annotation)) > 1:
            description = get_args(signature.parameters[name].annotation)[1]
        properties[name] = {**resolve_json_type(type_hints[name]), "description": description}
        if param.default == inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def build_mcp_output_schema(return_type: ToolReturnType) -> Dict[str, Any]:
    """Build an MCP output schema that preserves EAA tool return metadata.

    Parameters
    ----------
    return_type : ToolReturnType
        Declared EAA return type for the tool.

    Returns
    -------
    dict[str, Any]
        MCP output schema with a wrapped ``result`` field and an
        ``x-eaa-return-type`` extension.
    """
    json_type_by_return_type = {
        ToolReturnType.TEXT: "string",
        ToolReturnType.IMAGE_PATH: "string",
        ToolReturnType.NUMBER: "number",
        ToolReturnType.BOOL: "boolean",
        ToolReturnType.LIST: "array",
        ToolReturnType.DICT: "object",
        ToolReturnType.EXCEPTION: "string",
    }
    result_schema: Dict[str, Any] = {"type": json_type_by_return_type[return_type]}
    if return_type == ToolReturnType.LIST:
        result_schema["items"] = {}
    return {
        "type": "object",
        "properties": {"result": result_schema},
        "required": ["result"],
        "x-fastmcp-wrap-result": True,
        "x-eaa-return-type": return_type.value,
    }


def parse_mcp_output_schema_return_type(
    output_schema: Optional[Dict[str, Any]],
) -> ToolReturnType:
    """Recover an EAA tool return type from an MCP output schema.

    Parameters
    ----------
    output_schema : dict[str, Any] or None
        Output schema returned by the MCP server.

    Returns
    -------
    ToolReturnType
        Parsed return type. Defaults to ``ToolReturnType.TEXT`` when no EAA
        metadata is available.
    """
    if output_schema is None:
        return ToolReturnType.TEXT
    value = output_schema.get("x-eaa-return-type")
    if value is None:
        return ToolReturnType.TEXT
    try:
        return ToolReturnType(value)
    except ValueError:
        return ToolReturnType.TEXT
