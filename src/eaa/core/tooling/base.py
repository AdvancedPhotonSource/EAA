import base64
import inspect
import io
import json
import os
import re
import typing
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, List, Optional, get_args, get_type_hints

import eaa.matplotlib_setup  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from eaa.core.util import get_timestamp


TOOL_RESULT_FIELD = "result"
TOOL_IMAGE_PATH_FIELD = "img_path"
IMAGE_PATH_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}


@dataclass
class ExposedToolSpec:
    """Metadata for a single callable tool method."""

    name: str
    function: Callable[..., Any]
    require_approval: Optional[bool] = None
    schema: Optional[Dict[str, Any]] = None


def tool(name: str):
    """Mark a method as an exposed tool."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(func):
            raise TypeError("@tool can only decorate callables.")
        setattr(func, "is_tool", True)
        setattr(func, "tool_name", name)
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


def looks_like_image_path(value: str) -> bool:
    """Return whether a string looks like a filesystem image path."""
    suffix = os.path.splitext(value.strip())[1].lower()
    return suffix in IMAGE_PATH_SUFFIXES


def normalize_tool_result(result: Any) -> Dict[str, Any]:
    """Normalize a tool return value into a JSON-serializable object.

    Parameters
    ----------
    result : Any
        Raw value returned by a tool implementation.

    Returns
    -------
    dict[str, Any]
        Normalized JSON object. Scalar and list outputs are wrapped under
        ``result``. Image-bearing payloads should use ``img_path``.
    """
    payload: Dict[str, Any]
    if isinstance(result, dict):
        payload = dict(result)
    elif isinstance(result, str):
        stripped = result.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                payload = parsed
            elif looks_like_image_path(stripped):
                payload = {TOOL_IMAGE_PATH_FIELD: result}
            else:
                payload = {TOOL_RESULT_FIELD: result}
        else:
            payload = {TOOL_RESULT_FIELD: result}
    elif isinstance(result, list) and all(isinstance(value, str) for value in result):
        if all(looks_like_image_path(value) for value in result):
            payload = {TOOL_IMAGE_PATH_FIELD: result}
        else:
            payload = {TOOL_RESULT_FIELD: result}
    else:
        payload = {TOOL_RESULT_FIELD: result}
    if "image_path" in payload and TOOL_IMAGE_PATH_FIELD not in payload:
        payload[TOOL_IMAGE_PATH_FIELD] = payload.pop("image_path")
    if "image_paths" in payload and TOOL_IMAGE_PATH_FIELD not in payload:
        payload[TOOL_IMAGE_PATH_FIELD] = payload.pop("image_paths")
    return payload


def build_mcp_output_schema() -> Dict[str, Any]:
    """Build the normalized MCP output schema for EAA tools."""
    return {
        "type": "object",
        "properties": {
            TOOL_RESULT_FIELD: {},
            TOOL_IMAGE_PATH_FIELD: {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
        },
        "additionalProperties": True,
        "x-fastmcp-wrap-result": True,
    }
