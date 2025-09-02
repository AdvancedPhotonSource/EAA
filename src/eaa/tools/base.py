import typing
from typing import Optional, Dict, Callable, List, Any, get_args, get_type_hints
import base64
import os
import io
from enum import StrEnum, auto
import inspect

import matplotlib.pyplot as plt
import numpy as np

import eaa.util


class ToolReturnType(StrEnum):
    TEXT = auto()
    IMAGE_PATH = auto()
    NUMBER = auto()
    BOOL = auto()
    LIST = auto()
    DICT = auto()
    EXCEPTION = auto()


class BaseTool:
    
    name: str = "base_tool"
        
    def __init__(self, build: bool = True, *args, **kwargs):
        if build:
            self.build(*args, **kwargs)
        
        self.exposed_tools: List[Dict[str, Any]] = []

    def build(self, *args, **kwargs):
        pass

    def convert_image_to_base64(self, image: np.ndarray) -> str:
        """Convert an image to a base64 string."""
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64

    @staticmethod
    def save_image_to_temp_dir(
        fig: plt.Figure, 
        filename: Optional[str] = None, 
        add_timestamp: bool = False
    ) -> str:
        """Save a figure to the temporary directory.

        Parameters
        ----------
        fig : plt.Figure
            The figure to save.
        filename : str, optional
            The filename to save the image as. If not provided, the image is
            saved as "image.png".
        add_timestamp : bool, optional
            If True, the timestamp is added to the filename.
            
        Returns
        -------
        str
            The path to the saved image.
        """
        if not os.path.exists(".tmp"):
            os.makedirs(".tmp")
        if filename is None:
            filename = "image.png"
        else:
            if not filename.endswith(".png"):
                filename = filename + ".png"
        if add_timestamp:
            parts = os.path.splitext(filename)
            filename = parts[0] + "_" + eaa.util.get_timestamp() + parts[1]
        path = os.path.join(".tmp", filename)
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return path
    
    def create_image_message(self, image: np.ndarray, text: str) -> str:
        """Create a message with an image."""
        img_base64 = self.convert_image_to_base64(image)
        image_message = {
            "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ],
            "role": "user"
        }
        return image_message
    

def check(init_method: Callable):
    def wrapper(self, *args, **kwargs):
        return_value = init_method(self, *args, **kwargs)
        if (not hasattr(self, "exposed_tools") 
            or (hasattr(self, "exposed_tools") and len(self.exposed_tools) == 0)
        ):
            raise ValueError(
                "A subclass of BaseTool must have a non-empty `exposed_tools` attribute "
                "containing a dictionary of tool names and their corresponding callable functions."
            )
        return return_value
    return wrapper


def generate_openai_tool_schema(tool_name: str, func: Callable) -> Dict[str, Any]:
    """
    Generates an OpenAI-compatible tool schema from a Python function
    with type annotations and a docstring.

    Parameters
    ----------
    tool_name : str
        The name of the tool.
    func : Callable
        The function to generate the tool schema from.

    Returns
    -------
    dict
        The OpenAI-compatible tool schema.
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""

    # JSON schema type mapping
    python_type_to_json = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        tuple: "array",
        dict: "object"
    }

    def resolve_json_type(py_type):
        origin = typing.get_origin(py_type)
        args = typing.get_args(py_type)
        if origin is list or origin is typing.List:
            return {
                "type": "array",
                "items": {"type": python_type_to_json.get(args[0], "string")}
            }
        return {"type": python_type_to_json.get(py_type, "string")}

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name not in type_hints:
            continue
        json_type = resolve_json_type(type_hints[name])
        description = f"{name} parameter"
        if len(get_args(sig.parameters[name].annotation)) > 0:
            description = get_args(sig.parameters[name].annotation)[1]
        properties[name] = {**json_type, "description": description}
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
                "required": required
            }
        }
    }
