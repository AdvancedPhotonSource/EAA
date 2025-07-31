from typing import Optional, Dict, Callable, List, Any
import base64
import os
import io
from enum import StrEnum, auto

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
    
    def plot_2d_image(
        self, 
        image: np.ndarray, 
        add_axis_ticks: bool = False,
        x_ticks: Optional[List[float]] = None,
        y_ticks: Optional[List[float]] = None,
        add_grid_lines: bool = False,
        invert_yaxis: bool = False,
    ) -> plt.Figure:
        """Save an image to the temporary directory.

        Parameters
        ----------
        image : np.ndarray
            The image to save.
        add_axis_ticks : bool, optional
            If True, axis ticks are added to the image to indicate positions.
        x_ticks : List[float], optional
            The x-axis ticks to add to the image. Required when `add_axis_ticks` is True.
        y_ticks : List[float], optional
            The y-axis ticks to add to the image. Required when `add_axis_ticks` is True.
        add_grid_lines : bool, optional
            If True, grid lines are added to the image.
        invert_yaxis : bool, optional
            If True, the y-axis is inverted.
        """
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, cmap='gray')
        if add_axis_ticks:
            ax.set_xticks(np.linspace(0, len(x_ticks) - 1, 5, dtype=int))
            ax.set_yticks(np.linspace(0, len(y_ticks) - 1, 5, dtype=int))
            ax.set_xticklabels([np.round(x_ticks[i], 2) for i in ax.get_xticks()])
            ax.set_yticklabels([np.round(y_ticks[i], 2) for i in ax.get_yticks()])
        ax.grid(add_grid_lines)
        if invert_yaxis:
            ax.invert_yaxis()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        return fig
    
    def save_image_to_temp_dir(
        self, 
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
