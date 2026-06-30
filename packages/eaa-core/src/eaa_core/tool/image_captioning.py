from __future__ import annotations

from typing import Annotated, Any

from eaa_core.tool.base import BaseTool, tool


class ImageCaptioning(BaseTool):
    """Control automatic captions for images returned by tools."""

    name: str = "image_captioning"

    def __init__(self, task_manager: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the image-captioning control tool.

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

    @tool(name="image_captioning.toggle_auto_image_captioner")
    def toggle_auto_image_captioner(
        self,
        enabled: Annotated[
            bool,
            "Whether automatically captioning tool-returned images should be enabled.",
        ],
    ) -> dict[str, bool]:
        """Enable or disable automatic captions for images returned by tools.

        Parameters
        ----------
        enabled : bool
            Whether automatic image captioning should be enabled.

        Returns
        -------
        dict[str, bool]
            Current automatic image-captioning state.
        """
        self.task_manager.auto_image_captioner_enabled = bool(enabled)
        return {"auto_image_captioner_enabled": self.task_manager.auto_image_captioner_enabled}
