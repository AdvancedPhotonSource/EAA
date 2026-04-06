"""Helpers for rendering task-manager prompt templates."""

from importlib import resources
from typing import Any


def render_prompt_template(
    package: str,
    template_name: str,
    replacements: dict[str, Any],
) -> str:
    """Render a prompt template using ``str.format`` placeholders.

    Parameters
    ----------
    package : str
        Package containing the template resource.
    template_name : str
        Template filename within the package.
    replacements : dict[str, Any]
        Mapping from placeholder names to runtime values.

    Returns
    -------
    str
        Rendered prompt text.
    """
    template = resources.files(package).joinpath(template_name).read_text(
        encoding="utf-8"
    )
    normalized = {
        key: "" if value is None else value for key, value in replacements.items()
    }
    return template.format(**normalized)
