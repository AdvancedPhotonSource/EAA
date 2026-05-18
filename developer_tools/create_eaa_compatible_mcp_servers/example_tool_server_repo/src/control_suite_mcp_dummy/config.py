"""Configuration helpers for the dummy control-suite MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | None) -> dict[str, Any]:
    """Load an optional YAML configuration file.

    Parameters
    ----------
    path
        Path to a YAML configuration file. ``None`` returns an empty mapping.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config YAML must contain a mapping at the top level.")
    return config


def get_config_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a named YAML section as a mapping."""
    section = config.get(section_name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section {section_name!r} must be a mapping.")
    return section


def resolve_setting(
    config: dict[str, Any],
    section_name: str,
    key: str,
    cli_value: Any,
    default: Any,
) -> Any:
    """Resolve one setting with CLI values taking precedence over YAML.

    Parameters
    ----------
    config
        Parsed YAML configuration.
    section_name
        YAML section name to inspect first.
    key
        Setting key inside the section or as a top-level fallback.
    cli_value
        CLI value. ``None`` means the user did not provide an override.
    default
        Fallback value when neither CLI nor YAML provides the setting.

    Returns
    -------
    Any
        Resolved setting value.
    """
    if cli_value is not None:
        return cli_value
    section = get_config_section(config, section_name)
    if key in section:
        return section[key]
    return config.get(key, default)
