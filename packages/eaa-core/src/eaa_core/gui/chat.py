"""Compatibility wrappers for the default EAA WebUI.

The static HTML/CSS/JavaScript WebUI has been retired. This module keeps the
older ``eaa_core.gui.chat`` imports working while delegating runtime UI launch
to the NiceGUI implementation and SQL operations to the reusable relay.
"""

from typing import Any

from eaa_core.gui.nicegui import run_nicegui_webui
from eaa_core.gui.relay import (
    SQLiteWebUIRelay,
    get_default_relay,
    get_message_db_path,
    set_message_db_path as relay_set_message_db_path,
)


def set_message_db_path(path: str) -> None:
    """Set the path to the SQLite database that stores the chat history."""
    relay_set_message_db_path(path)


def _ensure_db() -> None:
    if get_message_db_path() is None:
        raise RuntimeError("Message DB path not set. Call set_message_db_path(path) first.")


def _query_messages(since_id: int | None = None) -> list[dict[str, Any]]:
    """Return checkpoint-backed transcript messages for the WebUI."""
    return get_default_relay().load_messages(since_id=since_id)


def _query_user_input_requested() -> int | None:
    """Return the current WebUI input-request status flag."""
    return get_default_relay().get_user_input_requested()


def _parse_images_field(image_value: Any) -> list[str]:
    """Parse one or multiple images from a persisted image value."""
    return SQLiteWebUIRelay.parse_images(image_value)


def _insert_user_message(content: str) -> None:
    """Queue a WebUI-originated message for the task manager."""
    get_default_relay().enqueue_user_input(content)


def _guess_mime_type_from_path(path: str) -> str:
    """Return a basic MIME type for an image path."""
    return SQLiteWebUIRelay.guess_mime_type_from_path(path)


def _ensure_tmp_dir() -> str:
    """Ensure the default WebUI upload directory exists."""
    return get_default_relay().ensure_upload_dir()


def run_webui(
    host: str = "127.0.0.1",
    port: int = 8008,
    static_dir: str | None = None,
) -> None:
    """Run the default NiceGUI WebUI.

    Parameters
    ----------
    host : str, default="127.0.0.1"
        Bind host.
    port : int, default=8008
        Bind port.
    static_dir : str | None, optional
        Retained for compatibility with the retired static WebUI. The value is
        ignored.
    """
    _ensure_db()
    if static_dir is not None:
        print("The static_dir argument is ignored by the NiceGUI WebUI.")
    db_path = get_message_db_path()
    if db_path is None:
        raise RuntimeError("Message DB path not set. Call set_message_db_path(path) first.")
    run_nicegui_webui(db_path, host=host, port=port)
