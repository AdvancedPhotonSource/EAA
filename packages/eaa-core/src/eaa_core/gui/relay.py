"""Reusable SQLite relay helpers for EAA WebUIs."""

import base64
import os
import sqlite3
from datetime import datetime
from email.utils import formatdate
from typing import Any, Optional

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from eaa_core.task_manager.persistence import (
    PrunableSqliteSaver,
    SQLiteMessageStore,
    parse_persisted_images,
)


class SQLiteWebUIRelay:
    """SQLite-backed communication relay between a WebUI and an EAA task manager.

    Parameters
    ----------
    db_path : str
        Path to the SQLite session database shared with the task manager.
    upload_dir : str, default=".tmp"
        Directory used to store images pasted or uploaded through the UI.
    """

    def __init__(self, db_path: str, upload_dir: str = ".tmp") -> None:
        self.db_path = db_path
        self.upload_dir = upload_dir

    def open_connection(self) -> sqlite3.Connection:
        """Open the relay database and ensure the WebUI schema exists.

        Returns
        -------
        sqlite3.Connection
            Open SQLite connection.
        """
        connection = sqlite3.connect(self.db_path)
        PrunableSqliteSaver(connection).setup()
        return connection

    def load_messages(self, since_id: int | None = None) -> list[dict[str, Any]]:
        """Load display messages for the WebUI.

        Parameters
        ----------
        since_id : int | None, optional
            Return only messages newer than this synthetic or table id.

        Returns
        -------
        list[dict[str, Any]]
            Messages formatted for WebUI display.
        """
        connection = self.open_connection()
        try:
            store = SQLiteMessageStore(self.db_path)
            store.connection = connection
            webui_messages = store.load_webui_messages(since_id=since_id)
            if len(webui_messages) > 0:
                return webui_messages
            saver = PrunableSqliteSaver(connection)
            return saver.load_latest_checkpoint_messages(since_id=since_id)
        finally:
            connection.close()

    def get_user_input_requested(self) -> int | None:
        """Return whether the task manager is currently waiting for UI input.

        Returns
        -------
        int | None
            ``1`` when input is requested, ``0`` while the agent is processing,
            or ``None`` if the status row cannot be read.
        """
        connection = self.open_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT user_input_requested FROM status ORDER BY rowid DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return int(row[0])
        except sqlite3.Error:
            return None
        finally:
            connection.close()

    def enqueue_user_input(self, content: str) -> int:
        """Queue one browser-submitted message for the task manager.

        Parameters
        ----------
        content : str
            Message content submitted by the user.

        Returns
        -------
        int
            Inserted queue row id.
        """
        connection = self.open_connection()
        try:
            store = SQLiteMessageStore(self.db_path)
            store.connection = connection
            return store.enqueue_webui_input(content)
        finally:
            connection.close()

    def ensure_upload_dir(self) -> str:
        """Ensure and return the configured upload directory.

        Returns
        -------
        str
            Upload directory path.
        """
        os.makedirs(self.upload_dir, exist_ok=True)
        return self.upload_dir

    def save_base64_image(self, image_data: str) -> str:
        """Save a base64 image payload from the browser.

        Parameters
        ----------
        image_data : str
            Base64 image bytes, optionally prefixed as a data URL.

        Returns
        -------
        str
            Path to the saved image file.
        """
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file_path = os.path.join(
            self.ensure_upload_dir(),
            f"pasted_image_{timestamp}.png",
        )
        with open(file_path, "wb") as image_file:
            image_file.write(image_bytes)
        return file_path

    @staticmethod
    def parse_images(image_value: Any) -> list[str]:
        """Parse one or multiple persisted image values.

        Parameters
        ----------
        image_value : Any
            Serialized image value from the database or checkpoint payload.

        Returns
        -------
        list[str]
            Normalized image values.
        """
        return parse_persisted_images(image_value)

    @staticmethod
    def guess_mime_type_from_path(path: str) -> str:
        """Return a basic MIME type for an image path.

        Parameters
        ----------
        path : str
            Image path.

        Returns
        -------
        str
            MIME type.
        """
        lower = path.lower()
        if lower.endswith(".png"):
            return "image/png"
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            return "image/jpeg"
        if lower.endswith(".gif"):
            return "image/gif"
        if lower.endswith(".webp"):
            return "image/webp"
        return "application/octet-stream"

    def image_response(self, path: str = Query(...)) -> FileResponse:
        """Build a FastAPI file response for an image path.

        Parameters
        ----------
        path : str
            Absolute or relative image path.

        Returns
        -------
        FileResponse
            Image response.
        """
        normalized_path = os.path.abspath(path)
        if not os.path.exists(normalized_path):
            raise HTTPException(
                status_code=404,
                detail=f"Image not found: {normalized_path}",
            )
        media_type = self.guess_mime_type_from_path(normalized_path)
        stat_result = os.stat(normalized_path)
        headers = {
            "Cache-Control": "public, max-age=3600",
            "ETag": f'"{stat_result.st_mtime_ns:x}-{stat_result.st_size:x}"',
            "Last-Modified": formatdate(stat_result.st_mtime, usegmt=True),
        }
        return FileResponse(normalized_path, media_type=media_type, headers=headers)

    def upload_image_response(self, payload: dict[str, Any]) -> JSONResponse:
        """Build a JSON response for a browser image upload.

        Parameters
        ----------
        payload : dict[str, Any]
            Request body containing an ``image_data`` field.

        Returns
        -------
        JSONResponse
            Upload result response.
        """
        image_data = payload.get("image_data", "")
        if not image_data:
            return JSONResponse({"error": "No image data provided"}, status_code=400)
        try:
            file_path = self.save_base64_image(str(image_data))
        except Exception as exc:
            return JSONResponse(
                {"error": f"Invalid image data: {exc}"},
                status_code=400,
            )
        return JSONResponse({"file_path": file_path}, status_code=201)


_message_db_path: Optional[str] = None
_default_upload_dir = ".tmp"


def set_message_db_path(path: str) -> None:
    """Set the default SQLite database path used by module-level helpers."""
    global _message_db_path
    _message_db_path = path


def get_message_db_path() -> str | None:
    """Return the default SQLite database path used by module-level helpers."""
    return _message_db_path


def set_default_upload_dir(path: str) -> None:
    """Set the default directory used by module-level image upload helpers."""
    global _default_upload_dir
    _default_upload_dir = path


def get_default_upload_dir() -> str:
    """Return the default image upload directory."""
    return _default_upload_dir


def get_default_relay() -> SQLiteWebUIRelay:
    """Return a relay for the configured default SQLite database.

    Returns
    -------
    SQLiteWebUIRelay
        Configured relay.
    """
    if _message_db_path is None:
        raise RuntimeError("Message DB path not set. Call set_message_db_path(path) first.")
    return SQLiteWebUIRelay(_message_db_path, upload_dir=_default_upload_dir)
