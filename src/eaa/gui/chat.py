"""Lightweight WebUI server for the EAA agent system (FastAPI version).

This module provides a standalone HTTP server with a minimal, modern Web UI
that interacts with a SQLite database as the relay between the agent workflow
and the frontend. It replaces the previous Chainlit-based implementation.

Usage:

Create a Python script (e.g. `start_webui.py`) with:

```
from eaa.gui.chat import set_message_db_path, run_webui
set_message_db_path("/absolute/path/to/messages.db")
run_webui(host="127.0.0.1", port=8008)
```

Then open your browser at http://127.0.0.1:8008
"""

import os
import re
import sqlite3
from typing import Any
import base64
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from eaa.util import get_timestamp


_message_db_path = None


def set_message_db_path(path: str):
    """Set the path to the SQLite database that stores the chat history."""
    global _message_db_path
    _message_db_path = path


def get_message_db_path():
    global _message_db_path
    return _message_db_path


def _ensure_db():
    if _message_db_path is None:
        raise RuntimeError("Message DB path not set. Call set_message_db_path(path) first.")


def _open_db_connection() -> sqlite3.Connection:
    _ensure_db()
    conn = sqlite3.connect(_message_db_path)
    return conn


def _query_messages(since_id: int | None = None) -> list[tuple[Any, ...]]:
    conn = _open_db_connection()
    try:
        cursor = conn.cursor()
        if since_id is None:
            cursor.execute(
                "SELECT rowid, timestamp, role, content, tool_calls, image FROM messages ORDER BY rowid"
            )
        else:
            cursor.execute(
                "SELECT rowid, timestamp, role, content, tool_calls, image FROM messages WHERE rowid > ? ORDER BY rowid",
                (since_id,),
            )
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()


def _insert_user_message(content: str):
    conn = _open_db_connection()
    try:
        conn.execute(
            "INSERT INTO messages (timestamp, role, content, tool_calls, image) VALUES (?, ?, ?, ?, ?)",
            (str(get_timestamp(as_int=True)), "user_webui", content, None, None),
        )
        conn.commit()
    finally:
        conn.close()


def _guess_mime_type_from_path(path: str) -> str:
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


def _ensure_tmp_dir():
    """Ensure .tmp directory exists for storing pasted images."""
    tmp_dir = ".tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return tmp_dir


def _get_static_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(here, "webui_static")
    return static_dir


def get_app(static_dir: str | None = None) -> FastAPI:
    _ensure_db()
    app = FastAPI()

    # --- API endpoints ---
    @app.get("/api/messages")
    def api_get_messages(since_id: int | None = Query(default=None)):
        try:
            rows = _query_messages(since_id=since_id)
            data = []
            for row in rows:
                rowid, timestamp, role, content, tool_calls, image_b64 = row
                image_url = None
                if image_b64 is not None:
                    if isinstance(image_b64, bytes):
                        image_b64 = image_b64.decode("utf-8", errors="ignore")
                    if image_b64.startswith("data:image"):
                        image_url = image_b64
                    else:
                        image_url = f"data:image/png;base64,{image_b64}"
                data.append(
                    {
                        "id": rowid,
                        "timestamp": timestamp,
                        "role": role,
                        "content": content or "",
                        "tool_calls": tool_calls,
                        "image": image_url,
                    }
                )
            return JSONResponse({"messages": data})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/messages")
    async def api_post_message(payload: dict):
        try:
            content = payload.get("content", "") if isinstance(payload, dict) else ""
            if not isinstance(content, str):
                content = str(content)
            _insert_user_message(content)
            return JSONResponse({"status": "ok"}, status_code=201)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/image")
    def api_get_image(path: str = Query(...)):
        normalized_path = os.path.abspath(path)
        if not os.path.exists(normalized_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {normalized_path}")
        media_type = _guess_mime_type_from_path(normalized_path)
        return FileResponse(normalized_path, media_type=media_type)

    @app.post("/api/upload-image")
    async def api_upload_image(payload: dict):
        """Handle clipboard image uploads from the frontend."""
        try:
            # Get base64 image data from payload
            image_data = payload.get("image_data", "")
            if not image_data:
                return JSONResponse({"error": "No image data provided"}, status_code=400)
            
            # Remove data URL prefix if present
            if image_data.startswith("data:image"):
                image_data = image_data.split(",", 1)[1]
            
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                return JSONResponse({"error": f"Invalid base64 image data: {str(e)}"}, status_code=400)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"pasted_image_{timestamp}.png"
            
            # Ensure .tmp directory exists
            tmp_dir = _ensure_tmp_dir()
            file_path = os.path.join(tmp_dir, filename)
            
            # Save image file
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            
            # Return the relative path for insertion into message
            return JSONResponse({"file_path": file_path}, status_code=201)
            
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # --- Static files ---
    directory = static_dir or _get_static_dir()
    app.mount("/", StaticFiles(directory=directory, html=True), name="static")

    return app


def run_webui(host: str = "127.0.0.1", port: int = 8008, static_dir: str | None = None):
    """Run the standalone WebUI server using Uvicorn."""
    app = get_app(static_dir=static_dir)
    print(f"EAA WebUI running at http://{host}:{port} (DB: {_message_db_path})")
    uvicorn.run(app, host=host, port=port, log_level="info")


# --- Minimal markdown helpers retained (client renders markdown) ---
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
CODE_BLOCK_RE = re.compile(r"```([\s\S]*?)```", re.MULTILINE)


def render_markdown_minimal(text: str) -> str:
    """Very small markdown renderer for inline use (fallback)."""
    def _escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    escaped = _escape_html(text)
    def _code_block_sub(match: re.Match):
        code = match.group(1)
        return f"<pre><code>{_escape_html(code)}</code></pre>"
    escaped = CODE_BLOCK_RE.sub(_code_block_sub, escaped)
    escaped = INLINE_CODE_RE.sub(r"<code>\1</code>", escaped)
    return escaped
