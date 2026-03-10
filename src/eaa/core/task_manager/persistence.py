from typing import Any, Optional
import json
import sqlite3

from eaa.core.message_proc import get_message_elements_as_text
from eaa.core.util import get_timestamp


class SQLiteMessageStore:
    """Adapter for the WebUI-compatible SQLite message schema."""

    def __init__(self, path: Optional[str] = None):
        """Initialize the store."""
        self.path = path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open the SQLite connection and ensure the schema exists."""
        if self.path is None:
            return
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS messages (timestamp TEXT, role TEXT, content TEXT, tool_calls TEXT, image TEXT)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS status (id INTEGER PRIMARY KEY, user_input_requested INTEGER)"
        )
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM status")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO status (id, user_input_requested) VALUES (1, 0)")
        self.connection.commit()

    def append_message(self, message: dict[str, Any]) -> None:
        """Persist one OpenAI-compatible message."""
        if self.connection is None:
            return
        elements = get_message_elements_as_text(message)
        image_value = elements["image"]
        if isinstance(image_value, list):
            image_value = json.dumps(image_value)
        self.connection.execute(
            "INSERT INTO messages (timestamp, role, content, tool_calls, image) VALUES (?, ?, ?, ?, ?)",
            (
                str(get_timestamp(as_int=True)),
                elements["role"],
                elements["content"],
                elements["tool_calls"],
                image_value,
            ),
        )
        self.connection.commit()

    def load_messages(self) -> list[dict[str, Any]]:
        """Load persisted messages into OpenAI-compatible message dictionaries."""
        if self.connection is None:
            return []
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT rowid, timestamp, role, content, tool_calls, image FROM messages ORDER BY rowid"
        )
        rows = cursor.fetchall()
        messages: list[dict[str, Any]] = []
        pending_tool_call_ids: list[str] = []
        for _, _, role, content, tool_calls, image in rows:
            normalized_role = "user" if role == "user_webui" else role
            content_text = (content or "").strip()
            if content_text:
                content_text = "\n".join(
                    line for line in content_text.splitlines() if line.strip() != "<image>"
                ).strip()
            message: dict[str, Any] = {"role": normalized_role, "content": content_text}
            image_urls = self.parse_images(image)
            if len(image_urls) > 0:
                message["content"] = [{"type": "text", "text": content_text}]
                message["content"].extend(
                    {"type": "image_url", "image_url": {"url": image_url}}
                    for image_url in image_urls
                )
            tool_call_list = self.parse_tool_calls(tool_calls)
            if tool_call_list:
                message["tool_calls"] = tool_call_list
                pending_tool_call_ids.extend(
                    [tool_call["id"] for tool_call in tool_call_list if "id" in tool_call]
                )
            elif normalized_role == "tool" and pending_tool_call_ids:
                message["tool_call_id"] = pending_tool_call_ids.pop(0)
            messages.append(message)
        return messages

    def set_user_input_requested(self, requested: bool) -> None:
        """Update the WebUI input request flag."""
        if self.connection is None:
            return
        self.connection.execute(
            "UPDATE status SET user_input_requested = ? WHERE id = 1",
            (1 if requested else 0,),
        )
        self.connection.commit()

    def get_latest_webui_input_timestamp(self) -> int:
        """Return the latest WebUI user-input timestamp."""
        if self.connection is None:
            return 0
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT timestamp FROM messages WHERE role = 'user_webui' ORDER BY rowid DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def poll_new_webui_input(self, last_timestamp: int) -> Optional[str]:
        """Return the newest WebUI input after the given timestamp."""
        if self.connection is None:
            return None
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT timestamp, content FROM messages WHERE role = 'user_webui' ORDER BY rowid DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None
        timestamp, content = int(row[0]), row[1]
        return content if timestamp > last_timestamp else None

    @staticmethod
    def parse_tool_calls(tool_calls: Optional[str]) -> list[dict[str, Any]]:
        """Parse the serialized DB tool-call column."""
        if not tool_calls:
            return []
        lines = [line.strip() for line in tool_calls.splitlines() if line.strip()]
        calls: list[dict[str, Any]] = []
        index = 0
        while index < len(lines):
            header = lines[index]
            if ":" not in header:
                index += 1
                continue
            tool_id, tool_name = header.split(":", 1)
            arguments = ""
            if index + 1 < len(lines) and lines[index + 1].startswith("Arguments:"):
                arguments = lines[index + 1].split("Arguments:", 1)[1].strip()
                index += 2
            else:
                index += 1
            calls.append(
                {
                    "id": tool_id.strip(),
                    "type": "function",
                    "function": {"name": tool_name.strip(), "arguments": arguments},
                }
            )
        return calls

    @staticmethod
    def parse_images(image_value: Any) -> list[str]:
        """Parse one or multiple image values from the DB."""
        if not image_value:
            return []
        if isinstance(image_value, bytes):
            image_value = image_value.decode("utf-8", errors="ignore")
        if not isinstance(image_value, str):
            return []
        try:
            parsed = json.loads(image_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            raw_images = [item for item in parsed if isinstance(item, str)]
        elif isinstance(parsed, str):
            raw_images = [parsed]
        else:
            raw_images = [image_value]
        image_urls = []
        for raw_image in raw_images:
            image_urls.append(raw_image if raw_image.startswith("data:image") else f"data:image/png;base64,{raw_image}")
        return image_urls
