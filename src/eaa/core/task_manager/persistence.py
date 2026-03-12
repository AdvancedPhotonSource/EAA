from typing import Any, Optional
import ast
import json
import sqlite3

from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.sqlite import SqliteSaver

from eaa.core.message_proc import get_message_elements_as_text
from eaa.core.util import get_timestamp


def parse_persisted_images(image_value: Any) -> list[str]:
    """Normalize persisted WebUI/session DB image values to data URLs.

    Parameters
    ----------
    image_value : Any
        Serialized image payload read from the shared SQLite session database.

    Returns
    -------
    list[str]
        Normalized image URLs. Base64 payloads without a data-URL prefix are
        converted to ``data:image/png;base64,...``.
    """
    if not image_value:
        return []
    if isinstance(image_value, list):
        raw_images = [item for item in image_value if isinstance(item, str)]
    else:
        if isinstance(image_value, bytes):
            image_value = image_value.decode("utf-8", errors="ignore")
        if not isinstance(image_value, str):
            return []
        parsed: Any = image_value
        for _ in range(2):
            if not isinstance(parsed, str):
                break
            try:
                parsed = json.loads(parsed)
                continue
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(parsed)
                except (ValueError, SyntaxError):
                    break
        if isinstance(parsed, list):
            raw_images = [item for item in parsed if isinstance(item, str)]
        elif isinstance(parsed, str):
            raw_images = [parsed]
        else:
            raw_images = [image_value]

    image_urls: list[str] = []
    for raw_image in raw_images:
        image_urls.append(
            raw_image
            if raw_image.startswith("data:image")
            else f"data:image/png;base64,{raw_image}"
        )
    return image_urls


class PrunableSqliteSaver(SqliteSaver):
    """SQLite checkpoint saver with optional pruning and WebUI relay tables.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection used by the saver.
    prune_checkpoints : bool, default=False
        Whether to retain only the newest checkpoint per thread and namespace.
        When enabled, older checkpoints and their writes are deleted after each
        successful checkpoint save.

    Notes
    -----
    The shared SQLite session database contains the following tables:

    ==============  ==========================================================
    Table           Purpose
    ==============  ==========================================================
    checkpoints     LangGraph checkpoint snapshots for each graph thread.
    writes          LangGraph intermediate writes associated with checkpoints.
    webui_messages  Explicit agent-to-WebUI display messages.
    webui_inputs    WebUI-to-agent user input queue.
    status          WebUI status flags such as `user_input_requested`.
    ==============  ==========================================================
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        prune_checkpoints: bool = False,
        serde: Any = None,
    ) -> None:
        """Initialize the saver.

        Parameters
        ----------
        conn : sqlite3.Connection
            Open SQLite connection used by the saver.
        prune_checkpoints : bool, default=False
            Whether to delete superseded checkpoints after each save.
        serde : Any, optional
            Optional LangGraph serializer override.
        """
        super().__init__(conn, serde=serde)
        self.prune_checkpoints = prune_checkpoints
        self.webui_tables_setup = False

    def setup(self) -> None:
        """Create checkpoint tables plus the shared WebUI relay tables."""
        if not self.is_setup:
            super().setup()
        if self.webui_tables_setup:
            return
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS webui_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                image TEXT
            );
            CREATE TABLE IF NOT EXISTS status (
                id INTEGER PRIMARY KEY,
                user_input_requested INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS webui_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO status (id, user_input_requested) VALUES (1, 0)"
        )
        self.conn.commit()
        self.webui_tables_setup = True

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict[str, Any]:
        """Store a checkpoint and prune older ones when configured."""
        saved_config = super().put(config, checkpoint, metadata, new_versions)
        if self.prune_checkpoints:
            self.prune_thread_history(
                thread_id=str(saved_config["configurable"]["thread_id"]),
                checkpoint_id=str(saved_config["configurable"]["checkpoint_id"]),
            )
        return saved_config

    def prune_thread_history(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> None:
        """Delete older checkpoints for one thread.

        Parameters
        ----------
        thread_id : str
            Thread identifier whose history should be pruned.
        checkpoint_id : str
            Checkpoint identifier to retain.
        """
        with self.cursor() as cur:
            cur.execute(
                """
                DELETE FROM writes
                WHERE thread_id = ?
                  AND checkpoint_id != ?
                """,
                (thread_id, checkpoint_id),
            )
            cur.execute(
                """
                DELETE FROM checkpoints
                WHERE thread_id = ?
                  AND checkpoint_id != ?
                """,
                (thread_id, checkpoint_id),
            )

    def load_latest_checkpoint_messages(
        self,
        since_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return WebUI-formatted messages from the newest checkpointed state.

        Parameters
        ----------
        since_id : Optional[int], optional
            Return only messages whose synthetic message id is greater than this
            value.

        Returns
        -------
        list[dict[str, Any]]
            Messages extracted from the newest checkpoint state. Synthetic ids
            are assigned from the message index in `full_history`.
        """
        latest_state, checkpoint_timestamp = self.load_latest_checkpoint_state()
        if latest_state is None:
            return []
        history = latest_state.get("full_history")
        if not isinstance(history, list) or len(history) == 0:
            history = latest_state.get("messages", [])
        messages: list[dict[str, Any]] = []
        start_id = 1 if since_id is None else since_id + 1
        for message_id, message in enumerate(history, start=1):
            if message_id < start_id or not isinstance(message, dict):
                continue
            messages.append(
                self.format_message_for_webui(
                    message=message,
                    message_id=message_id,
                    timestamp=checkpoint_timestamp,
                )
            )
        return messages

    def load_latest_checkpoint_state(
        self,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Return the newest checkpoint state that contains transcript data.

        Returns
        -------
        tuple[Optional[dict[str, Any]], Optional[str]]
            The checkpointed state dictionary and the checkpoint timestamp.
        """
        with self.cursor(transaction=False) as cur:
            try:
                cur.execute(
                    """
                    SELECT type, checkpoint
                    FROM checkpoints
                    ORDER BY checkpoint_id DESC
                    """
                )
                rows = cur.fetchall()
            except sqlite3.Error:
                return None, None

        latest_fallback_state: Optional[dict[str, Any]] = None
        latest_fallback_timestamp: Optional[str] = None
        for checkpoint_type, serialized_checkpoint in rows:
            checkpoint = self.serde.loads_typed((checkpoint_type, serialized_checkpoint))
            state = self.extract_state_from_checkpoint(checkpoint)
            if state is None:
                continue
            if latest_fallback_state is None:
                latest_fallback_state = state
                latest_fallback_timestamp = checkpoint.get("ts")
            history = state.get("full_history")
            if isinstance(history, list) and len(history) > 0:
                return state, checkpoint.get("ts")
            messages = state.get("messages")
            if isinstance(messages, list) and len(messages) > 0:
                return state, checkpoint.get("ts")
        return latest_fallback_state, latest_fallback_timestamp

    @staticmethod
    def extract_state_from_checkpoint(checkpoint: Any) -> Optional[dict[str, Any]]:
        """Extract the root graph state from a LangGraph checkpoint payload.

        Parameters
        ----------
        checkpoint : Any
            Deserialized checkpoint payload.

        Returns
        -------
        Optional[dict[str, Any]]
            Root state dictionary when available.
        """
        if not isinstance(checkpoint, dict):
            return None
        channel_values = checkpoint.get("channel_values")
        if not isinstance(channel_values, dict):
            return None
        if "full_history" in channel_values or "messages" in channel_values:
            return channel_values
        root_state = PrunableSqliteSaver.normalize_state_value(channel_values.get("__root__"))
        if root_state is not None:
            return root_state
        for value in channel_values.values():
            normalized_value = PrunableSqliteSaver.normalize_state_value(value)
            if normalized_value is not None:
                return normalized_value
        return None

    @staticmethod
    def normalize_state_value(value: Any) -> Optional[dict[str, Any]]:
        """Normalize checkpoint state values to dictionaries.

        Parameters
        ----------
        value : Any
            One deserialized channel value from the checkpoint payload.

        Returns
        -------
        Optional[dict[str, Any]]
            Dictionary representation of the state when possible.
        """
        if isinstance(value, dict):
            return value
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return None

    @classmethod
    def format_message_for_webui(
        cls,
        message: dict[str, Any],
        message_id: int,
        timestamp: Optional[str] = None,
    ) -> dict[str, Any]:
        """Convert one checkpointed message into the WebUI response shape.

        Parameters
        ----------
        message : dict[str, Any]
            OpenAI-style message payload.
        message_id : int
            Synthetic message id used by the WebUI polling API.
        timestamp : Optional[str], optional
            Timestamp associated with the checkpoint snapshot.

        Returns
        -------
        dict[str, Any]
            WebUI-formatted message payload.
        """
        elements = get_message_elements_as_text(message)
        image_urls = cls.parse_images(elements["image"])
        return {
            "id": message_id,
            "timestamp": timestamp,
            "role": elements["role"],
            "content": cls.normalize_content(elements["content"]),
            "tool_calls": elements["tool_calls"],
            "image": image_urls[0] if len(image_urls) > 0 else None,
            "images": image_urls,
        }

    @staticmethod
    def normalize_content(content: Optional[str]) -> str:
        """Strip transport-only placeholders from message text.

        Parameters
        ----------
        content : Optional[str]
            Raw message text.

        Returns
        -------
        str
            Cleaned message text.
        """
        content_text = (content or "").strip()
        if not content_text:
            return ""
        return "\n".join(
            line for line in content_text.splitlines() if line.strip() != "<image>"
        ).strip()

    @staticmethod
    def parse_images(image_value: Any) -> list[str]:
        """Parse one or multiple image values from persisted message data.

        Parameters
        ----------
        image_value : Any
            Serialized image value from a message payload.

        Returns
        -------
        list[str]
            Normalized image data URLs.
        """
        return parse_persisted_images(image_value)


class SQLiteMessageStore:
    """Adapter for the shared SQLite WebUI relay tables."""

    def __init__(self, path: Optional[str] = None):
        """Initialize the store."""
        self.path = path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open the SQLite connection and ensure the schema exists."""
        if self.path is None:
            return
        if self.connection is not None:
            return
        self.connection = sqlite3.connect(self.path)
        PrunableSqliteSaver(self.connection).setup()

    def append_message(self, message: dict[str, Any]) -> int:
        """Persist one display message for the WebUI.

        Parameters
        ----------
        message : dict[str, Any]
            OpenAI-compatible message payload.

        Returns
        -------
        int
            Inserted WebUI message row id.
        """
        if self.connection is None:
            self.connect()
        if self.connection is None:
            raise RuntimeError("The SQLite message store path is not configured.")
        elements = get_message_elements_as_text(message)
        image_value = elements["image"]
        if isinstance(image_value, list):
            image_value = json.dumps(image_value)
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO webui_messages (timestamp, role, content, tool_calls, image)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(get_timestamp(as_int=True)),
                elements["role"],
                elements["content"],
                elements["tool_calls"],
                image_value,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def load_webui_messages(self, since_id: Optional[int] = None) -> list[dict[str, Any]]:
        """Load persisted WebUI display messages.

        Parameters
        ----------
        since_id : Optional[int], optional
            Return only rows newer than this id.

        Returns
        -------
        list[dict[str, Any]]
            WebUI-formatted display messages.
        """
        if self.connection is None:
            return []
        cursor = self.connection.cursor()
        if since_id is None:
            cursor.execute(
                """
                SELECT id, timestamp, role, content, tool_calls, image
                FROM webui_messages
                ORDER BY id
                """
            )
        else:
            cursor.execute(
                """
                SELECT id, timestamp, role, content, tool_calls, image
                FROM webui_messages
                WHERE id > ?
                ORDER BY id
                """,
                (since_id,),
            )
        rows = cursor.fetchall()
        messages: list[dict[str, Any]] = []
        for message_id, timestamp, role, content, tool_calls, image in rows:
            image_urls = PrunableSqliteSaver.parse_images(image)
            messages.append(
                {
                    "id": int(message_id),
                    "timestamp": timestamp,
                    "role": role,
                    "content": PrunableSqliteSaver.normalize_content(content),
                    "tool_calls": tool_calls,
                    "image": image_urls[0] if len(image_urls) > 0 else None,
                    "images": image_urls,
                }
            )
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

    def dequeue_webui_input(self) -> Optional[str]:
        """Remove and return the next queued WebUI input.

        Returns
        -------
        Optional[str]
            The oldest queued input message, or `None` when the queue is
            empty.
        """
        if self.connection is None:
            return None
        cursor = self.connection.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            """
            SELECT id, content
            FROM webui_inputs
            ORDER BY id ASC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if row is None:
            self.connection.commit()
            return None
        cursor.execute(
            "DELETE FROM webui_inputs WHERE id = ?",
            (int(row[0]),),
        )
        self.connection.commit()
        return str(row[1])

    def enqueue_webui_input(self, content: str) -> int:
        """Insert a new WebUI-originated user message into the relay queue.

        Parameters
        ----------
        content : str
            Message text submitted from the WebUI.

        Returns
        -------
        int
            Inserted row id.
        """
        if self.connection is None:
            raise RuntimeError("The SQLite message store is not connected.")
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO webui_inputs (timestamp, content) VALUES (?, ?)",
            (str(get_timestamp(as_int=True)), content),
        )
        self.connection.commit()
        return int(cursor.lastrowid)
