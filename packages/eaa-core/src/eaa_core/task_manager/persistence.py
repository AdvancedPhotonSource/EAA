from typing import Any, Optional
import ast
import json
import sqlite3

from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.sqlite import SqliteSaver

from eaa_core.message_proc import get_message_elements_as_text
from eaa_core.util import get_timestamp


def configure_sqlite_connection(connection: sqlite3.Connection) -> sqlite3.Connection:
    """Apply shared SQLite settings for transcript/checkpoint databases.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open SQLite connection to configure.

    Returns
    -------
    sqlite3.Connection
        The configured connection.
    """
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def parse_persisted_images(image_value: Any) -> list[str]:
    """Normalize persisted transcript image values to data URLs.

    Parameters
    ----------
    image_value : Any
        Serialized image payload read from transcript persistence.

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


def normalize_message_content(content: Optional[str]) -> str:
    """Strip transport-only placeholders from message text."""
    content_text = (content or "").strip()
    if not content_text:
        return ""
    return "\n".join(
        line for line in content_text.splitlines() if line.strip() != "<image>"
    ).strip()


class PrunableSqliteSaver(SqliteSaver):
    """SQLite checkpoint saver with optional pruning.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection used by the saver.
    prune_checkpoints : bool, default=False
        Whether to retain only the newest checkpoint per thread and namespace.
        When enabled, older checkpoints and their writes are deleted after each
        successful checkpoint save.

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

    def setup(self) -> None:
        """Create checkpoint tables."""
        if not self.is_setup:
            super().setup()

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


class SQLiteTranscriptStore:
    """SQLite store for durable transcript messages."""

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
        self.connection = configure_sqlite_connection(
            sqlite3.connect(self.path, check_same_thread=False)
        )
        self.setup()

    def setup(self) -> None:
        """Create transcript tables."""
        if self.connection is None:
            return
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS transcript_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                image TEXT
            );
            """
        )
        self.connection.commit()

    def append_message(self, message: dict[str, Any]) -> int:
        """Persist one transcript message.

        Parameters
        ----------
        message : dict[str, Any]
            OpenAI-compatible message payload.

        Returns
        -------
        int
            Inserted transcript message row id.
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
            INSERT INTO transcript_messages (timestamp, role, content, tool_calls, image)
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

    def load_messages(self, since_id: Optional[int] = None) -> list[dict[str, Any]]:
        """Load persisted transcript messages.

        Parameters
        ----------
        since_id : Optional[int], optional
            Return only rows newer than this id.

        Returns
        -------
        list[dict[str, Any]]
            Transcript messages in display-friendly form.
        """
        if self.connection is None:
            self.connect()
        if self.connection is None:
            return []
        cursor = self.connection.cursor()
        if since_id is None:
            cursor.execute(
                """
                SELECT id, timestamp, role, content, tool_calls, image
                FROM transcript_messages
                ORDER BY id
                """
            )
        else:
            cursor.execute(
                """
                SELECT id, timestamp, role, content, tool_calls, image
                FROM transcript_messages
                WHERE id > ?
                ORDER BY id
                """,
                (since_id,),
            )
        rows = cursor.fetchall()
        messages: list[dict[str, Any]] = []
        for message_id, timestamp, role, content, tool_calls, image in rows:
            image_urls = parse_persisted_images(image)
            messages.append(
                {
                    "id": int(message_id),
                    "timestamp": timestamp,
                    "role": role,
                    "content": normalize_message_content(content),
                    "tool_calls": tool_calls,
                    "image": image_urls[0] if len(image_urls) > 0 else None,
                    "images": image_urls,
                }
            )
        return messages
