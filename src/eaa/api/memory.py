from dataclasses import dataclass
from typing import Any, Optional
from typing import TYPE_CHECKING

from eaa.api.base import BaseConfig

if TYPE_CHECKING:
    from eaa.api.llm_config import LLMConfig


@dataclass
class MemoryManagerConfig(BaseConfig):
    """Configuration for chat-graph long-term memory."""

    enabled: bool = False
    """Whether long-term memory is enabled for the chat graph."""

    save_enabled: bool = True
    """Whether keyword-triggered user memories should be persisted."""

    retrieval_enabled: bool = True
    """Whether semantic retrieval should run on user chat turns."""

    top_k: int = 5
    """Maximum number of memory candidates to retrieve per query."""

    score_threshold: float = 0.25
    """Minimum relevance score required for a retrieved memory to be injected."""

    embedding_model: str = "text-embedding-3-small"
    """Embedding model used by the Chroma vector store.

    This field always controls the embedding model name, even when
    `llm_config` is provided.
    """

    llm_config: Optional["LLMConfig | dict[str, Any]"] = None
    """Optional LLM config override for memory embeddings only.

    When provided, the base URL and API key from this config override the
    task manager's main `llm_config` for long-term-memory embedding calls.
    The embedding model name is still read from `embedding_model`, not from
    `llm_config.model`.
    """

    persist_directory: Optional[str] = None
    """Directory where the Chroma collection is persisted."""

    collection_name: str = "eaa_long_term_memory"
    """Chroma collection name used for long-term memory entries."""

    namespace: Optional[str] = None
    """Optional logical namespace used to isolate memories across task-manager sessions."""

    trigger_phrases: tuple[str, ...] = (
        "remember this",
        "remember that",
        "note that",
        "keep in mind",
        "please remember",
        "remember:",
    )
    """User phrases that trigger long-term memory saving."""
