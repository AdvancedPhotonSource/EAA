from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence
import logging

from .types import MemoryQueryResult, MemoryRecord, MemorySchema
from .vector_store import ChromaVectorStore, LocalVectorStore, VectorStore

logger = logging.getLogger(__name__)

EmbedderFn = Callable[[Sequence[str]], List[List[float]]]
FilterFn = Callable[[str, Dict[str, Any]], bool]
FormatterFn = Callable[[List[MemoryQueryResult]], str]


@dataclass
class MemoryManagerConfig(MemorySchema):
    enabled: bool = False
    write_enabled: bool = True
    retrieval_enabled: bool = True
    top_k: int = 5
    score_threshold: float = 0.25
    min_content_length: int = 12
    embedding_model: Optional[str] = None
    vector_store_path: Optional[str] = None
    injection_role: str = "system"


class MemoryManager:
    """Orchestrates storing and retrieving conversation memories for an agent.

    The manager wraps an embedder function and a vector-store backend so it can
    decide when to persist snippets (via :meth:`remember`) and fetch relevant
    context (:meth:`recall`). Implementations remain pluggable: pass in a custom
    embedder, a different vector store, or a notability filter to control what
    gets saved. This class coordinates those pieces and exposes a minimal API
    that agents can call during a conversation turn.
    """

    def __init__(
        self,
        embedder: EmbedderFn,
        *,
        config: Optional[MemoryManagerConfig] = None,
        vector_store: Optional[VectorStore] = None,
        notability_filter: Optional[FilterFn] = None,
        formatter: Optional[FormatterFn] = None,
    ) -> None:
        """Initialise a memory manager with the supplied components.

        Parameters
        ----------
        embedder : Callable[[Sequence[str]], List[List[float]]]
            Function that converts text snippets into numeric embeddings.
        config : MemoryManagerConfig, optional
            Runtime configuration controlling toggles, thresholds, and defaults.
        vector_store : VectorStore, optional
            Backend used to persist and query memories; defaults to a
            Chroma-backed store when available (falling back to the JSON-backed
            local store otherwise).
        notability_filter : Callable[[str, Dict[str, Any]], bool], optional
            Predicate that can veto storage even if heuristics would otherwise
            save the snippet.
        formatter : Callable[[List[MemoryQueryResult]], str], optional
            Converts recall results into text injected back into the agent
            prompt; falls back to a bullet-list formatter.
        """
        self.embedder = embedder
        self.config = config or MemoryManagerConfig()
        self.vector_store = vector_store or self._create_default_store()
        self.notability_filter = notability_filter
        self.formatter = formatter or self._default_formatter

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def write_enabled(self) -> bool:
        return self.config.write_enabled and self.enabled

    @property
    def retrieval_enabled(self) -> bool:
        return self.config.retrieval_enabled and self.enabled

    def set_enabled(self, value: bool) -> None:
        self.config.enabled = value

    def set_write_enabled(self, value: bool) -> None:
        self.config.write_enabled = value

    def set_retrieval_enabled(self, value: bool) -> None:
        self.config.retrieval_enabled = value

    def remember(self, content: str, *, metadata: Optional[Dict[str, Any]] = None) -> Optional[MemoryRecord]:
        if not self.write_enabled:
            return None
        if not content:
            return None
        if metadata is None:
            metadata = {}
        trimmed = content.strip()
        if len(trimmed) < self.config.min_content_length and not metadata.get("force_store"):
            return None
        if self.notability_filter is not None and not self.notability_filter(content, metadata):
            return None
        try:
            embedding = self.embedder([content])[0]
        except Exception as exc:  # pragma: no cover - dependent on runtime env.
            logger.warning("Failed to embed memory snippet: %s", exc)
            return None
        record = MemoryRecord(content=content, embedding=embedding, metadata=metadata)
        self.vector_store.add([record])
        return record

    def recall(self, prompt: str) -> List[MemoryQueryResult]:
        if not self.retrieval_enabled:
            return []
        if not prompt:
            return []
        try:
            embedding = self.embedder([prompt])[0]
        except Exception as exc:  # pragma: no cover - dependent on runtime env.
            logger.warning("Failed to embed recall prompt: %s", exc)
            return []
        return self.vector_store.search(
            embedding,
            top_k=self.config.top_k,
            score_threshold=self.config.score_threshold,
        )

    def format_results(self, results: List[MemoryQueryResult]) -> Optional[str]:
        if not results:
            return None
        return self.formatter(results)

    def flush(self) -> None:
        try:
            self.vector_store.persist()
        except AttributeError:
            logger.debug("Vector store does not implement `persist`; skipping flush.")

    def _create_default_store(self) -> VectorStore:
        persist_path = self.config.vector_store_path
        try:
            return ChromaVectorStore(persist_path)
        except ImportError:
            logger.debug("chromadb not available; falling back to LocalVectorStore.")
        except Exception as exc:  # pragma: no cover - defensive fallback.
            logger.warning(
                "Failed to initialise ChromaVectorStore, using LocalVectorStore instead: %s",
                exc,
            )
        return LocalVectorStore(persist_path)

    @staticmethod
    def _default_formatter(results: List[MemoryQueryResult]) -> str:
        parts = [
            "Relevant stored context (most similar first):"
        ]
        for item in results:
            content = item.record.content.strip().replace("\n", " ")
            score = f"{item.score:.2f}"
            note = item.record.metadata.get("note") if item.record.metadata else None
            if note:
                parts.append(f"- ({score}) {note}: {content}")
            else:
                parts.append(f"- ({score}) {content}")
        return "\n".join(parts)
