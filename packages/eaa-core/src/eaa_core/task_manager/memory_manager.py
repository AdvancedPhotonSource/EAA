from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from langchain_openai import OpenAIEmbeddings
from openai import UnprocessableEntityError

from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.message_proc import extract_message_text, generate_openai_message
from eaa_core.task_manager.state import ChatRuntimeContext

T = TypeVar("T")


class MemoryManager:
    """Manage long-term memory for the chat graph."""

    def __init__(self, task_manager: Any):
        """Initialize the memory manager.

        Parameters
        ----------
        task_manager : BaseTaskManager
            Task manager that owns this memory manager.
        """
        self.task_manager = task_manager
        self.check_embedding_ctx_length = True
        self.store: Optional[Any] = None

    @property
    def config(self) -> Optional[MemoryManagerConfig]:
        """Return the active memory configuration."""
        return self.task_manager.memory_config

    def build_store(self) -> None:
        """Build the long-term memory store used by the chat graph."""
        if self.config is None or not self.config.enabled:
            self.store = None
            return
        try:
            from langchain_chroma import Chroma
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Long-term memory requires the `langchain-chroma` package. "
                "Install the `memory_chroma` extra for `eaa-core`."
            ) from error
        self.store = Chroma(
            persist_directory=self.get_persist_directory(),
            collection_name=self.config.collection_name,
            embedding_function=self.build_embeddings_client(),
            collection_metadata={"hnsw:space": "cosine"},
            relevance_score_fn=lambda distance: 1.0 - distance,
        )

    def build_embeddings_client(self) -> OpenAIEmbeddings:
        """Build the embeddings client used for long-term memory.

        Returns
        -------
        OpenAIEmbeddings
            Embeddings client used by the memory store.
        """
        memory_llm_config = None if self.config is None else self.config.llm_config
        config = memory_llm_config or self.task_manager.llm_config
        if config is None:
            raise RuntimeError("Long-term memory requires an LLM config for embeddings.")
        if isinstance(config, dict):
            base_url = config.get("base_url") or config.get("server_base_url")
            api_key = config.get("api_key")
        else:
            base_url = getattr(config, "base_url", None) or getattr(
                config,
                "server_base_url",
                None,
            )
            api_key = getattr(config, "api_key", None)
        kwargs = {
            "model": self.config.embedding_model,
            "api_key": api_key,
        }
        if not self.check_embedding_ctx_length:
            kwargs["check_embedding_ctx_length"] = False
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIEmbeddings(**kwargs)

    def should_disable_embedding_ctx_length(
        self,
        error: UnprocessableEntityError,
    ) -> bool:
        """Return whether a 422 error indicates the embeddings endpoint needs string input.

        Parameters
        ----------
        error : UnprocessableEntityError
            Error raised by the OpenAI-compatible embeddings client.

        Returns
        -------
        bool
            Whether the embeddings client should be rebuilt with
            `check_embedding_ctx_length=False`.
        """
        if not self.check_embedding_ctx_length:
            return False
        body = getattr(error, "body", None)
        if not isinstance(body, dict):
            return "Input should be a valid string" in str(error)
        details = body.get("detail")
        if not isinstance(details, list):
            return "Input should be a valid string" in str(error)
        return any(
            isinstance(detail, dict)
            and "valid string" in str(detail.get("msg", "")).lower()
            and detail.get("loc") is not None
            and "input" in [str(item).lower() for item in detail.get("loc", [])]
            for detail in details
        )

    def run_with_string_input_fallback(self, operation: Callable[[], T]) -> T:
        """Run a memory-store operation and retry once with string-only embeddings.

        Parameters
        ----------
        operation : Callable[[], T]
            Memory-store operation to execute.

        Returns
        -------
        T
            Result returned by the memory-store operation.
        """
        try:
            return operation()
        except UnprocessableEntityError as error:
            if not self.should_disable_embedding_ctx_length(error):
                raise
            self.check_embedding_ctx_length = False
            self.build_store()
            return operation()

    def get_persist_directory(self) -> str:
        """Return the persistence directory for long-term memory."""
        if self.config is not None and self.config.persist_directory:
            return self.config.persist_directory
        if self.task_manager.session_db_path:
            return str(Path(self.task_manager.session_db_path).resolve().parent / ".eaa_memory")
        return str(Path.cwd() / ".eaa_memory")

    def get_namespace(self) -> str:
        """Return the namespace used for long-term memory."""
        if self.config is not None and self.config.namespace:
            return self.config.namespace
        if self.task_manager.session_db_path:
            return Path(self.task_manager.session_db_path).stem
        return self.task_manager.__class__.__name__

    def get_runtime_context(self) -> ChatRuntimeContext:
        """Return the runtime context used when invoking the chat graph."""
        return ChatRuntimeContext(
            memory_namespace=self.get_namespace(),
            memory_store=self.store,
        )

    def user_message_triggers_memory(self, message: Optional[dict[str, Any]]) -> bool:
        """Return whether a user message should be saved to long-term memory."""
        if (
            message is None
            or self.config is None
            or not self.config.enabled
            or not self.config.save_enabled
            or message.get("role") != "user"
        ):
            return False
        text = extract_message_text(message).lower()
        return any(trigger in text for trigger in self.config.trigger_phrases)

    def get_memory_text(self, message: dict[str, Any]) -> Optional[str]:
        """Extract the text payload that should be saved as long-term memory."""
        text = extract_message_text(message).strip()
        if not text:
            return None
        lower_text = text.lower()
        if self.config is None:
            return text
        for trigger in self.config.trigger_phrases:
            index = lower_text.find(trigger)
            if index < 0:
                continue
            suffix = text[index + len(trigger) :].lstrip(" :,-")
            return suffix or text
        return text

    def save_user_memory(self, message: dict[str, Any], namespace: str) -> None:
        """Persist a user memory in the vector store."""
        if self.store is None:
            return
        memory_text = self.get_memory_text(message)
        if not memory_text:
            return
        key = hashlib.sha1(memory_text.encode("utf-8")).hexdigest()
        self.run_with_string_input_fallback(
            lambda: self.store.add_texts(
                [memory_text],
                metadatas=[
                    {
                        "namespace": namespace,
                        "kind": "user_memory",
                        "source_message": extract_message_text(message),
                    }
                ],
                ids=[key],
            )
        )

    def retrieve_user_memories(self, message: dict[str, Any], namespace: str) -> list[Any]:
        """Retrieve semantically relevant long-term memories for a user message."""
        if (
            self.store is None
            or self.config is None
            or not self.config.enabled
            or not self.config.retrieval_enabled
        ):
            return []
        query = extract_message_text(message).strip()
        if not query:
            return []
        results = self.run_with_string_input_fallback(
            lambda: self.store.similarity_search_with_relevance_scores(
                query,
                k=self.config.top_k,
                filter={
                    "$and": [
                        {"namespace": namespace},
                        {"kind": "user_memory"},
                    ]
                },
            )
        )
        filtered = []
        for document, score in results:
            if score < self.config.score_threshold:
                continue
            filtered.append((document, score))
        return filtered

    def build_memory_context_message(self, memory_results: list[Any]) -> Optional[dict[str, Any]]:
        """Build a system message containing retrieved long-term memories."""
        if not memory_results:
            return None
        lines = ["Relevant long-term memory:"]
        for index, (document, _score) in enumerate(memory_results, start=1):
            text = document.page_content
            if isinstance(text, str) and text.strip():
                lines.append(f"{index}. {text.strip()}")
        if len(lines) == 1:
            return None
        return generate_openai_message("\n".join(lines), role="system")

    def inject_memory_context(
        self,
        context: list[dict[str, Any]],
        memory_message: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert retrieved memory context before the latest user message."""
        if memory_message is None:
            return context
        augmented = list(context)
        if augmented and augmented[-1].get("role") == "user":
            augmented.insert(len(augmented) - 1, memory_message)
        else:
            augmented.append(memory_message)
        return augmented
