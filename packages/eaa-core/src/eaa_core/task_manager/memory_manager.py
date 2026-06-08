from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from langchain_openai import OpenAIEmbeddings
from openai import UnprocessableEntityError

from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.message_proc import extract_message_text, generate_openai_message
from eaa_core.llm.model import invoke_chat_model
from eaa_core.task_manager.state import ChatRuntimeContext
from eaa_core.util import get_image_paths_from_text

T = TypeVar("T")
logger = logging.getLogger(__name__)


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
            logger.debug("Long-term memory store disabled.")
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
        logger.info(
            "Built long-term memory store: collection=%s namespace=%s persist_directory=%s",
            self.config.collection_name,
            self.get_namespace(),
            self.get_persist_directory(),
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
        if self.task_manager.checkpoint_db_path:
            return str(Path(self.task_manager.checkpoint_db_path).resolve().parent / ".eaa_memory")
        return str(Path.cwd() / ".eaa_memory")

    def get_namespace(self) -> str:
        """Return the namespace used for long-term memory."""
        if self.config is not None and self.config.namespace:
            return self.config.namespace
        if self.task_manager.checkpoint_db_path:
            return Path(self.task_manager.checkpoint_db_path).stem
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
            return self.remove_image_tags_from_text(suffix or text)
        return self.remove_image_tags_from_text(text)

    def remove_image_tags_from_text(self, text: str) -> str:
        """Return text with EAA ``<img ...>`` tags removed.

        Parameters
        ----------
        text : str
            Text that may include EAA image tags.

        Returns
        -------
        str
            Text with image tags removed and surrounding whitespace stripped.
        """
        _, cleaned_text = get_image_paths_from_text(text, return_text_without_image_tag=True)
        return cleaned_text.strip()

    def get_message_image_references(self, message: dict[str, Any]) -> list[dict[str, str]]:
        """Extract image references from an OpenAI-style message.

        Parameters
        ----------
        message : dict
            Message payload to inspect.

        Returns
        -------
        list[dict[str, str]]
            Image references as dictionaries with ``kind`` and ``value`` keys.
        """
        references: list[dict[str, str]] = []
        content = message.get("content")
        if isinstance(content, str):
            references.extend({"kind": "path", "value": path} for path in get_image_paths_from_text(content))
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    references.extend(
                        {"kind": "path", "value": path}
                        for path in get_image_paths_from_text(str(part.get("text", "")))
                    )
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url")
                    if isinstance(image_url, str) and image_url:
                        references.append({"kind": "image_url", "value": image_url})
        return references

    def caption_image_reference(self, image_reference: dict[str, str]) -> str:
        """Caption one image through the task manager's main chat model.

        Parameters
        ----------
        image_reference : dict[str, str]
            Image reference with ``kind`` set to ``path`` or ``image_url``.

        Returns
        -------
        str
            Concise model-generated image caption.
        """
        if self.task_manager.model is None:
            raise RuntimeError("Image memory captioning requires the main LLM model to be configured.")
        prompt = (
            "Describe this image for semantic retrieval memory. "
            "Focus on visible objects, scene structure, text, plots, measurements, "
            "and scientifically relevant details. Be concise."
        )
        if image_reference["kind"] == "path":
            message = generate_openai_message(
                content=prompt,
                role="user",
                image_path=image_reference["value"],
            )
        elif image_reference["kind"] == "image_url":
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_reference["value"]},
                    },
                ],
            }
        else:
            raise ValueError(f"Unsupported image reference kind: {image_reference['kind']}")
        response = invoke_chat_model(self.task_manager.model, messages=[message])
        caption = extract_message_text(response).strip()
        logger.debug("Captioned memory image reference kind=%s caption=%r", image_reference["kind"], caption)
        return caption

    def build_text_with_image_captions(self, text: str, message: dict[str, Any]) -> str:
        """Append LLM-generated image captions to text for embedding.

        Parameters
        ----------
        text : str
            Textual message content.
        message : dict
            Message payload that may include image references.

        Returns
        -------
        str
            Text suitable for a text embedding model.
        """
        image_references = self.get_message_image_references(message)
        if len(image_references) == 0:
            return text
        logger.info("Captioning %d image(s) for long-term memory embedding.", len(image_references))
        captions = [
            caption
            for caption in (self.caption_image_reference(reference) for reference in image_references)
            if caption
        ]
        if len(captions) == 0:
            return text
        sections = [text.strip()] if text.strip() else []
        sections.append("Image descriptions:")
        sections.extend(f"{index}. {caption}" for index, caption in enumerate(captions, start=1))
        return "\n".join(sections)

    def get_memory_embedding_text(self, message: dict[str, Any]) -> Optional[str]:
        """Build text to embed for a saved memory message.

        Parameters
        ----------
        message : dict
            User message being saved.

        Returns
        -------
        str or None
            Text and image captions to embed, or ``None`` when empty.
        """
        text = self.get_memory_text(message) or ""
        embedding_text = self.build_text_with_image_captions(text, message).strip()
        return embedding_text or None

    def get_memory_query_text(self, message: dict[str, Any]) -> str:
        """Build text to embed for memory retrieval.

        Parameters
        ----------
        message : dict
            User query message.

        Returns
        -------
        str
            Query text augmented with image captions when images are present.
        """
        text = self.remove_image_tags_from_text(extract_message_text(message))
        return self.build_text_with_image_captions(text, message).strip()

    def save_user_memory(self, message: dict[str, Any], namespace: str) -> None:
        """Persist a user memory in the vector store."""
        if self.store is None:
            logger.debug("Skipping long-term memory save because memory store is not available.")
            return
        memory_text = self.get_memory_embedding_text(message)
        if not memory_text:
            logger.debug("Skipping long-term memory save because extracted memory text is empty.")
            return
        image_count = len(self.get_message_image_references(message))
        key = hashlib.sha1(memory_text.encode("utf-8")).hexdigest()
        logger.info(
            "Saving long-term memory: namespace=%s id=%s text_length=%d image_count=%d",
            namespace,
            key,
            len(memory_text),
            image_count,
        )
        logger.debug("Long-term memory text preview: %r", memory_text[:500])
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
        logger.info("Saved long-term memory: namespace=%s id=%s", namespace, key)

    def save_triggered_user_memory(
        self,
        message: Optional[dict[str, Any]],
        runtime_context: Optional[ChatRuntimeContext],
    ) -> None:
        """Persist a user memory when runtime memory and trigger conditions match.

        Parameters
        ----------
        message : dict, optional
            Candidate OpenAI-style user message.
        runtime_context : ChatRuntimeContext, optional
            Active graph runtime context containing the memory store and
            namespace.
        """
        if (
            runtime_context is None
            or runtime_context.memory_store is None
            or not self.user_message_triggers_memory(message)
        ):
            logger.debug(
                "Skipping triggered long-term memory save: has_runtime_context=%s has_store=%s triggered=%s",
                runtime_context is not None,
                runtime_context is not None and runtime_context.memory_store is not None,
                self.user_message_triggers_memory(message),
            )
            return
        self.save_user_memory(message, runtime_context.memory_namespace)

    def retrieve_user_memories(self, message: dict[str, Any], namespace: str) -> list[Any]:
        """Retrieve semantically relevant long-term memories for a user message."""
        if (
            self.store is None
            or self.config is None
            or not self.config.enabled
            or not self.config.retrieval_enabled
        ):
            logger.debug(
                "Skipping long-term memory retrieval: has_store=%s has_config=%s enabled=%s retrieval_enabled=%s",
                self.store is not None,
                self.config is not None,
                self.config is not None and self.config.enabled,
                self.config is not None and self.config.retrieval_enabled,
            )
            return []
        query = self.get_memory_query_text(message)
        if not query:
            logger.debug("Skipping long-term memory retrieval because query text is empty.")
            return []
        image_count = len(self.get_message_image_references(message))
        logger.info(
            "Retrieving long-term memories: namespace=%s top_k=%d threshold=%s query_length=%d image_count=%d",
            namespace,
            self.config.top_k,
            self.config.score_threshold,
            len(query),
            image_count,
        )
        logger.debug("Long-term memory query preview: %r", query[:500])
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
        logger.info(
            "Retrieved %d long-term memor%s after filtering from %d candidate(s).",
            len(filtered),
            "y" if len(filtered) == 1 else "ies",
            len(results),
        )
        if filtered:
            logger.debug(
                "Long-term memory result scores: %s",
                [round(float(score), 4) for _document, score in filtered],
            )
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
