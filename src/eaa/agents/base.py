"""All routines in this module assume OpenAI-compatible API is used.
Subclasses of BaseAgent may implement message sending and receiving
with different APIs (for example, AskSage), but these methods should
still use OpenAI-compatible message dictionaries as input and/or output.
Format conversions should be done immediately before sending or after 
receiving. The message dictionaries passed between most routines should 
still be in OpenAI-compatible format.

This docstring illustrates the JSON formats of messages of various roles:

# Outgoing messages

## User, system (pure text)

```json
{
    "role": "user",
    "content": "Text content of the message."
}
```

## User (with image)

```json
{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Text content of the message."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,..."
            }
        }
    ]
}
```

## Tool

```json
{
    "role": "tool",
    "content": "Tool response.",
    "tool_call_id": "tool_call_id"
}
```

# Incoming messages (from AI)

The direct output of the OpenAI API is a JSON object with a `choices` key, 
which contains a list of `message` objects. We assume there is only one choice.

## Text response

```json
{
    "role": "assistant",
    "content": "Text content of the response."
}
```

## Tool call

```json
{
    "role": "assistant",
    "tool_calls": [
        {
            "id": "tool_call_id",
            "type": "function",
            "function": {
                "name": "function_name",
                "arguments": "{\"argument_name\": \"argument_value\", ...}"
            }
        }
    ]
}
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Optional,
    Literal,
    Sequence,
)
from dataclasses import dataclass
import json
import logging
import asyncio

import numpy as np
from openai.types.chat import ChatCompletionMessage

from eaa.tools.base import BaseTool, ToolReturnType, ExposedToolSpec, generate_openai_tool_schema
from eaa.tools.mcp import MCPTool
from eaa.comms import get_api_key
from eaa.util import encode_image_base64, get_image_path_from_text
from eaa.agents.memory import MemoryManager, MemoryManagerConfig, MemoryQueryResult

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    name: str
    parent: BaseTool
    call: Callable[..., Any]
    return_type: ToolReturnType
    schema: Dict[str, Any]
    require_approval: bool


class ToolManager:

    def __init__(self):
        self._tool_entries: Dict[str, ToolEntry] = {}
        self._base_tools: List[BaseTool] = []
        self.approval_handler: Optional[Callable[[str, Dict[str, Any]], bool]] = None

    def get_all_schema(self) -> List[Dict[str, Any]]:
        """Get the schema for all registered tools."""
        return [entry.schema for entry in self._tool_entries.values()]

    def set_approval_handler(self, handler: Optional[Callable[[str, Dict[str, Any]], bool]]) -> None:
        """Register a callback used to request user approval before running tools."""
        self.approval_handler = handler

    def add_tool(self, tool: BaseTool) -> None:
        """Add a BaseTool (either function tool or MCP tool) to the tool manager.
        
        Parameters
        ----------
        tool : BaseTool
            The tool to be added to the tool manager.
        """
        if not isinstance(tool, BaseTool):
            raise ValueError("tool must inherit from BaseTool")

        self._base_tools.append(tool)

        if isinstance(tool, MCPTool):
            self._register_mcp_tool(tool)
        else:
            self._register_function_tool(tool)

    def execute_tool(
        self,
        tool_name: str,
        tool_kwargs: Dict[str, Any]
    ) -> Any:
        """Execute a tool with the provided arguments."""
        entry = self._get_entry(tool_name)

        if entry.require_approval:
            if self.approval_handler is None:
                raise PermissionError(
                    f"Tool '{tool_name}' requires approval but no approval handler is configured."
                )
            if not self.approval_handler(tool_name, tool_kwargs):
                raise PermissionError(f"Tool '{tool_name}' execution denied by user.")

        return entry.call(**tool_kwargs)

    def get_tool(self, tool_name: str) -> ToolEntry:
        """Get the tool entry for a given tool name."""
        return self._get_entry(tool_name)

    def get_tool_return_type(self, tool_name: str) -> ToolReturnType:
        """Get the return type of a tool."""
        return self._get_entry(tool_name).return_type

    def get_tool_callable(self, tool_name: str) -> Callable[..., Any]:
        """Get the callable function for a given tool name."""
        return self._get_entry(tool_name).call

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the schema for a given tool name."""
        return self._get_entry(tool_name).schema

    def _register_function_tool(self, tool: BaseTool) -> None:
        for exposed in tool.exposed_tools:
            if not isinstance(exposed, ExposedToolSpec):
                raise TypeError(
                    "Items in `exposed_tools` must be ExposedToolSpec instances."
                )
            name = exposed.name
            if name in self._tool_entries:
                raise ValueError(
                    f"Tool '{name}' is already registered. Ensure tool names are unique."
                )

            call = exposed.function
            return_type = exposed.return_type
            schema = generate_openai_tool_schema(name, call)
            require_approval = (
                exposed.require_approval
                if exposed.require_approval is not None
                else tool.require_approval
            )

            self._tool_entries[name] = ToolEntry(
                name=name,
                parent=tool,
                call=call,
                return_type=return_type,
                schema=schema,
                require_approval=require_approval,
            )

    def _register_mcp_tool(self, tool: MCPTool) -> None:
        schemas = {schema["function"]["name"]: schema for schema in tool.get_all_schema()}
        for name in tool.get_all_tool_names():
            if name in self._tool_entries:
                raise ValueError(
                    f"Tool '{name}' is already registered. Ensure tool names are unique."
                )

            schema = schemas.get(name)
            if schema is None:
                raise ValueError(
                    f"Schema for MCP tool '{name}' could not be located."
                )

            call = self._make_mcp_call(tool, name)
            require_approval = getattr(tool, "require_approval", False)

            self._tool_entries[name] = ToolEntry(
                name=name,
                parent=tool,
                call=call,
                return_type=ToolReturnType.TEXT,
                schema=schema,
                require_approval=require_approval,
            )

    @staticmethod
    def _make_mcp_call(tool: MCPTool, tool_name: str) -> Callable[..., Any]:
        def runner(**kwargs: Any) -> Any:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(tool.call_tool(tool_name, kwargs))

        return runner

    def _get_entry(self, tool_name: str) -> ToolEntry:
        if tool_name not in self._tool_entries:
            raise ValueError(f"Tool {tool_name} not found.")
        return self._tool_entries[tool_name]
    
    
class BaseAgent:
    _MEMORY_KEYWORDS: List[Tuple[str, str]] = [
        ("remember", "User asked agent to remember"),
        ("note that", "User highlighted a note"),
        ("call me", "Preferred form of address"),
        ("my name is", "User shared their name"),
        ("i prefer", "User shared a preference"),
        ("i like", "User shared a preference"),
        ("store this", "User requested storage"),
        ("keep in mind", "User asked agent to keep context"),
    ]

    def __init__(
        self,
        llm_config: dict,
        system_message: str = "",
        memory_config: Optional[Dict[str, Any] | MemoryManagerConfig] = None,
    ) -> None:
        """The base agent class.

        Parameters
        ----------
        llm_config : dict
            Configuration for the OpenAI-compatible API. It should be a dictionary with
            the following keys:
            - `model`: The name of the model.
            - `api_key`: The API key for the OpenAI-compatible API.
            - `base_url`: The base URL for the OpenAI-compatible API.
        system_message : str, optional
            The system message for the OpenAI-compatible API.
        memory_config : dict | MemoryManagerConfig, optional
            Configuration for long-term memory. When provided, the agent can
            optionally store conversation snippets and retrieve them on future
            turns using retrieval-augmented generation. Pass a
            `MemoryManagerConfig` instance when you only need to adjust the
            dataclass fields. Supply a dictionary when you also want to include
            auxiliary hooks such as a custom vector store, notability filter,
            formatter, or embedder override. The dictionary may contain the
            config fields (`enabled`, `write_enabled`, `retrieval_enabled`,
            `top_k`, `score_threshold`, `min_content_length`, `embedding_model`,
            `vector_store_path`, `injection_role`) plus optional extras
            (`vector_store`, `notability_filter`, `formatter`, `embedder`).
        """
        self.llm_config = llm_config
        
        self.message_hooks = []
        
        self.system_messages = [
            {"role": "system", "content": system_message}
        ]
        
        self.tool_manager = ToolManager()
        self._default_approval_handler = self.request_tool_approval
        self._approval_handler = self._default_approval_handler
        self.tool_manager.set_approval_handler(self._approval_handler)

        self.client = self.create_client()

        self.memory_manager: Optional[MemoryManager] = None
        self._memory_injection_role = "system"
        self._initialize_memory(memory_config)
        
    @property
    def model(self) -> str:
        return self.llm_config.get("model")
    
    @property
    def base_url(self) -> str:
        if "base_url" in self.llm_config.keys():
            return self.llm_config["base_url"]
        elif "server_base_url" in self.llm_config.keys():
            return self.llm_config["server_base_url"]
        else:
            raise ValueError(
                "Unable to infer the base URL of the LLM. "
                "Please provide the base URL in the LLM configuration."
            )
    
    @property
    def api_key(self) -> str:
        api_key = self.llm_config.get("api_key")
        if api_key is None:
            logger.warning(
                "`api_key` is not set in the LLM configuration. "
                "Attempting to infer it from environment variables..."
            )
            api_key = get_api_key(
                model_name=self.model,
                model_base_url=self.base_url
            )
        return api_key
    
        
    def create_client(self) -> Any:
        raise NotImplementedError

    # --- Memory lifecycle -------------------------------------------------

    def supports_memory_embeddings(self) -> bool:
        """Indicate whether this agent can produce embeddings for memory."""
        return False

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement text embeddings."
        )

    def get_default_embedding_model(self) -> Optional[str]:
        return self.llm_config.get("embedding_model")

    def _initialize_memory(
        self,
        memory_config: Optional[Dict[str, Any] | MemoryManagerConfig],
    ) -> None:
        if memory_config is None:
            return

        config_obj: MemoryManagerConfig
        vector_store = None
        notability_filter = None
        formatter = None
        embedder_override = None

        if isinstance(memory_config, MemoryManagerConfig):
            config_obj = memory_config
        elif isinstance(memory_config, dict):
            config_obj = MemoryManagerConfig.from_dict(memory_config)
            vector_store = memory_config.get("vector_store")
            notability_filter = memory_config.get("notability_filter")
            formatter = memory_config.get("formatter")
            embedder_override = memory_config.get("embedder")
        else:
            raise ValueError("memory_config must be a dict or MemoryManagerConfig instance.")

        if config_obj.injection_role not in {"system", "user"}:
            raise ValueError(
                "Unsupported injection role '%s'; defaulting to 'system'.",
                config_obj.injection_role,
            )

        self._memory_injection_role = config_obj.injection_role

        if embedder_override is None and not self.supports_memory_embeddings():
            logger.warning(
                "%s does not support embeddings; ignoring memory configuration.",
                self.__class__.__name__,
            )
            return

        embedding_model = config_obj.embedding_model or self.get_default_embedding_model()
        embedder_fn = embedder_override or self._build_memory_embedder(embedding_model)

        try:
            self.memory_manager = MemoryManager(
                embedder_fn,
                config=config_obj,
                vector_store=vector_store,
                notability_filter=notability_filter,
                formatter=formatter,
            )
        except Exception as exc:  # pragma: no cover - defensive.
            logger.warning("Failed to initialise memory manager: %s", exc)
            self.memory_manager = None

    def _build_memory_embedder(self, embedding_model: Optional[str]) -> Callable[[Sequence[str]], List[List[float]]]:
        model_name = embedding_model

        def embedder(texts: Sequence[str]) -> List[List[float]]:
            selected_model = model_name or self.get_default_embedding_model()
            if selected_model is None:
                raise ValueError("No embedding model provided for memory usage.")
            return self.embed_texts(texts, model=selected_model)

        return embedder

    def _extract_message_text(self, message: Optional[Dict[str, Any]]) -> str:
        if message is None:
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts)
        return ""

    def _post_response_memory_update(
        self,
        *,
        user_message: Dict[str, Any],
        agent_response: Optional[Dict[str, Any]],
        recall_results: List[MemoryQueryResult],
    ) -> None:
        """Capture notable user content after a response has been produced.

        The method inspects the user message, determines whether it should be
        stored, and forwards the snippet to the memory manager along with
        metadata about the interaction (message id, prior retrieval count, etc.).

        Parameters
        ----------
        user_message : Dict[str, Any]
            The outgoing OpenAI-formatted user message for the current turn.
        agent_response : Optional[Dict[str, Any]]
            Processed agent reply, used to detect tool-call signals.
        recall_results : List[MemoryQueryResult]
            Memories that were retrieved for this turn; logged for metadata and
            potential deduplication.
        """
        if self.memory_manager is None or not self.memory_manager.write_enabled:
            return

        message_text = self._extract_message_text(user_message)
        if not message_text:
            return

        notable, note_reason = self._is_notable_message(message_text, agent_response)
        if not notable:
            return

        metadata: Dict[str, Any] = {
            "role": "user",
            "note": note_reason,
        }
        reference_id = user_message.get("id")
        if reference_id is not None:
            metadata["message_id"] = reference_id
        if recall_results:
            metadata["retrieved_count"] = len(recall_results)
        if len(message_text.strip()) < self.memory_manager.config.min_content_length:
            metadata["force_store"] = True

        self.memory_manager.remember(message_text, metadata=metadata)

    def _is_notable_message(
        self,
        content: str,
        response: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Determine whether a user message should be persisted in memory.

        A message is considered notable if any of the following hold:
        - It contains one of the heuristic keywords such as "remember", "my name is",
          or "keep in mind".
        - It is a long declarative statement (more than 140 characters terminating with
          punctuation).
        - The paired response triggered a `memorize` tool call.

        Parameters
        ----------
        content : str
            User-facing text content to inspect.
        response : Optional[Dict[str, Any]]
            Processed agent response, used to detect memory-specific tool calls.

        Returns
        -------
        Tuple[bool, str]
            Whether the message is notable and a short reason string describing
            the path that triggered the decision.
        """
        lowered = content.lower()
        for keyword, reason in self._MEMORY_KEYWORDS:
            if keyword in lowered:
                return True, reason

        # Treat long declarative statements as potentially notable.
        if len(content.strip()) > 140 and content.strip().endswith(('.', '!', '?')):
            return True, "Long-form user statement"

        if response is not None:
            tool_calls = response.get("tool_calls") if isinstance(response, dict) else None
            if tool_calls:
                for call in tool_calls:
                    if call.get("function", {}).get("name") == "memorize":
                        return True, "User triggered memory tool"

        return False, ""

    def configure_memory(
        self,
        *,
        enabled: Optional[bool] = None,
        write_enabled: Optional[bool] = None,
        retrieval_enabled: Optional[bool] = None,
    ) -> None:
        if self.memory_manager is None:
            raise ValueError("Memory manager is not initialised for this agent.")
        if enabled is not None:
            self.memory_manager.set_enabled(enabled)
        if write_enabled is not None:
            self.memory_manager.set_write_enabled(write_enabled)
        if retrieval_enabled is not None:
            self.memory_manager.set_retrieval_enabled(retrieval_enabled)

    def flush_memory(self) -> None:
        if self.memory_manager is None:
            return
        self.memory_manager.flush()
        
    def register_tools(self, tools: List[BaseTool]) -> None:
        """Register BaseTool instances with the agent."""
        if not isinstance(tools, List):
            raise ValueError("tools must be a list of BaseTool instances.")

        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError("All items in tools must inherit from BaseTool.")
            self.tool_manager.add_tool(tool)

    def set_tool_approval_handler(self, handler: Optional[Callable[[str, Dict[str, Any]], bool]]) -> None:
        """Configure a custom approval handler or revert to the default."""
        if handler is None:
            self._approval_handler = self._default_approval_handler
        else:
            self._approval_handler = handler
        self.tool_manager.set_approval_handler(self._approval_handler)

    def request_tool_approval(self, tool_name: str, tool_kwargs: Dict[str, Any]) -> bool:
        """Default approval flow that prompts in the current terminal."""
        serialized_args = json.dumps(tool_kwargs, default=str)
        prompt = (
            f"Tool '{tool_name}' requires approval before execution.\n"
            f"Arguments: {serialized_args}\n"
            "Approve? [y/N]: "
        )
        response = input(prompt)
        approved = response.strip().lower() in {"y", "yes"}
        logger.info(
            "Tool '%s' approval %s by user.",
            tool_name,
            "granted" if approved else "denied",
        )
        return approved

    def receive(
        self,
        message: Optional[str | Dict[str, Any]] = None, 
        role: Literal["user", "system", "tool"] = "user",
        context: Optional[List[Dict[str, Any]]] = None,
        image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        encoded_image: Optional[str] = None,
        request_response: bool = True,
        return_full_response: bool = True,
        return_outgoing_message: bool = False,
        with_system_message: bool = True,
    ) -> str | Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]] | None:
        """Receive a message from the user and generate a response.
        
        Parameters
        ----------
        message : Optional[str | Dict[str, Any]]
        
            The new message to be sent to the AI. It should be a string or a dictionary
            in an OpenAI-compatible format. To attach an image to the message,
            either provide the image as a numpy array with `image`, or provide the path
            to the image with `image_path`, or provide the base-64 encoded image with
            `encoded_image`. Alternatively, the path to the image can also be embedded
            in the message string with the following format as `<img path/to/image.png>`.
            Paths embedded in this way is only used when `image_path` is None and `message`
            is a string.
            
            If `message` is None, the function will send the chat history stored to AI
            if `with_history` is True.
            
        role : Literal["user", "system", "tool"], optional
            The role of the sender.
        context : Optional[List[Dict[str, Any]]], optional
            The context to be sent to the AI. This is a list of message dictionaries.
        image : np.ndarray, optional
            The image to be sent to the AI. Exclusive with `encoded_image` and `image_path`.
        image_path : str, optional
            The path to the image to be sent to the AI. Exclusive with `image` and `encoded_image`.
        encoded_image : str, optional
            The base-64 encoded image to be sent to the AI. Exclusive with `image` and `image_path`.
        request_response : bool, optional
            If True, the message will be sent to the AI and a response will be
            requested. Otherwise, the message will only be logged into the message history
            and will not be sent to the AI. The function returns None in this case.
        return_full_response : bool, optional
            If True, this function returns a dictionary containing the full response
            from the agent. Otherwise, only the content of the response is returned
            as a string.
        return_outgoing_message : bool, optional
            If True, the outgoing message will also be returned.
        with_system_message : bool, optional
            If True, the system message will be included in the message sent to the AI.
        Returns
        -------
        str | Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]] | None
            The response from the agent. If `return_full_response` is True, the
            response is a dictionary containing the full response from the agent.
            Otherwise, the response is a string containing the content of the response.
            If `return_outgoing_message` is True, the outgoing message will also be returned.
            If `request_response` is False, the function returns None.
        """
        if image is not None or image_path is not None or encoded_image is not None:
            if isinstance(message, dict):
                raise ValueError("`message` cannot be a dictionary if an image is provided.")
        if role == "tool" and not isinstance(message, Dict):
            raise ValueError(
                "When role is 'tool', `message` must be a dictionary of tool response that "
                "contains the tool_call_id."
            )
        if message is None and context is None:
            raise ValueError("`message` and `context` cannot be None at the same time.")
            
        # Extract image path from string message, if any.
        if image_path is None and isinstance(message, str):
            image_path, modified_message = get_image_path_from_text(message, return_text_without_image_tag=True)
            if image_path is not None:
                message = modified_message
            
        # Convert string message to JSON if it is not yet a dictionary.
        if isinstance(message, str):
            message = generate_openai_message(
                message, role=role, image=image, image_path=image_path, encoded_image=encoded_image
            )
        
        # Print message.
        if message is not None:
            print_message(message, response_requested=request_response)

        # Retrieve from vector store.
        recalled_memory_results = []
        memory_context_messages: List[Dict[str, Any]] = []
        if (
            request_response
            and self.memory_manager is not None
            and self.memory_manager.enabled
        ):
            recalled_memory_results, memory_context_messages = self.retrieve_from_vector_store(message)
        
        # Create the list of messages to send.
        sys_message = self.system_messages if with_system_message else []
        message_list = [message] if message is not None else []
        context_messages = list(context) if context is not None else []
        combined_messages = sys_message + memory_context_messages + context_messages + message_list
            
        # Send messages, get response and print it.
        if request_response:
            response = self.send_message_and_get_response(combined_messages)
            print_message(response)

        if request_response and self.memory_manager is not None and message is not None:
            self._post_response_memory_update(
                user_message=message,
                agent_response=response if request_response else None,
                recall_results=recalled_memory_results,
            )

        if not request_response:
            return None
        
        returns = []
        if return_full_response:
            returns.append(dict(response))
        else:
            returns.append(response.choices[0].message.content)
        if return_outgoing_message:
            returns.append(message_list[0] if len(message_list) > 0 else None)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns
        
    def retrieve_from_vector_store(self, message: Dict[str, Any]) -> List[MemoryQueryResult]:
        """Retrieve memories from the vector store.
        
        Parameters
        ----------
        message : Dict[str, Any]
            The message to be sent to the agent.
            
        Returns
        -------
        List[MemoryQueryResult]
            The memories retrieved from the vector store.
        List[Dict[str, Any]]
            The retrieved memories converted to OpenAI-compatible messages
            which can be inserted into the context.
        """
        recalled_memory_results = []
        memory_context_messages: List[Dict[str, Any]] = []

        query_text = self._extract_message_text(message)
        recalled_memory_results = self.memory_manager.recall(query_text)
        formatted_memory = self.memory_manager.format_results(recalled_memory_results)
        if formatted_memory:
            injection_role = self._memory_injection_role
            if injection_role not in {"system", "user"}:
                injection_role = "system"
            memory_context_messages.append(
                generate_openai_message(
                    formatted_memory,
                    role=injection_role,
                )
            )
        return recalled_memory_results, memory_context_messages
    
    def send_message_and_get_response(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Send a message to the agent and get the response.
        
        Parameters
        ----------
        messages : List[Dict[str, Any]]
            The list of messages to be sent to the agent. The messages
            should be in an OpenAI-compatible format.
        
        Returns
        -------
        Dict[str, Any]
            The response from the agent in an OpenAI-compatible format.
        """
        raise NotImplementedError
    
    def process_response(
        self, 
        response: Dict[str, Any],
        remove_empty_tool_calls_key: bool = True,
        remove_empty_reasoning_content_key: bool = True,
        move_reasoning_content_to_empty_content: bool = True,
    ) -> Dict[str, Any]:
        """Process the response from the agent. Models on OpenAI or OpenRouter
        should not need these processings, but some other model providers may
        require them.
        
        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent as a dictionary.
        remove_empty_tool_calls_key : bool, optional
            If True, the "tool_calls" key will be removed if it is an empty list.
        remove_empty_reasoning_content_key : bool, optional
            If True, the "reasoning_content" key will be removed if it exists.
        move_reasoning_content_to_empty_content : bool, optional
            If True, the "reasoning_content" key will be moved to the "content" key
            if the "content" key is None.
        
        Returns
        -------
        Dict[str, Any]
            The processed response from the agent.
        """
        if remove_empty_tool_calls_key:
            if "tool_calls" in response and isinstance(response["tool_calls"], list) and len(response["tool_calls"]) == 0:
                del response["tool_calls"]
        if remove_empty_reasoning_content_key:
            if ("reasoning_content" in response 
                and response["reasoning_content"] is not None
                and len(response["reasoning_content"]) == 0
            ):
                del response["reasoning_content"]
        if move_reasoning_content_to_empty_content:
            if (
                "reasoning_content" in response
                and response["content"] is None
            ):
                response["content"] = response["reasoning_content"]
                del response["reasoning_content"]
        return response
    
    def handle_tool_call(
        self, 
        message: Dict[str, Any],
        return_tool_return_types: bool = False
    ) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], List[ToolReturnType]]:
        """Handle the tool calls in the response of the agent.
        If tool call exists, the tools will be executed and tool
        responses will be returned. This function is able to handle 
        multiple tool calls.
        
        If an exception is encountered when executing a tool, the tool
        response will be a string containing the exception message. If
        `return_tool_return_types` is True, the return type of that tool
        execution will set to be ToolReturnType.EXCEPTION.
        
        Parameters
        ----------
        message : Dict[str, Any]
            The message to handle the tool call.
        return_tool_return_types : bool
            If True, the return types of the tool calls will be also returned.
        
        Returns
        -------
        List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], List[ToolReturnType]]
            The tool responses. If `return_tool_return_types` is True, the
            return types of the tool calls will also be returned.
        """
        if not has_tool_call(message):
            if return_tool_return_types:
                return [], []
            else:
                return []
        
        responses = []
        response_types = []
        tool_call_info_list = get_tool_call_info(message, index=None)
        for tool_call_info in tool_call_info_list:
            tool_call_id = tool_call_info["id"]
            tool_name = tool_call_info["function"]["name"]
            tool_call_kwargs = json.loads(tool_call_info["function"]["arguments"])
            
            exception_encountered = False
            try:
                result = self.tool_manager.execute_tool(tool_name, tool_call_kwargs)
            except Exception as e:
                exception_encountered = True
                result = str(e)
            
            response = generate_openai_message(
                content=str(result),
                role="tool",
                tool_call_id=tool_call_id
            )
            responses.append(response)
            response_types.append(
                self.tool_manager.get_tool_return_type(tool_name) if not exception_encountered 
                else ToolReturnType.EXCEPTION
            )
            
        if return_tool_return_types:
            return responses, response_types
        return responses
    
    def register_message_hook(self, hook: Callable) -> None:
        """Register a hook function that will be called to process the message
        received from the agent.
        
        The hook function should take a dictionary of message and return a
        dictionary of processed message.
        
        Parameters
        ----------
        hook : Callable
            The hook function.
        """
        self.message_hooks.append(hook)


def to_dict(message: ChatCompletionMessage | Dict[str, Any]) -> dict:
    """Convert a ChatCompletionMessage to a dictionary.
    """
    if isinstance(message, ChatCompletionMessage):
        return message.to_dict()
    else:
        return message


def generate_openai_message(
    content: str,
    role: Literal["user", "system", "tool"] = "user",
    tool_call_id: str = None,
    image: np.ndarray = None,
    image_path: str = None,
    encoded_image: str = None
) -> Dict[str, Any]:
    """Generate a dictionary in OpenAI-compatible format 
    containing the message to be sent to the agent.

    Parameters
    ----------
    content : str
        The content of the message.
    role : Literal["user", "system", "tool"], optional
        The role of the sender.
    image : np.ndarray, optional
        The image to be sent to the agent. Exclusive with `encoded_image` and `image_path`.
    image_path : str, optional
        The path to the image to be sent to the agent. Exclusive with `image` and `encoded_image`.
    encoded_image : str, optional
        The base-64 encoded image to be sent to the agent. Exclusive with `image` and `image_path`.
    """
    if sum([image is not None, encoded_image is not None, image_path is not None]) > 1:
        raise ValueError("Only one of `image`, `encoded_image`, or `image_path` should be provided.")
    if role not in ["user", "system", "tool"]:
        raise ValueError("Invalid role. Must be one of `user`, `system`, or `tool`.")

    if image is not None or image_path is not None:
        encoded_image = encode_image_base64(image=image, image_path=image_path)

    if role == "user":
        message = {
            "role": "user",
            "content": content
        }
    elif role == "system":
        message = {
            "role": "system",
            "content": content
        }
    elif role == "tool":
        message = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id
        }

    if encoded_image is not None:
        message["content"] = [
            {
                "type": "text",
                "text": content
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"
                }
            }
        ]
    return message


def has_tool_call(message: dict | ChatCompletionMessage) -> bool:
    """Check if the message has a tool call.
    
    Parameters
    ----------
    message : dict | ChatCompletionMessage
        A message in OpenAI-compatible format.
        
    Returns
    -------
    """
    message = to_dict(message)
    if "tool_calls" in message.keys():
        return True
    else:
        return False


def get_tool_call_info(
    message: dict | ChatCompletionMessage,
    index: Optional[int] = 0
) -> str | List[str]:
    """Get the tool call ID from the message.

    Parameters
    ----------
    message : dict | ChatCompletionMessage
        The message to get the tool call ID from. The message
        should be in OpenAI-compatible format.
    index : int, optional
        The index of the tool call to get the ID from. If None,
        all tool calls are returned as a list.

    Returns
    -------
    str | List[str]
        The tool call(s).
    """
    message = to_dict(message)
    if index is None:
        return message["tool_calls"]
    else:
        return message["tool_calls"][index]


def get_message_elements(message: Dict[str, Any]) -> Dict[str, Any]:
    """Get the elements of the message.

    Parameters
    ----------
    message : Dict[str, Any]
        The message to get the elements from. The message
        should be in OpenAI-compatible format.

    Returns
    -------
    Dict[str, Any]
        The elements of the message.
    """
    role = message["role"]

    image = None
    content = ""
    if "content" in message.keys():
        if isinstance(message["content"], str):
            content += message["content"] + "\n"
        elif isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    content += item["text"] + "\n"
                elif item["type"] == "image_url":
                    content += "<image> \n"
                    image = item["image_url"]["url"]

    tool_calls = None
    if "tool_calls" in message.keys():
        tool_calls = ""
        for tool_call in message["tool_calls"]:
            tool_calls += f"{tool_call['id']}: {tool_call['function']['name']}\n"
            tool_calls += f"Arguments: {tool_call['function']['arguments']}\n"

    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        "image": image
    }


def print_message(
    message: Dict[str, Any],
    response_requested: Optional[bool] = None,
    return_string: bool = False
) -> None:
    """Print the message.

    Parameters
    ----------
    message : Dict[str, Any]
        The message to be printed. The message should be in
        OpenAI-compatible format.
    response_requested : bool, optional
        Whether a response is requested for the message.
    return_string : bool, optional
        If True, the message is returned as a string instead of printed.
    """
    color_dict = {
        "user": "\033[94m",
        "system": "\033[92m",
        "tool": "\033[93m",
        "assistant": "\033[91m"
    }
    color = color_dict[message["role"]]

    text = f"[Role] {message['role']}\n"
    if response_requested is not None:
        text += f"[Response requested] {response_requested}\n"

    elements = get_message_elements(message)

    text += "[Content]\n"
    text += elements["content"] + "\n"

    if elements["tool_calls"] is not None:
        text += "[Tool call]\n"
        text += elements["tool_calls"] + "\n"

    text += "\n ========================================= \n"

    if return_string:
        return text
    else:
        print(f"{color}{text}\033[0m")
