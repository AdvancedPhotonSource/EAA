from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from eaa.agents.openai import OpenAIAgent
from eaa.agents.memory import MemoryQueryResult, VectorStore
from eaa.api.llm_config import OpenAIConfig
from eaa.api.memory import MemoryManagerConfig


class ArgoClient:
    def __init__(self, base_url: str, user: str):
        self.base_url = base_url
        self.user = user

    def create(
        self, 
        messages: List[Dict[str, Any]], 
        model: str, 
        tools: List[Dict[str, Any]], 
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        payload = {
            "user": self.user,
            "model": model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        
        sep = "" if self.base_url.endswith("/") else "/"
        response = requests.post(
            f"{self.base_url}{sep}chat/",
            json=payload,
        )
        if response.status_code != 200:
            raise Exception(f"Error from Argo: {response.text}")
        return response.json()


class ArgoAgent(OpenAIAgent):
    
    def __init__(
        self,
        llm_config: OpenAIConfig,
        system_message: str = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        *,
        memory_vector_store: Optional[VectorStore] = None,
        memory_notability_filter: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        memory_formatter: Optional[Callable[[List[MemoryQueryResult]], str]] = None,
        memory_embedder: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
    ) -> None:
        """An agent that uses OpenAI-compatible API to generate responses.

        Parameters
        ----------
        llm_config : OpenAIConfig
            Configuration for the OpenAI-compatible API. It should be an instance
            of OpenAIConfig. Refer to the documentation of the config class for
            more details.
        system_message : str, optional
            The system message for the OpenAI-compatible API.
        memory_config : MemoryManagerConfig, optional
            Optional configuration for persistent memory.
        memory_vector_store, memory_notability_filter, memory_formatter,
        memory_embedder : optional
            Overrides for the memory backend, storage filter, result
            formatting, and embedding function respectively.
        """
        super().__init__(
            llm_config=llm_config,
            system_message=system_message,
            memory_config=memory_config,
            memory_vector_store=memory_vector_store,
            memory_notability_filter=memory_notability_filter,
            memory_formatter=memory_formatter,
            memory_embedder=memory_embedder,
        )
        
    def create_client(self) -> ArgoClient:
        return ArgoClient(
            base_url=self.llm_config.base_url,
            user=self.llm_config.user,
        )

    def supports_memory_embeddings(self) -> bool:
        return False

    def send_message_and_get_response(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Send a message to the agent and get the response.
        
        Parameters
        ----------
        message : List[Dict[str, Any]]
            The list of messages to be sent to the agent.
        
        Returns
        -------
        Dict[str, Any]
            The response from the agent.
        """
        tool_schema = self.tool_manager.get_all_schema()
        response = self.client.create(
            messages=messages,
            model=self.model,
            tools=tool_schema if len(tool_schema) > 0 else None,
            tool_choice="auto" if len(tool_schema) > 0 else None,
        )
        response_dict = self.argo_response_to_openai_response(response)
        response_dict = self.process_response(
            response_dict,
            remove_empty_tool_calls_key=True,
            remove_empty_reasoning_content_key=True,
            move_reasoning_content_to_empty_content=True,
        )
        for hook in self.message_hooks:
            response_dict = hook(response_dict)
        return response_dict

    def argo_response_to_openai_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        oai_response = {
            "role": "assistant",
            "content": response["response"]["content"],
        }
        if "tool_calls" in response["response"] and len(response["response"]["tool_calls"]) > 0:
            oai_response["tool_calls"] = response["response"]["tool_calls"]
        return oai_response
