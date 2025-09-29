from typing import Any, Dict, List, Optional, Sequence, Union

from openai import OpenAI

from eaa.agents.base import BaseAgent
from eaa.agents.memory import MemoryManagerConfig


class OpenAIAgent(BaseAgent):
    
    def __init__(
        self,
        llm_config: dict,
        system_message: str = None,
        memory_config: Optional[Union[dict, MemoryManagerConfig]] = None,
    ) -> None:
        """An agent that uses OpenAI-compatible API to generate responses.

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
            Optional configuration for persistent memory.
        """
        super().__init__(
            llm_config=llm_config,
            system_message=system_message,
            memory_config=memory_config,
        )
        
    @property
    def base_url(self) -> str:
        return self.llm_config.get("base_url", "https://api.openai.com/v1")
        
    def create_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def supports_memory_embeddings(self) -> bool:
        return True

    def get_default_embedding_model(self) -> Optional[str]:
        return self.llm_config.get("embedding_model", "text-embedding-3-small")

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        selected_model = model or self.get_default_embedding_model()
        if selected_model is None:
            raise ValueError("No embedding model configured for OpenAIAgent.")
        response = self.client.embeddings.create(
            model=selected_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]
    
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tool_schema if len(tool_schema) > 0 else None,
            tool_choice="auto" if len(tool_schema) > 0 else None,
        )
        response_dict = response.choices[0].message.to_dict()
        response_dict = self.process_response(
            response_dict,
            remove_empty_tool_calls_key=True,
            remove_empty_reasoning_content_key=True,
            move_reasoning_content_to_empty_content=True,
        )
        for hook in self.message_hooks:
            response_dict = hook(response_dict)
        return response_dict
