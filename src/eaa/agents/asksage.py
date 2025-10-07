import os
import re

from typing import Any, Callable, Dict, List, Optional, Sequence

from asksageclient import AskSageClient

from eaa.agents.base import BaseAgent
from eaa.api.llm_config import AskSageConfig
from eaa.agents.memory import MemoryManagerConfig, MemoryQueryResult, VectorStore
from eaa.util import decode_image_base64, get_timestamp


class AskSageAgent(BaseAgent):
    
    def __init__(
        self,
        llm_config: AskSageConfig,
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
        llm_config : AskSageConfig
            Configuration for the AskSage API. It should be an instance
            of AskSageConfig. Refer to the documentation of the config class for
            more details.
        system_message : str, optional
            The system message for the OpenAI-compatible API.
        memory_config : MemoryManagerConfig, optional
            Optional configuration for persistent memory support.
        memory_vector_store, memory_notability_filter, memory_formatter,
        memory_embedder : optional
            Overrides for the memory backend, storage filter, recall formatting,
            and embedding function respectively.
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
    
    @property
    def user_base_url(self) -> str:
        return self.llm_config.user_base_url
    
    @property
    def server_base_url(self) -> str:
        return self.llm_config.server_base_url
        
    def create_client(self) -> AskSageClient:
        if self.llm_config.cacert_path is not None:
            os.environ["REQUESTS_CA_BUNDLE"] = self.llm_config.cacert_path
        
        return AskSageClient(
            email=self.llm_config.email,
            api_key=self.api_key,
            user_base_url=self.user_base_url,
            server_base_url=self.server_base_url,
        )
        
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
        tool_schema = self.tool_manager.get_all_schema()
        asksage_messages = [
            self.openai_message_to_asksage_message(m) 
            for m in messages 
            if m["role"] != "system"
        ]
        file = asksage_messages[-1].get("file", None)
        response = self.client.query(
            message=asksage_messages,
            persona="default",
            model=self.model,
            system_prompt=self.system_messages[0]["content"],
            tools=tool_schema if len(tool_schema) > 0 else None,
            file=file,
        )
        response_dict = self.asksage_response_to_openai_response(response)
        response_dict = self.process_response(
            response_dict,
            remove_empty_tool_calls_key=True,
            remove_empty_reasoning_content_key=True,
            move_reasoning_content_to_empty_content=True,
        )
        for hook in self.message_hooks:
            response_dict = hook(response_dict)
        return response_dict
    
    def openai_message_to_asksage_message(self, msg: dict) -> dict:
        """Convert a message with OpenAI-compatible format to AskSage format.

        Parameters
        ----------
        msg : dict
            A message in OpenAI-compatible format.
            
        Returns
        -------
        dict
            A message in AskSage format.
        """
        text = ""
        image_url = None
        image_path = None
        role = {"user": "me", "assistant": "gpt", "tool": "tool", "system": "system"}[msg["role"]]
        
        if "content" in msg.keys():
            if isinstance(msg["content"], str):
                text = msg["content"]
            elif isinstance(msg["content"], list):
                text = ""
                for item in msg["content"]:
                    if item["type"] == "text":
                        text += item["text"] + "\n"
                    elif item["type"] == "image_url":
                        image_url = item["image_url"]["url"]
                if len(text) > 0:
                    # Remove the last newline character
                    text = text[:-1]
            
        if image_url is not None:
            image_base64 = re.sub("^data:image/.+;base64,", "", image_url)
            image = decode_image_base64(image_base64, return_type="pil")
            if not os.path.exists(".tmp"):
                os.makedirs(".tmp")
            image_path = f".tmp/{get_timestamp()}.png"
            image.save(image_path)

        message = {
            "user": role,
            "message": text,
            "model": self.model,
            "system_prompt": self.system_messages[0]["content"],
        }
        if role == "me":
            message["tools"] = self.tool_manager.get_all_schema()
        if image_path is not None:
            message["file"] = image_path
        if role == "gpt" and "tool_calls" in msg.keys():
            message["tool_calls"] = msg["tool_calls"]
        if role == "tool":
            message["tool_call_id"] = msg["tool_call_id"]
        return message
        
    def asksage_response_to_openai_response(self, response: dict) -> dict:
        """Convert a response with AskSage format to OpenAI format.

        Parameters
        ----------
        response : dict
            A response in AskSage format.
        """
        oai_response = {
            "role": "assistant",
            "content": response["message"],
        }
        if (
            "tool_calls" in response.keys()
        ) and (
            response["tool_calls"] is not None
        ) and (
            len(response["tool_calls"]) > 0
        ):
            oai_response["tool_calls"] = response["tool_calls"]
        return oai_response
        
