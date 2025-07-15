from typing import Dict, Any, List

from openai import OpenAI

from eaa.agents.base import BaseAgent


class OpenAIAgent(BaseAgent):
    
    def __init__(
        self,
        llm_config: dict,
        system_message: str = None,
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
        """
        super().__init__(
            llm_config=llm_config, 
            system_message=system_message
        )
        
    def create_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
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
