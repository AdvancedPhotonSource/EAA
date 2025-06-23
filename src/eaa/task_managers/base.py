from typing import Any, Dict

from eaa.tools.base import BaseTool
from eaa.comms import get_api_key
from eaa.agents.openai import OpenAIAgent, generate_openai_message
from eaa.tools.base import ToolReturnType


class BaseTaskManager:

    assistant_system_message = ""
            
    def __init__(
        self, 
        model_name: str = "gpt-4o", 
        model_base_url: str = None, 
        access_token: str = None,
        tools: list[BaseTool] = [], 
        *args, **kwargs
    ):
        self.context = []
        self.full_history = []
        self.model = model_name
        self.model_base_url = model_base_url
        self.access_token = access_token
        self.agent = None
        self.tools = tools
        self.build()
        
    def build(self, *args, **kwargs):
        self.build_agent()
        self.build_tools()
    
    def build_agent(self, *args, **kwargs):
        """Build the assistant(s)."""
        llm_config = self.get_llm_config(*args, **kwargs)
        self.agent = OpenAIAgent(
            llm_config=llm_config,
            system_message=self.assistant_system_message,
        )
    
    def build_tools(self, *args, **kwargs):
        pass

    def register_tools(
        self, 
        tools: BaseTool | list[BaseTool], 
    ) -> None:
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        self.agent.register_tools(
            self.create_tool_list(tools)
        )
        
    def create_tool_list(self, tools: list[BaseTool]) -> list[dict]:
        """Create a list of tool dictionaries by concatenating the exposed_tools
        of all BaseTool objects.
        
        Parameters
        ----------
        tools : list[BaseTool]
            A list of BaseTool objects.
        
        Returns
        -------
        list[dict]
            A list of tool dictionaries.
        """
        tool_list = []
        registered_tool_names = []
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError("Input should be a list of BaseTool objects.")
            if (
                not hasattr(tool, "exposed_tools")
                or (hasattr(tool, "exposed_tools") and len(tool.exposed_tools) == 0)
            ):
                raise ValueError(
                    "A subclass of BaseTool must have a non-empty `exposed_tools` attribute "
                    "containing a dictionary of tool names and their corresponding callable functions."
                )
            for tool_dict in tool.exposed_tools:
                if tool_dict["name"] in registered_tool_names:
                    raise ValueError(
                        f"Tool {tool_dict['name']} is already registered. Make sure no two callables have the same name."
                    )
                tool_list.append(tool_dict)
                registered_tool_names.append(tool_dict["name"])
        return tool_list

    def get_llm_config(self, *args, **kwargs):
        llm_config = {
            "model": self.model,
            "api_key": (
                get_api_key(self.model, self.model_base_url) 
                if self.access_token is None 
                else self.access_token
            ),
        }
        if self.model_base_url:
            llm_config["base_url"] = self.model_base_url
        return llm_config

    def prerun_check(self, *args, **kwargs) -> bool:
        return True
    
    def update_message_history(
        self,
        message: Dict[str, Any],
        update_context: bool = True,
        update_full_history: bool = True
    ) -> None:
        if update_context:
            self.context.append(message)
        if update_full_history:
            self.full_history.append(message)

    def run(self, *args, **kwargs) -> None:
        self.prerun_check()

    def run_conversation(
        self, 
        store_all_images_in_context: bool = False, 
        *args, **kwargs
    ) -> None:
        """Start a free-dtyle conversation with the assistant.

        Parameters
        ----------
        store_all_images_in_context : bool, optional
            Whether to store all images in the context. If False, only the image
            in the initial prompt, if any, is stored in the context. Keep this
            False to reduce the context size and save costs.
        """
        while True:
            message = input("Enter a message: ")
            if message.lower() == "exit":
                break
            
            # Send message and get response
            response, outgoing_message = self.agent.receive(message, return_outgoing_message=True)
            self.update_message_history(outgoing_message, update_context=True, update_full_history=True)
            self.update_message_history(response, update_context=True, update_full_history=True)
            
            # Handle tool calls
            tool_responses, tool_response_types = self.agent.handle_tool_call(response, return_tool_return_types=True)
            if len(tool_responses) >= 1:        
                for tool_response, tool_response_type in zip(tool_responses, tool_response_types):
                    self.update_message_history(tool_response, update_context=True, update_full_history=True)
                    # If the tool returns an image path, load the image and send it to 
                    # the assistant in a follow-up message as user.
                    if tool_response_type == ToolReturnType.IMAGE_PATH:
                        image_path = tool_response["content"]
                        image_message = generate_openai_message(
                            content="Here is the image the tool returned.",
                            image_path=image_path,
                            role="user",
                        )
                        self.update_message_history(
                            image_message, update_context=store_all_images_in_context, update_full_history=True
                        )
                response = self.agent.receive(message=None, context=self.context, return_outgoing_message=False)
                self.update_message_history(response, update_context=True, update_full_history=True)
