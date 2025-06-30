from typing import Any, Dict, Optional
import sqlite3

from eaa.tools.base import BaseTool
from eaa.comms import get_api_key
from eaa.agents.openai import (
    OpenAIAgent, 
    generate_openai_message, 
    get_message_elements,
)
from eaa.tools.base import ToolReturnType


class BaseTaskManager:

    assistant_system_message = ""
            
    def __init__(
        self, 
        model_name: str = "gpt-4o", 
        model_base_url: str = None, 
        access_token: str = None,
        tools: list[BaseTool] = [], 
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args, **kwargs
    ):
        """The base task manager.
        
        Parameters
        ----------
        model_name : str
            The name of the model to use.
        model_base_url : str
            The base URL of the model.
        access_token : str
            The access token for the model.
        tools : list[BaseTool]
            A list of tools provided to the agent.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        build : bool
            Whether to build the internal state of the task manager.
        """
        self.context = []
        self.full_history = []
        self.model = model_name
        self.model_base_url = model_base_url
        self.access_token = access_token
        self.agent = None
        self.tools = tools
        self.message_db_path = message_db_path
        self.message_db_conn = None
        if build:
            self.build()
        
    def build(self, *args, **kwargs):
        self.build_db()
        self.build_agent()
        self.build_tools()
    
    def build_db(self, *args, **kwargs):
        if self.message_db_path:
            self.message_db_conn = sqlite3.connect(self.message_db_path)
            self.message_db_conn.execute(
                "CREATE TABLE IF NOT EXISTS messages (role TEXT, content TEXT, tool_calls TEXT, image TEXT)"
            )
            self.message_db_conn.commit()
    
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
        if self.message_db_conn:
            self.add_message_to_db(message)
            
    def add_message_to_db(self, message: Dict[str, Any]) -> None:
        elements = get_message_elements(message)
        self.message_db_conn.execute(
            "INSERT INTO messages (role, content, tool_calls, image) VALUES (?, ?, ?, ?)",
            (elements["role"], elements["content"], elements["tool_calls"], elements["image"])
        )
        self.message_db_conn.commit()

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
