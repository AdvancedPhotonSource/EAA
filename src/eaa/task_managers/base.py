from eaa.tools.base import BaseTool
from eaa.comms import get_api_key
from eaa.agents.openai import OpenAIAgent


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

    def run(self, *args, **kwargs) -> None:
        self.prerun_check()

    def run_conversation(self, *args, **kwargs) -> None:
        """Run a conversation with the assistant."""
        message = input("Enter a message: ")
        while True:
            if message.lower() == "exit":
                break
            _ = self.agent.receive(message, store_message=True, store_response=True)
            message = input("Enter a message: ")