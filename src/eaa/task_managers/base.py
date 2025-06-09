from eaa.tools.base import BaseTool
from eaa.comms import get_api_key


class BaseTaskManager:
            
    def __init__(
        self, 
        model_name: str = "gpt-4o", 
        model_base_url: str = None, 
        tools: list[BaseTool] = [], 
        *args, **kwargs
    ):
        self.model = model_name
        self.model_base_url = model_base_url
        self.agent = None
        self.tools = tools
        self.build()
        
    def build(self, *args, **kwargs):
        self.build_agent()
        self.build_tools()
    
    def build_agent(self, *args, **kwargs):
        pass
    
    def build_tools(self, *args, **kwargs):
        pass

    def register_tools(
        self, 
        tools: BaseTool | list[BaseTool], 
    ) -> None:
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        self.agent.register_tools(
            self.create_tool_dict(tools)
        )
        
    def create_tool_dict(self, tools: list[BaseTool]) -> dict:
        """Create a dictionary containing the callable tools of all BaseTool objects.
        
        Parameters
        ----------
        tools : list[BaseTool]
            A list of BaseTool objects.
        """
        d = {}
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError(f"Input should be a list of BaseTool objects.")
            if (
                not hasattr(tool, "exposed_tools")
                or (hasattr(tool, "exposed_tools") and len(tool.exposed_tools) == 0)
            ):
                raise ValueError(
                    "A subclass of BaseTool must have a non-empty `exposed_tools` attribute "
                    "containing a dictionary of tool names and their corresponding callable functions."
                )
            for tool_name, tool_function in tool.exposed_tools.items():
                if tool_name in d.keys():
                    raise ValueError(
                        f"Tool {tool_name} is already registered. Make sure no two callables have the same name."
                    )
                d[tool_name] = tool_function
        return d
            
    def get_llm_config(self, *args, **kwargs):
        llm_config = {
            "model": self.model,
            "api_key": get_api_key(self.model, self.model_base_url),
        }
        if self.model_base_url:
            llm_config["base_url"] = self.model_base_url
        return llm_config

    def prerun_check(self, *args, **kwargs) -> bool:
        return True

    def run(self, *args, **kwargs) -> None:
        self.prerun_check()
