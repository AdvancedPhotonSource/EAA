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
            {tool.name: tool.__call__ for tool in tools}
        )
            
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
