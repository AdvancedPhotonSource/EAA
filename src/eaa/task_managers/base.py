import autogen

from eaa.tools.base import BaseTool


class BaseTaskManager:
    
    class AgentGroup(dict):
        user_proxy: autogen.ConversableAgent = None
        assistant: autogen.ConversableAgent = None
            
    def __init__(
        self, 
        model_name: str = "gpt-4o", 
        model_base_url: str = None, 
        tools: list[BaseTool] = [], 
        *args, **kwargs
    ):
        self.model = model_name
        self.model_base_url = model_base_url
        self.agents = self.AgentGroup()
        self.tools = tools
        self.build()
        
    def build(self, *args, **kwargs):
        self.build_agents()
        self.build_tools()
    
    def build_agents(self, *args, **kwargs):
        pass
    
    def build_tools(self, *args, **kwargs):
        pass

    def register_tools(
        self, 
        tools: BaseTool | list[BaseTool], 
        caller: autogen.ConversableAgent, 
        executor: autogen.ConversableAgent
    ) -> None:
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        for tool in tools:
            wrapped_tool = autogen.tools.Tool(
                name=tool.name,
                description=tool.__call__.__doc__,
                func_or_tool=tool.__call__,
            )
            
            autogen.register_function(
                wrapped_tool,
                caller=caller,
                executor=executor,
                name=tool.name,
                description=tool.__call__.__doc__,
            )

    def prerun_check(self, *args, **kwargs) -> bool:
        return True

    def run(self, *args, **kwargs) -> None:
        self.prerun_check()
