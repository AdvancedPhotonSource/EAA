from textwrap import dedent

import autogen

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool
from eaa.comms import get_api_key


class SetupParameterSearchTaskManager(BaseTaskManager):
    
    class AgentGroup(dict):
        user_proxy: autogen.ConversableAgent = None
        tool_executor: autogen.ConversableAgent = None
        assistant: autogen.ConversableAgent = None
        group_chat_manager: autogen.GroupChatManager = None
    
    def __init__(
        self,
        model_name: str = "gpt-4o", 
        model_base_url: str = None,
        tools: list[BaseTool] = [],
        *args, **kwargs
    ) -> None:
        """An agent that searches for the best setup parameters
        for an imaging system.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use.
        model_base_url : str, optional
            The base URL of the model. This is only needed for self-hosted models.
        """
        super().__init__(
            model_name=model_name, 
            model_base_url=model_base_url,
            tools=tools, 
            *args, **kwargs
        )
                
    def build_agents(self, *args, **kwargs) -> None:
        """Build the assistant(s)."""
        
        llm_config = {
            "model": self.model,
            "api_key": get_api_key(self.model, self.model_base_url),
        }
        if self.model_base_url:
            llm_config["base_url"] = self.model_base_url

        self.agents.assistant = autogen.ConversableAgent(
            name="assistant",
            system_message=dedent(
                """\
                You are helping scientists at a microscopy facility to
                to find the best setup parameters for their imaging system.
                These include the field of view of the microscope, the beam energy,
                and the optics. You have the following tool(s) at your disposal:
                - A tool that acquires an image of a sub-region of a sample at
                  given location and with given size (the field of view, or FOV),
                  analyzes that image internally, and reports back the features
                  identified in the image in text.
                When using tools, only make one call at a time. Do not make 
                multiple calls simultaneously.\
                """
            ),
            llm_config=llm_config,
        )
        
        self.agents.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            llm_config=False,
            human_input_mode="ALWAYS",
            code_execution_config={
                "use_docker": False
            },
        )
        
        self.agents.tool_executor = autogen.ConversableAgent(
            name="tool_executor",
            human_input_mode="NEVER",
            code_execution_config={
                "use_docker": False
            },
        )
        
        group_chat =  autogen.GroupChat(
            agents=[self.agents.user_proxy, self.agents.tool_executor, self.agents.assistant],
            messages=[],
            max_round=999,
        )
        
        self.agents.group_chat_manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config,
        )

    def build_tools(self, *args, **kwargs):
        for tool in self.tools:
            self.register_tools(
                tool,
                caller=self.agents.assistant,
                executor=self.agents.tool_executor
            )

    def prerun_check(self, *args, **kwargs) -> bool:
        if self.agents.assistant.tools is None:
            raise ValueError("No tools registered for the main agent.")
        return super().prerun_check(*args, **kwargs)
        
    def run(self, *args, **kwargs) -> None:
        """Run the task manager."""
        super().run(*args, **kwargs)
        
        self.run_fov_search(*args, **kwargs)
        
    def run_conversation(self, *args, **kwargs) -> None:
        """Run a conversation with the assistant."""
        message = input("Enter a message: ")
        self.agents.user_proxy.initiate_chat(
            self.agents.assistant,
            message=message
        )
        
    def run_fov_search(
        self,
        feature_description: str,
        y_range: tuple[float, float],
        x_range: tuple[float, float],
        fov_size: tuple[float, float] = (100, 100),
        step_size: tuple[float, float] = (100, 100),
        *args, **kwargs
    ) -> None:
        """Run a search for the best field of view for the microscope.
        
        Parameters
        ----------
        feature_description : str
            A text description of the feature to search for.
        y_range : tuple[float, float]
            The range of y coordinates to search for the feature.
        x_range : tuple[float, float]
            The range of x coordinates to search for the feature.
        fov_size : tuple[float, float], optional
            The size of the field of view.
        step_size : float, optional
            The step size to move the field of view each time.
        """
        message = dedent(f"""\
            You are given a tool that acquires an image of a sub-region
            of a sample at given location and with given size (the field
            of view, or FOV). Use this tool to find a subregion that contains
            the following feature: {feature_description}.
            The feature should be roughly centered in the field of view. 
            The field of view size should always be {fov_size}. Start from 
            position (y={y_range[0]}, x={x_range[0]}), and gradually move the FOV 
            with a step size of {step_size[0]} in the y direction and 
            {step_size[1]} in the x direction, and examine the image until you 
            find the feature of interest. Positions should not go beyond 
            y={y_range[1]} and x={x_range[1]}. 
            When you find the feature of interest, report the coordinates of 
            the FOV. When calling tools, make only one call at a time. Do not make
            another call before getting the response of a previous one. When you 
            finish the search, say 'TERMINATE'.\
            """
        )
        
        self.agents.user_proxy.initiate_chat(
            self.agents.group_chat_manager,
            message=message
        )
