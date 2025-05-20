from typing import Literal
from textwrap import dedent

import autogen

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool
from eaa.comms import get_api_key
from eaa.hooks import register_hooks


class ImagingBaseTaskManager(BaseTaskManager):
    
    class AgentGroup(dict):
        user_proxy: autogen.ConversableAgent = None
        tool_executor: autogen.ConversableAgent = None
        assistant: autogen.ConversableAgent = None
        group_chat_manager: autogen.GroupChatManager = None
        
    assistant_system_message = dedent(
        """\
        You are helping scientists at a microscopy facility to
        to find the best setup parameters for their imaging system.
        You are given the tools that adjust the imaging system based
        on given parameters, and acquire images with those parameters.
        When using tools, only make one call at a time. Do not make 
        multiple calls simultaneously.\
        """
    )
    
    def __init__(
        self,
        model_name: str = "gpt-4o", 
        model_base_url: str = None,
        tools: list[BaseTool] = [],
        speaker_selection_method: Literal["round_robin", "random", "auto"] = "auto",
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
        tools : list[BaseTool], optional
            A list of tools given to the agent.
        speaker_selection_method : Literal["round_robin", "random", "auto"], optional
            The method to select the next speaker in the group chat.
            - "round_robin": select the next speaker in a round-robin fashion.
            - "random": select the next speaker randomly.
            - "auto": let the LLM decide the next speaker. Some models might have issues
              with suggesting the speaker in the right format when used as the group chat
              manager. In that case, use "round_robin" or "random" instead.
        """
        self.speaker_selection_method = speaker_selection_method
        
        super().__init__(
            model_name=model_name, 
            model_base_url=model_base_url,
            tools=tools, 
            *args, **kwargs
        )

    def build_agents(self, *args, **kwargs) -> None:
        """Build the assistant(s)."""
        self.build_assistant(*args, **kwargs)
        self.build_user_proxy(*args, **kwargs)
        self.build_tool_executor(*args, **kwargs)
        self.build_group_chat(*args, **kwargs)
        
    def get_llm_config(self, *args, **kwargs):
        llm_config = {
            "model": self.model,
            "api_key": get_api_key(self.model, self.model_base_url),
        }
        if self.model_base_url:
            llm_config["base_url"] = self.model_base_url
        return llm_config
        
    def build_assistant(self, *args, **kwargs) -> None:
        llm_config = self.get_llm_config(*args, **kwargs)
        self.agents.assistant = autogen.ConversableAgent(
            name="assistant",
            system_message=self.assistant_system_message,
            llm_config=llm_config,
        )
        register_hooks(self.agents.assistant)
        
    def build_user_proxy(self, *args, **kwargs):
        self.agents.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            llm_config=False,
            human_input_mode="ALWAYS",
            code_execution_config={
                "use_docker": False
            },
        )
        
    def build_tool_executor(self, *args, **kwargs):
        self.agents.tool_executor = autogen.ConversableAgent(
            name="tool_executor",
            human_input_mode="NEVER",
            code_execution_config={
                "use_docker": False
            },
        )
        
    def build_group_chat(self, max_round: int = 999, *args, **kwargs):
        group_chat =  autogen.GroupChat(
            agents=[self.agents.user_proxy, self.agents.tool_executor, self.agents.assistant],
            allow_repeat_speaker=False,
            messages=[],
            max_round=max_round,
            speaker_selection_method=self.speaker_selection_method,
        )
        
        llm_config = self.get_llm_config(*args, **kwargs)
        self.agents.group_chat_manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config,
        )
        register_hooks(self.agents.group_chat_manager)
    
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
