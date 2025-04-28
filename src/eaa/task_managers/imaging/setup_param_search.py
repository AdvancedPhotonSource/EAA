import autogen

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool

class SetupParameterSearchTaskManager(BaseTaskManager):
    
    def __init__(
        self,
        model_name: str = "gpt-4o", 
        tools: list[BaseTool] = [],
        *args, **kwargs
    ) -> None:
        """An agent that searches for the best setup parameters
        for an imaging system.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use.
        """
        super().__init__(
            model_name=model_name, 
            tools=tools, 
            *args, **kwargs
        )
                
    def build_agents(self, *args, **kwargs) -> None:
        """Build the assistant(s)."""

        self.agents.assistant = autogen.ConversableAgent(
            name="assistant",
            system_message=
                "You are helping scientists at a microscopy facility to "
                "to find the best setup parameters for their imaging system. "
                "These include the field of view of the microscope, the beam energy, "
                "and the optics. You have tools that controls the sample stage position, "
                "the beam energy, and the optics at your disposal.",
            llm_config={
                "model": self.model,
                "api_key": self.get_api_key(),
            }
        )
        
        self.agents.user_proxy = autogen.ConversableAgent(
            name="user_proxy",
            llm_config=False,
            human_input_mode="ALWAYS"
        )

    def build_tools(self, *args, **kwargs):
        for tool in self.tools:
            self.register_tools(
                tool,
                caller=self.agents.assistant,
                executor=self.agents.user_proxy
            )

    def prerun_check(self, *args, **kwargs) -> bool:
        if self.agents.assistant.tools is None:
            raise ValueError("No tools registered for the main agent.")
        return super().prerun_check(*args, **kwargs)
        
    def run(self, *args, **kwargs) -> None:
        """Run the task manager."""
        super().run(*args, **kwargs)
        
        message = input("Enter a message to the assistant: ")
        chat_result = self.agents.user_proxy.initiate_chat(
            self.agents.assistant,
            message=message
        )
