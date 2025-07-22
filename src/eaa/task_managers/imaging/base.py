from textwrap import dedent
import logging
from typing import Optional

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool, ToolReturnType
from eaa.agents.base import print_message
from eaa.api.llm_config import LLMConfig

logger = logging.getLogger(__name__)


class ImagingBaseTaskManager(BaseTaskManager):
        
    assistant_system_message = dedent(
        """\
        You are helping scientists at a microscopy facility to
        to calibrate their imaging system and set up their experiments.
        You are given the tools that adjust the imaging system, move
        the sample stage, and acquire images.
        When using tools, only make one call at a time. Do not make 
        multiple calls simultaneously.\
        """
    )
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        tools: list[BaseTool] = (), 
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args, **kwargs
    ) -> None:
        """An agent that searches for the best setup parameters
        for an imaging system.

        Parameters
        ----------
        llm_config : LLMConfig
            The configuration for the LLM.
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
        super().__init__(
            llm_config=llm_config,
            tools=tools, 
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
    
    def build_tools(self, *args, **kwargs):
        self.register_tools(self.tools)
            
    def prerun_check(self, *args, **kwargs) -> bool:
        if len(self.agent.tool_manager.tools) == 0:
            logger.warning("No tools registered for the main agent.")
        return super().prerun_check(*args, **kwargs)
        
    def run(self, *args, **kwargs) -> None:
        """Run the task manager."""
        super().run(*args, **kwargs)
        
        self.run_fov_search(*args, **kwargs)

    def run_imaging_feedback_loop(
        self,
        initial_prompt: str,
        initial_image_path: Optional[str] = None,
        message_with_acquired_image: str = "Here is the image the tool returned.",
        max_rounds: int = 99,
        store_all_images_in_context: bool = False,
        allow_non_image_tool_responses: bool = True
    ) -> None:
        """Run an agent-involving feedback loop.
        
        The loop workflow is as follows:
        1. The agent is given an initial prompt, optionally with an image. 
        2. The agent makes a tool call; the return of the tool is 
           expected to be a path to an acquired image.
        3. The tool call is executed.
        4. The tool response is added to the context.
        5. The actual image is loaded and encoded.
        6. The tool response and image are sent to the agent.
        7. Go back to 2 and repeat until the agent responds with "TERMINATE".
        
        Each time the agent calls a tool, only one tool call is allowed. If multiple
        tool calls are made, the agent will be asked to redo the tool calls.

        Parameters
        ----------
        initial_prompt : str
            The initial prompt for the agent.
        initial_image_path : str, optional
            The initial image path for the agent.
        message_with_acquired_image : str, optional
            The message to send to the agent along with the acquired image.
        max_rounds : int, optional
            The maximum number of rounds to run.
        store_all_images_in_context : bool, optional
            Whether to store all images in the context. If False, only the image
            in the initial prompt, if any, is stored in the context. Keep this
            False to reduce the context size and save costs.
        allow_non_image_tool_responses : bool, optional
            If False, the agent will be asked to redo the tool call if it returns
            anything that is not an image path.
        """
        round = 0
        image_path = None
        response, outgoing = self.agent.receive(
            initial_prompt,
            context=self.context,
            image_path=initial_image_path,
            return_outgoing_message=True
        )
        self.update_message_history(outgoing, update_context=True, update_full_history=True)
        self.update_message_history(response, update_context=True, update_full_history=True)
        while round < max_rounds:
            if response["content"] is not None and "TERMINATE" in response["content"]:
                message = self.get_user_input(
                    "Termination condition triggered. What to do next? Type \"exit\" to exit. "
                )
                if message.lower() == "exit":
                    return
                else:
                    response, outgoing = self.agent.receive(
                        message,
                        context=self.context,
                        image_path=None,
                        return_outgoing_message=True
                    )
                    self.update_message_history(outgoing, update_context=True, update_full_history=True)
                    self.update_message_history(response, update_context=True, update_full_history=True)
                    continue
            
            tool_responses, tool_response_types = self.agent.handle_tool_call(response, return_tool_return_types=True)
            if len(tool_responses) == 1:
                tool_response = tool_responses[0]
                tool_response_type = tool_response_types[0]
                # Just save the tool response to context, but don't send yet. We will send it later;
                # that will be together with the image if the tool returns an image path.
                print_message(tool_response)
                self.update_message_history(tool_response, update_context=True, update_full_history=True)
                
                if tool_response_type == ToolReturnType.IMAGE_PATH:
                    image_path = tool_response["content"]
                    response, outgoing = self.agent.receive(
                        message_with_acquired_image,
                        image_path=image_path,
                        context=self.context,
                        return_outgoing_message=True
                    )
                elif tool_response_type == ToolReturnType.EXCEPTION:
                    response, outgoing = self.agent.receive(
                        "The tool returned an exception. Please fix the exception and try again.",
                        image_path=None,
                        context=self.context,
                        return_outgoing_message=True
                    )
                else:
                    if not allow_non_image_tool_responses:
                        response, outgoing = self.agent.receive(
                            f"The tool should return an image path, but got {str(tool_response_type)}. "
                            "Make sure you call the right tool correctly.",
                            image_path=None,
                            context=self.context,
                            return_outgoing_message=True
                        )
                    else:
                        # Tool response is already added to the context so just send it.
                        response, outgoing = self.agent.receive(
                            message=None,
                            image_path=None,
                            context=self.context,
                            return_outgoing_message=True
                        )
                self.update_message_history(outgoing, update_context=store_all_images_in_context, update_full_history=True)
                self.update_message_history(response, update_context=True, update_full_history=True)
            elif len(tool_responses) > 1:
                response, outgoing = self.agent.receive(
                    "There are more than one tool calls in your response. "
                    "Make sure you only make one call at a time. Please redo "
                    "your tool calls.",
                    image_path=None,
                    context=self.context,
                    return_outgoing_message=True
                )
                self.update_message_history(outgoing, update_context=True, update_full_history=True)
                self.update_message_history(response, update_context=True, update_full_history=True)
            else:
                response, outgoing = self.agent.receive(
                    "There is no tool call in the response. Make sure you call the tool correctly. "
                    "If you need human intervention, say \"TERMINATE\".",
                    image_path=None,
                    context=self.context,
                    return_outgoing_message=True
                )
                self.update_message_history(outgoing, update_context=True, update_full_history=True)
                self.update_message_history(response, update_context=True, update_full_history=True)
            
            round += 1
