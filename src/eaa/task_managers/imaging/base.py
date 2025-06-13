from textwrap import dedent
import logging
from typing import Optional

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool, ToolReturnType


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
        tools : list[BaseTool], optional
            A list of tools given to the agent.
        """        
        super().__init__(
            model_name=model_name, 
            model_base_url=model_base_url,
            tools=tools, 
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
        store_all_images_in_context: bool = False
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
        """
        round = 0
        image_path = None
        response = self.agent.receive(
            initial_prompt,
            image_path=initial_image_path,
            store_message=True, 
            store_response=True
        )
        while round < max_rounds:
            if response["content"] is not None and "TERMINATE" in response["content"]:
                message = input(
                    "Termination condition triggered. What to do next? Type \"exit\" to exit. "
                )
                if message.lower() == "exit":
                    return
                else:
                    response = self.agent.receive(
                        message,
                        image_path=None,
                        store_message=True,
                        store_response=True,
                        request_response=True
                    )
                    continue
            
            tool_responses, tool_response_types = self.agent.handle_tool_call(response, return_tool_return_types=True)
            if len(tool_responses) == 1:
                tool_response = tool_responses[0]
                tool_response_type = tool_response_types[0]
                # Just save the tool response, but don't send yet. We will send it
                # together with the image later.
                self.agent.receive(
                    tool_response, 
                    role="tool", 
                    request_response=False,
                    store_message=True, 
                    store_response=True
                )
                if not tool_response_type == ToolReturnType.IMAGE_PATH:
                    raise ValueError(
                        "The tool returned a response that is not an image path. "
                        "Make sure the tool returns an image path."
                    )
                image_path = tool_response["content"]
                response = self.agent.receive(
                    message_with_acquired_image,
                    image_path=image_path,
                    store_message=store_all_images_in_context,
                    store_response=True,
                    request_response=True
                )
            elif len(tool_responses) > 1:
                response = self.agent.receive(
                    "There are more than one tool calls in your response. "
                    "Make sure you only make one call at a time. Please redo "
                    "your tool calls.",
                    image_path=None,
                    store_message=True,
                    store_response=True,
                    request_response=True
                )
            else:
                response = self.agent.receive(
                    "There is no tool call in the response. Make sure you call the tool correctly.",
                    image_path=None,
                    store_message=True,
                    store_response=True,
                    request_response=True
                )
            
            round += 1
