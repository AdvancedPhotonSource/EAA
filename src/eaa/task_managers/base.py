from typing import Any, Dict, Optional
import sqlite3
import logging
import time

from eaa.agents.base import generate_openai_message, get_message_elements, print_message
from eaa.tools.base import BaseTool
from eaa.agents.openai import OpenAIAgent
from eaa.util import get_timestamp
from eaa.tools.base import ToolReturnType
from eaa.api.llm_config import LLMConfig, OpenAIConfig, AskSageConfig
try:
    from eaa.agents.asksage import AskSageAgent
except ImportError:
    logging.warning(
        "AskSage endpoint is not supported because the `asksageclient` "
        "package is not installed. To use AskSage endpoints, install it with "
        "`pip install asksageclient`."
    )
    AskSageAgent = None

logger = logging.getLogger(__name__)


class BaseTaskManager:

    assistant_system_message = ""
            
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        tools: list[BaseTool] = (), 
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args, **kwargs
    ):
        """The base task manager.
        
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
        self.context = []
        self.full_history = []
        self.llm_config = llm_config
        self.agent = None
        self.tools = tools
        
        self.message_db_path = message_db_path
        self.message_db_conn = None
        self.webui_user_input_last_timestamp = 0
        
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
                "CREATE TABLE IF NOT EXISTS messages (timestamp TEXT, role TEXT, content TEXT, tool_calls TEXT, image TEXT)"
            )
            self.message_db_conn.commit()
            
            # Set timestamp buffer to the timestamp of the last user input in the database
            # if it exists..
            cursor = self.message_db_conn.cursor()
            cursor.execute("SELECT timestamp, role, content, tool_calls, image FROM messages WHERE role = 'user_webui' ORDER BY rowid")
            messages = cursor.fetchall()
            if len(messages) > 0 and self.webui_user_input_last_timestamp == 0:
                self.webui_user_input_last_timestamp = int(messages[-1][0])
    
    def build_agent(self, *args, **kwargs):
        """Build the assistant(s)."""
        if self.llm_config is None:
            logger.info(
                "Skipping agent build because `llm_config` is not provided."
            )
            return
        
        if not isinstance(self.llm_config, LLMConfig):
            raise ValueError(
                "`llm_config` must be an instance of `LLMConfig`. The type of this "
                "config object will be used to determine the API type of the LLM."
            )
        
        agent_class = {
            OpenAIConfig: OpenAIAgent,
            AskSageConfig: AskSageAgent,
        }[type(self.llm_config)]
        
        if agent_class is None:
            raise RuntimeError(
                f"Dependencies required for {agent_class.__name__} is unavailable."
            )
        self.agent = agent_class(
            llm_config=self.llm_config.to_dict(),
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
            "INSERT INTO messages (timestamp, role, content, tool_calls, image) VALUES (?, ?, ?, ?, ?)",
            (
                str(get_timestamp(as_int=True)), 
                elements["role"], 
                elements["content"], 
                elements["tool_calls"], 
                elements["image"]
            )
        )
        self.message_db_conn.commit()
        
    def get_user_input(self, prompt: Optional[str] = None, *args, **kwargs) -> str:
        """Get user input. If the task manager has a SQL message database connection,
        it will be assumed that the user input is coming from the WebUI and is relayed
        by the database. Otherwise, the user will be prompted to enter a message from
        terminal.
        
        Parameters
        ----------
        prompt : Optional[str], optional
            The prompt to display to the user in the terminal.

        Returns
        -------
        str
            The user input.
        """
        if self.message_db_conn:
            logger.info("Getting user input from relay database. Please enter your message in the WebUI.")
            cursor = self.message_db_conn.cursor()
            while True:
                cursor.execute("SELECT timestamp, role, content, tool_calls, image FROM messages WHERE role = 'user_webui' ORDER BY rowid")
                messages = cursor.fetchall()
                if len(messages) > 0 and int(messages[-1][0]) > self.webui_user_input_last_timestamp:
                    self.webui_user_input_last_timestamp = int(messages[-1][0])
                    return messages[-1][2]
                time.sleep(1)
        else:
            if prompt is None:
                prompt = "Enter a message: "
            message = input(prompt)
            return message

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
            message = self.get_user_input(prompt="Enter a message: ")
            if message.lower() == "exit":
                break
            
            # Send message and get response
            response, outgoing_message = self.agent.receive(
                message, 
                context=self.context, 
                return_outgoing_message=True
            )
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
                # Send tool responses stored in the context
                response = self.agent.receive(
                    message=None, 
                    context=self.context, 
                    return_outgoing_message=False
                )
                self.update_message_history(response, update_context=True, update_full_history=True)

    def purge_context_images(self, keep_first_n: int = 0, keep_last_n: int = 0) -> None:
        """Remove image-containing messages from the context, only keeping
        the ones in the first `keep_fist_n` and last `keep_last_n`.

        Parameters
        ----------
        keep_fist_n : int, optional
            The first n image-containing messages to keep.
        keep_last_n : int, optional
            The last n image-containing messages to keep.
        """
        if keep_first_n < 0 or keep_last_n < 0:
            raise ValueError("`keep_fist_n` and `keep_last_n` must be non-negative.")
        n_image_messages = 0
        image_message_indices = []
        for i, message in enumerate(self.context):
            elements = get_message_elements(message)
            if elements["image"] is not None:
                n_image_messages += 1
                image_message_indices.append(i)
        ind_range_to_remove = [keep_first_n, n_image_messages - keep_last_n - 1]
        new_context = []
        i_img_msg = 0
        for i, message in enumerate(self.context):
            if i in image_message_indices:
                if i_img_msg < ind_range_to_remove[0] or i_img_msg > ind_range_to_remove[1]:
                    new_context.append(message)
                i_img_msg += 1
            else:
                new_context.append(message)
        self.context = new_context

    def run_feedback_loop(
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
