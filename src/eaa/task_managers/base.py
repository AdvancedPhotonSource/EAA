from typing import Any, Dict, Optional, Callable, Literal, List, Sequence
import json
import sqlite3
import logging
import time

from eaa.agents.base import generate_openai_message, get_message_elements, print_message
from eaa.tools.base import BaseTool
from eaa.tools.mcp import MCPTool
from eaa.agents.openai import OpenAIAgent
from eaa.util import get_timestamp
from eaa.tools.base import ToolReturnType
from eaa.api.llm_config import LLMConfig, OpenAIConfig, AskSageConfig
from eaa.agents.memory import MemoryManagerConfig, MemoryQueryResult, VectorStore
from eaa.exceptions import MaxRoundsReached
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
        memory_config: Optional[MemoryManagerConfig] = None,
        tools: list[BaseTool] = (), 
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args,
        memory_vector_store: Optional[VectorStore] = None,
        memory_notability_filter: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        memory_formatter: Optional[Callable[[List[MemoryQueryResult]], str]] = None,
        memory_embedder: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
        **kwargs
    ):
        """The base task manager.
        
        Parameters
        ----------
        llm_config : LLMConfig
            The configuration for the LLM.
        memory_config : MemoryManagerConfig, optional
            Optional configuration for long-term memory. Use
            `MemoryManagerConfig.from_dict` to build from a mapping if needed.
        memory_vector_store, memory_notability_filter, memory_formatter, memory_embedder : optional
            Overrides propagated to the agent for custom memory behaviour.
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
        if isinstance(memory_config, dict):  # type: ignore[unreachable]
            memory_config = MemoryManagerConfig.from_dict(memory_config)  # pragma: no cover - legacy path
        self.memory_config = memory_config
        self.agent = None
        self.tools = tools
        
        self.message_db_path = message_db_path
        self.message_db_conn = None
        self.webui_user_input_last_timestamp = 0

        self._memory_vector_store = memory_vector_store
        self._memory_notability_filter = memory_notability_filter
        self._memory_formatter = memory_formatter
        self._memory_embedder = memory_embedder
        
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
            memory_config=self.memory_config,
            memory_vector_store=self._memory_vector_store,
            memory_notability_filter=self._memory_notability_filter,
            memory_formatter=self._memory_formatter,
            memory_embedder=self._memory_embedder,
        )
        self.agent.set_tool_approval_handler(self._request_tool_approval_via_task_manager)
    
    def build_tools(self, *args, **kwargs):
        if self.agent is not None:
            self.register_tools(self.tools)

    def register_tools(
        self, 
        tools: BaseTool | list[BaseTool], 
    ) -> None:
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError("Input should be a list of BaseTool objects.")
            if (
                not isinstance(tool, MCPTool) and
                (
                    not hasattr(tool, "exposed_tools")
                    or (hasattr(tool, "exposed_tools") and len(tool.exposed_tools) == 0)
                )
            ):
                raise ValueError(
                    "A subclass of BaseTool that is not MCPTool must provide at least one ExposedToolSpec in `exposed_tools`."
                )
        self.agent.register_tools(list(tools))

    def prerun_check(self, *args, **kwargs) -> bool:
        return True
    
    def update_message_history(
        self,
        message: Dict[str, Any],
        update_context: bool = True,
        update_full_history: bool = True,
        update_db: bool = True
    ) -> None:
        if update_context:
            self.context.append(message)
        if update_full_history:
            self.full_history.append(message)
        if self.message_db_conn and update_db:
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
        
    def get_user_input(
        self, 
        prompt: str = "Enter a message: ",
        display_prompt_in_webui: bool = False,
        *args, **kwargs
    ) -> str:
        """Get user input. If the task manager has a SQL message database connection,
        it will be assumed that the user input is coming from the WebUI and is relayed
        by the database. Otherwise, the user will be prompted to enter a message from
        terminal.
        
        Parameters
        ----------
        prompt : Optional[str], optional
            The prompt to display to the user in the terminal.
        display_prompt_in_webui : bool, optional
            If True, the prompt will be displayed in the WebUI.

        Returns
        -------
        str
            The user input.
        """
        if self.message_db_conn:
            logger.info("Getting user input from relay database. Please enter your message in the WebUI.")
            cursor = self.message_db_conn.cursor()
            if display_prompt_in_webui:
                self.add_message_to_db({"role": "system", "content": prompt})
            while True:
                cursor.execute("SELECT timestamp, role, content, tool_calls, image FROM messages WHERE role = 'user_webui' ORDER BY rowid")
                messages = cursor.fetchall()
                if len(messages) > 0 and int(messages[-1][0]) > self.webui_user_input_last_timestamp:
                    self.webui_user_input_last_timestamp = int(messages[-1][0])
                    return messages[-1][2]
                time.sleep(1)
        else:
            message = input(prompt)
            return message

    def _request_tool_approval_via_task_manager(self, tool_name: str, tool_kwargs: Dict[str, Any]) -> bool:
        """Relay approval requests through the task manager input channels."""
        serialized_args = json.dumps(tool_kwargs, default=str)
        prompt = (
            f"Tool '{tool_name}' requires approval before execution.\n"
            f"Arguments: {serialized_args}\n"
            "Approve? [y/N]: "
        )
        response = self.get_user_input(
            prompt,
            display_prompt_in_webui=bool(self.message_db_conn),
        )
        approved = response.strip().lower() in {"y", "yes"}
        logger.info(
            "Tool '%s' approval %s by user via task manager.",
            tool_name,
            "granted" if approved else "denied",
        )
        return approved

    def run(self, *args, **kwargs) -> None:
        self.prerun_check()
        
    def display_command_help(self) -> str:            
        s = (
            "Below are supported commands. Note that not all commands are availble "
            "in your current environment. Refer to the input prompt for available commands.\n"
            "\\exit: exit the current loop\n"
            "\\chat: enter chat mode. The task manager will always ask for you input "
            "(instead of sending workflow-determined replies to the agent) when the "
            "agent finishes its response.\n"
            "\\monitor <task description>: enter monitoring mode. The agent will perform "
            "the described task periodically. Example: `\\monitor check the content of status.txt `"
            "every 60 seconds`\n"
            "\\return: return to upper level task\n"
        )
        if self.message_db_conn:
            self.add_message_to_db({"role": "system", "content": s})
        else:
            print(s)
        return s

    def run_conversation(
        self, 
        store_all_images_in_context: bool = True, 
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
            message = self.get_user_input(
                prompt=(
                    "Enter a message (\\exit: exit; \\return: return to upper level task; "
                    "\\help: show command help): "
                )
            )
            if message.lower() == "\\exit":
                break
            elif message.lower() == "\\return":
                return
            elif message.lower().startswith("\\monitor "):
                self.enter_monitoring_mode(message.lower().split("\\monitor ")[1])
                continue
            elif message.lower() == "\\help":
                self.display_command_help()
                continue
            
            # Send message and get response
            response, outgoing_message = self.agent.receive(
                message, 
                context=self.context, 
                return_outgoing_message=True
            )
            # If message DB is used, user input should come from WebUI which writes
            # to the DB, so we don't update DB again.
            self.update_message_history(
                outgoing_message, 
                update_context=True, 
                update_full_history=True, 
                update_db=(self.message_db_conn is None)
            )
            self.update_message_history(response, update_context=True, update_full_history=True)
            
            # Handle tool calls
            tool_responses, tool_response_types = self.agent.handle_tool_call(response, return_tool_return_types=True)
            if len(tool_responses) >= 1:
                for tool_response, tool_response_type in zip(tool_responses, tool_response_types):
                    print_message(tool_response)
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

    def enter_monitoring_mode(
        self,
        task_description: str,
    ):
        local_context = []
        parsing_prompt = (
            "Parse the following task description and return a JSON object with the following fields:\n"
            "- task_description: str\n"
            "- time_interval: float; time interval in seconds\n"
            "Example: given \"check the content of status.txt every 60 seconds\", the JSON object should be:\n"
            "{\n"
            "    \"task_description\": \"check the content of status.txt\",\n"
            "    \"time_interval\": 60\n"
            "}\n"
            "If enough information is provided, return the only the JSON object. Do not respond with anything else. "
            "Do not enclose the JSON object in triple backticks."
            "If any information is missing, ask the user for clarification.\n"
            f"Task description to be parsed: {task_description}\n"
        )
        
        while True:
            response, outgoing = self.agent.receive(
                message=parsing_prompt,
                context=local_context,
                return_outgoing_message=True,
            )
            self.update_message_history(outgoing, update_context=False, update_full_history=True)
            self.update_message_history(response, update_context=False, update_full_history=True)
            local_context.append(outgoing)
            local_context.append(response)
            try:
                parsed_task_description = json.loads(response["content"])
            except json.JSONDecodeError:
                parsing_prompt = self.get_user_input(
                    prompt=f"Failed to parse the task description. Please try again. {response['content']}",
                    display_prompt_in_webui=bool(self.message_db_conn),
                )
                continue
            break
        
        self.run_monitoring(
            task_description=parsed_task_description["task_description"],
            time_interval=parsed_task_description["time_interval"],
        )

    def run_monitoring(
        self,
        task_description: str,
        time_interval: float,
        initial_prompt: Optional[str] = None,
    ):
        """Run a monitoring task.

        Parameters
        ----------
        task_description : str
            The task description.
        time_interval : float
            The time interval in seconds to run the task.
        initial_prompt : Optional[str], optional
            If provided, the default initial prompt will be overridden.
        """
        if initial_prompt is None:
            initial_prompt = (
                "You are given the following task needed to monitor the status of an experiment: "
                f"{task_description}\n"
                "Use proper trigger words in your response in the following scenarios:\n"
                "- You have checked all the statuses and everything is right - add \"TERMINATE\".\n"
                "- Something is wrong, but you have fixed it (if user instructed you to fix it) - add \"TERMINATE\".\n"
                "- Something is wrong, and you need immediate human input - add \"NEED HUMAN\".\n\n"
            )
        
        while True:
            try:
                self.run_feedback_loop(
                    initial_prompt=initial_prompt,
                    termination_behavior="return",
                )
                time.sleep(time_interval)
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt detected. Terminating monitoring task.")
                self.add_message_to_db(
                    generate_openai_message(
                        content="Keyboard interrupt detected. Terminating monitoring task.",
                        role="system",
                    )
                )
                return

    def purge_context_images(
        self, 
        keep_first_n: Optional[int] = None,
        keep_last_n: Optional[int] = None,
        keep_text: bool = True
    ) -> None:
        """Remove image-containing messages from the context, only keeping
        the ones in the first `keep_fist_n` and last `keep_last_n`.

        Parameters
        ----------
        keep_first_n, keep_last_n : int, optional
            The first and last n image-containing messages to keep. If both them
            is None, no images will be removed. If one and only one of them is None,
            it will be set to 0. If there is an overlap between the
            ranges given by `keep_first_n` and `keep_last_n`, the overlap will be
            kept. For example, if `keep_first_n` is 3 and `keep_last_n` is 3 and there
            are 5 image-containing messages in the context, all the 5 images will be
            kept.
        keep_text : bool, optional
            Whether to keep the text in image-containing messages. If True,
            these messages will be preserved with only the images removed.
            Otherwise, these messages will be removed completely.
        """
        if keep_first_n is None and keep_last_n is None:
            return
        if keep_first_n is None and keep_last_n is not None:
            keep_first_n = 0
        if keep_first_n is not None and keep_last_n is None:
            keep_last_n = 0
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
                else:
                    if keep_text:
                        elements = get_message_elements(message)
                        new_context.append(
                            generate_openai_message(
                                content=elements["content"],
                                role=elements["role"],
                            )
                        )
                i_img_msg += 1
            else:
                new_context.append(message)
        self.context = new_context
        
    def enforce_tool_call_sequence(
        self,
        expected_tool_call_sequence: list[str],
        tolerance: int = 0,
    ) -> None:
        """Check whether the current tool call is the one expected to be
        made after the last tool call as in the given sequence. If not,
        a warning is added to the context.

        Parameters
        ----------
        expected_tool_call_sequence : list[str]
            A list of tool names that are expected to be called in order.
        tolerance : int, optional
            The number of tool calls to tolerate in the expected tool call sequence.
            For example, if the actual tool call sequence is [X, B, C] while the
            expected tool call sequence is [B, C], the check will still pass if the
            tolerance is 1 because only [B, C] is used for matching. This is useful
            when the first several calls are only done once instead of iteratively.
        """
        if len(self.agent.tool_manager.tool_execution_history) <= 1:
            return
        
        n_actual_calls_to_fetch = min(
            len(self.agent.tool_manager.tool_execution_history), len(expected_tool_call_sequence)
        ) - tolerance
        
        if n_actual_calls_to_fetch <= 0:
            return
        actual_tool_call_sequence = [
            entry["tool_name"] for entry in self.agent.tool_manager.tool_execution_history[-n_actual_calls_to_fetch:]
        ]
        
        # Replicate the expected sequence so that it matches circularly.
        expanded_expected_tool_call_sequence = list(expected_tool_call_sequence) * 2
        
        # Match the actual tool call sequence with the expanded expected tool call sequence.
        for i in range(len(expanded_expected_tool_call_sequence) - len(actual_tool_call_sequence) + 1):
            if expanded_expected_tool_call_sequence[i:i+len(actual_tool_call_sequence)] == actual_tool_call_sequence:
                return
        self.context.append(
            generate_openai_message(
                content=(
                    f"The tool call sequence {actual_tool_call_sequence} is not as expected. "
                    f"Are you making the right tool calls in the right order? If this is intended "
                    f"to address an exception, ignore this message."
                ),
                role="user"
            )
        )

    def run_feedback_loop(
        self,
        initial_prompt: str,
        initial_image_path: Optional[str] = None,
        message_with_acquired_image: str = "Here is the image the tool returned.",
        max_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_past_images_to_keep_in_context: Optional[int] = None,
        allow_non_image_tool_responses: bool = True,
        hook_functions: Optional[dict[str, Callable]] = None,
        expected_tool_call_sequence: Optional[list[str]] = None,
        expected_tool_call_sequence_tolerance: int = 0,
        termination_behavior: Literal["ask", "return"] = "ask",
        max_arounds_reached_behavior: Literal["return", "raise"] = "return",
        *args, **kwargs
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
        
        Termination signals: "TERMINATE" and "NEED HUMAN". When "TERMINATE" is
        present, the function either returns or asks for user input depending on
        the setting of `termination_behavior`. When "NEED HUMAN" is present, the
        function always asks for user input.

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
        n_first_images_to_keep, n_past_images_to_keep : int, optional
            The number of first and last images to keep in the context. If both of
            them are None, all images will be kept.
        allow_non_image_tool_responses : bool, optional
            If False, the agent will be asked to redo the tool call if it returns
            anything that is not an image path.
        hook_functions : dict[str, Callable], optional
            A dictionary of hook functions to call at certain points in the loop.
            The keys specify the points where the hook functions are called, and
            the values are the callables. Allowed keys are:
            - `image_path_tool_response`: 
              args: {"img_path": str}
              return: {"response": Dict[str, Any], "outgoing": Dict[str, Any]}
              Executed when the tool response is an image path, after the tool
              response is added to the context but before the image is loaded and
              sent to the agent. When this function is given, it **replaces** the
              `agent.receive` call so be sure to send the image to the agent in
              the hook if this is intended.
        expected_tool_call_sequence : list[str], optional
            A list of tool call names that are expected to be made in the loop.
            If provided, the function will check if the tool call sequence is
            as expected. A warning is issued to the agent if the sequence is 
            not as expected.
        expected_tool_call_sequence_tolerance : int, optional
            If given, the actual tool call sequence used for matching will be the
            last `len(expected_tool_call_sequence) - tolerance` calls. For example,
            if the actual tool call sequence is [X, B, C] while the expected tool
            call sequence is [B, C], the check will still pass if the tolerance is
            1 because only [B, C] is used for matching. This is useful when the
            first several calls are only done once instead of iteratively.
        termination_behavior : Literal["ask", "return"]
            Decides what to do when the agent sends termination signal ("TERMINATE")
            in the response. If "ask", the user will be asked to provide further
            instructions. If "return", the function will return directly.
        max_arounds_reached_behavior : Literal["return", "raise"], optional
            Decides what to do when the agent reaches the maximum number of
            rounds. If "return", the function will return directly. If "raise",
            the function will raise an error.
        """
        if termination_behavior not in ["ask", "return"]:
            raise ValueError("`termination_behavior` must be either 'ask' or 'return'.")
        
        hook_functions = hook_functions or {}
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
            if response["content"] is not None:
                if "TERMINATE" in response["content"] and termination_behavior == "return":
                    return
                if (
                    ("TERMINATE" in response["content"] and termination_behavior == "ask") 
                    or "NEED HUMAN" in response["content"]
                ):
                    message = self.get_user_input(
                        prompt=(
                            "Termination condition triggered. What to do next? "
                            "(\\exit: exit; \\chat: chat mode; \\help: show command help): "
                        ),
                        display_prompt_in_webui=True
                    )
                    if message.lower() == "\\exit":
                        return
                    elif message.lower() == "\\chat":
                        self.run_conversation(store_all_images_in_context=True)
                    elif message.lower() == "\\help":
                        self.display_command_help()
                        continue
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
                if expected_tool_call_sequence is not None:
                    self.enforce_tool_call_sequence(
                        expected_tool_call_sequence,
                        expected_tool_call_sequence_tolerance
                    )
                
                tool_response = tool_responses[0]
                tool_response_type = tool_response_types[0]
                # Just save the tool response to context, but don't send yet. We will send it later;
                # that will be together with the image if the tool returns an image path.
                print_message(tool_response)
                self.update_message_history(tool_response, update_context=True, update_full_history=True)
                
                if tool_response_type == ToolReturnType.IMAGE_PATH:
                    image_path = tool_response["content"]
                    if hook_functions.get("image_path_tool_response", None) is not None:
                        response, outgoing = hook_functions["image_path_tool_response"](image_path)
                    else:
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
                if outgoing is not None:
                    self.update_message_history(outgoing, update_context=True, update_full_history=True)
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
            
            if n_past_images_to_keep_in_context is not None or n_first_images_to_keep_in_context is not None:
                n_past_images_to_keep_in_context = n_past_images_to_keep_in_context if n_past_images_to_keep_in_context is not None else 0
                n_first_images_to_keep_in_context = n_first_images_to_keep_in_context if n_first_images_to_keep_in_context is not None else 0
                self.purge_context_images(
                    keep_first_n=n_first_images_to_keep_in_context, 
                    keep_last_n=n_past_images_to_keep_in_context - 1,
                    keep_text=True
                )
            round += 1
        logger.warning(f"Maximum number of rounds ({max_rounds}) reached.")
        if max_arounds_reached_behavior == "raise":
            raise MaxRoundsReached()
        else:
            return
