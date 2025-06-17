"""All routines in this module assume OpenAI-compatible API is used.
This docstring illustrates the JSON formats of messages of various roles:

# Outgoing messages

## User, system (pure text)

```json
{
    "role": "user",
    "content": "Text content of the message."
}
```

## User (with image)

```json
{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Text content of the message."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,..."
            }
        }
    ]
}
```

## Tool

```json
{
    "role": "tool",
    "content": "Tool response.",
    "tool_call_id": "tool_call_id"
}
```

# Incoming messages (from AI)

The direct output of the OpenAI API is a JSON object with a `choices` key, 
which contains a list of `message` objects. We assume there is only one choice.

## Text response

```json
{
    "role": "assistant",
    "content": "Text content of the response."
}
```

## Tool call

```json
{
    "role": "assistant",
    "tool_calls": [
        {
            "id": "tool_call_id",
            "type": "function",
            "function": {
                "name": "function_name",
                "arguments": "{\"argument_name\": \"argument_value\", ...}"
            }
        }
    ]
}
"""

import inspect
import typing
import json

import numpy as np
from typing import (
    get_type_hints, Callable, Dict, Any, Literal, List, Tuple, Optional
)

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from eaa.tools.base import ToolReturnType
from eaa.util import encode_image_base64, get_image_path_from_text


class ToolManager:

    def __init__(self):
        self.tools: List[Dict[str, Any]] = []
    
    def get_all_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool.
        """
        return [generate_openai_tool_schema(tool["name"], tool["function"]) for tool in self.tools]
    
    def add_tool(self, name: str, tool_function: Callable, return_type: ToolReturnType) -> None:
        """Add a tool to the tool manager.
        
        Parameters
        ----------
        name : str
            The name of the tool.
        tool_function : Callable
            The function to be used as a tool. The function should be type-annotated,
            and have a docstring to be used as the description of the tool.
        return_type : ToolReturnType
            The type of the return value of the tool.
        """
        self.tools.append(
            {
                "name": name,
                "function": tool_function,
                "return_type": return_type,
                "schema": generate_openai_tool_schema(name, tool_function)
            }
        )
        
    def execute_tool(
        self,
        tool_name: str,
        tool_kwargs: Dict[str, Any]
    ) -> Any:
        """Execute a tool.
        
        Parameters
        ----------
        tool_name : str
            The name of the tool.
        tool_kwargs : Dict[str, Any]
            The arguments to be passed to the tool.
        """
        return self.get_tool_callable(tool_name)(**tool_kwargs)
    
    def get_tool_dict(self, tool_name: str) -> Dict[str, Any]:
        """Get the tool dictionary for a given tool name.
        """
        for tool in self.tools:
            if tool["name"] == tool_name:
                return tool
        raise ValueError(f"Tool {tool_name} not found.")
    
    def get_tool_return_type(self, tool_name: str) -> ToolReturnType:
        """Get the return type of a tool.
        """
        return self.get_tool_dict(tool_name)["return_type"]
    
    def get_tool_callable(self, tool_name: str) -> Callable:
        """Get the callable function for a given tool name.
        """
        return self.get_tool_dict(tool_name)["function"]
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the schema for a given tool name.
        """
        return self.get_tool_dict(tool_name)["schema"]


class OpenAIAgent:
    
    def __init__(
        self,
        llm_config: dict,
        system_message: str = None,
    ) -> None:
        """An agent that uses OpenAI-compatible API to generate responses.

        Parameters
        ----------
        llm_config : dict
            Configuration for the OpenAI-compatible API. It should be a dictionary with
            the following keys:
            - `model`: The name of the model.
            - `api_key`: The API key for the OpenAI-compatible API.
            - `base_url`: The base URL for the OpenAI-compatible API.
        system_message : str, optional
            The system message for the OpenAI-compatible API.
        """
        self.model = llm_config.get("model")
        self.api_key = llm_config.get("api_key", "")
        self.base_url = llm_config.get("base_url", "https://api.openai.com/v1")
        
        self.message_hooks = []
        
        self.system_messages = [
            {"role": "system", "content": system_message}
        ]
        self.tool_manager = ToolManager()
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
    def register_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Register tools with the OpenAI-compatible API.
        
        Parameters
        ----------
        tools : List[Dict[str, Any]]
            A list of dictionaries, each containing the name of the tool, the
            callable function, and the return type of the tool.
        """
        if not isinstance(tools, List):
            raise ValueError(
                "tools must be a list of dictionaries, each containing the name of the tool, the "
                "callable function, and the return type of the tool."
            )
        
        for tool_dict in tools:
            self.tool_manager.add_tool(
                name=tool_dict["name"],
                tool_function=tool_dict["function"],
                return_type=tool_dict["return_type"]
            )
        
    def receive(
        self,
        message: Optional[str | Dict[str, Any]] = None, 
        role: Literal["user", "system", "tool"] = "user",
        context: Optional[List[Dict[str, Any]]] = None,
        image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        encoded_image: Optional[str] = None,
        request_response: bool = True,
        return_full_response: bool = True,
        return_outgoing_message: bool = False,
        with_system_message: bool = True,
    ) -> str | Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]] | None:
        """Receive a message from the user and generate a response.
        
        Parameters
        ----------
        message : Optional[str | Dict[str, Any]]
        
            The new message to be sent to the AI. It should be a string or a dictionary
            complying with OpenAI's message format. To attach an image to the message,
            either provide the image as a numpy array with `image`, or provide the path
            to the image with `image_path`, or provide the base-64 encoded image with
            `encoded_image`. Alternatively, the path to the image can also be embedded
            in the message string with the following format as `<img path/to/image.png>`.
            Paths embedded in this way is only used when `image_path` is None and `message`
            is a string.
            
            If `message` is None, the function will send the chat history stored to AI
            if `with_history` is True.
            
        role : Literal["user", "system", "tool"], optional
            The role of the sender.
        context : Optional[List[Dict[str, Any]]], optional
            The context to be sent to the AI. This is a list of message dictionaries.
        image : np.ndarray, optional
            The image to be sent to the AI. Exclusive with `encoded_image` and `image_path`.
        image_path : str, optional
            The path to the image to be sent to the AI. Exclusive with `image` and `encoded_image`.
        encoded_image : str, optional
            The base-64 encoded image to be sent to the AI. Exclusive with `image` and `image_path`.
        request_response : bool, optional
            If True, the message will be sent to the AI and a response will be
            requested. Otherwise, the message will only be logged into the message history
            and will not be sent to the AI. The function returns None in this case.
        return_full_response : bool, optional
            If True, this function returns a dictionary containing the full response
            from the agent. Otherwise, only the content of the response is returned
            as a string.
        return_outgoing_message : bool, optional
            If True, the outgoing message will also be returned.
        with_system_message : bool, optional
            If True, the system message will be included in the message sent to the AI.
        Returns
        -------
        str | Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]] | None
            The response from the agent. If `return_full_response` is True, the
            response is a dictionary containing the full response from the agent.
            Otherwise, the response is a string containing the content of the response.
            If `return_outgoing_message` is True, the outgoing message will also be returned.
            If `request_response` is False, the function returns None.
        """
        if image is not None or image_path is not None or encoded_image is not None:
            if isinstance(message, dict):
                raise ValueError("`message` cannot be a dictionary if an image is provided.")
        if role == "tool" and not isinstance(message, Dict):
            raise ValueError(
                "When role is 'tool', `message` must be a dictionary of tool response that "
                "contains the tool_call_id."
            )
        if message is None and context is None:
            raise ValueError("`message` and `context` cannot be None at the same time.")
            
        # Extract image path from string message, if any.
        if image_path is None and isinstance(message, str):
            image_path, modified_message = get_image_path_from_text(message, return_text_without_image_tag=True)
            if image_path is not None:
                message = modified_message
            
        # Convert string message to JSON if it is not yet a dictionary.
        if isinstance(message, str):
            message = generate_openai_message(
                message, role=role, image=image, image_path=image_path, encoded_image=encoded_image
            )
            
        # Print message.
        if message is not None:
            print_message(message, response_requested=request_response)
            
        # Create the list of messages to send.
        sys_message = self.system_messages if with_system_message else []
        message = [message] if message is not None else []
        if context is None:
            context = []
        combined_messages = sys_message + context + message
            
        # Send messages, get response and print it.
        if request_response:
            response = self.send_message_and_get_response(combined_messages)
            print_message(response)
        
        if not request_response:
            return None
        
        returns = []
        if return_full_response:
            returns.append(dict(response))
        else:
            returns.append(response.choices[0].message.content)
        if return_outgoing_message:
            returns.append(message[0] if len(message) > 0 else None)
        return returns
    
    def send_message_and_get_response(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Send a message to the agent and get the response.
        
        Parameters
        ----------
        message : List[Dict[str, Any]]
            The list of messages to be sent to the agent.
        
        Returns
        -------
        Dict[str, Any]
            The response from the agent.
        """
        tool_schema = self.tool_manager.get_all_schema()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tool_schema if len(tool_schema) > 0 else None,
            tool_choice="auto" if len(tool_schema) > 0 else None,
        )
        response_dict = response.choices[0].message.to_dict()
        response_dict = self.process_response(
            response_dict,
            remove_empty_tool_calls_key=True,
            remove_empty_reasoning_content_key=True,
            move_reasoning_content_to_empty_content=True,
        )
        for hook in self.message_hooks:
            response_dict = hook(response_dict)
        return response_dict
    
    def process_response(
        self, 
        response: Dict[str, Any],
        remove_empty_tool_calls_key: bool = True,
        remove_empty_reasoning_content_key: bool = True,
        move_reasoning_content_to_empty_content: bool = True,
    ) -> Dict[str, Any]:
        """Process the response from the agent. Models on OpenAI or OpenRouter
        should not need these processings, but some other model providers may
        require them.
        
        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent as a dictionary.
        remove_empty_tool_calls_key : bool, optional
            If True, the "tool_calls" key will be removed if it is an empty list.
        remove_empty_reasoning_content_key : bool, optional
            If True, the "reasoning_content" key will be removed if it exists.
        move_reasoning_content_to_empty_content : bool, optional
            If True, the "reasoning_content" key will be moved to the "content" key
            if the "content" key is None.
        
        Returns
        -------
        Dict[str, Any]
            The processed response from the agent.
        """
        if remove_empty_tool_calls_key:
            if "tool_calls" in response and isinstance(response["tool_calls"], list) and len(response["tool_calls"]) == 0:
                del response["tool_calls"]
        if remove_empty_reasoning_content_key:
            if ("reasoning_content" in response 
                and response["reasoning_content"] is not None
                and len(response["reasoning_content"]) == 0
            ):
                del response["reasoning_content"]
        if move_reasoning_content_to_empty_content:
            if (
                "reasoning_content" in response
                and response["content"] is None
            ):
                response["content"] = response["reasoning_content"]
                del response["reasoning_content"]
        return response
    
    def handle_tool_call(
        self, 
        message: Dict[str, Any],
        return_tool_return_types: bool = False
    ) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], List[ToolReturnType]]:
        """Handle the tool calls in the response of the agent.
        If tool call exists, the tools will be executed and tool
        responses will be returned. This function is able to handle 
        multiple tool calls.
        
        If an exception is encountered when executing a tool, the tool
        response will be a string containing the exception message. If
        `return_tool_return_types` is True, the return type of that tool
        execution will set to be ToolReturnType.EXCEPTION.
        
        Parameters
        ----------
        message : Dict[str, Any]
            The message to handle the tool call.
        return_tool_return_types : bool
            If True, the return types of the tool calls will be also returned.
        
        Returns
        -------
        List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], List[ToolReturnType]]
            The tool responses. If `return_tool_return_types` is True, the
            return types of the tool calls will also be returned.
        """
        if not has_tool_call(message):
            if return_tool_return_types:
                return [], []
            else:
                return []
        
        responses = []
        response_types = []
        tool_call_info_list = get_tool_call_info(message, index=None)
        for tool_call_info in tool_call_info_list:
            tool_call_id = tool_call_info["id"]
            tool_name = tool_call_info["function"]["name"]
            tool_call_kwargs = json.loads(tool_call_info["function"]["arguments"])
            
            exception_encountered = False
            try:
                result = self.tool_manager.execute_tool(tool_name, tool_call_kwargs)
            except Exception as e:
                exception_encountered = True
                result = str(e)
            
            response = generate_openai_message(
                content=str(result),
                role="tool",
                tool_call_id=tool_call_id
            )
            responses.append(response)
            response_types.append(
                self.tool_manager.get_tool_return_type(tool_name) if not exception_encountered 
                else ToolReturnType.EXCEPTION
            )
            
        if return_tool_return_types:
            return responses, response_types
        return responses
    
    def register_message_hook(self, hook: Callable) -> None:
        """Register a hook function that will be called to process the message
        received from the agent.
        
        The hook function should take a dictionary of message and return a
        dictionary of processed message.
        
        Parameters
        ----------
        hook : Callable
            The hook function.
        """
        self.message_hooks.append(hook)
        
    
def to_dict(message: ChatCompletionMessage | Dict[str, Any]) -> dict:
    """Convert a ChatCompletionMessage to a dictionary.
    """
    if isinstance(message, ChatCompletionMessage):
        return message.to_dict()
    else:
        return message


def generate_openai_message(
    content: str,
    role: Literal["user", "system", "tool"] = "user",
    tool_call_id: str = None,
    image: np.ndarray = None,
    image_path: str = None,
    encoded_image: str = None
) -> Dict[str, Any]:
    """Generate a dictionary containing the message to be sent
    to the agent.
    
    Parameters
    ----------
    content : str
        The content of the message.
    role : Literal["user", "system", "tool"], optional
        The role of the sender.
    image : np.ndarray, optional
        The image to be sent to the agent. Exclusive with `encoded_image` and `image_path`.
    image_path : str, optional
        The path to the image to be sent to the agent. Exclusive with `image` and `encoded_image`.
    encoded_image : str, optional
        The base-64 encoded image to be sent to the agent. Exclusive with `image` and `image_path`.
    """
    if sum([image is not None, encoded_image is not None, image_path is not None]) > 1:
        raise ValueError("Only one of `image`, `encoded_image`, or `image_path` should be provided.")
    
    if image is not None or image_path is not None:
        encoded_image = encode_image_base64(image=image, image_path=image_path)
        
    if role == "user":
        message = {
            "role": "user",
            "content": content
        }
    elif role == "system":
        message = {
            "role": "system",
            "content": content
        }
    elif role == "tool":
        message = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id
        }
        
    if encoded_image is not None:
        message["content"] = [
            {
                "type": "text",
                "text": content
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"
                }
            }
        ]
    return message


def has_tool_call(message: dict | ChatCompletionMessage) -> bool:
    """Check if the message has a tool call.
    """
    message = to_dict(message)
    if "tool_calls" in message.keys():
        return True
    else:
        return False


def get_tool_call_info(
    message: dict | ChatCompletionMessage, 
    index: Optional[int] = 0
) -> str | List[str]:
    """Get the tool call ID from the message.
    
    Parameters
    ----------
    message : dict | ChatCompletionMessage
        The message to get the tool call ID from.
    index : int, optional
        The index of the tool call to get the ID from. If None,
        all tool calls are returned as a list.
        
    Returns
    -------
    str | List[str]
        The tool call(s).
    """
    message = to_dict(message)
    if index is None:
        return message["tool_calls"]
    else:
        return message["tool_calls"][index]

def generate_openai_tool_schema(tool_name: str, func: Callable) -> Dict[str, Any]:
    """
    Generates an OpenAI-compatible tool schema from a Python function
    with type annotations and a docstring.
    
    Parameters
    ----------
    tool_name : str
        The name of the tool.
    func : Callable
        The function to generate the tool schema from.
        
    Returns
    -------
    dict
        The OpenAI-compatible tool schema.
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""
    
    # JSON schema type mapping
    python_type_to_json = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        tuple: "array",
        dict: "object"
    }

    def resolve_json_type(py_type):
        origin = typing.get_origin(py_type)
        args = typing.get_args(py_type)
        if origin is list or origin is typing.List:
            return {
                "type": "array",
                "items": {"type": python_type_to_json.get(args[0], "string")}
            }
        return {"type": python_type_to_json.get(py_type, "string")}

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name not in type_hints:
            continue
        json_type = resolve_json_type(type_hints[name])
        description = f"{name} parameter"
        properties[name] = {**json_type, "description": description}
        if param.default == inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


def print_message(
    message: Dict[str, Any], 
    response_requested: Optional[bool] = None,
    return_string: bool = False
) -> None:
    """Print the message.
    
    Parameters
    ----------
    message : Dict[str, Any]
        The message to be printed.
    response_requested : bool, optional
        Whether a response is requested for the message.
    return_string : bool, optional
        If True, the message is returned as a string instead of printed.
    """
    color_dict = {
        "user": "\033[94m",
        "system": "\033[92m",
        "tool": "\033[93m",
        "assistant": "\033[91m"
    }
    color = color_dict[message["role"]]
    
    text = f"[Role] {message['role']}\n"
    if response_requested is not None:
        text += f"[Response requested] {response_requested}\n"
    
    if "content" in message.keys():
        text += "[Content]\n"
        if isinstance(message["content"], str):
            text += message["content"] + "\n"
        elif isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    text += item["text"] + "\n"
                elif item["type"] == "image_url":
                    text += "<image> \n"
    if "tool_calls" in message.keys():
        text += "[Tool call]\n"
        for tool_call in message["tool_calls"]:
            text += f"{tool_call['id']}: {tool_call['function']['name']}\n"
            text += f"Arguments: {tool_call['function']['arguments']}\n"
    text += "\n ========================================= \n"
    
    if return_string:
        return text
    else:
        print(f"{color}{text}\033[0m")
