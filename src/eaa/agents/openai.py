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
from typing import get_type_hints, Callable, Dict, Any, Literal, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from eaa.util import encode_image_base64


class ToolManager:
    
    def __init__(self):
        self.tools: List[Dict[str, Any]] = []
    
    def get_all_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool.
        """
        return [generate_openai_tool_schema(tool["name"], tool["function"]) for tool in self.tools]
    
    def add_tool(self, name: str, tool_function: Callable) -> None:
        """Add a tool to the tool manager.
        
        Parameters
        ----------
        name : str
            The name of the tool.
        tool_function : Callable
            The function to be used as a tool. The function should be type-annotated,
            and have a docstring to be used as the description of the tool.
        """
        self.tools.append(
            {
                "name": name,
                "function": tool_function,
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
        for tool in self.tools:
            if tool["name"] == tool_name:
                return tool["function"](**tool_kwargs)
        raise ValueError(f"Tool {tool_name} not found.")


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
        
        self.messages = [
            {"role": "system", "content": system_message}
        ]
        self.tool_manager = ToolManager()
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
    def register_tools(self, tools: Dict[str, Callable]) -> None:
        """Register tools with the OpenAI-compatible API.
        
        Parameters
        ----------
        tools : Dict[str, Callable]
            A dictionary of tool names and their corresponding callable functions.
        """
        if not isinstance(tools, Dict):
            raise ValueError(
                "tools must be a dictionary of tool names and their corresponding callable functions."
            )
        
        for tool_name, tool_function in tools.items():
            self.tool_manager.add_tool(tool_name, tool_function)
        
    def receive(
        self,
        message: str | Dict[str, Any], 
        role: Literal["user", "system", "tool"] = "user",
        image: np.ndarray = None,
        image_path: str = None,
        encoded_image: str = None,
        request_response: bool = True,
        return_full_response: bool = True,
        with_history: bool = True,
        store_message: bool = False,
        store_response: bool = True,
    ) -> str | Dict[str, Any] | None:
        """Receive a message from the user and generate a response.
        
        Parameters
        ----------
        message : str | Dict[str, Any]
            The message to be sent to the AI. It should be a string or a dictionary
            complying with OpenAI's message format.
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
        with_history : bool, optional
            If True, the history of the conversation is included in the message sent
            to the agent so that it has the memory of the conversation.
        store_message : bool, optional
            If True, the message will be stored in the message history. This is necessary
            for maintaining the memory of the conversation, but it is strongly recommended
            to turn it off for messages containing images or large amounts of text to control
            the speed and cost.
        store_response : bool, optional
            If True, the response of AI will be stored in the message history.
        
        Returns
        -------
        str | Dict[str, Any] | None
            The response from the agent. If `return_full_response` is True, the
            response is a dictionary containing the full response from the agent.
            Otherwise, the response is a string containing the content of the response.
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
            
        if isinstance(message, str):
            message = generate_openai_message(
                message, role=role, image=image, image_path=image_path, encoded_image=encoded_image
            )
        print_message(message, response_requested=request_response)
            
        if request_response:
            response = self.send_message_and_get_response(message, with_history=with_history)
            print_message(response)
        
        # Add incoming message to history (this must be done after send_message_and_get_response)
        # so that the message sent does not get duplicated.
        if store_message:
            self.messages.append(message)
        
        if not request_response:
            return None
        
        # Add response to history
        if store_response:
            self.messages.append(response)
        
        if return_full_response:
            return dict(response)
        else:
            return response.choices[0].message.content
    
    def send_message_and_get_response(
        self,
        message: Dict[str, Any],
        with_history: bool = True
    ) -> Dict[str, Any]:
        """Send a message to the agent and get the response.
        
        Parameters
        ----------
        message : Dict[str, Any]
            The message to be sent to the agent.
        with_history : bool, optional
            If True, the history of the conversation is included in the message sent
            to the agent so that it has the memory of the conversation.
        
        Returns
        -------
        Dict[str, Any]
            The response from the agent.
        """
        if with_history:
            messages = self.messages + [message]
        else:
            messages = self.messages[0:1] + [message]
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tool_manager.get_all_schema(),
            tool_choice="auto",
        )
        return response.choices[0].message.to_dict()
    
    def handle_tool_call(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Handle the tool call in the response of the agent.
        If tool call exists, the tool will be executed and a tool
        response will be returned. Otherwise, this function returns
        None.
        """
        if not has_tool_call(message):
            return None
        tool_call_info = get_tool_call_info(message)
        tool_call_id = tool_call_info["id"]
        tool_name = tool_call_info["function"]["name"]
        tool_call_kwargs = json.loads(tool_call_info["function"]["arguments"])
        result = self.tool_manager.execute_tool(tool_name, tool_call_kwargs)
        
        response = generate_openai_message(
            content=str(result),
            role="tool",
            tool_call_id=tool_call_id
        )
        return response
    
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


def get_tool_call_info(message: dict | ChatCompletionMessage, index=0) -> str:
    """Get the tool call ID from the message.
    """
    message = to_dict(message)
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


def print_message(message: Dict[str, Any], response_requested: Optional[bool] = None) -> None:
    """Print the message.
    
    Parameters
    ----------
    message : Dict[str, Any]
        The message to be printed.
    response_requested : bool, optional
        Whether a response is requested for the message.
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
    
    print(f"{color}{text}\033[0m")
