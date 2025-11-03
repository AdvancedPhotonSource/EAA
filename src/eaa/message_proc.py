from typing import Any, Dict, List, Literal, Optional
import json
import logging

from openai.types.chat import ChatCompletionMessage
import numpy as np

from eaa.util import encode_image_base64

logger = logging.getLogger(__name__)


def to_dict(message: str | ChatCompletionMessage | dict) -> dict:
    """Convert a message to a dictionary.
    
    Parameters
    ----------
    message : str | ChatCompletionMessage | dict
        The message to convert.
    
    Returns
    -------
    dict
        The dictionary representation of the message.
    """
    if isinstance(message, dict):
        return message
    elif isinstance(message, ChatCompletionMessage):
        return message.to_dict()
    elif isinstance(message, str):
        return json.loads(message)
    else:
        raise ValueError(f"Invalid message type: {type(message)}")


def generate_openai_message(
    content: str,
    role: Literal["user", "system", "tool"] = "user",
    tool_call_id: str = None,
    image: np.ndarray = None,
    image_path: str = None,
    encoded_image: str = None
) -> Dict[str, Any]:
    """Generate a dictionary in OpenAI-compatible format 
    containing the message to be sent to the agent.

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
    if role not in ["user", "system", "tool"]:
        raise ValueError("Invalid role. Must be one of `user`, `system`, or `tool`.")

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

    Parameters
    ----------
    message : dict | ChatCompletionMessage
        A message in OpenAI-compatible format.

    Returns
    -------
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
    """Get the tool call information from the message.

    Parameters
    ----------
    message : dict | ChatCompletionMessage
        The message to get the tool call ID from. The message
        should be in OpenAI-compatible format.
    index : int, optional
        The index of the tool call to get the ID from. If None,
        all tool calls are returned as a list.

    Returns
    -------
    str | List[str]
        The tool call(s).
    """
    message = to_dict(message)
    if not has_tool_call(message):
        logger.warning("No tool call found in message.")
        return None
    
    if index is None:
        return message["tool_calls"]
    else:
        return message["tool_calls"][index]


def get_message_elements_as_text(message: Dict[str, Any]) -> Dict[str, Any]:
    """Get the elements of the message as human readable text.

    Parameters
    ----------
    message : Dict[str, Any]
        The message to get the elements from. The message
        should be in OpenAI-compatible format.

    Returns
    -------
    Dict[str, Any]
        The elements of the message.
    """
    role = message["role"]

    image = None
    content = ""
    if "content" in message.keys():
        if isinstance(message["content"], str):
            content += message["content"] + "\n"
        elif isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    content += item["text"] + "\n"
                elif item["type"] == "image_url":
                    content += "<image> \n"
                    image = item["image_url"]["url"]

    tool_calls = None
    if "tool_calls" in message.keys():
        tool_calls = ""
        for tool_call in message["tool_calls"]:
            tool_calls += f"{tool_call['id']}: {tool_call['function']['name']}\n"
            tool_calls += f"Arguments: {tool_call['function']['arguments']}\n"

    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        "image": image
    }


def get_message_elements(message: Dict[str, Any]) -> Dict[str, Any]:
    """Get the elements of the message as a structured dictionary.

    Parameters
    ----------
    message : Dict[str, Any]
        The message to get the elements from. The message
        should be in OpenAI-compatible format.

    Returns
    -------
    Dict[str, Any]
        The elements of the message.
    """
    role = message["role"]

    image = []
    content = []
    if "content" in message.keys():
        if isinstance(message["content"], str):
            content.append(message["content"])
        elif isinstance(message["content"], list):
            content = message["content"]
            for item in content:
                if item["type"] == "image_url":
                    image.append(item["image_url"]["url"])

    tool_calls = None
    if "tool_calls" in message.keys():
        tool_calls = message["tool_calls"]

    tool_response_id = None
    if "tool_call_id" in message.keys():
        tool_response_id = message["tool_call_id"]

    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        "image": image,
        "tool_response_id": tool_response_id
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
        The message to be printed. The message should be in
        OpenAI-compatible format.
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

    elements = get_message_elements_as_text(message)

    text += "[Content]\n"
    text += elements["content"] + "\n"

    if elements["tool_calls"] is not None:
        text += "[Tool call]\n"
        text += elements["tool_calls"] + "\n"

    text += "\n ========================================= \n"

    if return_string:
        return text
    else:
        print(f"{color}{text}\033[0m")


def purge_context_images(
    context: list[Dict[str, Any]], 
    keep_first_n: Optional[int] = None,
    keep_last_n: Optional[int] = None,
    keep_text: bool = True
) -> None:
    """Remove image-containing messages from the context, only keeping
    the ones in the first `keep_fist_n` and last `keep_last_n`.

    Parameters
    ----------
    context : list[Dict[str, Any]]
        The context to purge.
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
    for i, message in enumerate(context):
        elements = get_message_elements_as_text(message)
        if elements["image"] is not None:
            n_image_messages += 1
            image_message_indices.append(i)
    ind_range_to_remove = [keep_first_n, n_image_messages - keep_last_n - 1]
    new_context = []
    i_img_msg = 0
    for i, message in enumerate(context):
        if i in image_message_indices:
            if i_img_msg < ind_range_to_remove[0] or i_img_msg > ind_range_to_remove[1]:
                new_context.append(message)
            else:
                if keep_text:
                    elements = get_message_elements_as_text(message)
                    new_context.append(
                        generate_openai_message(
                            content=elements["content"],
                            role=elements["role"],
                        )
                    )
            i_img_msg += 1
        else:
            new_context.append(message)
    return new_context


def complete_unresponded_tool_calls(
    context: list[Dict[str, Any]],
) -> None:
    """Look for any tool calls from "assistant" in the context
    that are not followed by a tool response from "tool" with the
    same tool call ID. If found, a placeholder tool response message
    is added to the context.
    
    Parameters
    ----------
    context : list[Dict[str, Any]]
        The context to complete unresponded tool calls in.
    """
    tool_call_ids = []
    tool_call_responded = []
    
    for message in context:
        elements = get_message_elements(message)
        if elements["role"] == "assistant" and elements["tool_calls"] is not None:
            for tool_call in elements["tool_calls"]:
                tool_call_ids.append(tool_call["id"])
                tool_call_responded.append(False)
    
        elif elements["role"] == "tool" and elements["tool_response_id"] is not None:
            if elements["tool_response_id"] in tool_call_ids:
                tool_call_responded[tool_call_ids.index(elements["tool_response_id"])] = True
    
    for i, responded in enumerate(tool_call_responded):
        if not responded:
            context.append(
                generate_openai_message(
                    content="<Incomplete tool response>",
                    role="tool",
                    tool_call_id=tool_call_ids[i],
                )
            )
    return context
