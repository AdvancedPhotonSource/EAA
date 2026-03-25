from typing import Any, Dict, List, Literal, Optional, Sequence
import json
import logging

import numpy as np
from langchain_core.messages import AIMessage
from openai.types.chat import ChatCompletionMessage
from PIL import Image

from eaa_core.util import encode_image_base64

logger = logging.getLogger(__name__)


def to_dict(message: str | ChatCompletionMessage | dict) -> dict:
    """Convert a message object into a dictionary."""
    if isinstance(message, dict):
        return message
    if isinstance(message, ChatCompletionMessage):
        return message.to_dict()
    if isinstance(message, str):
        return json.loads(message)
    raise ValueError(f"Invalid message type: {type(message)}")


def generate_openai_message(
    content: str,
    role: Literal["user", "system", "tool", "assistant"] = "user",
    tool_call_id: str = None,
    image: np.ndarray | Image.Image | Sequence[np.ndarray | Image.Image] = None,
    image_path: str | Sequence[str] = None,
    encoded_image: str | Sequence[str] = None,
) -> Dict[str, Any]:
    """Generate an OpenAI-compatible message dictionary."""

    def normalize_to_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, Sequence):
            return list(value)
        return [value]

    image_objects = normalize_to_list(image)
    path_values = normalize_to_list(image_path)
    encoded_values = normalize_to_list(encoded_image)

    if any(not isinstance(path, str) for path in path_values):
        raise ValueError("`image_path` must be a string or a sequence of strings.")
    if any(not isinstance(value, str) for value in encoded_values):
        raise ValueError("`encoded_image` must be a string or a sequence of strings.")
    if sum([len(image_objects) > 0, len(path_values) > 0, len(encoded_values) > 0]) > 1:
        raise ValueError(
            "Only one image source kind can be provided: image object(s), image path(s), or base64-encoded image(s)."
        )
    if role not in ["user", "system", "tool", "assistant"]:
        raise ValueError("Invalid role.")

    if len(image_objects) > 0:
        encoded_values = [encode_image_base64(image=value) for value in image_objects]
    elif len(path_values) > 0:
        encoded_values = [encode_image_base64(image_path=value) for value in path_values]

    message: Dict[str, Any] = {"role": role, "content": content}
    if role == "tool":
        message["tool_call_id"] = tool_call_id

    if len(encoded_values) > 0:
        message_content: list[dict[str, Any]] = [{"type": "text", "text": content}]
        message_content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_value}"},
            }
            for encoded_value in encoded_values
        )
        message["content"] = message_content
    return message


def ai_message_to_openai_dict(message: AIMessage) -> Dict[str, Any]:
    """Convert a LangChain AI message into an OpenAI-compatible dict.

    Parameters
    ----------
    message : AIMessage
        LangChain assistant message to convert.

    Returns
    -------
    dict
        OpenAI-compatible assistant message payload.
    """
    payload = generate_openai_message(content=message.content or "", role="assistant")
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call.get("args", {})),
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def has_tool_call(message: dict | ChatCompletionMessage) -> bool:
    """Return whether the message contains tool calls."""
    payload = to_dict(message)
    tool_calls = payload.get("tool_calls")
    return isinstance(tool_calls, list) and len(tool_calls) > 0


def get_tool_call_info(
    message: dict | ChatCompletionMessage,
    index: Optional[int] = 0,
) -> str | List[str]:
    """Get one or all tool call payloads from a message."""
    payload = to_dict(message)
    if not has_tool_call(payload):
        logger.warning("No tool call found in message.")
        return None
    return payload["tool_calls"] if index is None else payload["tool_calls"][index]


def get_message_elements_as_text(message: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a message payload into text fields used by the WebUI DB."""
    image: list[str] = []
    content = ""
    if "content" in message and message["content"] is not None:
        if isinstance(message["content"], str):
            content += f"{message['content']}\n"
        elif isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    content += f"{item['text']}\n"
                elif item["type"] == "image_url":
                    content += "<image>\n"
                    image_url = item.get("image_url", {}).get("url")
                    if isinstance(image_url, str):
                        image.append(image_url)
    tool_calls = None
    if isinstance(message.get("tool_calls"), list):
        tool_calls = ""
        for tool_call in message["tool_calls"]:
            tool_calls += f"{tool_call['id']}: {tool_call['function']['name']}\n"
            tool_calls += f"Arguments: {tool_call['function']['arguments']}\n"
    return {
        "role": message["role"],
        "content": content,
        "tool_calls": tool_calls,
        "image": image if len(image) > 0 else None,
    }


def extract_message_text(message: Dict[str, Any]) -> str:
    """Extract textual content from an OpenAI-style message payload.

    Parameters
    ----------
    message : dict
        Message payload to read.

    Returns
    -------
    str
        Concatenated text content from the message.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        return "\n".join(part for part in parts if part)
    return ""


def get_message_elements(message: Dict[str, Any]) -> Dict[str, Any]:
    """Return structured message elements."""
    image = []
    content = []
    if "content" in message:
        if isinstance(message["content"], str):
            content.append(message["content"])
        elif isinstance(message["content"], list):
            content = message["content"]
            for item in content:
                if item["type"] == "image_url":
                    image.append(item["image_url"]["url"])
    return {
        "role": message["role"],
        "content": content,
        "tool_calls": message.get("tool_calls"),
        "image": image,
        "tool_response_id": message.get("tool_call_id"),
    }


def print_message(
    message: Dict[str, Any],
    response_requested: Optional[bool] = None,
    return_string: bool = False,
) -> Optional[str]:
    """Pretty-print a message in the terminal."""
    color_dict = {
        "user": "\033[94m",
        "system": "\033[92m",
        "tool": "\033[93m",
        "assistant": "\033[91m",
    }
    color = color_dict.get(message["role"], "")
    text = f"[Role] {message['role']}\n"
    if response_requested is not None:
        text += f"[Response requested] {response_requested}\n"
    elements = get_message_elements_as_text(message)
    text += "[Content]\n"
    text += f"{elements['content']}\n"
    if elements["tool_calls"] is not None:
        text += "[Tool call]\n"
        text += f"{elements['tool_calls']}\n"
    text += "\n ========================================= \n"
    if return_string:
        return text
    print(f"{color}{text}\033[0m")
    return None


def purge_context_images(
    context: list[Dict[str, Any]],
    keep_first_n: Optional[int] = None,
    keep_last_n: Optional[int] = None,
    keep_text: bool = True,
) -> list[Dict[str, Any]]:
    """Remove image payloads from middle context messages."""
    if keep_first_n is None and keep_last_n is None:
        return context
    keep_first_n = 0 if keep_first_n is None else keep_first_n
    keep_last_n = 0 if keep_last_n is None else keep_last_n
    if keep_first_n < 0 or keep_last_n < 0:
        raise ValueError("`keep_first_n` and `keep_last_n` must be non-negative.")

    image_indices = []
    for index, message in enumerate(context):
        if get_message_elements_as_text(message)["image"] is not None:
            image_indices.append(index)

    remove_start = keep_first_n
    remove_end = len(image_indices) - keep_last_n - 1
    new_context = []
    image_counter = 0
    for index, message in enumerate(context):
        if index in image_indices:
            if image_counter < remove_start or image_counter > remove_end:
                new_context.append(message)
            elif keep_text:
                elements = get_message_elements_as_text(message)
                new_context.append(
                    generate_openai_message(content=elements["content"], role=elements["role"])
                )
            image_counter += 1
        else:
            new_context.append(message)
    return new_context


def complete_unresponded_tool_calls(context: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Append placeholder tool responses for dangling assistant tool calls."""
    tool_call_ids = []
    responded = []
    for message in context:
        elements = get_message_elements(message)
        if elements["role"] == "assistant" and elements["tool_calls"] is not None:
            for tool_call in elements["tool_calls"]:
                tool_call_ids.append(tool_call["id"])
                responded.append(False)
        elif elements["role"] == "tool" and elements["tool_response_id"] is not None:
            if elements["tool_response_id"] in tool_call_ids:
                responded[tool_call_ids.index(elements["tool_response_id"])] = True
    for index, value in enumerate(responded):
        if not value:
            context.append(
                generate_openai_message(
                    content="<Incomplete tool response>",
                    role="tool",
                    tool_call_id=tool_call_ids[index],
                )
            )
    return context
