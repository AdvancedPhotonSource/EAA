import json


def to_dict(message: str) -> dict:
    """Convert a message to a dictionary.
    
    Parameters
    ----------
    message : str
        The message to convert.
    
    Returns
    -------
    dict
        The dictionary representation of the message.
    """
    if isinstance(message, dict):
        return message
    return json.loads(message)


def get_tool_call_kwargs(message: str) -> dict:
    """Get the tool call kwargs from a message.
    
    Parameters
    ----------
    message : str
        The message to get the tool call kwargs from.
    
    Returns
    -------
    dict
        The tool call kwargs.
    """
    message = to_dict(message)
    if "tool_calls" in message.keys():
        return to_dict(message["tool_calls"][0]["function"]["arguments"])
    return None


def get_tool_call_id(message: str) -> str:
    """Get the tool call id from a message.
    
    Parameters
    ----------
    message : str
        The message to get the tool call id from.
    
    Returns
    -------
    str
        The tool call id.
    """
    message = to_dict(message)
    if "tool_calls" in message.keys():
        return message["tool_calls"][0]["id"]
    return None


def create_tool_response(tool_call_id: str, response: str) -> dict:
    """Create a tool response.
    
    Parameters
    ----------
    tool_call_id : str
        The tool call id.
    response : str
        The response to the tool call.
    
    Returns
    -------
    dict
        The tool response.
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": response
    }
